"""One fixed-prompt request against an installed plugin; print structural diagnostics only.

This standalone helper does not update the installed plugin, read chat logs, print
response text or expose credentials. Normal OAuth refresh may update its own token file.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EVENT_TYPES = {
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.refusal.delta",
    "response.refusal.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.completed",
    "response.failed",
    "response.incomplete",
    "error",
}
ITEM_TYPES = {
    "message",
    "reasoning",
    "function_call",
    "web_search_call",
    "custom_tool_call",
    "mcp_call",
    "image_generation_call",
    "computer_call",
    "file_search_call",
    "code_interpreter_call",
    "shell_call",
}
PART_TYPES = {"output_text", "text", "input_text", "refusal", "reasoning_text", "summary_text"}
RESPONSE_KEYS = {
    "id",
    "object",
    "status",
    "model",
    "output",
    "output_text",
    "usage",
    "error",
    "incomplete_details",
    "text",
    "reasoning",
    "choices",
    "data",
    "result",
    "response",
}
ITEM_KEYS = {
    "type",
    "id",
    "role",
    "status",
    "phase",
    "content",
    "summary",
    "text",
    "output_text",
    "name",
    "arguments",
    "call_id",
    "namespace",
    "encrypted_content",
}


def _enum(value: Any, allowed: set[str]) -> str:
    return value if isinstance(value, str) and value in allowed else "unknown"


def _length(value: Any) -> int | None:
    return len(value) if isinstance(value, (str, list, dict)) else None


def item_shape(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"is_object": False}
    result = {
        "type": _enum(item.get("type"), ITEM_TYPES),
        "status": _enum(item.get("status"), {"completed", "in_progress", "incomplete", "failed"}),
        "role": _enum(item.get("role"), {"assistant", "user", "system", "developer", "tool"}),
        "present_fields": sorted(ITEM_KEYS.intersection(item)),
        "unknown_field_count": len(set(item) - ITEM_KEYS),
        "text_chars": _length(item.get("text")),
        "output_text_chars": _length(item.get("output_text")),
    }
    for key in ("content", "summary"):
        parts = item.get(key)
        if isinstance(parts, list):
            result[key] = {
                "count": len(parts),
                "parts": [
                    {
                        "type": _enum(part.get("type"), PART_TYPES),
                        "text_chars": _length(part.get("text")),
                        "refusal_chars": _length(part.get("refusal")),
                    }
                    if isinstance(part, dict)
                    else {"is_object": False}
                    for part in parts[:8]
                ],
            }
        else:
            result[key] = {
                "is_list": False,
                "is_string": isinstance(parts, str),
                "length": _length(parts),
            }
    return result


def response_shape(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"is_object": False}
    output = response.get("output")
    result = {
        "present_fields": sorted(RESPONSE_KEYS.intersection(response)),
        "unknown_field_count": len(set(response) - RESPONSE_KEYS),
        "status": _enum(
            response.get("status"),
            {"completed", "in_progress", "incomplete", "failed", "cancelled"},
        ),
        "output_is_list": isinstance(output, list),
        "output_count": len(output) if isinstance(output, list) else None,
        "output_text_chars": _length(response.get("output_text")),
        "has_error": response.get("error") is not None,
        "has_incomplete_details": response.get("incomplete_details") is not None,
    }
    if isinstance(output, list):
        result["output_items"] = [item_shape(item) for item in output[:12]]
    usage = response.get("usage")
    if isinstance(usage, dict):
        result["usage"] = {
            key: value
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance(value := usage.get(key), int)
            and not isinstance(value, bool)
            and 0 <= value <= 10**9
        }
    return result


class StreamTrace:
    def __init__(self):
        self.events = Counter()
        self.item_done = []
        self.text_delta_chars = 0
        self.text_done_chars = 0
        self.completed = None
        self._response_id = None
        self.ids_consistent = True

    def observe(self, event: dict[str, Any]) -> None:
        kind = _enum(event.get("type"), EVENT_TYPES)
        self.events[kind] += 1
        response = event.get("response")
        response_id = response.get("id") if isinstance(response, dict) else event.get("response_id")
        if isinstance(response_id, str) and response_id:
            if self._response_id is None:
                self._response_id = response_id
            elif self._response_id != response_id:
                self.ids_consistent = False
        if kind == "response.output_item.done" and len(self.item_done) < 12:
            self.item_done.append(item_shape(event.get("item")))
        if kind == "response.output_text.delta" and isinstance(event.get("delta"), str):
            self.text_delta_chars += len(event["delta"])
        if kind == "response.output_text.done" and isinstance(event.get("text"), str):
            self.text_done_chars += len(event["text"])
        if kind == "response.completed":
            self.completed = response_shape(response)

    def report(self):
        return {
            "event_counts": dict(self.events),
            "text_delta_chars": self.text_delta_chars,
            "text_done_chars": self.text_done_chars,
            "output_item_done": self.item_done,
            "completed_snapshot": self.completed,
            "has_response_id": self._response_id is not None,
            "response_ids_consistent": self.ids_consistent,
        }


async def diagnose(args) -> tuple[dict[str, Any], int]:
    plugin_dir = args.plugin_dir.expanduser().resolve()
    sys.path.insert(0, str(plugin_dir))
    package = importlib.import_module("chatgpt_maibot")
    loaded_path = await asyncio.to_thread(Path(package.__file__).resolve)
    if not loaded_path.is_relative_to(plugin_dir / "chatgpt_maibot"):
        raise ValueError("加载位置与指定插件目录不符。")
    auth_module = importlib.import_module("chatgpt_maibot.auth")
    provider_module = importlib.import_module("chatgpt_maibot.provider")
    errors = importlib.import_module("chatgpt_maibot.errors")
    store_module = importlib.import_module("chatgpt_maibot.store")
    httpx = importlib.import_module("httpx")
    version = getattr(package, "VERSION", "unknown")
    report = {
        "plugin_version": version
        if isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version)
        else "unknown",
        "fixed_prompt": True,
        "web_search": False,
    }
    trace = StreamTrace()
    original_events = provider_module.sse_events

    async def watched_events(response):
        async for event in original_events(response):
            trace.observe(event)
            yield event

    provider_module.sse_events = watched_events
    auth = None
    code = 1
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30, connect=15), follow_redirects=False
        ) as client:
            auth = auth_module.OpenAIPlanAuth(store_module.CredentialStore(args.data_dir), client)
            await auth.initialize()
            provider = provider_module.PlanProvider(auth, client)
            result = await provider.response(
                {
                    "model_info": {"model_identifier": args.model},
                    "message_list": [{"role": "user", "content": "请只回复 CONNECT_OK。"}],
                },
                enable_web_search=False,
            )
            content = result.get("content")
            report["result"] = {
                "accepted": True,
                "content_chars": _length(content),
                "matches_connect_ok": isinstance(content, str) and content.strip() == "CONNECT_OK",
            }
            code = 0
    except errors.PluginError:
        report["result"] = {"accepted": False, "error_category": "plugin_error"}
    except Exception:
        # Unexpected exception messages may contain HTTP headers or local credentials.
        report["result"] = {"accepted": False, "error_category": "unexpected_error"}
    finally:
        provider_module.sse_events = original_events
        if auth is not None:
            await auth.cancel_login()
        report["stream"] = trace.report()
    return report, code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        report, code = asyncio.run(diagnose(args))
    except KeyboardInterrupt:
        print("已取消。")
        return 130
    except Exception:
        print("无法加载指定插件或依赖；请检查 --plugin-dir 与 Python 环境。")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
