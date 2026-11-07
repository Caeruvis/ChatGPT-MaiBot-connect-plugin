"""Explicit conversion of MaiBot's Chat Completions snapshots to Responses."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, SchemaError

from .context import ITEM_SCHEMA_VERSION, context_messages
from .errors import PluginError

_INTERNAL_CITATION_RE = re.compile(r"cite[^\r\n]+")


def clean_internal_citations(text: str) -> str:
    """Remove ChatGPT UI citation placeholders that MaiBot cannot render."""
    cleaned = _INTERNAL_CITATION_RE.sub("", text)
    if cleaned == text:
        return text
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?，。；：！？])", r"\1", cleaned)
    return cleaned.strip()


def clean_internal_citations_in_value(value: Any) -> Any:
    """Clean citation placeholders from nested model-generated tool arguments."""
    if isinstance(value, str):
        return clean_internal_citations(value)
    if isinstance(value, list):
        return [clean_internal_citations_in_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_internal_citations_in_value(item) for key, item in value.items()}
    return value


def require_string(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise PluginError(f"{label} 必须是有效字符串。")
    return value


def text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise PluginError("文本消息必须是字符串或文本块列表。")
    parts = []
    for block in value:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text"}:
            parts.append(require_string(block.get("text"), "文本块", empty=True))
        else:
            raise PluginError("该角色只支持文本消息块，未知块未发送。")
    return "\n".join(parts)


def user_content(value: Any) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise PluginError("用户消息必须是字符串或消息块列表。")
    result = []
    for block in value:
        if isinstance(block, str):
            result.append({"type": "input_text", "text": block})
        elif isinstance(block, dict):
            kind = block.get("type")
            if kind in {"text", "input_text"}:
                result.append(
                    {
                        "type": "input_text",
                        "text": require_string(block.get("text"), "文本块", empty=True),
                    }
                )
            elif kind in {"image_url", "input_image", "image"}:
                image = block.get("image_url", block.get("url"))
                detail = block.get("detail", "auto")
                if isinstance(image, dict):
                    detail, image = image.get("detail", detail), image.get("url")
                image = require_string(image, "图片 URL")
                if not (
                    image.startswith("https://")
                    or re.match(r"^data:image/(png|jpeg|webp|gif);base64,", image)
                ):
                    raise PluginError("图片仅接受 HTTPS URL 或受支持的 base64 data URL。")
                if detail not in {"auto", "low", "high", "original"}:
                    raise PluginError("图片 detail 无效。")
                result.append({"type": "input_image", "image_url": image, "detail": detail})
            elif kind == "input_file":
                sources = [key for key in ("file_data", "file_url", "file_id") if block.get(key)]
                if len(sources) != 1 or sources[0] == "file_id":
                    raise PluginError(
                        "文件必须使用内联 file_data 或 HTTPS file_url；不使用 Files API。"
                    )
                key = sources[0]
                source = require_string(block[key], "文件来源")
                if key == "file_url" and urlsplit(source).scheme != "https":
                    raise PluginError("文件 URL 必须使用 HTTPS。")
                converted = {"type": "input_file", key: source}
                if "filename" in block:
                    converted["filename"] = require_string(block["filename"], "文件名")
                result.append(converted)
            else:
                raise PluginError("不支持该用户消息块；音频、视频及未知块未发送。")
        else:
            raise PluginError("用户消息块必须是字符串或字典。")
    return result


def wire_tool_name(original: str) -> str:
    """Stable across requests and tool ordering, including history-only tool names."""
    require_string(original, "工具名")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", original) and not original.startswith("mbt_"):
        return original
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", original)[:35]
    return "mbt_" + stem + "_" + hashlib.sha256(original.encode()).hexdigest()[:20]


def validate_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise PluginError("JSON Schema 必须是字典。")
    try:
        Draft202012Validator.check_schema(schema)
    except (SchemaError, TypeError, ValueError, RecursionError):
        raise PluginError("JSON Schema 无效。") from None
    # No network resolution of untrusted remote schema references.
    pending = [schema]
    count = 0
    while pending:
        value = pending.pop()
        count += 1
        if count > 20000:
            raise PluginError("JSON Schema 过大。")
        if isinstance(value, dict):
            for key in ("$ref", "$dynamicRef"):
                if key in value and (
                    not isinstance(value[key], str) or not value[key].startswith("#")
                ):
                    raise PluginError("JSON Schema 仅允许本地引用。")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return schema


def convert_tools(options: Any):
    if options is None:
        return [], {}
    if not isinstance(options, list):
        raise PluginError("tool_options 必须是列表。")
    functions, reverse = [], {}
    for option in options:
        if not isinstance(option, dict) or option.get("type", "function") != "function":
            raise PluginError("仅支持本地 function 工具，托管工具未发送。")
        definition = option.get("function", option)
        if not isinstance(definition, dict):
            raise PluginError("工具 function 必须是字典。")
        original = require_string(definition.get("name"), "工具名")
        wire = wire_tool_name(original)
        if wire in reverse:
            raise PluginError("工具名重复或冲突。")
        reverse[wire] = original
        schema = validate_schema(definition.get("parameters", {"type": "object", "properties": {}}))
        strict = definition.get("strict", False)
        if not isinstance(strict, bool):
            raise PluginError("工具 strict 必须为布尔值。")
        functions.append(
            {
                "type": "function",
                "name": wire,
                "description": require_string(
                    definition.get("description", ""), "工具描述", empty=True
                ),
                "parameters": schema,
                "strict": strict,
            }
        )
    return (
        [
            {
                "type": "namespace",
                "name": "maibot",
                "description": "MaiBot 本地工具",
                "tools": functions,
            }
        ]
        if functions
        else []
    ), reverse


def structured_format(value: Any):
    if isinstance(value, dict) and "format_type" in value:
        # MaiBot's native RespFormat.to_dict() differs from Chat Completions.
        value = {"type": value["format_type"], "json_schema": value.get("schema")}
    if value is None or (isinstance(value, dict) and value.get("type") == "text"):
        return None
    if not isinstance(value, dict):
        raise PluginError("response_format 必须是字典。")
    if value.get("type") == "json_object":
        return {"format": {"type": "json_object"}}
    info = value.get("json_schema")
    if value.get("type") != "json_schema" or not isinstance(info, dict):
        raise PluginError("不支持此 response_format。")
    name = require_string(info.get("name", "maibot_response"), "结构化输出名称")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise PluginError("结构化输出名称无效。")
    strict = info.get("strict", True)
    if not isinstance(strict, bool):
        raise PluginError("结构化输出 strict 必须为布尔值。")
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": validate_schema(info.get("schema")),
            "strict": strict,
        }
    }


def build_responses_body(
    request: dict[str, Any], *, enable_web_search: bool = False, enable_tool_controls: bool = False
):
    if not isinstance(request, dict) or not isinstance(request.get("model_info"), dict):
        raise PluginError("请求缺少 model_info。")
    info = request["model_info"]
    model = require_string(info.get("model_identifier"), "model_identifier")
    tools, reverse = convert_tools(request.get("tool_options"))
    if enable_web_search:
        tools.append({"type": "web_search"})
    messages = (
        context_messages(request) if "context_items" in request else request.get("message_list")
    )
    if not isinstance(messages, list):
        raise PluginError("message_list 必须是列表。")
    instructions, items = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise PluginError("消息必须是字典。")
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            instructions.append(text_content(content))
            continue
        if role == "tool":
            call_id = require_string(
                message.get("tool_call_id") or message.get("call_id"), "工具结果 call ID"
            )
            output = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False, allow_nan=False)
            )
            items.append({"type": "function_call_output", "call_id": call_id, "output": output})
            continue
        if role not in {"user", "assistant", "developer"}:
            raise PluginError("不支持该消息角色。")
        converted = user_content(content) if role == "user" else text_content(content)
        if converted:
            message_item = {"role": role, "content": converted}
            if role == "assistant" and message.get("phase") is not None:
                if message["phase"] not in ("commentary", "final_answer"):
                    raise PluginError("assistant phase 无效。")
                message_item["phase"] = message["phase"]
            items.append(message_item)
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list) or (calls and role != "assistant"):
            raise PluginError("tool_calls 只能位于 assistant 消息中。")
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise PluginError("历史工具调用格式无效。")
            function = call["function"]
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
            items.append(
                {
                    "type": "function_call",
                    "namespace": "maibot",
                    "call_id": require_string(
                        call.get("id") or call.get("call_id"), "历史工具 call ID"
                    ),
                    "name": wire_tool_name(function.get("name")),
                    "arguments": arguments,
                }
            )
    if not items:
        raise PluginError("Responses input 为空。")
    body = {"model": model, "input": items, "store": False, "stream": True}
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    if tools:
        body["tools"] = tools
    structured = structured_format(request.get("response_format"))
    if structured:
        body["text"] = structured
    for source in (info.get("extra_params"), request.get("extra_params")):
        if not isinstance(source, dict):
            continue
        candidates = [source]
        if isinstance(source.get("body"), dict):
            candidates.append(source["body"])
        for candidate in candidates:
            if "reasoning" in candidate:
                reasoning = candidate["reasoning"]
                if not isinstance(reasoning, dict) or set(reasoning) - {"effort", "summary"}:
                    raise PluginError("reasoning 仅支持 effort 与 summary。")
                if "effort" in reasoning and reasoning["effort"] not in {
                    "none",
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                    "ultra",
                }:
                    raise PluginError("reasoning.effort 无效。")
                if "summary" in reasoning and reasoning["summary"] not in {
                    "auto",
                    "concise",
                    "detailed",
                }:
                    raise PluginError("reasoning.summary 无效。")
                body["reasoning"] = dict(reasoning)
            if enable_tool_controls:
                if "parallel_tool_calls" in candidate:
                    if not isinstance(candidate["parallel_tool_calls"], bool):
                        raise PluginError("parallel_tool_calls 必须为布尔值。")
                    body["parallel_tool_calls"] = candidate["parallel_tool_calls"]
                if "tool_choice" in candidate:
                    if candidate["tool_choice"] not in ("auto", "none", "required"):
                        raise PluginError("tool_choice 仅支持 auto、none、required。")
                    body["tool_choice"] = candidate["tool_choice"]
    return body, reverse


def parse_completed_response(response: dict[str, Any], reverse: dict[str, str]):
    if response.get("status") != "completed" or not isinstance(response.get("output"), list):
        raise PluginError("completed 事件中的响应快照无效。")
    text, reasoning, calls, output_items = [], [], [], []
    try:
        for item in response["output"]:
            kind = item.get("type")
            if kind == "message":
                parts = []
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        value = clean_internal_citations(
                            require_string(block.get("text"), "输出文本", empty=True)
                        )
                        text.append(value)
                        if value:
                            parts.append({"type": "text", "text": value})
                    elif block.get("type") == "refusal":
                        value = clean_internal_citations(
                            require_string(block.get("refusal"), "拒绝说明", empty=True)
                        )
                        text.append(value)
                        if value:
                            parts.append({"type": "refusal", "refusal": value})
                if parts:
                    output_item = {"item_type": "AssistantMessageItem", "parts": parts}
                    if item.get("phase") in ("commentary", "final_answer"):
                        output_item["phase"] = item["phase"]
                    output_items.append(output_item)
            elif kind == "reasoning":
                summary = [
                    clean_internal_citations(
                        require_string(block.get("text"), "推理摘要", empty=True)
                    )
                    for block in item.get("summary", [])
                    if block.get("type") == "summary_text"
                ]
                reasoning.extend(summary)
                if any(summary):
                    output_items.append(
                        {
                            "item_type": "ReasoningItem",
                            "summary_parts": summary,
                            "text_parts": [],
                            "representation": "summary",
                        }
                    )
            elif kind == "function_call":
                name = item.get("name")
                if name not in reverse or item.get("namespace") not in (None, "maibot"):
                    raise PluginError("服务端返回未声明的工具调用，未交给 Host 执行。")
                arguments = require_string(item.get("arguments"), "工具参数")
                try:
                    args = clean_internal_citations_in_value(json.loads(arguments))
                    if not isinstance(args, dict):
                        raise ValueError
                except ValueError:
                    raise PluginError(
                        "工具参数必须是有效 JSON object，未交给 Host 执行。"
                    ) from None
                call_id = require_string(item.get("call_id"), "输出工具 call ID")
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": reverse[name],
                            "arguments": args,
                        },
                    }
                )
                output_items.append(
                    {
                        "item_type": "FunctionCallItem",
                        "tool_call": {
                            "call_id": call_id,
                            "func_name": reverse[name],
                            "args": args,
                            "extra_content": None,
                        },
                    }
                )
        usage = response.get("usage") or {}
        if not isinstance(usage, dict):
            raise PluginError("响应 usage 格式无效。")
        usage = dict(usage)
        for target, source in (
            ("prompt_tokens", "input_tokens"),
            ("completion_tokens", "output_tokens"),
        ):
            count = usage.get(source, 0)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise PluginError("响应 token 用量无效。")
            usage[target] = count
        details = usage.get("input_tokens_details")
        if isinstance(details, dict) and "cached_tokens" in details:
            cached = details["cached_tokens"]
            if (
                not isinstance(cached, int)
                or isinstance(cached, bool)
                or not 0 <= cached <= usage["prompt_tokens"]
            ):
                raise PluginError("响应缓存 token 用量无效。")
            usage["prompt_cache_hit_tokens"] = cached
            usage["prompt_cache_miss_tokens"] = usage["prompt_tokens"] - cached
        if not output_items:
            raise PluginError("完成快照未包含可交给 MaiBot 的文本、摘要或工具调用。")
        result = {
            "content": "".join(text),
            "reasoning_content": "\n".join(reasoning),
            "usage": usage,
            "item_schema_version": ITEM_SCHEMA_VERSION,
            "output_items": output_items,
            "status": "completed",
        }
        if isinstance(response.get("id"), str):
            result["response_id"] = response["id"]
        if calls:
            result["tool_calls"] = calls
        return result
    except (AttributeError, KeyError, TypeError):
        raise PluginError("Responses 完成快照格式无效。") from None
