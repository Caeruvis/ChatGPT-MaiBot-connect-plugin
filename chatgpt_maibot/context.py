"""MaiBot Context Item schema v1 snapshots, without importing any Host internals."""

from typing import Any

from .errors import PluginError

ITEM_SCHEMA_VERSION = 1


def context_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    if request.get("item_schema_version") != ITEM_SCHEMA_VERSION:
        raise PluginError("不支持此 MaiBot Context Item schema 版本。")
    context = request["context_items"]
    if not isinstance(context, list):
        raise PluginError("context_items 必须是列表。")
    messages = []
    for item in context:
        if not isinstance(item, dict):
            raise PluginError("Context Item 必须是字典。")
        kind = item.get("item_type")
        roles = {
            "SystemMessageItem": "system",
            "UserMessageItem": "user",
            "AssistantMessageItem": "assistant",
        }
        if kind in roles:
            role = roles[kind]
            raw_parts = item.get("parts")
            if not isinstance(raw_parts, list):
                raise PluginError("Context Message 缺少 parts 列表。")
            parts = []
            for part in raw_parts:
                if not isinstance(part, dict):
                    raise PluginError("Context part 必须是字典。")
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part.get("text")})
                elif part.get("type") == "refusal" and role == "assistant":
                    parts.append({"type": "text", "text": part.get("refusal")})
                elif part.get("type") == "image" and role == "user":
                    image_format, image_base64 = part.get("image_format"), part.get("image_base64")
                    if (
                        not isinstance(image_format, str)
                        or not isinstance(image_base64, str)
                        or not image_base64
                    ):
                        raise PluginError("Context 图片缺少有效格式或 base64。")
                    mime = image_format.lower().replace("jpg", "jpeg")
                    if mime not in {"jpeg", "png", "webp", "gif"}:
                        raise PluginError("不支持此 Context 图片格式。")
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{image_base64}"},
                        }
                    )
                else:
                    raise PluginError("不支持此 Context part。")
            message = {"role": role, "content": parts}
            if role == "assistant" and item.get("phase") is not None:
                message["phase"] = item["phase"]
            messages.append(message)
        elif kind == "FunctionCallItem":
            call = item.get("tool_call")
            if not isinstance(call, dict) or not isinstance(call.get("args"), dict):
                raise PluginError("Context function call 缺少有效 tool_call 或 args。")
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call.get("call_id"),
                            "function": {"name": call.get("func_name"), "arguments": call["args"]},
                        }
                    ],
                }
            )
        elif kind == "FunctionCallOutputItem":
            messages.append(
                {"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output")}
            )
        elif kind in {"ReasoningItem", "ProviderActivityItem", "ProviderOpaqueItem"}:
            # Host snapshots carry no provider replay fragments. Like Host's native Responses
            # portable projection, do not invent opaque provider state from display summaries.
            continue
        else:
            raise PluginError("不支持此 Context Item 类型。")
    return messages
