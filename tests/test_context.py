import pytest
from conftest import completed

from chatgpt_maibot.errors import PluginError
from chatgpt_maibot.mapping import build_responses_body, parse_completed_response, wire_tool_name


def native_request(items):
    return {
        "item_schema_version": 1,
        "context_items": items,
        "model_info": {"model_identifier": "test-model"},
    }


def test_native_context_items_include_images_and_function_history():
    request = native_request(
        [
            {"item_type": "SystemMessageItem", "parts": [{"type": "text", "text": "rules"}]},
            {
                "item_type": "UserMessageItem",
                "parts": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image", "image_format": "jpg", "image_base64": "ZmFrZQ=="},
                ],
            },
            {
                "item_type": "FunctionCallItem",
                "tool_call": {
                    "call_id": "call_1",
                    "func_name": "weather.query",
                    "args": {"city": "南京"},
                },
            },
            {
                "item_type": "FunctionCallOutputItem",
                "call_id": "call_1",
                "output": "晴",
                "success": True,
                "tool_name": "weather.query",
            },
            {
                "item_type": "AssistantMessageItem",
                "parts": [{"type": "refusal", "refusal": "No"}],
                "phase": "final_answer",
            },
        ]
    )
    body, _ = build_responses_body(request)
    assert body["instructions"] == "rules"
    assert body["input"][0]["content"][1]["image_url"] == "data:image/jpeg;base64,ZmFrZQ=="
    assert body["input"][1]["name"] == wire_tool_name("weather.query")
    assert body["input"][2] == {"type": "function_call_output", "call_id": "call_1", "output": "晴"}
    assert body["input"][3]["phase"] == "final_answer"


@pytest.mark.parametrize("version", [None, 0, 2, "1"])
def test_unknown_native_schema_cannot_fall_back_to_legacy(version):
    request = native_request([])
    request["item_schema_version"] = version
    request["message_list"] = [{"role": "user", "content": "must not fall back"}]
    with pytest.raises(PluginError, match="schema"):
        build_responses_body(request)


def test_display_only_provider_items_are_not_invented_as_replay():
    request = native_request(
        [
            {
                "item_type": "ReasoningItem",
                "representation": "opaque",
                "summary_parts": [],
                "text_parts": [],
            },
            {"item_type": "ProviderActivityItem", "provider_type": "web_search"},
            {"item_type": "ProviderOpaqueItem", "provider_type": "other-provider"},
            {"item_type": "UserMessageItem", "parts": [{"type": "text", "text": "Next"}]},
        ]
    )
    body, _ = build_responses_body(request)
    assert len(body["input"]) == 1
    request["context_items"].append({"item_type": "UnsupportedFutureItem"})
    with pytest.raises(PluginError):
        build_responses_body(request)


@pytest.mark.parametrize("format_type", ["text", "json_object", "json_schema"])
def test_native_response_format_snapshot(format_type):
    request = native_request(
        [{"item_type": "UserMessageItem", "parts": [{"type": "text", "text": "JSON"}]}]
    )
    request["response_format"] = {"format_type": format_type}
    if format_type == "json_schema":
        request["response_format"]["schema"] = {
            "name": "native_result",
            "schema": {"type": "object"},
            "strict": False,
        }
    body, _ = build_responses_body(request)
    if format_type == "text":
        assert "text" not in body
    else:
        assert body["text"]["format"]["type"] == format_type


def test_native_output_items_and_host_usage_fields_preserve_arguments():
    wire = wire_tool_name("weather.query")
    response = completed(
        output=[
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Plan"}]},
            {
                "type": "function_call",
                "namespace": "maibot",
                "call_id": "call_1",
                "name": wire,
                "arguments": '{"city":"南京"}',
            },
        ],
        usage={
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "input_tokens_details": {"cached_tokens": 3},
        },
    )["response"]
    result = parse_completed_response(response, {wire: "weather.query"})
    assert result["item_schema_version"] == 1
    assert result["output_items"][0]["item_type"] == "ReasoningItem"
    assert result["output_items"][1]["tool_call"]["args"] == {"city": "南京"}
    assert result["tool_calls"][0]["function"]["arguments"] == {"city": "南京"}
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 4
    assert result["usage"]["prompt_cache_hit_tokens"] == 3
    assert result["usage"]["prompt_cache_miss_tokens"] == 7


@pytest.mark.parametrize("arguments", ["not-json", "[]", "null", '"text"'])
def test_bad_json_arguments_fail_before_host_execution(arguments):
    response = completed(
        output=[
            {
                "type": "function_call",
                "name": "weather",
                "call_id": "call_1",
                "arguments": arguments,
            }
        ]
    )["response"]
    with pytest.raises(PluginError, match="JSON object"):
        parse_completed_response(response, {"weather": "weather"})
