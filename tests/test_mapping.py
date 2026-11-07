import copy
import json

import pytest
from conftest import completed, request_body

from chatgpt_maibot.errors import PluginError
from chatgpt_maibot.mapping import (
    build_responses_body,
    parse_completed_response,
    validate_schema,
    wire_tool_name,
)


def test_required_fields_and_strict_allowlist():
    request = request_body()
    forbidden = {
        "temperature": 0.1,
        "max_tokens": 10,
        "max_output_tokens": 20,
        "top_p": 0.5,
        "metadata": {},
        "previous_response_id": "x",
        "background": True,
        "conversation": "x",
        "truncation": "auto",
        "api_key": "do-not-forward",
    }
    request.update(forbidden)
    request["model_info"]["extra_params"] = forbidden
    request["extra_params"] = {
        "body": {
            **forbidden,
            "store": True,
            "stream": False,
            "tools": [{"type": "image_generation"}],
        },
        "headers": {"Authorization": "do-not-forward"},
    }
    snapshot = copy.deepcopy(request)
    body, _ = build_responses_body(request)
    assert body == {
        "model": "test-model",
        "input": [{"role": "user", "content": "Hello"}],
        "store": False,
        "stream": True,
    }
    assert request == snapshot


def test_system_instructions_and_user_images():
    request = request_body()
    request["message_list"] = [
        {"role": "system", "content": "first"},
        {"role": "system", "content": [{"type": "text", "text": "second"}]},
        {"role": "developer", "content": "developer text"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ZmFrZQ==", "detail": "low"},
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "history"}]},
    ]
    body, _ = build_responses_body(request)
    assert body["instructions"] == "first\n\nsecond"
    assert body["input"][1]["content"] == [
        {"type": "input_text", "text": "Describe"},
        {"type": "input_image", "image_url": "data:image/png;base64,ZmFrZQ==", "detail": "low"},
    ]
    assert all(m.get("role") != "system" for m in body["input"])


@pytest.mark.parametrize(
    "block",
    [
        {"type": "input_audio", "input_audio": {"data": "fake"}},
        {"type": "video"},
        {"type": "unknown", "data": "fake"},
        {"type": "image_url", "image_url": {"url": "file:///private/image.png"}},
        {"type": "input_file", "file_id": "file_fake"},
    ],
)
def test_unsupported_inputs_fail_closed(block):
    request = request_body()
    request["message_list"][0]["content"] = [block]
    with pytest.raises(PluginError):
        build_responses_body(request)


def test_tool_namespace_history_and_name_restoration():
    names = ["weather.query", "weather_query", "天气", "1_start", "a" * 100]
    request = request_body()
    request["tool_options"] = [{"type": "function", "function": {"name": name}} for name in names]
    request["message_list"].extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": names[0], "arguments": {"city": "南京"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": {"temperature": 20}},
        ]
    )
    body, reverse = build_responses_body(request)
    namespace = body["tools"][0]
    assert namespace["type"] == "namespace" and namespace["name"] == "maibot"
    wires = [tool["name"] for tool in namespace["tools"]]
    assert len(set(wires)) == len(names)
    assert body["input"][1]["name"] == wires[0]
    assert body["input"][1]["namespace"] == "maibot"
    assert body["input"][2]["type"] == "function_call_output"
    response = completed(
        output=[
            {
                "type": "function_call",
                "namespace": "maibot",
                "call_id": "call_b",
                "name": wires[0],
                "arguments": '{"city":"南京"}',
            }
        ]
    )["response"]
    parsed = parse_completed_response(response, reverse)
    assert parsed["tool_calls"][0]["function"]["name"] == names[0]
    request["tool_options"].reverse()
    assert build_responses_body(request)[0]["input"][1]["name"] == wires[0]


def test_wire_names_do_not_collide_with_reserved_prefix():
    wire = wire_tool_name("tool.name")
    assert wire_tool_name(wire) != wire
    assert wire_tool_name("normal_name") == "normal_name"
    assert len(wire_tool_name("长" * 200)) <= 64


def test_historical_tool_name_stable_when_current_tool_options_absent():
    request = request_body()
    request["message_list"].append(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_old", "function": {"name": "old.tool", "arguments": "{}"}},
            ],
        }
    )
    body, _ = build_responses_body(request)
    assert body["input"][1]["name"] == wire_tool_name("old.tool")


@pytest.mark.parametrize(
    "tool", [{"type": "mcp"}, {"type": "image_generation"}, {"type": "tool_search"}]
)
def test_unsupported_tools_rejected(tool):
    request = request_body()
    request["tool_options"] = [tool]
    with pytest.raises(PluginError):
        build_responses_body(request)


def test_duplicate_tools_rejected():
    request = request_body()
    request["tool_options"] = [{"name": "same"}, {"name": "same"}]
    with pytest.raises(PluginError, match="重复"):
        build_responses_body(request)


def test_structured_output_and_validated_reasoning():
    request = request_body()
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    request["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "result", "schema": schema},
    }
    request["extra_params"] = {
        "body": {
            "reasoning": {"effort": "low", "summary": "auto"},
            "parallel_tool_calls": True,
            "tool_choice": "required",
        }
    }
    body, _ = build_responses_body(request)
    assert body["text"]["format"]["schema"] == schema
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert "parallel_tool_calls" not in body and "tool_choice" not in body
    enabled, _ = build_responses_body(request, enable_tool_controls=True, enable_web_search=True)
    assert enabled["parallel_tool_calls"] is True and enabled["tool_choice"] == "required"
    assert enabled["tools"] == [{"type": "web_search"}]


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "not_a_type"},
        {"$ref": "https://evil.invalid/schema"},
        {"properties": {"x": {"$dynamicRef": "http://evil.invalid"}}},
    ],
)
def test_invalid_or_remote_schema_rejected(schema):
    with pytest.raises(PluginError):
        validate_schema(schema)


@pytest.mark.parametrize(
    "reasoning", [{"effort": "invented"}, {"secret": "do-not-pass"}, {"summary": "raw"}, "high"]
)
def test_reasoning_nested_allowlist(reasoning):
    request = request_body()
    request["extra_params"] = {"reasoning": reasoning}
    with pytest.raises(PluginError):
        build_responses_body(request)


def test_refusal_reasoning_usage_are_from_completed_snapshot():
    response = completed(
        output=[
            {"type": "message", "content": [{"type": "refusal", "refusal": "Cannot comply"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Brief summary"}]},
        ]
    )["response"]
    parsed = parse_completed_response(response, {})
    assert parsed["content"] == "Cannot comply"
    assert parsed["reasoning_content"] == "Brief summary"
    assert parsed["usage"]["input_tokens"] == 5


def test_internal_citations_are_removed_from_all_model_output_paths():
    single = "citeturn0search0"
    multiple = "citeturn0search0turn0search1"
    arguments = {
        "reply_reference": f"结论 {multiple}，请直接回答。",
        "nested": [f"依据 {single}", {"unchanged": "  preserve spacing  "}],
    }
    response = completed(
        output=[
            {
                "type": "message",
                "content": [{"type": "output_text", "text": f"正文 {single}。"}],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": f"推理 {multiple} 完成"}],
            },
            {
                "type": "function_call",
                "namespace": "maibot",
                "call_id": "call_reply",
                "name": "reply",
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        ]
    )["response"]

    parsed = parse_completed_response(response, {"reply": "reply"})

    assert parsed["content"] == "正文。"
    assert parsed["reasoning_content"] == "推理 完成"
    cleaned = parsed["tool_calls"][0]["function"]["arguments"]
    assert cleaned["reply_reference"] == "结论，请直接回答。"
    assert cleaned["nested"][0] == "依据"
    assert cleaned["nested"][1]["unchanged"] == "  preserve spacing  "
    assert parsed["output_items"][-1]["tool_call"]["args"] == cleaned


def test_unknown_tool_cannot_be_sent_to_host():
    response = completed(
        output=[
            {
                "type": "function_call",
                "call_id": "call_x",
                "name": "not_declared",
                "arguments": "{}",
            }
        ]
    )["response"]
    with pytest.raises(PluginError, match="未声明"):
        parse_completed_response(response, {})
