import json

import httpx
import pytest
from conftest import ACCESS, completed, request_body, save_account, sse

from chatgpt_maibot.errors import OpenAIError, PluginError
from chatgpt_maibot.provider import PlanProvider


def stream_response(*events):
    return httpx.Response(
        200,
        text=sse(*events),
        headers={"content-type": "text/event-stream", "x-request-id": "req_stream"},
    )


async def test_success_uses_same_token_for_models_and_inference(env):
    await save_account(env)
    result = await env.provider.response(request_body())
    assert result["content"] == "Hello"
    models, inference = env.state.requests
    assert (
        models.headers["authorization"] == inference.headers["authorization"] == "Bearer " + ACCESS
    )
    assert inference.url == "https://api.openai.com/v1/responses"
    body = json.loads(inference.content)
    assert body["store"] is False and body["stream"] is True


async def test_model_visibility_requires_explicit_list_and_slug(env):
    await save_account(env)
    env.state.models = [
        {"slug": "hidden", "visibility": "hidden"},
        {"slug": "no-visibility"},
        {"id": "no-slug", "visibility": "list"},
        {"slug": "visible", "visibility": "list", "display_name": "Visible"},
    ]
    assert await env.provider.models() == [{"slug": "visible", "display_name": "Visible"}]
    with pytest.raises(PluginError, match="可见目录"):
        await env.provider.response(request_body())
    assert all(r.url.path != "/v1/responses" for r in env.state.requests)


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete"])
async def test_failure_after_delta_never_returns_partial_success(env, terminal):
    await save_account(env)
    env.state.response_queue = [
        stream_response(
            {"type": "response.output_text.delta", "delta": "partial SECRET CHAT"},
            {
                "type": terminal,
                "response": {
                    "status": "failed",
                    "error": {"code": "server_error", "message": "SECRET CHAT"},
                },
            },
        )
    ]
    with pytest.raises(OpenAIError) as caught:
        await env.provider.response(request_body())
    assert caught.value.status == 200
    assert caught.value.code == "server_error"
    assert caught.value.request_id == "req_stream"
    assert "SECRET CHAT" not in str(caught.value)


@pytest.mark.parametrize(
    "body",
    ["", "data: [DONE]\n\n", 'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'],
)
async def test_eof_without_completed_fails(env, body):
    await save_account(env)
    env.state.response_queue = [
        httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    ]
    with pytest.raises(PluginError, match="未收到 response.completed|未返回可确认的 SSE"):
        await env.provider.response(request_body())


async def test_sse_multiline_comments_event_name_and_utf8(env):
    await save_account(env)
    payload = json.dumps(completed("南京"), ensure_ascii=False, indent=2)
    framed = (
        ": heartbeat\r\nevent: response.completed\r\n"
        + "\r\n".join("data: " + line for line in payload.splitlines())
        + "\r\n\r\n"
    )
    env.state.response_queue = [
        httpx.Response(200, text=framed, headers={"content-type": "text/event-stream"})
    ]
    assert (await env.provider.response(request_body()))["content"] == "南京"


async def test_401_refreshes_once_and_retries(env):
    await save_account(env)
    env.state.response_queue = [
        httpx.Response(401, json={"error": {"code": "subscription_sharing_invalid_user"}}),
        stream_response(completed("after-refresh")),
    ]
    result = await env.provider.response(request_body())
    assert result["content"] == "after-refresh" and env.state.refresh_count == 1
    requests = [r for r in env.state.requests if r.url.path == "/v1/responses"]
    assert (
        len(requests) == 2 and requests[1].headers["authorization"] == "Bearer " + ACCESS + "-fresh"
    )


async def test_second_401_stops(env):
    await save_account(env)
    env.state.response_queue = [
        httpx.Response(401, json={"detail": "SECRET"}),
        httpx.Response(401, json={"detail": "SECRET"}, headers={"x-request-id": "req_second"}),
    ]
    with pytest.raises(OpenAIError) as caught:
        await env.provider.response(request_body())
    assert caught.value.status == 401 and caught.value.shape == "detail"
    assert caught.value.request_id == "req_second" and "SECRET" not in str(caught.value)
    assert env.state.refresh_count == 1
    assert len([r for r in env.state.requests if r.url.path == "/v1/responses"]) == 2


@pytest.mark.parametrize("in_stream", [False, True])
async def test_usage_exhaustion_pauses_until_manual_resume(env, in_stream):
    await save_account(env)
    error = {"code": "subscription_sharing_usage_limit_exceeded", "param": "model"}
    env.state.response_queue = [
        stream_response({"type": "response.failed", "response": {"error": error}})
        if in_stream
        else httpx.Response(429, json={"error": error})
    ]
    with pytest.raises(OpenAIError, match="usage_limit_exceeded"):
        await env.provider.response(request_body())
    count = len(env.state.requests)
    # A new provider instance also respects the persisted pause.
    with pytest.raises(PluginError, match="暂停"):
        await PlanProvider(env.auth, env.client).response(request_body())
    assert len(env.state.requests) == count
    await env.provider.resume()
    assert (await env.provider.response(request_body()))["content"] == "Hello"


async def test_503_retry_is_bounded_and_preserves_credentials(env):
    await save_account(env)
    env.state.response_queue = [
        httpx.Response(503, json={"error": {"code": "subscription_sharing_usage_unavailable"}})
        for _ in range(3)
    ]
    with pytest.raises(OpenAIError):
        await env.provider.response(request_body())
    assert len(env.state.response_queue) == 1
    assert env.store.account()["access_token"] == ACCESS


@pytest.mark.parametrize(
    ("code", "hint"),
    [
        ("subscription_sharing_user_not_eligible", "账号、工作区或策略"),
        ("unsupported_country_region_territory", "网络来自不受支持的国家或地区"),
    ],
)
async def test_403_not_retried_and_no_api_key_fallback(env, code, hint):
    await save_account(env)
    env.state.response_queue = [httpx.Response(403, json={"error": {"code": code}})]
    request = request_body()
    request["api_provider"] = {"base_url": "https://evil.invalid", "api_key": "fake-api-key"}
    with pytest.raises(OpenAIError, match=hint):
        await env.provider.response(request)
    assert len([r for r in env.state.requests if r.url.path == "/v1/responses"]) == 1
    assert all(r.url.host == "api.openai.com" for r in env.state.requests)


async def test_network_exception_does_not_replay_or_leak(env):
    await save_account(env)
    env.state.response_queue = [httpx.ReadError("SECRET CHAT " + ACCESS)]
    with pytest.raises(PluginError) as caught:
        await env.provider.response(request_body())
    assert "SECRET" not in str(caught.value) and ACCESS not in str(caught.value)
    assert len([r for r in env.state.requests if r.url.path == "/v1/responses"]) == 1


async def test_tool_round_trip_snapshot_can_be_returned_as_history(env):
    await save_account(env)
    request = request_body()
    request["tool_options"] = [{"type": "function", "function": {"name": "weather.query"}}]
    from chatgpt_maibot.mapping import wire_tool_name

    call = {
        "type": "function_call",
        "namespace": "maibot",
        "call_id": "call_weather",
        "name": wire_tool_name("weather.query"),
        "arguments": '{"city":"Nanjing"}',
    }
    env.state.response_queue = [
        stream_response(completed(output=[call])),
        stream_response(completed("Sunny")),
    ]
    result = await env.provider.response(request)
    request["message_list"].extend(
        [
            {"role": "assistant", "content": result["content"], "tool_calls": result["tool_calls"]},
            {"role": "tool", "tool_call_id": "call_weather", "content": "Sunny, 20 C"},
        ]
    )
    assert (await env.provider.response(request))["content"] == "Sunny"
    wire = json.loads(env.state.requests[-1].content)
    assert wire["input"][-2]["namespace"] == "maibot"
    assert wire["input"][-1]["call_id"] == "call_weather"


async def test_catalog_refresh_on_token_change(env):
    await save_account(env)
    await env.provider.response(request_body())
    async with env.store.locked():
        account = env.store.account()
        account["access_token"] = "fake-changed-token"
        env.store.save(account)
    await env.provider.response(request_body())
    assert len([r for r in env.state.requests if r.url.path == "/v1/models"]) == 2
