import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import ACCESS, REFRESH, completed, request_body, save_account, sse

from chatgpt_maibot.__main__ import parser, run
from chatgpt_maibot.errors import OpenAIError, PluginError
from chatgpt_maibot.transport import MAX_SAMPLE_BYTES, ResponseFormatError


@pytest.mark.parametrize(
    "mime",
    [
        "Text/Event-Stream; charset=UTF-8",
        "TEXT/EVENT-STREAM",
        "text/event-stream; charset=utf-8",
        "application/octet-stream",
        "application/json",
        "text/plain",
        None,
    ],
)
async def test_verified_sse_frames_do_not_depend_on_mime_case_or_label(env, mime):
    await save_account(env)
    headers = {"x-request-id": "req_mime"}
    if mime is not None:
        headers["content-type"] = mime
    env.state.response_queue = [
        httpx.Response(200, content=sse(completed("ok")).encode(), headers=headers)
    ]
    result = await env.provider.response(request_body())
    assert result["content"] == "ok"
    assert result["raw_data"]["transport"]["framing"] == "sse"
    assert result["raw_data"]["transport"]["request_id"] == "req_mime"


@pytest.mark.parametrize(
    "body,shape",
    [
        ('{"detail":"SECRET CHAT"}', "json_detail"),
        ('{"status":"completed","output":[]}', "json_object"),
        ("<!DOCTYPE html><html>SECRET CHAT</html>", "html"),
        ("<html>SECRET CHAT</html>", "html"),
        ("plain SECRET CHAT", "other"),
        ("", "empty"),
    ],
)
async def test_non_sse_is_classified_without_exposing_response_body(env, body, shape):
    await save_account(env)
    env.state.response_queue = [
        httpx.Response(
            200,
            content=body.encode(),
            headers={
                "content-type": "application/json; token=" + ACCESS,
                "x-request-id": "req_format",
            },
        )
    ]
    with pytest.raises(ResponseFormatError) as caught:
        await env.provider.response(request_body())
    assert caught.value.metadata["body_shape"] == shape
    message = str(caught.value)
    assert (
        "HTTP 200" in message
        and "content_type=application/json" in message
        and "req_format" in message
    )
    assert ACCESS not in message and "SECRET CHAT" not in message
    assert len([r for r in env.state.requests if r.url.path == "/v1/responses"]) == 1


async def test_json_completed_snapshot_is_not_a_streaming_success(env):
    await save_account(env)
    env.state.response_queue = [httpx.Response(200, json=completed("do-not-return")["response"])]
    with pytest.raises(ResponseFormatError):
        await env.provider.response(request_body())


async def test_json_error_preserves_machine_code_and_usage_pause(env):
    await save_account(env)
    env.state.response_queue = [
        httpx.Response(
            200,
            json={
                "error": {
                    "code": "subscription_sharing_usage_limit_exceeded",
                    "param": "model",
                    "message": "SECRET CHAT " + ACCESS,
                }
            },
            headers={"x-request-id": "req_json_error"},
        )
    ]
    with pytest.raises(OpenAIError) as caught:
        await env.provider.response(request_body())
    assert caught.value.status == 200
    assert caught.value.code == "subscription_sharing_usage_limit_exceeded"
    assert "SECRET CHAT" not in str(caught.value) and ACCESS not in str(caught.value)
    assert env.store.account()["usage_paused"] is True


async def test_header_diagnostics_redact_access_token(env):
    await save_account(env)
    env.state.response_queue = [
        httpx.Response(
            200,
            content=b"not SSE",
            headers={
                "content-type": "application/" + ACCESS,
                "x-request-id": ACCESS,
            },
        )
    ]
    with pytest.raises(ResponseFormatError) as caught:
        await env.provider.response(request_body())
    assert ACCESS not in str(caught.value)
    assert caught.value.metadata["content_type"] == "redacted"
    assert caught.value.metadata["request_id"] == "unknown"


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.read_count = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.read_count += 1
            yield chunk


async def test_sniffing_handles_split_bom_and_event_prefix_without_losing_bytes(env):
    await save_account(env)
    payload = sse(completed("南京")).encode()
    stream = ChunkedStream(
        [b"\xef", b"\xbb", b"\xbf\n", b"d", b"at", b"a:", payload[5:18], payload[18:]]
    )
    env.state.response_queue = [httpx.Response(200, stream=stream)]
    assert (await env.provider.response(request_body()))["content"] == "南京"


async def test_mislabeled_partial_stream_still_fails(env):
    await save_account(env)
    env.state.response_queue = [
        httpx.Response(
            200,
            content=sse({"type": "response.output_text.delta", "delta": "partial"}).encode(),
            headers={"content-type": "text/plain"},
        )
    ]
    with pytest.raises(PluginError, match="未收到 response.completed"):
        await env.provider.response(request_body())


async def test_bad_body_sample_is_bounded(env):
    await save_account(env)
    stream = ChunkedStream([b"<html>", b"a" * MAX_SAMPLE_BYTES, b"must-not-read"])
    env.state.response_queue = [
        httpx.Response(200, stream=stream, headers={"content-type": "text/html"})
    ]
    with pytest.raises(ResponseFormatError):
        await env.provider.response(request_body())
    assert stream.read_count == 2


async def test_completed_search_count_excludes_unfinished_calls(env):
    await save_account(env)
    event = completed()
    event["response"]["output"].extend(
        [
            {"type": "web_search_call", "status": "completed", "action": {"query": "not-logged"}},
            {"type": "web_search_call", "status": "in_progress"},
        ]
    )
    env.state.response_queue = [httpx.Response(200, content=sse(event).encode())]
    result = await env.provider.response(request_body(), enable_web_search=True)
    assert result["raw_data"]["web_search_completed"] == 1
    assert "not-logged" not in json.dumps(result["raw_data"])


@pytest.mark.parametrize("web", [False, True])
async def test_local_test_command_uses_fixed_prompt_and_never_prints_tokens(
    env, monkeypatch, capsys, web
):
    import chatgpt_maibot.__main__ as cli

    await save_account(env)
    monkeypatch.setattr(cli, "OpenAIPlanAuth", lambda *_: env.auth)
    monkeypatch.setattr(cli, "PlanProvider", lambda *_: env.provider)
    args = SimpleNamespace(
        command="test", data_dir=env.store.data_dir, model="test-model", web_search=web
    )
    assert await run(args) == 0
    output = capsys.readouterr().out
    assert ACCESS not in output and REFRESH not in output
    assert '"framing": "sse"' in output
    body = json.loads(env.state.requests[-1].content)
    assert body["stream"] is True and body["store"] is False
    assert ("tools" in body) is web
    assert len(body["input"]) == 1
    if not web:
        assert "CONNECT_OK" in body["input"][0]["content"]


def test_cli_test_requires_explicit_model():
    with pytest.raises(SystemExit):
        parser().parse_args(["--data-dir", "/tmp/unused", "test"])
