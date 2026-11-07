import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from conftest import ACCESS, REFRESH, completed, save_account, sse

from tools.diagnose_response import StreamTrace, diagnose, response_shape


def test_shape_report_never_echoes_values_or_unknown_field_names():
    payload = {
        ACCESS: REFRESH,
        "status": ACCESS,
        "output_text": "PRIVATE CHAT",
        "error": {"message": REFRESH},
        "output": [
            {
                "type": ACCESS,
                "role": REFRESH,
                "status": ACCESS,
                REFRESH: ACCESS,
                "content": [{"type": ACCESS, "text": "PRIVATE CHAT", ACCESS: REFRESH}],
            }
        ],
    }
    encoded = json.dumps(response_shape(payload))
    for secret in (ACCESS, REFRESH, "PRIVATE CHAT"):
        assert secret not in encoded
    assert response_shape(payload)["unknown_field_count"] == 1


def test_trace_distinguishes_empty_terminal_output_from_completed_item_events():
    trace = StreamTrace()
    trace.observe({"type": "response.created", "response": {"id": "private-response-id"}})
    trace.observe({"type": "response.output_text.delta", "delta": "PRIVATE CHAT"})
    trace.observe({"type": "response.output_text.done", "text": "PRIVATE CHAT"})
    trace.observe(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "PRIVATE CHAT"}],
            },
        }
    )
    trace.observe(
        {
            "type": "response.completed",
            "response": {"id": "private-response-id", "status": "completed", "output": []},
        }
    )
    report = trace.report()
    assert report["completed_snapshot"]["output_count"] == 0
    assert report["output_item_done"][0]["content"]["parts"][0]["text_chars"] == 12
    assert report["text_delta_chars"] == 12
    assert "PRIVATE CHAT" not in json.dumps(report)
    assert "private-response-id" not in json.dumps(report)
    assert report["response_ids_consistent"] is True


@pytest.mark.parametrize("empty_terminal", [False, True])
async def test_standalone_report_works_with_current_plugin_and_does_not_change_parser(
    env, monkeypatch, empty_terminal
):
    import chatgpt_maibot.auth as auth_module
    import chatgpt_maibot.provider as provider_module

    await save_account(env)
    monkeypatch.setattr(auth_module, "OpenAIPlanAuth", lambda *_: env.auth)
    monkeypatch.setattr(provider_module, "PlanProvider", lambda *_: env.provider)
    original = provider_module.sse_events
    terminal = completed("CONNECT_OK")
    if empty_terminal:
        terminal["response"]["output"] = []
    env.state.response_queue = [
        httpx.Response(
            200,
            content=sse(
                {"type": "response.output_text.delta", "delta": "CONNECT_OK"},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "CONNECT_OK"}],
                    },
                },
                terminal,
            ).encode(),
        )
    ]
    report, code = await diagnose(
        SimpleNamespace(plugin_dir=Path.cwd(), data_dir=env.store.data_dir, model="test-model")
    )
    assert code == 0
    assert report["result"]["accepted"] is True
    assert report["stream"]["completed_snapshot"]["output_count"] == (0 if empty_terminal else 1)
    assert provider_module.sse_events is original
    serialized = json.dumps(report)
    assert ACCESS not in serialized and REFRESH not in serialized and "CONNECT_OK" not in serialized
    wire = json.loads(env.state.requests[-1].content)
    assert "tools" not in wire
    assert wire["input"] == [{"role": "user", "content": "请只回复 CONNECT_OK。"}]
