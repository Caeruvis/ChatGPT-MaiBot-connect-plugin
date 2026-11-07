import asyncio
import json

import httpx
import pytest
from conftest import completed, save_account, sse

from chatgpt_maibot.errors import PluginError
from chatgpt_maibot.search_service import SearchService, normalize_search


def search_event(*, status="completed", action="search", sources=None):
    return {
        "type": "web_search_call",
        "status": status,
        "action": {"type": action, "sources": sources or []},
    }


def answer(text, annotations=None):
    return {
        "type": "message",
        "content": [{"type": "output_text", "text": text, "annotations": annotations or []}],
    }


def normalize(output, count=5):
    return normalize_search({"status": "completed", "output": output}, "query", count, {})


async def test_hosted_search_uses_shared_auth_and_only_web_tool(env):
    await save_account(env)
    output = [
        search_event(sources=[{"url": "https://python.org", "title": "Python"}]),
        answer(json.dumps({"results": [{"url": "https://python.org", "summary": "版本说明"}]})),
    ]
    # Exercise the same output-item recovery used by ordinary chat.
    env.state.response_queue = [
        httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(
                *[
                    {"type": "response.output_item.done", "output_index": i, "item": row}
                    for i, row in enumerate(output)
                ],
                completed(output=[]),
            ),
        )
    ]
    result = await SearchService(env.provider).search("test-model", " Python ", 2)
    wire = json.loads(env.state.requests[-1].content)
    assert wire["tools"] == [{"type": "web_search"}]
    assert wire["tool_choice"] == "required"
    assert wire["input"] == [{"role": "user", "content": "Python"}]
    assert wire["include"] == ["web_search_call.action.sources"]
    assert wire["store"] is False and wire["stream"] is True
    assert result["results"][0]["snippet"] == "[GPT 生成摘要] 版本说明"
    assert result["results"][0]["published_at"] is None
    assert result["web_search_completed"] == 1


def test_only_tool_sources_and_annotations_can_create_hits():
    data = {
        "results": [
            {"url": "https://made-up.example", "summary": "无来源"},
            {"url": "https://real.example", "summary": "有来源"},
        ]
    }
    result = normalize(
        [
            search_event(sources=[{"url": "https://real.example"}]),
            answer(
                json.dumps(data),
                [
                    {"type": "url_citation", "url": "https://real.example", "title": "真实标题"},
                    {"type": "url_citation", "url": "file:///private", "title": "非法"},
                ],
            ),
        ]
    )
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "真实标题"
    assert "无来源" not in str(result)


def test_malformed_answer_keeps_sources_without_assigning_answer_to_every_source():
    result = normalize(
        [
            search_event(),
            answer(
                "answer citeturn0search0",
                [
                    {"type": "url_citation", "url": "https://one.example", "title": "One"},
                    {"type": "url_citation", "url": "https://two.example", "title": "Two"},
                ],
            ),
        ],
        count=1,
    )
    assert len(result["results"]) == 1
    assert result["results"][0]["summary"] == ""
    assert "answer" not in str(result)


@pytest.mark.parametrize(
    "output",
    [
        [answer("我搜索了")],
        [search_event(status="failed"), answer("解释")],
        [search_event(action="open_page")],
        [search_event(), {"type": "function_call", "name": "web_search"}],
    ],
)
def test_missing_failed_search_and_local_tool_recursion_are_errors(output):
    with pytest.raises(PluginError):
        normalize(output)


def test_completed_search_with_no_sources_is_successful_empty_result():
    assert normalize([search_event(), answer("没有找到资料")])["results"] == []


@pytest.mark.parametrize("query,count", [(" ", 1), ("q" * 1001, 1), ("q", True), ("q", 21)])
async def test_invalid_search_never_calls_upstream(env, query, count):
    with pytest.raises(PluginError):
        await SearchService(env.provider).search("test-model", query, count)
    assert not env.state.requests


async def test_search_cancellation_propagates(env, monkeypatch):
    started = asyncio.Event()

    async def wait(body):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(env.provider, "complete", wait)
    task = asyncio.create_task(SearchService(env.provider).search("test-model", "q"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
