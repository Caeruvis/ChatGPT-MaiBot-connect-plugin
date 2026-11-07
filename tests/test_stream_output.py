import copy

import httpx
import pytest
from conftest import completed, request_body, save_account, sse

from chatgpt_maibot.errors import OpenAIError, PluginError


def message(text="CONNECT_OK", *, item_id="msg_test", **kwargs):
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": text}],
        **kwargs,
    }


def done(item=None, index=0, **kwargs):
    event = {"type": "response.output_item.done", "item": item or message(), **kwargs}
    if index is not None:
        event["output_index"] = index
    return event


def terminal(**kwargs):
    return completed(output=[], id="resp_test", **kwargs)


def stream(*events):
    return httpx.Response(
        200, content=sse(*events).encode(), headers={"content-type": "text/event-stream"}
    )


async def test_reported_empty_terminal_snapshot_recovers_completed_message(env):
    """Synthetic event values reproduce the exact structural pattern in the user's report."""
    await save_account(env)
    final = terminal(usage={"input_tokens": 12, "output_tokens": 6, "total_tokens": 18})
    snapshot = copy.deepcopy(final)
    env.state.response_queue = [
        stream(
            {"type": "response.created", "response": {"id": "resp_test", "status": "in_progress"}},
            {
                "type": "response.in_progress",
                "response": {"id": "resp_test", "status": "in_progress"},
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": message("", status="in_progress"),
            },
            {
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
            {"type": "response.output_text.delta", "delta": "CONNE"},
            {"type": "response.output_text.delta", "delta": "CT_OK"},
            {"type": "synthetic.unknown.event"},
            {"type": "response.output_text.done", "text": "CONNECT_OK"},
            {
                "type": "response.content_part.done",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "CONNECT_OK"},
            },
            done(),
            final,
        )
    ]
    result = await env.provider.response(request_body())
    assert result["content"] == "CONNECT_OK"
    assert result["output_items"][0]["phase"] == "final_answer"
    assert result["usage"]["prompt_tokens"] == 12
    assert result["usage"]["completion_tokens"] == 6
    assert result["raw_data"]["output_source"] == "output_item_done"
    assert final == snapshot


@pytest.mark.parametrize("failure", ["response.failed", "response.incomplete", None])
async def test_item_done_alone_never_counts_as_response_success(env, failure):
    await save_account(env)
    events = [done()]
    if failure:
        events.append(
            {"type": failure, "response": {"status": "failed", "error": {"code": "server_error"}}}
        )
    env.state.response_queue = [stream(*events)]
    with pytest.raises(PluginError):
        await env.provider.response(request_body())


async def test_text_deltas_and_text_done_are_not_substituted_for_complete_item(env):
    await save_account(env)
    env.state.response_queue = [
        stream(
            {"type": "response.output_text.delta", "delta": "partial"},
            {"type": "response.output_text.done", "text": "partial"},
            terminal(),
        )
    ]
    with pytest.raises(PluginError, match="完成快照未包含"):
        await env.provider.response(request_body())


async def test_full_terminal_snapshot_is_authoritative_and_not_duplicated(env):
    await save_account(env)
    env.state.response_queue = [stream(done(message("item text")), completed("final text"))]
    result = await env.provider.response(request_body())
    assert result["content"] == "final text"
    assert len(result["output_items"]) == 1
    assert result["raw_data"]["output_source"] == "completed_snapshot"


async def test_done_items_sorted_by_output_index_even_if_they_finish_out_of_order(env):
    await save_account(env)
    env.state.response_queue = [
        stream(
            done(message("second", item_id="msg_2"), 1),
            done(message("first", item_id="msg_1"), 0),
            terminal(),
        )
    ]
    result = await env.provider.response(request_body())
    assert result["content"] == "firstsecond"


async def test_identical_repeated_item_done_is_not_duplicated(env):
    await save_account(env)
    env.state.response_queue = [stream(done(), done(), terminal())]
    result = await env.provider.response(request_body())
    assert result["content"] == "CONNECT_OK" and len(result["output_items"]) == 1


@pytest.mark.parametrize(
    "events",
    [
        [done(message("one")), done(message("different"))],
        [done(message("one"), 0), done(message("one"), 1)],
        [done(message(status="incomplete"))],
        [done(message(status="in_progress"))],
        [done(index=1)],
        [done(index=-1)],
        [done(index=True)],
        [{"type": "response.output_item.done", "output_index": 0, "item": "invalid"}],
        [done(index=None), done(message("other", item_id="msg_other"), index=None)],
        [
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": message(status="in_progress"),
            },
            done(),
        ],
    ],
)
async def test_conflicting_unfinished_or_unordered_items_fail_closed(env, events):
    await save_account(env)
    env.state.response_queue = [stream(*events, terminal())]
    with pytest.raises(PluginError):
        await env.provider.response(request_body())


async def test_single_unindexed_done_item_can_be_recovered_unambiguously(env):
    await save_account(env)
    env.state.response_queue = [stream(done(index=None), terminal())]
    assert (await env.provider.response(request_body()))["content"] == "CONNECT_OK"


async def test_mismatched_response_ids_rejected_without_echoing_ids(env):
    await save_account(env)
    env.state.response_queue = [
        stream(
            {"type": "response.created", "response": {"id": "resp_PRIVATE"}},
            done(),
            terminal(),
        )
    ]
    with pytest.raises(PluginError) as caught:
        await env.provider.response(request_body())
    assert "resp_PRIVATE" not in str(caught.value)


async def test_sparse_terminal_function_call_preserves_real_arguments(env):
    await save_account(env)
    request = request_body()
    request["tool_options"] = [{"name": "weather"}]
    tool = {
        "id": "fc_test",
        "type": "function_call",
        "status": "completed",
        "namespace": "maibot",
        "call_id": "call_test",
        "name": "weather",
        "arguments": '{"city":"Nanjing"}',
    }
    env.state.response_queue = [stream(done(tool), terminal())]
    result = await env.provider.response(request)
    assert result["tool_calls"][0]["function"]["arguments"] == {"city": "Nanjing"}
    assert result["output_items"][0]["tool_call"]["args"] == {"city": "Nanjing"}


async def test_sparse_terminal_retains_completed_search_count(env):
    await save_account(env)
    search = {"id": "ws_test", "type": "web_search_call", "status": "completed"}
    env.state.response_queue = [stream(done(search, 0), done(message(), 1), terminal())]
    result = await env.provider.response(request_body(), enable_web_search=True)
    assert result["content"] == "CONNECT_OK"
    assert result["raw_data"]["web_search_completed"] == 1


async def test_retry_does_not_reuse_previous_attempt_items(env):
    await save_account(env)
    env.state.response_queue = [
        stream(
            done(message("stale")),
            {
                "type": "response.failed",
                "response": {"error": {"code": "subscription_sharing_usage_unavailable"}},
            },
        ),
        stream(terminal()),
    ]
    with pytest.raises(PluginError, match="完成快照未包含"):
        await env.provider.response(request_body())


async def test_completed_with_error_is_not_recovered_as_success(env):
    await save_account(env)
    env.state.response_queue = [stream(done(), terminal(error={"code": "server_error"}))]
    with pytest.raises((PluginError, OpenAIError)):
        await env.provider.response(request_body())
