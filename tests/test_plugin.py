import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import httpx
import pytest

import plugin
from chatgpt_maibot import CLIENT_TYPE
from chatgpt_maibot.errors import PluginError


@pytest.fixture
async def loaded(tmp_path):
    instance = plugin.create_plugin()
    sent = AsyncMock()
    instance._set_context(
        SimpleNamespace(
            paths=SimpleNamespace(data_dir=tmp_path / "data"),
            logger=logging.getLogger("plugin-test"),
            send=SimpleNamespace(text=sent),
        )
    )
    instance.set_plugin_config({})
    await instance.on_load()
    yield instance, sent
    await instance.on_unload()


def test_sdk_discovers_manifest_provider_and_commands():
    instance = plugin.create_plugin()
    manifest = json.loads(open("_manifest.json").read())
    providers = instance.get_llm_providers()
    assert providers[0]["client_type"] == manifest["llm_providers"][0]["client_type"] == CLIENT_TYPE
    assert providers[0]["version"] == manifest["version"]
    assert len(instance.get_components()) == 6
    apis = [item for item in instance.get_components() if item["type"] == "API"]
    assert {item["name"] for item in apis} == {"search", "fetch"}
    assert all(item["metadata"]["public"] for item in apis)
    config, changed = instance.normalize_plugin_config({})
    assert changed and config["plugin"]["config_version"] == "0.1.0"
    assert config["plugin"]["admin_stream_ids"] == []
    assert instance.get_webui_config_schema()


async def test_web_apis_disabled_by_default_and_after_plugin_disable(loaded):
    instance, _ = loaded
    instance.search_service.search = AsyncMock()
    instance.page_fetcher.fetch = AsyncMock()
    for method, kwargs in [("search", {"query": "q"}), ("fetch", {"url": "https://example.com"})]:
        with pytest.raises(PluginError, match="未启用"):
            await getattr(instance, method)(**kwargs)
    instance.set_plugin_config(
        {"plugin": {"config_version": "0.1.0", "enabled": False}, "web": {"enabled": True}}
    )
    with pytest.raises(PluginError, match="未启用"):
        await instance.fetch("https://example.com")
    instance.search_service.search.assert_not_awaited()
    instance.page_fetcher.fetch.assert_not_awaited()


async def test_web_apis_use_selected_model_and_are_independent_of_chat_search(loaded):
    instance, _ = loaded
    instance.set_plugin_config(
        {
            "plugin": {"config_version": "0.1.0"},
            "web": {"enabled": True, "model": "chosen", "fetch_timeout_seconds": 20},
        }
    )
    instance.search_service.search = AsyncMock(return_value={"results": []})
    instance.page_fetcher.fetch = AsyncMock(return_value={"content": "正文"})
    assert (await instance.search(query="q", max_results=3))["results"] == []
    instance.search_service.search.assert_awaited_once_with("chosen", "q", 3)
    assert not instance.config.plugin.enable_web_search
    await instance.fetch(url="https://example.com")
    instance.page_fetcher.fetch.assert_awaited_once_with("https://example.com", timeout_seconds=20)


async def test_deepseek_web_routes_and_key_rotation(loaded, unused_tcp_port):
    instance, _ = loaded
    url = "https://example.com/guide"
    instance.search_service.search = AsyncMock(
        return_value={
            "results": [
                {
                    "url": url,
                    "title": "指南",
                    "snippet": "摘要",
                    "summary": "说明",
                    "published_at": "2026-11-07",
                }
            ]
        }
    )
    instance.page_fetcher.fetch = AsyncMock(return_value={"title": "指南", "content": "实际正文"})

    async def configure(key, port=unused_tcp_port):
        instance.set_plugin_config(
            {
                "plugin": {"config_version": "0.1.0"},
                "web": {
                    "enabled": True,
                    "model": "chosen",
                    "deepseek_port": port,
                    "deepseek_api_key": key,
                },
            }
        )
        await instance.on_config_update("self", {}, "0.1.0")

    await configure("A" * 32)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{unused_tcp_port}", trust_env=False
    ) as client:
        search = {"query": "test", "count": 3, "need_content": False}
        headers = {"Authorization": "Bearer " + "A" * 32}
        assert (await client.post("/v1/tools/search", json=search)).status_code == 401
        assert (
            await client.post("/v1/tools/search", json={**search, "count": True}, headers=headers)
        ).status_code == 400
        instance.search_service.search.assert_not_awaited()
        response = await client.post("/v1/tools/search", json=search, headers=headers)
        assert response.json() == {
            "results": [
                {
                    "url": url,
                    "title": "指南",
                    "snippet": "摘要",
                    "summary": "说明",
                    "date": "2026-11-07",
                }
            ]
        }
        instance.search_service.search.assert_awaited_once_with("chosen", "test", 3)
        await configure("B" * 32)
        assert (
            await client.post("/v1/tools/fetch", json={"url": url}, headers=headers)
        ).status_code == 401
        headers = {"Authorization": "Bearer " + "B" * 32}
        response = await client.post("/v1/tools/fetch", json={"url": url}, headers=headers)
        assert response.json() == {"title": "指南", "content": "实际正文"}
        instance.page_fetcher.fetch.assert_awaited_once_with(url, timeout_seconds=30)
        await configure("", port=0)
        with pytest.raises(httpx.ConnectError):
            await client.post("/v1/tools/fetch", json={"url": url}, headers=headers)


@pytest.mark.parametrize("operation", ["embedding", "audio_transcription", "unknown"])
async def test_unsupported_operations_rejected_before_http(loaded, operation):
    instance, _ = loaded
    instance.provider.response = AsyncMock()
    with pytest.raises(PluginError, match="仅支持 response"):
        await instance.handle_llm_provider(operation, {})
    instance.provider.response.assert_not_called()


async def test_disabled_provider_does_not_call_openai(loaded):
    instance, _ = loaded
    instance.set_plugin_config({"plugin": {"config_version": "0.1.0", "enabled": False}})
    instance.provider.response = AsyncMock()
    with pytest.raises(PluginError, match="禁用"):
        await instance.handle_llm_provider("response", {})
    instance.provider.response.assert_not_called()


@pytest.mark.parametrize("method", ["login", "models", "status", "logout"])
async def test_management_closed_by_default(loaded, method):
    instance, sent = loaded
    success, text, _ = await getattr(instance, method)(stream_id="untrusted-stream")
    assert success is False and "未授权" in text
    sent.assert_not_called()
    assert not instance.auth._login_task


async def test_denial_reports_actual_stream_without_disclosing_context(loaded, caplog):
    instance, sent = loaded
    instance.set_plugin_config(
        {
            "plugin": {
                "config_version": "0.1.0",
                "admin_stream_ids": ["configured-admin"],
                "enable_tool_controls": True,
            }
        }
    )
    with caplog.at_level(logging.WARNING, logger="plugin-test"):
        success, text, _ = await instance.status(
            stream_id="actual-private-stream", message="private-message", token="private-token"
        )
    assert success is False
    assert "stream_id='actual-private-stream'" in text
    assert "stream_id='actual-private-stream'" in caplog.text
    assert "enable_tool_controls" in text
    for hidden in ("configured-admin", "private-message", "private-token"):
        assert hidden not in text + caplog.text
    sent.assert_not_called()
    assert not instance.auth._login_task


@pytest.mark.parametrize("stream", [None, "", 123, ["private-admin"]])
async def test_missing_or_invalid_stream_is_rejected_even_if_empty_is_allowlisted(loaded, stream):
    instance, sent = loaded
    instance.set_plugin_config({"plugin": {"config_version": "0.1.0", "admin_stream_ids": [""]}})
    success, text, _ = await instance.status(stream_id=stream)
    assert success is False and "Host 未提供有效的 stream_id" in text
    sent.assert_not_called()


async def test_admin_policy_and_status_do_not_expose_tokens(loaded):
    instance, sent = loaded
    instance.set_plugin_config(
        {"plugin": {"config_version": "0.1.0", "admin_stream_ids": ["private-admin"]}}
    )
    result = await instance.status(stream_id="private-admin")
    assert result[0] and "尚未注册" in sent.call_args.args[0]
    assert (await instance.status(stream_id="other"))[0] is False


async def test_unload_closes_listener_and_cancels_reports(loaded):
    instance, sent = loaded
    instance.set_plugin_config(
        {"plugin": {"config_version": "0.1.0", "admin_stream_ids": ["private-admin"]}}
    )
    assert (await instance.login(stream_id="private-admin"))[0] is True
    url = sent.call_args.args[0].splitlines()[-1]
    from urllib.parse import parse_qs

    callback = parse_qs(urlsplit(url).query)["redirect_uri"][0]
    await instance.on_unload()
    assert not instance._reports and instance._client.is_closed
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", urlsplit(callback).port)


async def test_reload_applies_timeouts_and_cancels_pending_login(loaded):
    instance, _ = loaded
    attempt = await instance.auth.begin_login()
    instance.set_plugin_config(
        {
            "plugin": {
                "config_version": "0.1.0",
                "request_timeout_seconds": 180,
                "oauth_timeout_seconds": 120,
            }
        }
    )
    await instance.on_config_update("self", {}, "0.1.0")
    assert attempt.task.cancelled()
    assert instance.provider.timeout_seconds == 180
    assert instance.auth.timeout_seconds == 120


def test_local_consistency_and_expected_unfinished_release_metadata():
    from tools.check_release import check

    assert check() == []
    failures = check(release=True)
    assert len(failures) == 3
    assert any("作者" in message for message in failures)
