"""Offline integration: real SDK ctx.web -> Host -> bridge -> mocked Responses SSE."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

ROOT = Path(__file__).resolve().parents[1]


async def check(host: Path, runtime: Path) -> None:
    sys.path[:0] = [str(ROOT), str(host)]
    os.environ["MAIBOT_INSTANCE_DIR"] = str(runtime / "instance")
    from maibot_sdk.context import PluginContext, PluginPaths
    from src.plugin_runtime.capabilities import web
    from src.plugin_runtime.host.api_registry import APIRegistry
    from src.plugin_runtime.host.authorization import AuthorizationManager
    from src.plugin_runtime.host.capability_service import CapabilityService
    from src.plugin_runtime.protocol.envelope import Envelope, MessageType
    from src.web_search.config import WebRuntimeConfig
    from src.web_search.service import WebSearchService

    from chatgpt_maibot import page_fetch
    from chatgpt_maibot.page_fetch import PageFetcher
    from plugin import create_plugin

    bridge = create_plugin()
    bridge._set_context(
        PluginContext(
            "local.chatgpt-maibot-connect",
            paths=PluginPaths(
                data_dir=runtime / "bridge",
                runtime_dir=runtime / "temp",
            ),
        )
    )
    bridge.set_plugin_config(
        {"plugin": {"config_version": "0.1.0"}, "web": {"enabled": True, "model": "test-model"}}
    )
    await bridge.on_load()
    requests = []

    def upstream(request):
        requests.append(request)
        assert request.url == "https://api.openai.com/v1/responses"
        body = json.loads(request.content)
        assert body["tools"] == [{"type": "web_search"}] and body["tool_choice"] == "required"
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "sources": [{"url": "https://docs.example/a", "title": "官方资料"}],
                    },
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "results": [
                                        {"url": "https://docs.example/a", "summary": "来源中的事实"}
                                    ]
                                }
                            ),
                            "annotations": [],
                        }
                    ],
                },
            ],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: "
            + json.dumps({"type": "response.completed", "response": response})
            + "\n\n",
        )

    session = SimpleNamespace(client_id="test-client", token="test-token")
    bridge.provider.auth = SimpleNamespace(session=AsyncMock(return_value=session))
    bridge.provider._models = AsyncMock(return_value=([{"slug": "test-model"}], session))
    bridge.provider._usage_pause = AsyncMock(return_value=False)
    registry = APIRegistry()
    for component in bridge.get_components():
        if component["type"] == "API":
            registry.register_api(
                component["name"], "local.chatgpt-maibot-connect", component["metadata"]
            )

    async def invoke_api(*, plugin_id, component_name, args, timeout_ms):
        assert plugin_id == "local.chatgpt-maibot-connect" and timeout_ms == 270000
        result = await getattr(bridge, component_name)(**args)
        return SimpleNamespace(error=None, payload={"success": True, "result": result})

    supervisor = SimpleNamespace(api_registry=registry, invoke_api=invoke_api)
    # Only the process boundary is replaced. Registry, authorization, SDK normalization,
    # service routing, API methods and Responses stream processing run their real code.
    sys.modules["src.plugin_runtime.integration"] = SimpleNamespace(
        get_plugin_runtime_manager=lambda: SimpleNamespace(supervisors=[supervisor])
    )
    settings = WebRuntimeConfig(
        enabled=True,
        search="gpt",
        fetch="gpt",
        services={
            "gpt": {
                "adapter": "plugin",
                "plugin_id": "local.chatgpt-maibot-connect",
                "timeout": 270,
            }
        },
    )
    web.get_config = lambda: settings
    web.get_service = lambda: WebSearchService(None)
    auth = AuthorizationManager()
    auth.register_plugin("maibot-community.mai-life", ["web.search", "web.fetch"])
    capabilities = CapabilityService(auth)
    for name in ("web.search", "web.fetch"):
        capabilities.register_capability(name, web.call)

    async def rpc(method, plugin_id, payload, timeout_ms=None):
        assert method == "cap.call" and timeout_ms == 310000
        envelope = Envelope(
            request_id=1,
            message_type=MessageType.REQUEST,
            plugin_id=plugin_id,
            method=method,
            payload=payload,
        )
        result = await capabilities.handle_capability_request(envelope)
        if result.error:
            raise RuntimeError(result.error["message"])
        return result.payload["result"]

    ctx = PluginContext("maibot-community.mai-life", rpc_call=rpc)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        bridge.provider.client = client
        try:
            result = await ctx.web.search("test query", max_results=2)
            assert result["results"][0]["snippet"] == "[GPT 生成摘要] 来源中的事实"
            assert result["service"] == "gpt" and "result_id" not in result

            async def resolve(host, port):
                return "93.184.216.34"

            page_fetch.resolve_public = resolve

            def page(request):
                assert "authorization" not in request.headers
                return httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text="<title>标题</title><main>真实网页正文</main>",
                )

            bridge.page_fetcher = PageFetcher(transport=httpx.MockTransport(page))
            result = await ctx.web.fetch("https://docs.example/a")
            assert result["content_kind"] == "page_text" and result["content"] == "真实网页正文"
            assert len(requests) == 1
            print(
                json.dumps(
                    {
                        "sdk_ctx_web_search": "passed",
                        "sdk_ctx_web_fetch": "passed",
                        "bridge_sse": "passed",
                        "real_cloud_inference": "not_run",
                    },
                    indent=2,
                )
            )
        finally:
            await bridge.on_unload()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-source", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gpt-web-integration-") as temp:
        asyncio.run(check(args.host_source.resolve(), Path(temp)))
