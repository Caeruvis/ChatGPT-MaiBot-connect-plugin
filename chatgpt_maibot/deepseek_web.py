"""Loopback HTTP adapter for MaiBot's DeepSeek Search / Fetch protocol."""

from __future__ import annotations

import asyncio
import hmac
import re
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import web

from .errors import PluginError
from .page_fetch import validate_page_url


class DeepSeekWebServer:
    def __init__(
        self,
        search: Callable[[str, int], Awaitable[dict[str, Any]]],
        fetch: Callable[[str], Awaitable[dict[str, Any]]],
    ) -> None:
        self._search = search
        self._fetch = fetch
        self._runner: web.AppRunner | None = None
        self._port = 0
        self._authorization = b""
        self._lock = asyncio.Lock()

    async def configure(self, port: int, api_key: str = "") -> None:
        if port and not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", api_key):
            raise PluginError(
                "启用 DeepSeek 兼容接口需要至少 32 位字母、数字、短横线或下划线组成的本地密钥。"
            )
        async with self._lock:
            if port == self._port:
                self._authorization = f"Bearer {api_key}".encode() if port else b""
                return

            runner = None
            if port:
                app = web.Application(client_max_size=16 * 1024)
                app.router.add_post("/v1/tools/search", self._handle_search)
                app.router.add_post("/v1/tools/fetch", self._handle_fetch)
                runner = web.AppRunner(
                    app, access_log=None, shutdown_timeout=2, handler_cancellation=True
                )
                try:
                    await runner.setup()
                    await web.TCPSite(runner, "127.0.0.1", port).start()
                except BaseException:
                    await runner.cleanup()
                    raise

            old_runner = self._runner
            self._runner, self._port = runner, port
            self._authorization = f"Bearer {api_key}".encode() if port else b""
            if old_runner is not None:
                await old_runner.cleanup()

    async def _body(self, request: web.Request) -> dict[str, Any]:
        authorization = request.headers.getall("Authorization", [])
        if (
            not self._authorization
            or len(authorization) != 1
            or not authorization[0].isascii()
            or not hmac.compare_digest(authorization[0].encode(), self._authorization)
        ):
            raise web.HTTPUnauthorized()
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            body = await request.json()
        except (ValueError, LookupError):
            raise web.HTTPBadRequest(text="请求体须为 JSON 对象") from None
        if not isinstance(body, dict):
            raise web.HTTPBadRequest(text="请求体须为 JSON 对象")
        return body

    async def _handle_search(self, request: web.Request) -> web.Response:
        body = await self._body(request)
        query = body.get("query")
        count = body.get("count", 5)
        if (
            set(body) - {"query", "count", "need_content"}
            or not isinstance(query, str)
            or not 1 <= len(query.strip()) <= 128
            or type(count) is not int
            or not 1 <= count <= 20
            or body.get("need_content", False) is not False
        ):
            raise web.HTTPBadRequest(
                text="搜索须提供 1–128 字符的 query、1–20 的 count，并使用独立抓取接口读取正文"
            )
        try:
            result = await self._search(query.strip(), count)
            return web.json_response(
                {
                    "results": [
                        {
                            "title": hit["title"],
                            "url": hit["url"],
                            "snippet": hit["snippet"],
                            "summary": hit["summary"],
                            "date": hit["published_at"],
                        }
                        for hit in result["results"][:count]
                    ]
                }
            )
        except PluginError as exc:
            return web.json_response({"error": str(exc)}, status=502)

    async def _handle_fetch(self, request: web.Request) -> web.Response:
        body = await self._body(request)
        if set(body) != {"url"}:
            raise web.HTTPBadRequest(text="抓取仅接受 url")
        try:
            validate_page_url(body["url"])
        except PluginError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from None
        try:
            page = await self._fetch(body["url"])
            return web.json_response({"title": page["title"], "content": page["content"]})
        except PluginError as exc:
            return web.json_response({"error": str(exc)}, status=502)
