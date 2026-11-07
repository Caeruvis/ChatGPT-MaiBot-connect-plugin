"""Fixed-origin HTTP requests and terminal-event-only SSE success handling."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .auth import RESOURCE, AccessSession, OpenAIPlanAuth
from .errors import OpenAIError, PluginError
from .mapping import build_responses_body, parse_completed_response
from .stream_output import CompletedOutput
from .transport import inspect_event_stream


async def sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """SSE frames can contain multiple data lines and heartbeat comments."""
    data: list[str] = []
    size = 0
    event_name = ""
    first_line = True
    async for line in response.aiter_lines():
        if first_line:
            line = line.removeprefix("\ufeff")
            first_line = False
        if line == "":
            if data:
                raw = "\n".join(data)
                if raw != "[DONE]":
                    try:
                        event = json.loads(raw)
                        if not isinstance(event, dict):
                            raise ValueError
                    except ValueError:
                        raise PluginError("OpenAI SSE 事件 JSON 无效，未返回部分文本。") from None
                    if event_name and event.get("type", event_name) != event_name:
                        raise PluginError("OpenAI SSE event 与 data.type 不一致。")
                    event.setdefault("type", event_name)
                    yield event
            data, size, event_name = [], 0, ""
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            size += len(value)
            if size > 16 * 1024 * 1024:
                raise PluginError("OpenAI SSE 事件过大。")
            data.append(value)
        elif field == "event":
            event_name = value
    # An unterminated event is not dispatched. EOF never implies success.


class PlanProvider:
    def __init__(
        self, auth: OpenAIPlanAuth, client: httpx.AsyncClient, *, timeout_seconds: int = 240
    ):
        self.auth, self.client, self.timeout_seconds = auth, client, timeout_seconds
        self._catalog: dict[str, tuple[str, float, list[dict[str, str]]]] = {}

    async def _request(self, method: str, path: str, session: AccessSession):
        for attempt in range(2):
            response = await self.client.request(
                method,
                RESOURCE + path,
                follow_redirects=False,
                headers={"Authorization": f"Bearer {session.token}"},
            )
            if response.status_code == 401 and attempt == 0:
                session = await self.auth.session(
                    client_id=session.client_id, rejected_token=session.token
                )
                continue
            if not 200 <= response.status_code < 300:
                raise self._error(response, session)
            try:
                payload = response.json()
            except ValueError:
                raise PluginError("OpenAI 模型目录不是有效 JSON。") from None
            return payload, session
        raise PluginError("OpenAI 刷新后仍未接受身份。")

    @staticmethod
    def _error(response: httpx.Response, session: AccessSession):
        try:
            payload = response.json() if len(response.content) <= 1024 * 1024 else None
        except ValueError:
            payload = None
        return OpenAIError(
            response.status_code,
            payload,
            response.headers.get("x-request-id"),
            secrets=(session.token,),
        )

    async def _models(self, session: AccessSession, *, cached: bool):
        previous = self._catalog.get(session.client_id)
        if cached and previous and previous[0] == session.token and time.monotonic() < previous[1]:
            return previous[2], session
        payload, session = await self._request("GET", "/models", session)
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise PluginError("账号模型目录缺少 models 数组。")
        models = [
            {"slug": item["slug"], "display_name": item.get("display_name") or item["slug"]}
            for item in payload["models"]
            if isinstance(item, dict)
            and item.get("visibility") == "list"
            and isinstance(item.get("slug"), str)
            and item["slug"]
        ]
        self._catalog[session.client_id] = (session.token, time.monotonic() + 60, models)
        return models, session

    async def models(self):
        try:
            async with asyncio.timeout(self.timeout_seconds):
                result, _ = await self._models(await self.auth.session(), cached=False)
                return result
        except (httpx.HTTPError, TimeoutError):
            raise PluginError("OpenAI 模型目录请求超时或网络失败。") from None

    async def _usage_pause(self, client_id: str, *, paused: bool | None = None):
        async with self.auth.store.locked():
            account = self.auth.store.account(client_id)
            if paused is not None:
                account["usage_paused"] = paused
                self.auth.store.save(account)
            return account.get("usage_paused", False)

    async def resume(self):
        async with self.auth.store.locked():
            account = self.auth.store.account()
            account["usage_paused"] = False
            self.auth.store.save(account)

    async def response(
        self, request: dict[str, Any], *, enable_web_search=False, enable_tool_controls=False
    ):
        body, reverse = build_responses_body(
            request, enable_web_search=enable_web_search, enable_tool_controls=enable_tool_controls
        )
        completed, diagnostics = await self.complete(body)
        result = parse_completed_response(completed, reverse)
        result["raw_data"] = diagnostics
        return result

    async def complete(self, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal transport shared by chat and hosted search; never exposed as an RPC API."""
        try:
            async with asyncio.timeout(self.timeout_seconds):
                session = await self.auth.session()
                if await self._usage_pause(session.client_id):
                    raise PluginError(
                        "此账号因用量错误暂停推理；请在 ChatGPT Usage 检查后运行 resume。"
                    )
                models, session = await self._models(session, cached=True)
                if body["model"] not in {item["slug"] for item in models}:
                    raise PluginError(
                        "模型不在当前账号可见目录中；请重新列出模型并使用返回的 slug。"
                    )
                refreshed, transient_retries = False, 0
                while True:
                    try:
                        # A logout or token rotation in another process must affect retries too.
                        session = await self.auth.session(client_id=session.client_id)
                        models, session = await self._models(session, cached=True)
                        if body["model"] not in {item["slug"] for item in models}:
                            raise PluginError("模型已不在此账号的可见目录中。")
                        async with self.client.stream(
                            "POST",
                            RESOURCE + "/responses",
                            follow_redirects=False,
                            headers={
                                "Authorization": f"Bearer {session.token}",
                                "Accept": "text/event-stream",
                            },
                            json=body,
                        ) as response:
                            if response.status_code == 401 and not refreshed:
                                await response.aread()
                                session = await self.auth.session(
                                    client_id=session.client_id, rejected_token=session.token
                                )
                                refreshed = True
                                continue
                            if response.status_code != 200:
                                await response.aread()
                                raise self._error(response, session)
                            event_response, metadata = await inspect_event_stream(
                                response, secrets=(session.token,)
                            )
                            output = CompletedOutput()
                            async for event in sse_events(event_response):
                                output.observe(event)
                                kind = event.get("type")
                                if kind in {"response.failed", "response.incomplete", "error"}:
                                    failed = event.get("response") or event
                                    if kind == "error" and "error" not in failed:
                                        failed = {
                                            "error": {
                                                "code": event.get("code"),
                                                "param": event.get("param"),
                                            }
                                        }
                                    raise OpenAIError(
                                        response.status_code,
                                        failed,
                                        response.headers.get("x-request-id"),
                                        secrets=(session.token,),
                                    )
                                if kind == "response.completed":
                                    completed = event.get("response")
                                    if not isinstance(completed, dict):
                                        raise PluginError("完成事件缺少响应快照。")
                                    completed, output_source = output.finish(completed)
                                    diagnostics = {
                                        "transport": metadata,
                                        "output_source": output_source,
                                        "web_search_completed": sum(
                                            1
                                            for item in completed["output"]
                                            if isinstance(item, dict)
                                            and item.get("type") == "web_search_call"
                                            and item.get("status") == "completed"
                                        ),
                                    }
                                    return completed, diagnostics
                            raise PluginError(
                                "SSE 已结束但未收到 response.completed，未返回部分文本。"
                            )
                    except OpenAIError as exc:
                        if exc.code == "subscription_sharing_usage_limit_exceeded":
                            await self._usage_pause(session.client_id, paused=True)
                        if (
                            exc.status == 503
                            or exc.code
                            in {
                                "subscription_sharing_usage_unavailable",
                                "subscription_sharing_user_unavailable",
                            }
                        ) and transient_retries == 0:
                            transient_retries += 1
                            await asyncio.sleep(0.5)
                            continue
                        raise
        except (httpx.HTTPError, TimeoutError):
            raise PluginError("OpenAI 请求超时或连接中断，未返回部分文本；凭据已保留。") from None
