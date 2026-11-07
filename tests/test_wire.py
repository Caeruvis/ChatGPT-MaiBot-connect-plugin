"""Real loopback HTTP/SSE, with a test-only transport redirecting fixed OpenAI URLs."""

import asyncio
import json

import httpx
import pytest
from conftest import completed, request_body, save_account, sse

from chatgpt_maibot.errors import PluginError
from chatgpt_maibot.provider import PlanProvider


class LoopbackTransport(httpx.AsyncBaseTransport):
    def __init__(self, port):
        self.port = port
        self.inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request):
        assert request.url.host == "api.openai.com"
        local = httpx.Request(
            request.method,
            request.url.copy_with(scheme="http", host="127.0.0.1", port=self.port),
            headers=request.headers,
            content=await request.aread(),
        )
        return await self.inner.handle_async_request(local)

    async def aclose(self):
        await self.inner.aclose()


@pytest.mark.parametrize("disconnect", [False, True])
@pytest.mark.parametrize("empty_terminal", [False, True])
@pytest.mark.parametrize(
    "sse_content_type",
    ["text/event-stream", "Text/Event-Stream; charset=UTF-8", "application/json"],
)
async def test_actual_http_fragmentation_and_disconnect(
    env, disconnect, empty_terminal, sse_content_type
):
    await save_account(env)
    received = []

    async def handle(reader, writer):
        raw = await reader.readuntil(b"\r\n\r\n")
        headers = {}
        for line in raw.decode().split("\r\n")[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.lower()] = value.strip()
        body = await reader.readexactly(int(headers.get("content-length", 0)))
        received.append((raw.split(b"\r\n", 1)[0], body))
        if raw.startswith(b"GET"):
            content_type = "application/json"
            data = json.dumps({"models": [{"slug": "test-model", "visibility": "list"}]}).encode()
        else:
            content_type = sse_content_type
            final = completed("南京：晴")
            events = [{"type": "response.output_text.delta", "delta": "partial"}]
            if empty_terminal:
                item = {**final["response"]["output"][0], "id": "msg_wire", "status": "completed"}
                events.append(
                    {"type": "response.output_item.done", "output_index": 0, "item": item}
                )
                final["response"]["output"] = []
            data = sse(*events, final).encode()
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode()
        )
        if disconnect and raw.startswith(b"POST"):
            writer.write(data[:40])
        else:
            # Deliberately split in the middle of event lines and UTF-8 characters.
            for start in range(0, len(data), 7):
                writer.write(data[start : start + 7])
                await writer.drain()
                await asyncio.sleep(0)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with httpx.AsyncClient(transport=LoopbackTransport(port)) as client:
            provider = PlanProvider(env.auth, client)
            if disconnect:
                with pytest.raises(PluginError, match="连接中断"):
                    await provider.response(request_body())
            else:
                assert (await provider.response(request_body()))["content"] == "南京：晴"
        assert len(received) == 2
        assert json.loads(received[1][1])["store"] is False
    finally:
        server.close()
        await server.wait_closed()
