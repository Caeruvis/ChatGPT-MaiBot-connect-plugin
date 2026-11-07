"""Inspect response framing without exposing upstream bodies or credentials."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .errors import OpenAIError, PluginError, safe_field

MAX_SAMPLE_BYTES = 65536


def media_type(response: httpx.Response, secrets: tuple[str, ...]) -> str:
    value = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if any(secret and secret.lower() in value for secret in secrets):
        return "redacted"
    if not value:
        return "missing"
    return (
        value
        if re.fullmatch(r"[a-z0-9!#$&^_.+-]{1,80}/[a-z0-9!#$&^_.+-]{1,80}", value)
        else "invalid"
    )


def response_metadata(response: httpx.Response, secrets: tuple[str, ...]) -> dict[str, Any]:
    return {
        "http_status": response.status_code,
        "content_type": media_type(response, secrets),
        "request_id": safe_field(response.headers.get("x-request-id"), secrets) or "unknown",
    }


class ResponseFormatError(PluginError):
    def __init__(self, metadata: dict[str, Any], shape: str):
        self.metadata = {**metadata, "body_shape": shape}
        super().__init__(
            "OpenAI 未返回可确认的 SSE 响应；"
            f"HTTP {metadata['http_status']}; content_type={metadata['content_type']}; "
            f"body_shape={shape}; request_id={metadata['request_id']}。"
            "响应正文已隐藏；请用本机 test 命令检查基础文本调用。"
        )


class _PrefixedStream(httpx.AsyncByteStream):
    def __init__(self, prefix: bytes, remaining: AsyncIterator[bytes]):
        self.prefix, self.remaining = prefix, remaining

    async def __aiter__(self):
        yield self.prefix
        async for chunk in self.remaining:
            yield chunk


def _looks_like_sse(prefix: bytes) -> bool | None:
    # A UTF-8 BOM is allowed only at the beginning of the stream.
    if prefix and b"\xef\xbb\xbf".startswith(prefix):
        return None
    if prefix.startswith(b"\xef\xbb\xbf"):
        prefix = prefix[3:]
    prefix = prefix.lstrip(b"\r\n")
    if not prefix:
        return None
    markers = (b"data:", b"event:", b"id:", b"retry:", b":")
    if any(prefix.startswith(marker) for marker in markers):
        return True
    if any(marker.startswith(prefix) for marker in markers):
        return None
    return False


async def inspect_event_stream(
    response: httpx.Response, *, secrets: tuple[str, ...]
) -> tuple[httpx.Response, dict[str, Any]]:
    """Replay peeked bytes once; a completed JSON snapshot is never treated as SSE.

    The real response context owns and closes the underlying HTTP connection.
    A stale or rewritten MIME header alone cannot turn a valid event stream into failure.
    """
    metadata = response_metadata(response, secrets)
    chunks = response.aiter_bytes()
    prefix = bytearray()
    eof = False
    while len(prefix) < MAX_SAMPLE_BYTES:
        try:
            chunk = await anext(chunks)
        except StopAsyncIteration:
            eof = True
            break
        if not chunk:
            continue
        probe = bytes(prefix) + chunk[: MAX_SAMPLE_BYTES - len(prefix)]
        framing = _looks_like_sse(probe)
        if framing is True:
            replay = httpx.Response(
                response.status_code,
                headers={"content-type": "text/event-stream; charset=utf-8"},
                stream=_PrefixedStream(bytes(prefix) + chunk, chunks),
            )
            metadata["framing"] = "sse"
            metadata["header_matches_sse"] = metadata["content_type"] == "text/event-stream"
            return replay, metadata
        prefix.extend(chunk[: MAX_SAMPLE_BYTES - len(prefix)])
        if framing is False:
            break

    # Capture only a bounded body for machine-code extraction and shape classification.
    # A gateway that keeps a non-SSE connection open cannot consume the whole request timeout.
    if not eof and len(prefix) < MAX_SAMPLE_BYTES:
        try:
            async with asyncio.timeout(5):
                async for chunk in chunks:
                    prefix.extend(chunk[: MAX_SAMPLE_BYTES - len(prefix)])
                    if len(prefix) >= MAX_SAMPLE_BYTES:
                        break
                else:
                    eof = True
        except TimeoutError:
            pass
    payload = None
    try:
        payload = json.loads(prefix)
    except (ValueError, UnicodeError):
        pass
    if isinstance(payload, dict) and payload.get("error") is not None:
        raise OpenAIError(response.status_code, payload, metadata["request_id"], secrets=secrets)
    raw = bytes(prefix).lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if isinstance(payload, dict):
        shape = "json_detail" if "detail" in payload else "json_object"
    elif isinstance(payload, list):
        shape = "json_array"
    elif not prefix and eof:
        shape = "empty"
    elif raw.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        shape = "html"
    else:
        shape = "other" if eof else "other_or_truncated"
    raise ResponseFormatError(metadata, shape)
