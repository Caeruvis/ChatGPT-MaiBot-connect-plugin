"""Credential-free static page extraction with public-IP pinning on every redirect."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .errors import PluginError


def validate_page_url(value: str) -> httpx.URL:
    try:
        if not isinstance(value, str) or not value.strip() or len(value) > 8192:
            raise ValueError
        url = httpx.URL(value.strip())
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
            raise ValueError
        if url.port not in (None, 80, 443):
            raise ValueError
        if any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError
        return url.copy_with(fragment=None)
    except (ValueError, httpx.InvalidURL):
        raise PluginError(
            "网页地址须为公开 HTTP(S) 地址，仅允许 80/443 端口且不能含凭据。"
        ) from None


async def resolve_public(host: str, port: int) -> str:
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )
    except OSError:
        raise PluginError("网页域名解析失败。") from None
    ips = list(dict.fromkeys(row[4][0] for row in addresses))
    if not ips:
        raise PluginError("网页域名未返回地址。")
    for address in ips:
        ip = ipaddress.ip_address(address)
        if (
            not ip.is_global
            or ip.is_multicast
            or ip.is_reserved
            or (
                isinstance(ip, ipaddress.IPv6Address)
                and (ip.ipv4_mapped or ip.sixtofour or ip.teredo)
            )
        ):
            raise PluginError("拒绝读取本机、内网或保留地址的网页。")
    return ips[0]


def extract_page(data: bytes, content_type: str, url: str, max_chars: int) -> dict[str, Any]:
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type in {"text/html", "application/xhtml+xml"}:
        soup = BeautifulSoup(data, "html.parser")
        title = soup.title.get_text(" ", strip=True)[:500] if soup.title else ""
        for node in soup.select("script,style,noscript,template,nav,header,footer,form,iframe,svg"):
            node.decompose()
        root = (
            soup.select_one("article")
            or soup.select_one("main")
            or soup.select_one('[role="main"]')
            or soup.body
            or soup
        )
        text = root.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
    elif media_type in {"text/plain", "text/markdown", "text/x-markdown"}:
        response = httpx.Response(200, content=data, headers={"Content-Type": content_type})
        title, text = "", response.text.strip()
    else:
        raise PluginError("网页类型不支持；仅提取 HTML、纯文本或 Markdown。")
    text = text.replace("\x00", "")
    if not text.strip():
        raise PluginError("网页没有可提取的正文；动态页面可能需要浏览器。")
    if len(text) > max_chars:
        raise PluginError("网页正文超过传输上限，请读取更小的页面。")
    return {
        "url": url,
        "title": title or httpx.URL(url).host,
        "content": text,
        "content_kind": "page_text",
        "service": "local_page_fetch",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


class PageFetcher:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport

    async def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float = 30,
        max_bytes: int = 8 * 1024 * 1024,
        max_chars: int = 200000,
    ) -> dict[str, Any]:
        current = validate_page_url(url)
        try:
            async with asyncio.timeout(timeout_seconds):
                for _ in range(6):
                    address = await resolve_public(
                        current.host, current.port or (443 if current.scheme == "https" else 80)
                    )
                    pinned = current.copy_with(host=address)
                    # A new client per hop prevents cookie propagation. The IP is pinned while
                    # Host, TLS SNI and certificate verification retain the original hostname.
                    async with httpx.AsyncClient(
                        transport=self.transport,
                        trust_env=False,
                        follow_redirects=False,
                        timeout=httpx.Timeout(timeout_seconds, connect=min(15, timeout_seconds)),
                    ) as client:
                        async with client.stream(
                            "GET",
                            pinned,
                            headers={
                                "Host": current.netloc.decode("ascii"),
                                "User-Agent": "MaiBot-PageReader/0.2",
                                "Accept": "text/html,text/plain,text/markdown,application/xhtml+xml",
                            },
                            extensions={"sni_hostname": current.host},
                        ) as response:
                            if response.status_code in {301, 302, 303, 307, 308}:
                                location = response.headers.get("location")
                                if not location:
                                    raise PluginError("网页重定向缺少目标地址。")
                                current = validate_page_url(str(current.join(location)))
                                continue
                            if not 200 <= response.status_code < 300:
                                raise PluginError(f"网页读取失败，HTTP {response.status_code}。")
                            chunks, size = [], 0
                            async for chunk in response.aiter_bytes():
                                size += len(chunk)
                                if size > max_bytes:
                                    raise PluginError("网页响应超过大小上限。")
                                chunks.append(chunk)
                            return await asyncio.to_thread(
                                extract_page,
                                b"".join(chunks),
                                response.headers.get("content-type", ""),
                                str(current),
                                max_chars,
                            )
                raise PluginError("网页重定向次数超过上限。")
        except (httpx.HTTPError, httpx.InvalidURL, TimeoutError):
            raise PluginError("网页读取超时、网络失败或重定向地址无效。") from None
