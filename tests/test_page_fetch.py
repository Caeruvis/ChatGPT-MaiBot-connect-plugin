import asyncio
import socket

import httpx
import pytest

from chatgpt_maibot import page_fetch
from chatgpt_maibot.errors import PluginError
from chatgpt_maibot.page_fetch import PageFetcher, extract_page, resolve_public, validate_page_url


@pytest.fixture
def public_dns(monkeypatch):
    calls = []

    async def resolve(host, port):
        calls.append((host, port))
        if host == "127.0.0.1":
            raise PluginError("内网")
        return "93.184.216.34"

    monkeypatch.setattr(page_fetch, "resolve_public", resolve)
    return calls


async def test_fetch_pins_public_ip_keeps_tls_host_and_never_forwards_auth(public_dns):
    def handler(request):
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "docs.example"
        assert request.extensions["sni_hostname"] == "docs.example"
        assert not {"authorization", "cookie", "proxy-authorization"} & set(request.headers)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=(
                "<title>页面</title><nav>菜单</nav><main><h1>正文标题</h1>"
                "<p>原始事实与数字123。</p><script>bad()</script></main>"
            ).encode(),
        )

    result = await PageFetcher(transport=httpx.MockTransport(handler)).fetch(
        "https://docs.example/a#b"
    )
    assert result["url"] == "https://docs.example/a"
    assert result["title"] == "页面"
    assert result["content_kind"] == "page_text"
    assert "123" in result["content"] and "菜单" not in result["content"]
    assert "bad()" not in result["content"]


async def test_redirect_is_revalidated_and_cookie_is_not_forwarded(public_dns):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                302, headers={"location": "https://other.example/b", "set-cookie": "secret=x"}
            )
        assert "cookie" not in request.headers
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="真实文本")

    result = await PageFetcher(transport=httpx.MockTransport(handler)).fetch(
        "https://first.example"
    )
    assert public_dns == [("first.example", 443), ("other.example", 443)]
    assert result["url"] == "https://other.example/b"


async def test_redirect_to_private_address_is_rejected_before_second_request(public_dns):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    with pytest.raises(PluginError, match="内网"):
        await PageFetcher(transport=httpx.MockTransport(handler)).fetch("https://public.example")
    assert len(requests) == 1


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "::1",
        "fc00::1",
        "::ffff:8.8.8.8",
        "224.0.0.1",
        "ff02::1",
        "2002:7f00:1::",
    ],
)
async def test_private_and_mapped_dns_addresses_are_blocked(monkeypatch, address):
    async def resolve(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    with pytest.raises(PluginError, match="拒绝"):
        await resolve_public("host.example", 80)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:password@host",
        "https://host:8080",
        "https://host/\nsecret",
    ],
)
def test_unsafe_url_forms(url):
    with pytest.raises(PluginError):
        validate_page_url(url)


@pytest.mark.parametrize(
    "kind,data,limit", [("application/pdf", b"pdf", 100), ("text/plain", b"long", 1)]
)
def test_binary_and_oversized_body_are_explicit_errors(kind, data, limit):
    with pytest.raises(PluginError):
        extract_page(data, kind, "https://example.com", limit)


async def test_oversized_response_is_rejected(public_dns):
    fetcher = PageFetcher(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"12345")
        )
    )
    with pytest.raises(PluginError, match="大小上限"):
        await fetcher.fetch("https://example.com", max_bytes=3)
