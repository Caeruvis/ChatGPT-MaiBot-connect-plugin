"""Public-client OAuth + PKCE. No other application's credentials are read."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt

from . import APP_NAME
from .errors import OpenAIError, PluginError
from .store import CredentialStore

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
JWKS_URL = ISSUER + "/.well-known/jwks.json"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPES = {"resource.invoke", "chatgpt.tokens.use.direct"}
TERMINAL_REFRESH_ERRORS = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}


def pkce_challenge(verifier: str) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode()
        .rstrip("=")
    )


def plan_enabled(account: dict[str, Any]) -> bool:
    return PLAN_SCOPES.issubset(set(account.get("scopes", [])))


@dataclass(frozen=True)
class AccessSession:
    client_id: str
    token: str = field(repr=False)


@dataclass(frozen=True)
class LoginAttempt:
    url: str = field(repr=False)
    task: asyncio.Task = field(repr=False)
    redirect_uri: str


class OpenAIPlanAuth:
    def __init__(
        self, store: CredentialStore, client: httpx.AsyncClient, timeout_seconds: int = 300
    ):
        self.store, self.client, self.timeout_seconds = store, client, timeout_seconds
        self._login_lock = asyncio.Lock()
        self._login_task: asyncio.Task | None = None
        self._server: asyncio.Server | None = None
        self._handlers: set[asyncio.Task] = set()
        self._keys: list[dict[str, Any]] = []
        self._keys_expire = 0.0

    async def initialize(self) -> None:
        async with self.store.locked():
            self.store.host_id()

    async def _json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        try:
            async with asyncio.timeout(30):
                response = await self.client.request(method, url, follow_redirects=False, **kwargs)
            try:
                payload = response.json() if len(response.content) <= 1024 * 1024 else None
            except ValueError:
                payload = None
            if not 200 <= response.status_code < 300:
                secret_values = tuple(
                    str(value)
                    for key, value in kwargs.get("data", {}).items()
                    if key in {"code", "code_verifier", "refresh_token", "token", "id_token"}
                )
                raise OpenAIError(
                    response.status_code,
                    payload,
                    response.headers.get("x-request-id"),
                    secrets=secret_values,
                )
            if not isinstance(payload, dict):
                raise PluginError("OpenAI 返回了无效 JSON，未保存凭据。")
            return payload
        except (httpx.HTTPError, TimeoutError):
            raise PluginError("OpenAI 认证网络请求失败，凭据已保留。") from None

    async def begin_login(
        self, *, new: bool = False, client_id: str | None = None, consent: bool = False
    ) -> LoginAttempt:
        async with self._login_lock:
            if self._login_task and not self._login_task.done():
                raise PluginError("已有授权流程等待完成。")
            async with self.store.locked():
                host_id = self.store.host_id()
                data = self.store.read()
                selected = None if new else client_id or data.get("active_client_id")
                saved = self.store.account(selected) if selected else None
            verifier, state, nonce = (secrets.token_urlsafe(n) for n in (64, 32, 32))
            future = asyncio.get_running_loop().create_future()
            port = 0

            async def callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                current = asyncio.current_task()
                self._handlers.add(current)
                status, body = "400 Bad Request", "无效授权回调。"
                try:
                    async with asyncio.timeout(5):
                        raw = await reader.readuntil(b"\r\n\r\n")
                    if len(raw) > 16384:
                        raise ValueError
                    lines = raw.decode("ascii").split("\r\n")
                    method, target, version = lines[0].split(" ")
                    headers = {}
                    for line in lines[1:]:
                        if line:
                            key, value = line.split(":", 1)
                            if key.lower() in headers:
                                raise ValueError
                            headers[key.lower()] = value.strip()
                    parsed = urlsplit(target)
                    if (
                        method != "GET"
                        or version != "HTTP/1.1"
                        or parsed.scheme
                        or parsed.netloc
                        or parsed.path != "/auth/callback"
                        or parsed.fragment
                        or headers.get("host") != f"127.0.0.1:{port}"
                    ):
                        raise ValueError
                    query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=20)
                    if any(len(v) != 1 for v in query.values()):
                        raise ValueError
                    values = {k: v[0] for k, v in query.items()}
                    if not hmac.compare_digest(values.get("state", ""), state):
                        raise ValueError
                    if future.done():
                        status, body = "409 Conflict", "此授权回调已处理。"
                    elif not values.get("error") and not values.get("code"):
                        raise ValueError
                    else:
                        future.set_result(values)
                        status, body = "200 OK", "已收到授权回调。请返回应用确认最终授权结果。"
                except (
                    ValueError,
                    TimeoutError,
                    asyncio.IncompleteReadError,
                    asyncio.LimitOverrunError,
                ):
                    pass
                finally:
                    try:
                        data = body.encode("utf-8")
                        writer.write(
                            (
                                f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                                f"Content-Length: {len(data)}\r\nCache-Control: no-store\r\n"
                                "Referrer-Policy: no-referrer\r\nConnection: close\r\n\r\n"
                            ).encode()
                            + data
                        )
                        async with asyncio.timeout(2):
                            await writer.drain()
                    except (ConnectionError, TimeoutError):
                        pass
                    writer.close()
                    self._handlers.discard(current)

            self._server = await asyncio.start_server(callback, "127.0.0.1", 0, limit=16384)
            port = self._server.sockets[0].getsockname()[1]
            redirect_uri = f"http://127.0.0.1:{port}/auth/callback"
            params = {
                "client_id": selected or "dynamic_agent_client",
                "ext_agent_host_id": host_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": SCOPES,
                "resource": RESOURCE,
                "state": state,
                "nonce": nonce,
                "code_challenge_method": "S256",
                "code_challenge": pkce_challenge(verifier),
            }
            if not selected:
                params["agent_name_hint"] = APP_NAME
            # Deliberately omit identity hints: authorization URLs may be delivered to an admin chat.
            if consent:
                params["prompt"] = "consent"
            self._login_task = asyncio.create_task(
                self._finish_login(
                    self._server, future, selected, saved, nonce, verifier, redirect_uri
                )
            )
            return LoginAttempt(
                f"{AUTHORIZE_URL}?{urlencode(params)}", self._login_task, redirect_uri
            )

    async def _finish_login(self, server, future, selected, saved, nonce, verifier, redirect_uri):
        try:
            try:
                result = await asyncio.wait_for(future, self.timeout_seconds)
            except TimeoutError:
                raise PluginError("本次 OAuth 授权已超时，请重新开始。") from None
            finally:
                server.close()
                await server.wait_closed()
            if result.get("error"):
                raise PluginError("用户拒绝授权或 OpenAI 未完成授权，未交换令牌。")
            issued = result.get("client_id") or selected
            self.store.validate_client_id(issued)
            if selected and issued != selected:
                raise PluginError("回调 client ID 与所选账号不匹配，未交换令牌。")
            # Retain registration even if the one-use code later expires; do not activate it.
            async with self.store.locked():
                existing = self.store.read()["accounts"].get(issued)
                if existing is None:
                    self.store.save({"client_id": issued, "registered_at": time.time()})
                expected_subject = (
                    saved.get("subject") if saved else (existing or {}).get("subject")
                )
            tokens = await self._json(
                "POST",
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": issued,
                    "code": result["code"],
                    "code_verifier": verifier,
                    "redirect_uri": redirect_uri,
                    "resource": RESOURCE,
                },
            )
            claims = await self._verify_identity(tokens.get("id_token"), issued, nonce)
            if expected_subject and claims["sub"] != expected_subject:
                raise PluginError("重新授权的身份与所选账号不同，原凭据未替换。")
            account = {"client_id": issued, "subject": claims["sub"], "issuer": claims["iss"]}
            account.update(self._token_fields(tokens))
            async with self.store.locked():
                self.store.save(account, activate=True)
            return {"client_id": issued, "plan_enabled": plan_enabled(account)}
        finally:
            server.close()
            await server.wait_closed()

    async def _verify_identity(self, token: Any, client_id: str, nonce: str | None = None):
        phase = "登录" if nonce is not None else "刷新"

        def invalid(reason: str) -> PluginError:
            return PluginError(f"OIDC ID token 校验失败（{phase}，{reason}），未保存新凭据。")

        if token is None:
            raise invalid("OpenAI 未返回 ID token")
        if not isinstance(token, str) or not token:
            raise invalid("ID token 格式无效")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise invalid("ID token 头部格式无效") from None
        if header.get("alg") != "RS256":
            raise invalid("签名算法不受支持")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise invalid("缺少有效的签名密钥标识 kid")

        if time.monotonic() >= self._keys_expire or not any(
            key.get("kid") == kid for key in self._keys
        ):
            payload = await self._json("GET", JWKS_URL)
            keys = payload.get("keys")
            if not isinstance(keys, list) or any(not isinstance(key, dict) for key in keys):
                raise invalid("OpenAI 签名公钥列表格式无效")
            self._keys, self._keys_expire = keys, time.monotonic() + 300
        matches = [
            key
            for key in self._keys
            if key.get("kid") == kid
            and key.get("kty") == "RSA"
            and key.get("use", "sig") == "sig"
            and key.get("alg", "RS256") == "RS256"
        ]
        if len(matches) != 1:
            raise invalid("找不到唯一匹配的 OpenAI 签名公钥")
        try:
            key = jwt.PyJWK.from_dict(matches[0], algorithm="RS256").key
        except (jwt.PyJWTError, ValueError, TypeError, KeyError):
            raise invalid("OpenAI 签名公钥格式无效") from None
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=client_id,
                issuer=ISSUER,
                options={"require": ["iss", "aud", "exp", "iat", "sub"]},
            )
        except (jwt.PyJWTError, ValueError, TypeError, OverflowError, KeyError) as exc:
            reason = {
                jwt.InvalidSignatureError: "签名不匹配",
                jwt.InvalidIssuerError: "签发方 iss 不匹配",
                jwt.InvalidAudienceError: "客户端 aud 不匹配",
                jwt.ExpiredSignatureError: "令牌已过期，请检查系统时间",
                jwt.ImmatureSignatureError: "令牌尚未生效，请检查系统时间",
                jwt.MissingRequiredClaimError: "缺少必需的身份字段",
                jwt.exceptions.InvalidSubjectError: "账号标识 sub 无效",
            }.get(type(exc), "令牌格式或字段无效")
            raise invalid(reason) from None
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise invalid("账号标识 sub 无效")
        multiple_audiences = isinstance(claims["aud"], list) and len(claims["aud"]) > 1
        azp = claims.get("azp", None if multiple_audiences else client_id)
        if azp != client_id:
            raise invalid("授权方 azp 缺失或不匹配")
        if nonce is not None:
            token_nonce = claims.get("nonce")
            if not isinstance(token_nonce, str) or not token_nonce:
                raise invalid("缺少有效的登录 nonce")
            if not token_nonce.isascii() or not hmac.compare_digest(token_nonce, nonce):
                raise invalid("登录 nonce 不匹配")
        return claims

    @staticmethod
    def _token_fields(tokens: dict[str, Any], previous: dict[str, Any] | None = None):
        previous = previous or {}
        token = tokens.get("access_token")
        if (
            not isinstance(token, str)
            or not token
            or tokens.get("token_type", "Bearer").lower() != "bearer"
        ):
            raise PluginError("OpenAI 未返回有效 Bearer access token。")
        scopes = tokens.get("scope", " ".join(previous.get("scopes", [])))
        if not isinstance(scopes, str):
            raise PluginError("OpenAI scope 格式无效。")
        try:
            lifetime = float(tokens["expires_in"])
            if not 0 < lifetime <= 31536000:
                raise ValueError
        except (ValueError, TypeError, KeyError):
            raise PluginError("OpenAI 未返回有效 token 有效期。") from None
        fields = {
            "access_token": token,
            "token_type": "Bearer",
            "scopes": scopes.split(),
            "expires_at": time.time() + lifetime,
            "saved_at": time.time(),
        }
        for name in ("refresh_token", "id_token"):
            value = tokens.get(name, previous.get(name))
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise PluginError("OpenAI 返回的 token 格式无效。")
                fields[name] = value
        fields["plan_enabled"] = plan_enabled(fields)
        return fields

    async def session(
        self, *, client_id: str | None = None, rejected_token: str | None = None
    ) -> AccessSession:
        # The file lock spans HTTP refresh, serializing rotating-token refresh across processes too.
        async with self.store.locked():
            account = self.store.account(client_id)
            if not account.get("access_token"):
                raise PluginError("当前账号没有可用的授权凭据，请在本机重新完成 ChatGPT 授权。")
            if not plan_enabled(account):
                raise PluginError(
                    "当前账号缺少 ChatGPT 方案调用权限（需要 resource.invoke 和 "
                    "chatgpt.tokens.use.direct）。请使用本机助手 login --consent 重新授权。"
                )
            refresh = time.time() >= float(account.get("expires_at", 0)) - 60
            refresh |= rejected_token is not None and account["access_token"] == rejected_token
            if refresh:
                if not account.get("refresh_token"):
                    raise PluginError("缺少 refresh token，需要重新授权。")
                try:
                    fresh = await self._json(
                        "POST",
                        TOKEN_URL,
                        data={
                            "grant_type": "refresh_token",
                            "client_id": account["client_id"],
                            "refresh_token": account["refresh_token"],
                            "resource": RESOURCE,
                        },
                    )
                except OpenAIError as exc:
                    if exc.code in TERMINAL_REFRESH_ERRORS:
                        self.store.clear_tokens(account["client_id"])
                    raise
                if "id_token" in fresh:
                    claims = await self._verify_identity(fresh["id_token"], account["client_id"])
                    if claims["sub"] != account.get("subject"):
                        raise PluginError("刷新返回了其他身份，未覆盖凭据。")
                account.update(self._token_fields(fresh, account))
                self.store.save(account)
                if not plan_enabled(account):
                    raise PluginError("刷新后的 grant 不再包含方案调用权限。")
            return AccessSession(account["client_id"], account["access_token"])

    async def accounts(self) -> dict[str, Any]:
        async with self.store.locked():
            data = self.store.read()
            return {
                "active_client_id": data.get("active_client_id"),
                "accounts": [
                    {
                        "client_id": key,
                        "connected": bool(value.get("access_token")),
                        "plan_enabled": plan_enabled(value),
                        "expires_at": value.get("expires_at"),
                    }
                    for key, value in data["accounts"].items()
                ],
            }

    async def switch(self, client_id: str) -> None:
        async with self.store.locked():
            self.store.switch(client_id)

    async def logout(self, client_id: str | None = None) -> bool:
        await self.cancel_login()
        confirmed = False
        async with self.store.locked():
            account = self.store.account(client_id)
            try:
                if account.get("refresh_token"):
                    discovery = await self._json("GET", DISCOVERY_URL)
                    endpoint = discovery.get("revocation_endpoint", "")
                    parsed = urlsplit(endpoint)
                    if (
                        parsed.scheme != "https"
                        or parsed.netloc != "auth.openai.com"
                        or not parsed.path.startswith("/")
                        or parsed.query
                        or parsed.fragment
                    ):
                        raise PluginError("拒绝向非 OpenAI 认证端点发送撤销凭据。")
                    for attempt in range(2):
                        try:
                            async with asyncio.timeout(30):
                                response = await self.client.post(
                                    endpoint,
                                    follow_redirects=False,
                                    data={
                                        "token": account["refresh_token"],
                                        "token_type_hint": "refresh_token",
                                        "client_id": account["client_id"],
                                    },
                                )
                            if response.status_code == 200:
                                confirmed = True
                                break
                            if response.status_code < 500:
                                break
                        except (httpx.HTTPError, TimeoutError):
                            pass
                        if attempt == 0:
                            await asyncio.sleep(0.5)
            except PluginError:
                pass
            finally:
                self.store.clear_tokens(account["client_id"])
        return confirmed

    async def cancel_login(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._login_task and not self._login_task.done():
            self._login_task.cancel()
            await asyncio.gather(self._login_task, return_exceptions=True)
        handlers = list(self._handlers)
        for task in handlers:
            task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
