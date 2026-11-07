from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from chatgpt_maibot.auth import ISSUER, SCOPES, OpenAIPlanAuth
from chatgpt_maibot.provider import PlanProvider
from chatgpt_maibot.store import CredentialStore

CLIENT_ID = "oaiapp_test_account"
ACCESS = "fake-access-sentinel-DO-NOT-LOG"
REFRESH = "fake-refresh-sentinel-DO-NOT-LOG"


def sse(*events):
    return "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events)


def completed(text="Hello", **kwargs):
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            **kwargs,
        },
    }


@pytest.fixture(scope="session")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
async def env(tmp_path, signing_key):
    state = SimpleNamespace(
        nonce="test-nonce",
        token_overrides={},
        claims_overrides={},
        requests=[],
        response_queue=[],
        refresh_count=0,
        token_error=None,
        models=[{"slug": "test-model", "display_name": "Test model", "visibility": "list"}],
    )
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    jwk.update(kid="test-key", use="sig", alg="RS256")

    def id_token(**overrides):
        claims = {
            "iss": ISSUER,
            "aud": CLIENT_ID,
            "sub": "test-user",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "nonce": state.nonce,
            **state.claims_overrides,
            **overrides,
        }
        return jwt.encode(claims, signing_key, algorithm="RS256", headers={"kid": "test-key"})

    async def handler(request):
        state.requests.append(request)
        path = request.url.path
        if path == "/.well-known/jwks.json":
            return httpx.Response(200, json={"keys": [jwk]})
        if path == "/api/accounts/oauth/token":
            from urllib.parse import parse_qs

            form = parse_qs(request.content.decode())
            refresh = form["grant_type"] == ["refresh_token"]
            if refresh:
                state.refresh_count += 1
            if state.token_error:
                return httpx.Response(
                    400, json={"error": state.token_error}, headers={"x-request-id": "req_auth"}
                )
            tokens = {
                "access_token": ACCESS + ("-fresh" if refresh else ""),
                "refresh_token": REFRESH + ("-rotated" if refresh else ""),
                "expires_in": 3600,
                "scope": SCOPES,
                "token_type": "Bearer",
            }
            if not refresh:
                tokens["id_token"] = id_token()
            tokens.update(state.token_overrides)
            return httpx.Response(200, json=tokens)
        if path == "/v1/models":
            return httpx.Response(200, json={"models": state.models})
        if path == "/v1/responses":
            result = (
                state.response_queue.pop(0)
                if state.response_queue
                else httpx.Response(
                    200, text=sse(completed()), headers={"content-type": "text/event-stream"}
                )
            )
            if isinstance(result, Exception):
                raise result
            return result
        if path == "/.well-known/openid-configuration":
            return httpx.Response(
                200, json={"revocation_endpoint": ISSUER + "/api/accounts/oauth/revoke"}
            )
        if path == "/api/accounts/oauth/revoke":
            return httpx.Response(200)
        raise AssertionError(f"Unexpected mocked route: {request.method} {path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    store = CredentialStore(tmp_path / "credentials")
    auth = OpenAIPlanAuth(store, client, timeout_seconds=2)
    await auth.initialize()
    result = SimpleNamespace(
        state=state,
        client=client,
        store=store,
        auth=auth,
        provider=PlanProvider(auth, client),
        id_token=id_token,
    )
    yield result
    await auth.cancel_login()
    await client.aclose()


async def save_account(env, *, client_id=CLIENT_ID, expires_at=None, scopes=SCOPES):
    account = {
        "client_id": client_id,
        "issuer": ISSUER,
        "subject": "test-user",
        "access_token": ACCESS,
        "refresh_token": REFRESH,
        "id_token": env.id_token(),
        "expires_at": expires_at if expires_at is not None else time.time() + 3600,
        "scopes": scopes.split(),
        "plan_enabled": True,
    }
    async with env.store.locked():
        env.store.save(account, activate=True)
    return account


def request_body():
    return {
        "model_info": {"model_identifier": "test-model"},
        "message_list": [{"role": "user", "content": "Hello"}],
    }
