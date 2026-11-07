import asyncio
import json
import os
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
import pytest
from conftest import ACCESS, CLIENT_ID, REFRESH, save_account
from cryptography.hazmat.primitives.asymmetric import rsa

from chatgpt_maibot.auth import OpenAIPlanAuth, pkce_challenge
from chatgpt_maibot.errors import OpenAIError, PluginError
from chatgpt_maibot.store import CredentialStore


async def start_login(env, **kwargs):
    attempt = await env.auth.begin_login(**kwargs)
    params = {k: v[0] for k, v in parse_qs(urlsplit(attempt.url).query).items()}
    env.state.nonce = params["nonce"]
    return attempt, params


async def callback(attempt, params, **overrides):
    query = {
        "state": params["state"],
        "code": "fake-one-use-code",
        "client_id": CLIENT_ID,
        **overrides,
    }
    async with httpx.AsyncClient(trust_env=False) as browser:
        return await browser.get(attempt.redirect_uri + "?" + urlencode(query))


async def test_login_pkce_and_private_atomic_storage(env, caplog):
    attempt, params = await start_login(env)
    assert params["client_id"] == "dynamic_agent_client"
    assert params["code_challenge_method"] == "S256"
    assert "id_token_hint" not in params and "login_hint" not in params
    assert urlsplit(attempt.redirect_uri).hostname == "127.0.0.1"
    assert (await callback(attempt, params)).status_code == 200
    assert (await attempt.task)["plan_enabled"] is True
    account = env.store.account()
    assert account["client_id"] == CLIENT_ID
    assert account["access_token"] == ACCESS
    exchange = next(r for r in env.state.requests if r.url.path.endswith("oauth/token"))
    form = parse_qs(exchange.content.decode())
    assert form["client_id"] == [CLIENT_ID]
    assert pkce_challenge(form["code_verifier"][0]) == params["code_challenge"]
    assert form["redirect_uri"] == [attempt.redirect_uri]
    assert "client_secret" not in form
    if os.name != "nt":
        assert env.store.accounts_path.stat().st_mode & 0o777 == 0o600
        assert env.store.host_id_path.stat().st_mode & 0o777 == 0o600
        assert env.store.data_dir.stat().st_mode & 0o777 == 0o700
    for secret in (ACCESS, REFRESH, attempt.url, form["code_verifier"][0], "fake-one-use-code"):
        assert secret not in caplog.text
    assert not list(env.store.data_dir.glob(".*.tmp"))


@pytest.mark.parametrize("bad_state", ["", "wrong-state"])
async def test_bad_state_does_not_poison_pending_login(env, bad_state):
    attempt, params = await start_login(env)
    assert (await callback(attempt, params, state=bad_state)).status_code == 400
    assert not attempt.task.done()
    assert not env.state.requests
    await callback(attempt, params)
    await attempt.task


async def test_missing_state_and_duplicate_parameters_rejected(env):
    attempt, params = await start_login(env)
    async with httpx.AsyncClient(trust_env=False) as browser:
        r = await browser.get(attempt.redirect_uri + "?code=fake-code")
        assert r.status_code == 400
        r = await browser.get(
            attempt.redirect_uri
            + "?"
            + urlencode({"state": params["state"]})
            + "&state=duplicate&code=x"
        )
        assert r.status_code == 400
    assert not env.state.requests


async def test_denial_does_not_exchange_code(env):
    attempt, params = await start_login(env)
    await callback(attempt, params, error="access_denied")
    with pytest.raises(PluginError, match="拒绝"):
        await attempt.task
    assert not env.state.requests


@pytest.mark.parametrize("client_id", ["", "dynamic_agent_client", "invalid client ID"])
async def test_first_registration_requires_issued_id(env, client_id):
    attempt, params = await start_login(env)
    await callback(attempt, params, client_id=client_id)
    with pytest.raises(PluginError, match="issued client ID"):
        await attempt.task
    assert env.store.read()["accounts"] == {}
    assert not env.state.requests


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"iss": "https://wrong.invalid"}, "签发方 iss 不匹配"),
        ({"aud": "wrong-client"}, "客户端 aud 不匹配"),
        ({"exp": int(time.time()) - 100}, "令牌已过期"),
        ({"iat": int(time.time()) + 3600}, "令牌尚未生效"),
        ({"nonce": "wrong"}, "登录 nonce 不匹配"),
        ({"nonce": "错误的 nonce"}, "登录 nonce 不匹配"),
        ({"nonce": None}, "缺少有效的登录 nonce"),
        ({"azp": "other-client"}, "授权方 azp 缺失或不匹配"),
        ({"sub": ""}, "账号标识 sub 无效"),
        ({"sub": 123}, "账号标识 sub 无效"),
    ],
)
async def test_invalid_oidc_does_not_save_tokens(env, override, reason):
    env.state.claims_overrides = override
    attempt, params = await start_login(env)
    await callback(attempt, params)
    with pytest.raises(PluginError, match=f"OIDC ID token 校验失败（登录，{reason}"):
        await attempt.task
    assert "access_token" not in env.store.account(CLIENT_ID)
    assert env.store.read()["active_client_id"] is None


async def test_invalid_signature_rejected(env):
    wrong = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    claims = jwt.decode(env.id_token(), options={"verify_signature": False})
    env.state.token_overrides["id_token"] = jwt.encode(
        claims, wrong, algorithm="RS256", headers={"kid": "test-key"}
    )
    attempt, params = await start_login(env)
    # Signature check is independent of the nonce chosen for this attempt.
    await callback(attempt, params)
    with pytest.raises(PluginError, match="签名不匹配"):
        await attempt.task


async def test_missing_login_id_token_is_reported(env):
    env.state.token_overrides["id_token"] = None
    attempt, params = await start_login(env)
    await callback(attempt, params)
    with pytest.raises(PluginError, match="OpenAI 未返回 ID token"):
        await attempt.task
    assert env.store.read()["active_client_id"] is None


async def test_missing_plan_scope_keeps_identity_but_blocks_inference(env):
    env.state.token_overrides["scope"] = "openid profile email offline_access"
    attempt, params = await start_login(env)
    await callback(attempt, params)
    assert (await attempt.task)["plan_enabled"] is False
    assert env.store.account()["subject"] == "test-user"
    with pytest.raises(PluginError, match="resource.invoke"):
        await env.auth.session()


async def test_resource_invoke_is_required_independently(env):
    await save_account(env, scopes="chatgpt.tokens.use.direct")
    with pytest.raises(PluginError, match="login --consent"):
        await env.auth.session()


async def test_selected_account_without_credentials_reports_missing_login(env):
    await save_account(env)
    async with env.store.locked():
        env.store.clear_tokens(CLIENT_ID)
    with pytest.raises(PluginError, match="没有可用的授权凭据"):
        await env.auth.session()
    assert not env.state.requests


async def test_invalid_refresh_id_token_is_reported_and_old_credentials_preserved(env):
    original = await save_account(env, expires_at=0)
    env.state.token_overrides["id_token"] = env.id_token(aud="other-client")
    with pytest.raises(PluginError, match="OIDC ID token 校验失败（刷新，客户端 aud 不匹配"):
        await env.auth.session()
    assert env.store.account() == original


async def test_concurrent_refresh_rotates_once_and_is_atomic(env):
    await save_account(env, expires_at=time.time() - 1)
    sessions = await asyncio.gather(*(env.auth.session() for _ in range(20)))
    assert env.state.refresh_count == 1
    assert {s.token for s in sessions} == {ACCESS + "-fresh"}
    account = env.store.account()
    assert account["access_token"] == ACCESS + "-fresh"
    assert account["refresh_token"] == REFRESH + "-rotated"


async def test_separate_store_instances_serialize_rotation(env):
    await save_account(env, expires_at=time.time() - 1)
    other = OpenAIPlanAuth(CredentialStore(env.store.data_dir), env.client)
    a, b = await asyncio.gather(env.auth.session(), other.session())
    assert a.token == b.token and env.state.refresh_count == 1


async def test_concurrent_401_recovery_refreshes_once(env):
    await save_account(env)
    sessions = await asyncio.gather(*(env.auth.session(rejected_token=ACCESS) for _ in range(10)))
    assert env.state.refresh_count == 1
    assert all(s.token == ACCESS + "-fresh" for s in sessions)


async def test_refresh_scope_downgrade_is_saved_and_blocks_use(env):
    await save_account(env, expires_at=0)
    env.state.token_overrides["scope"] = "openid"
    with pytest.raises(PluginError, match="不再包含"):
        await env.auth.session()
    assert env.store.account()["scopes"] == ["openid"]


@pytest.mark.parametrize("code", ["invalid_grant", "refresh_token_reused", "refresh_token_expired"])
async def test_terminal_refresh_clears_tokens_keeps_registration(env, code):
    await save_account(env, expires_at=0)
    env.state.token_error = code
    with pytest.raises(OpenAIError):
        await env.auth.session()
    account = env.store.account()
    assert account["client_id"] == CLIENT_ID
    assert all(key not in account for key in ("access_token", "refresh_token", "id_token"))


async def test_transient_refresh_does_not_erase_credentials(env):
    await save_account(env, expires_at=0)
    env.state.token_error = "temporarily_unavailable"
    with pytest.raises(OpenAIError):
        await env.auth.session()
    assert env.store.account()["refresh_token"] == REFRESH


async def test_reauthorization_reuses_registration_and_checks_identity(env):
    original = await save_account(env)
    attempt, params = await start_login(env)
    assert params["client_id"] == CLIENT_ID
    assert "agent_name_hint" not in params
    env.state.claims_overrides = {"sub": "other-user"}
    await callback(attempt, params, client_id="")
    with pytest.raises(PluginError, match="身份"):
        await attempt.task
    assert env.store.account() == original


async def test_reauthorization_rejects_other_client_before_exchange(env):
    await save_account(env)
    attempt, params = await start_login(env)
    await callback(attempt, params, client_id="oaiapp_other")
    with pytest.raises(PluginError, match="client ID"):
        await attempt.task
    assert not env.state.requests


async def test_timeout_cancellation_and_overlapping_login(env):
    env.auth.timeout_seconds = 0.05
    attempt, _ = await start_login(env)
    with pytest.raises(PluginError, match="已有授权"):
        await env.auth.begin_login()
    with pytest.raises(PluginError, match="超时"):
        await attempt.task
    port = urlsplit(attempt.redirect_uri).port
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", port)
    attempt, _ = await start_login(env)
    await env.auth.cancel_login()
    assert attempt.task.cancelled()


async def test_logout_revokes_then_clears_only_selected_account(env):
    await save_account(env, client_id="oaiapp_other")
    await save_account(env)
    assert await env.auth.logout() is True
    revoked = next(r for r in env.state.requests if r.url.path.endswith("revoke"))
    assert parse_qs(revoked.content.decode())["client_id"] == [CLIENT_ID]
    assert "access_token" not in env.store.account()
    assert env.store.account("oaiapp_other")["access_token"] == ACCESS


async def test_bad_discovery_cannot_exfiltrate_refresh_token(env):
    await save_account(env)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"revocation_endpoint": "https://evil.invalid/revoke"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        env.auth.client = client
        assert await env.auth.logout() is False
    assert len(requests) == 1 and requests[0].url.host == "auth.openai.com"
    assert "refresh_token" not in env.store.account()


async def test_host_id_is_stable_and_not_shared_by_other_data_dir(env, tmp_path):
    first = env.store.host_id()
    assert CredentialStore(env.store.data_dir).host_id() == first
    assert CredentialStore(tmp_path / "other").host_id() != first


async def test_storage_rejects_symlink_and_does_not_overwrite_corruption(env, tmp_path):
    env.store.accounts_path.write_text("invalid json")
    with pytest.raises(PluginError, match="格式无效"):
        env.store.read()
    assert env.store.accounts_path.read_text() == "invalid json"
    env.store.accounts_path.unlink()
    target = tmp_path / "other.json"
    target.write_text(json.dumps({"accounts": {}}))
    env.store.accounts_path.symlink_to(target)
    with pytest.raises(PluginError, match="符号链接"):
        env.store.read()
