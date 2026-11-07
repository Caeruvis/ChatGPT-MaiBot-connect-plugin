"""Owner-only, atomic credential storage with an inter-process async lock."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .errors import PluginError

if os.name == "nt":
    import msvcrt
else:
    import fcntl

TOKEN_KEYS = ("access_token", "refresh_token", "id_token", "expires_at", "scopes", "plan_enabled")


class CredentialStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        if self.data_dir.is_symlink():
            raise PluginError("凭据目录不能是符号链接。")
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            self.data_dir.chmod(0o700)
        self.accounts_path = self.data_dir / "accounts.json"
        self.host_id_path = self.data_dir / "host_id.txt"
        self._local_lock = asyncio.Lock()

    @asynccontextmanager
    async def locked(self):
        """All read-modify-write operations, including refresh, use this lock."""
        async with self._local_lock:
            path = self.data_dir / ".credentials.lock"
            if path.is_symlink():
                raise PluginError("凭据锁文件不能是符号链接。")
            fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
            acquired = False
            try:
                if os.name == "nt" and os.fstat(fd).st_size == 0:
                    os.write(fd, b"0")
                deadline = time.monotonic() + 90
                while not acquired:
                    try:
                        if os.name == "nt":
                            os.lseek(fd, 0, os.SEEK_SET)
                            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        else:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                    except (BlockingIOError, OSError):
                        if time.monotonic() >= deadline:
                            raise PluginError("凭据正被其他进程使用，请稍后重试。") from None
                        await asyncio.sleep(0.05)
                yield
            finally:
                if acquired:
                    if os.name == "nt":
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    @staticmethod
    def _read_private(path: Path) -> str:
        if path.is_symlink():
            raise PluginError("凭据文件不能是符号链接。")
        if os.name != "nt":
            path.chmod(0o600)
        return path.read_text(encoding="utf-8")

    def host_id(self) -> str:
        if self.host_id_path.exists():
            value = self._read_private(self.host_id_path).strip()
            try:
                if not value.startswith("urn:uuid:") or uuid.UUID(value[9:]).version != 4:
                    raise ValueError
            except ValueError:
                raise PluginError("host_id.txt 无效；请保留每台主机独立的 UUIDv4。") from None
            return value
        value = f"urn:uuid:{uuid.uuid4()}"
        self._atomic_write(self.host_id_path, value + "\n")
        return value

    def read(self) -> dict[str, Any]:
        if not self.accounts_path.exists():
            return {"version": 1, "active_client_id": None, "accounts": {}}
        try:
            data = json.loads(self._read_private(self.accounts_path))
            if not isinstance(data, dict) or not isinstance(data.get("accounts"), dict):
                raise ValueError
            for key, account in data["accounts"].items():
                if not isinstance(account, dict) or account.get("client_id") != key:
                    raise ValueError
                self.validate_client_id(key)
            return data
        except (ValueError, TypeError):
            raise PluginError("accounts.json 格式无效；未覆盖原文件。") from None

    @staticmethod
    def validate_client_id(client_id: Any) -> None:
        if (
            not isinstance(client_id, str)
            or client_id == "dynamic_agent_client"
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", client_id)
        ):
            raise PluginError(
                "需要回调返回的 OpenAI issued client ID，不能使用 dynamic_agent_client。"
            )

    def account(self, client_id: str | None = None) -> dict[str, Any]:
        data = self.read()
        selected = client_id or data.get("active_client_id")
        account = data["accounts"].get(selected)
        if not isinstance(account, dict):
            raise PluginError("尚未选择账号，请先在本机完成 Sign in with ChatGPT。")
        return dict(account)

    def save(self, account: dict[str, Any], *, activate: bool = False) -> None:
        self.validate_client_id(account.get("client_id"))
        data = self.read()
        data["accounts"][account["client_id"]] = account
        if activate:
            data["active_client_id"] = account["client_id"]
        self._write_data(data)

    def switch(self, client_id: str) -> None:
        account = self.account(client_id)
        if not account.get("access_token"):
            raise PluginError("该账号已登出，需要重新授权。")
        data = self.read()
        data["active_client_id"] = client_id
        self._write_data(data)

    def clear_tokens(self, client_id: str) -> None:
        account = self.account(client_id)
        for key in TOKEN_KEYS:
            account.pop(key, None)
        self.save(account)

    def _write_data(self, data: dict[str, Any]) -> None:
        self._atomic_write(
            self.accounts_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )

    def _atomic_write(self, path: Path, content: str) -> None:
        temp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp, path)
            if os.name != "nt":
                path.chmod(0o600)
        finally:
            temp.unlink(missing_ok=True)
