"""Apply source files to another installation while preserving runtime configuration and tokens."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install(target: Path) -> tuple[int, Path]:
    target = target.expanduser().resolve()
    if target == ROOT:
        raise ValueError("请指定另一份已安装的插件目录，不能覆盖更新包自身。")
    manifest = json.loads((ROOT / "_manifest.json").read_text())
    installed = json.loads((target / "_manifest.json").read_text())
    if manifest["id"] != installed.get("id"):
        raise ValueError("目标插件 ID 不匹配，未执行更新。")
    sources = [
        ROOT / name
        for name in (
            "plugin.py",
            "_manifest.json",
            "requirements.txt",
            "pyproject.toml",
            "README.md",
            "TESTING.md",
            "UNIFIED_WEB_BACKEND.md",
            "LICENSE",
        )
    ]
    sources.extend((ROOT / "chatgpt_maibot").glob("*.py"))
    sources.extend((ROOT / "tools").glob("*.py"))
    sources.extend((ROOT / "examples").glob("*.toml"))
    bundle = ROOT / "host_backend"
    if bundle.is_dir():
        backend_manifest = json.loads((bundle / "manifest.json").read_text())
        sources.extend(
            bundle / name
            for name in ("manifest.json", "README.md", "LICENSE", "unified-web-backend.patch")
        )
        sources.extend(bundle / "files" / entry["path"] for entry in backend_manifest["files"])
    for source in sources:
        relative = source.relative_to(ROOT)
        if ".." in relative.parts:
            raise ValueError("更新文件路径无效。")
        for base in (ROOT, target):
            candidate = base
            for part in relative.parts:
                candidate /= part
                if candidate.is_symlink():
                    raise ValueError("更新目标中存在符号链接，未执行更新。")
    backup = (
        target
        / ".update-backups"
        / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4))
    )
    if backup.parent.is_symlink():
        raise ValueError("更新备份目录不能是符号链接。")
    backup.mkdir(parents=True)
    previous = []
    for source in sources:
        destination = target / source.relative_to(ROOT)
        if destination.exists():
            saved = backup / source.relative_to(ROOT)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
            previous.append((destination, saved))
        else:
            previous.append((destination, None))
    try:
        for source in sources:
            destination = target / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                "." + destination.name + ".update-" + secrets.token_hex(4)
            )
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    except BaseException:
        for destination, saved in previous:
            if saved is None:
                destination.unlink(missing_ok=True)
            else:
                shutil.copy2(saved, destination)
        raise
    return len(sources), backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    try:
        count, backup = install(args.target)
    except (OSError, ValueError, KeyError):
        print("更新未完成；请检查目标路径、插件 ID 与文件权限。")
        return 1
    print(f"已更新 {count} 个源码和说明文件；原源码备份在 {backup}")
    print("config.toml、账号授权与 host ID 保持原样。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
