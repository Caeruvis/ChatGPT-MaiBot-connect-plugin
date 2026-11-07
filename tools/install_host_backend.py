"""Install the bundled MaiBot backend after exact-source checks; preserve instance data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

BUNDLE = Path(__file__).resolve().parents[1] / "host_backend"


def digest(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or not relative.parts:
        raise ValueError("补丁路径无效。")
    if relative.parts[0] not in {"src", "dashboard"}:
        raise ValueError("补丁仅允许修改主程序 src 与 dashboard 源码。")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("补丁路径包含符号链接。")
    return path


def read(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def write(path: Path, data: bytes | None) -> None:
    if data is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".backend-" + secrets.token_hex(4))
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def probe_runtime(python: Path) -> Path | None:
    """Query the selected interpreter without importing or starting MaiBot."""
    try:
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.metadata as m,sysconfig; m.version('MaiBot'); print(sysconfig.get_paths()['purelib'])",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        root = Path(result.stdout.strip()) if result.returncode == 0 else None
        if root is not None and (root / "src/web_search/service.py").is_file():
            return root.resolve()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def resolve_target(target: Path, python: Path | None = None) -> Path:
    target = target.expanduser().resolve()
    if (target / "src/web_search/service.py").is_file():
        return target
    if not ((target / "config").is_dir() and (target / "plugins").is_dir()):
        raise ValueError("目标不是 MaiBot 实例或包含统一联网服务的主程序目录。")
    if python is not None:
        runtime = probe_runtime(python.expanduser().absolute())
        if runtime is None:
            raise ValueError("指定 Python 中未找到 MaiBot 统一联网服务。")
        return runtime
    # Instance directories contain config/plugins/data. The uv tool launcher points
    # to the active environment; a legacy .venv inside an instance may be unrelated.
    candidates = [Path(sys.executable)]
    launcher = shutil.which("maibot")
    if launcher:
        candidates.append(
            Path(launcher).resolve().parent / ("python.exe" if os.name == "nt" else "python")
        )
    candidates.extend([target / ".venv/bin/python", target / ".venv/Scripts/python.exe"])
    roots = set()
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            runtime = probe_runtime(candidate)
            if runtime is not None:
                roots.add(runtime)
    if len(roots) != 1:
        raise ValueError("无法唯一确定实例使用的主程序，请用 --python 指定启动 MaiBot 的 Python。")
    return roots.pop()


def is_packaged_runtime(target: Path) -> bool:
    return target.name in {"site-packages", "dist-packages"} or any(
        target.glob("maibot-*.dist-info")
    )


def install(
    target: Path, *, check_only: bool = False, python: Path | None = None
) -> tuple[int, Path | None]:
    target = resolve_target(target, python)
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    changes = []
    runtime_only = is_packaged_runtime(target)
    for entry in manifest["files"]:
        name = entry["path"]
        if runtime_only and not (name.startswith("src/") and name.endswith(".py")):
            continue
        source = safe_path(BUNDLE / "files", name)
        destination = safe_path(target, name)
        data, previous = source.read_bytes(), read(destination)
        if digest(data) != entry["sha256"]:
            raise ValueError("补丁包校验失败。")
        current = digest(previous)
        if current == entry["sha256"]:
            continue
        if current != entry["base_sha256"]:
            raise ValueError(f"目标源码与补丁基线不兼容或已有本地修改：{name}；未写入任何文件。")
        changes.append((name, destination, previous, data))
    if check_only or not changes:
        return len(changes), None
    backup = (
        target
        / ".gpt-web-backups"
        / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4))
    )
    if backup.parent.is_symlink():
        raise ValueError("备份目录不能是符号链接。")
    backup.mkdir(parents=True)
    state = []
    for name, _, previous, data in changes:
        if previous is not None:
            saved = safe_path(backup, name)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(previous)
        state.append(
            {"path": name, "previous_sha256": digest(previous), "installed_sha256": digest(data)}
        )
    (backup / "state.json").write_text(
        json.dumps({"target": str(target), "files": state}, indent=2)
    )
    applied = []
    try:
        for _, destination, previous, data in changes:
            applied.append((destination, previous))
            write(destination, data)
    except BaseException:
        for destination, previous in reversed(applied):
            write(destination, previous)
        raise
    return len(changes), backup


def rollback(target: Path, backup: Path, *, python: Path | None = None) -> int:
    target, backup = resolve_target(target, python), backup.expanduser().resolve()
    if backup.parent != target / ".gpt-web-backups":
        raise ValueError("请使用该 MaiBot 目录中安装器生成的备份。")
    state = json.loads((backup / "state.json").read_text())
    if state["target"] != str(target):
        raise ValueError("备份对应的主程序目录不匹配。")
    changes = []
    for entry in state["files"]:
        destination = safe_path(target, entry["path"])
        saved = safe_path(backup, entry["path"])
        previous = read(saved)
        if digest(previous) != entry["previous_sha256"]:
            raise ValueError("备份文件校验失败。")
        current = read(destination)
        if digest(current) == entry["previous_sha256"]:
            continue
        if digest(current) != entry["installed_sha256"]:
            raise ValueError("安装后的源码又有改动，未执行回滚。")
        changes.append((destination, previous, current))
    applied = []
    try:
        for destination, previous, current in changes:
            applied.append((destination, current))
            write(destination, previous)
    except BaseException:
        for destination, current in reversed(applied):
            write(destination, current)
        raise
    return len(changes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--python", type=Path, help="实例实际使用的 Python；多环境时显式指定")
    parser.add_argument("--check", action="store_true", help="只检查兼容性，不写文件")
    parser.add_argument("--rollback", type=Path, help="恢复指定安装备份")
    args = parser.parse_args()
    try:
        if args.rollback:
            if args.check:
                raise ValueError("--check 与 --rollback 不能同时使用。")
            print(f"已恢复 {rollback(args.target, args.rollback, python=args.python)} 个源码文件。")
        else:
            target = resolve_target(args.target, args.python)
            print(f"主程序位置：{target}")
            if is_packaged_runtime(target):
                print("已识别包安装布局，仅更新 Python 后端；面板包和实例数据保持原样。")
            count, backup = install(target, check_only=args.check)
            print(
                f"兼容性检查通过，待更新 {count} 个源码文件。"
                if args.check
                else f"已更新 {count} 个源码文件。"
            )
            if backup:
                print(f"备份：{backup}")
            print("实例配置、数据与账号授权保持原样。修改后请重启主程序。")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
