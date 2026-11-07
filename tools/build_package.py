"""Build a local source ZIP from an explicit allowlist; never bundle runtime credentials."""

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "plugin.py",
    "_manifest.json",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "README.md",
    "TESTING.md",
    "UNIFIED_WEB_BACKEND.md",
    "LICENSE",
    ".gitignore",
]


def main():
    manifest = __import__("json").loads((ROOT / "_manifest.json").read_text())
    target = ROOT / "dist" / f"ChatGPT-MaiBot-connect-plugin-{manifest['version']}.zip"
    archive_root = f"{ROOT.name}-{manifest['version']}"
    target.parent.mkdir(exist_ok=True)
    sources = [ROOT / name for name in FILES]
    for folder in ("chatgpt_maibot", "examples", "tools", "tests"):
        sources.extend(
            path
            for path in (ROOT / folder).rglob("*")
            if path.is_file() and path.suffix in {".py", ".toml"}
        )
    bundle = ROOT / "host_backend"
    backend_manifest = __import__("json").loads((bundle / "manifest.json").read_text())
    sources.extend(
        bundle / name
        for name in ("manifest.json", "README.md", "LICENSE", "unified-web-backend.patch")
    )
    sources.extend(bundle / "files" / entry["path"] for entry in backend_manifest["files"])
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            if path.is_symlink():
                raise RuntimeError("拒绝将符号链接打包。")
            archive.write(path, archive_root + "/" + path.relative_to(ROOT).as_posix())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(digest + "  " + target.name + "\n")
    print(target)
    print("SHA256 " + digest)


if __name__ == "__main__":
    main()
