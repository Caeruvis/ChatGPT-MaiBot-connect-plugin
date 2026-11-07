import json

import pytest

from tools import install_host_backend as installer


@pytest.fixture
def setup(tmp_path, monkeypatch):
    target = tmp_path / "MaiBot"
    (target / "src/web_search").mkdir(parents=True)
    (target / "src/web_search/service.py").write_bytes(b"old")
    (target / "config").mkdir()
    (target / "config/model_config.toml").write_text("configuration sentinel")
    bundle = tmp_path / "bundle"
    files = []
    for name, old, new in [
        ("src/web_search/service.py", b"old", b"new"),
        ("src/web_search/adapters/plugin.py", None, b"adapter"),
    ]:
        path = bundle / "files" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(new)
        files.append(
            {"path": name, "base_sha256": installer.digest(old), "sha256": installer.digest(new)}
        )
    (bundle / "manifest.json").write_text(json.dumps({"files": files}))
    monkeypatch.setattr(installer, "BUNDLE", bundle)
    return target, bundle


def test_install_check_idempotence_and_rollback_preserve_configuration(setup):
    target, _ = setup
    assert installer.install(target, check_only=True) == (2, None)
    assert not (target / ".gpt-web-backups").exists()
    count, backup = installer.install(target)
    assert count == 2
    assert (target / "src/web_search/service.py").read_bytes() == b"new"
    assert installer.install(target) == (0, None)
    assert installer.rollback(target, backup) == 2
    assert (target / "src/web_search/service.py").read_bytes() == b"old"
    assert not (target / "src/web_search/adapters/plugin.py").exists()
    assert (target / "config/model_config.toml").read_text() == "configuration sentinel"


def test_hash_mismatch_rejects_all_writes(setup):
    target, _ = setup
    (target / "src/web_search/adapters").mkdir()
    (target / "src/web_search/adapters/plugin.py").write_bytes(b"local modification")
    with pytest.raises(ValueError, match="不兼容"):
        installer.install(target)
    assert (target / "src/web_search/service.py").read_bytes() == b"old"
    assert not (target / ".gpt-web-backups").exists()


def test_corrupt_bundle_rejects_writes(setup):
    target, bundle = setup
    (bundle / "files/src/web_search/adapters/plugin.py").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="校验失败"):
        installer.install(target)
    assert (target / "src/web_search/service.py").read_bytes() == b"old"


def test_failed_write_restores_previous_files(setup, monkeypatch):
    target, _ = setup
    original = installer.write

    def fail(path, data):
        if data == b"adapter":
            raise OSError("simulated write failure")
        original(path, data)

    monkeypatch.setattr(installer, "write", fail)
    with pytest.raises(OSError):
        installer.install(target)
    assert (target / "src/web_search/service.py").read_bytes() == b"old"
    assert not (target / "src/web_search/adapters/plugin.py").exists()


def test_rollback_rejects_later_local_edits(setup):
    target, _ = setup
    _, backup = installer.install(target)
    (target / "src/web_search/service.py").write_bytes(b"later edit")
    with pytest.raises(ValueError, match="又有改动"):
        installer.rollback(target, backup)
    assert (target / "src/web_search/service.py").read_bytes() == b"later edit"


@pytest.mark.parametrize(
    "name",
    ["../config/model_config.toml", "config/model_config.toml", "/src/evil", "src/../config/token"],
)
def test_paths_cannot_escape_source_scope(tmp_path, name):
    with pytest.raises(ValueError):
        installer.safe_path(tmp_path, name)


def test_symlink_source_targets_are_rejected(setup):
    target, _ = setup
    outside = target.parent / "outside"
    outside.mkdir()
    (target / "src/web_search/adapters").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        installer.install(target)
    assert not list(outside.iterdir())


def test_packaged_runtime_skips_absent_dashboard_and_docs(setup):
    source, bundle = setup
    target = source.parent / "site-packages"
    source.rename(target)
    path = bundle / "manifest.json"
    data = json.loads(path.read_text())
    data["files"].extend(
        [
            {"path": "dashboard/src/view.tsx", "base_sha256": "absent", "sha256": "unused"},
            {"path": "src/web_search/README.md", "base_sha256": "absent", "sha256": "unused"},
        ]
    )
    path.write_text(json.dumps(data))
    count, backup = installer.install(target)
    assert count == 2
    assert not (target / "dashboard").exists()
    assert installer.rollback(target, backup) == 2


def test_instance_resolution_uses_explicit_runtime_and_preserves_instance(setup, monkeypatch):
    target, _ = setup
    instance = target.parent / "instance"
    (instance / "config").mkdir(parents=True)
    (instance / "plugins").mkdir()
    python = instance / "python"
    monkeypatch.setattr(
        installer, "probe_runtime", lambda candidate: target if candidate == python else None
    )
    assert installer.resolve_target(instance, python) == target
    assert installer.install(instance, python=python, check_only=True) == (2, None)
    assert not (instance / "src").exists()


def test_invalid_python_is_rejected_for_instance(setup, monkeypatch):
    target, _ = setup
    instance = target.parent / "instance"
    (instance / "config").mkdir(parents=True)
    (instance / "plugins").mkdir()
    monkeypatch.setattr(installer, "probe_runtime", lambda candidate: None)
    with pytest.raises(ValueError, match="指定 Python"):
        installer.resolve_target(instance, instance / "missing")


def test_probe_uses_venv_interpreter_path_without_resolving_its_python_symlink(
    tmp_path, monkeypatch
):
    runtime = tmp_path / "site-packages"
    (runtime / "src/web_search").mkdir(parents=True)
    (runtime / "src/web_search/service.py").write_text("service")
    python = tmp_path / ".venv/bin/python"
    calls = []

    def run(args, **kwargs):
        from types import SimpleNamespace

        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=str(runtime) + "\n")

    monkeypatch.setattr(installer.subprocess, "run", run)
    assert installer.probe_runtime(python) == runtime
    assert calls[0][0] == str(python)
