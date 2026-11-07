import json
import shutil
from pathlib import Path

import pytest

from tools.install_update import ROOT, install

CURRENT_VERSION = json.loads((ROOT / "_manifest.json").read_text())["version"]


def installation(tmp_path):
    target = tmp_path / "installed-plugin"
    target.mkdir()
    (target / "_manifest.json").write_text(
        (ROOT / "_manifest.json").read_text().replace(CURRENT_VERSION, "0.1.0")
    )
    (target / "plugin.py").write_text("old source")
    (target / "config.toml").write_text("user configuration must remain")
    (target / "accounts.json").write_text("fake-credential-sentinel")
    (target / "host_id.txt").write_text("fake-host-sentinel")
    return target


def test_update_preserves_configuration_and_credentials_and_backs_up_source(tmp_path):
    target = installation(tmp_path)
    count, backup = install(target)
    assert count > 3
    assert json.loads((target / "_manifest.json").read_text())["version"] == CURRENT_VERSION
    assert (target / "config.toml").read_text() == "user configuration must remain"
    assert (target / "accounts.json").read_text() == "fake-credential-sentinel"
    assert (target / "host_id.txt").read_text() == "fake-host-sentinel"
    assert (backup / "plugin.py").read_text() == "old source"
    assert not (backup / "accounts.json").exists()
    assert (target / "chatgpt_maibot" / "transport.py").is_file()


def test_failed_update_restores_old_source(tmp_path, monkeypatch):
    target = installation(tmp_path)
    original = shutil.copy2

    def fail_on_second_update(source, destination, *args, **kwargs):
        if Path(destination).name.startswith("._manifest.json.update-"):
            raise OSError("simulated write failure")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_on_second_update)
    with pytest.raises(OSError):
        install(target)
    assert (target / "plugin.py").read_text() == "old source"
    assert json.loads((target / "_manifest.json").read_text())["version"] == "0.1.0"
    assert (target / "accounts.json").read_text() == "fake-credential-sentinel"


def test_wrong_target_is_rejected_before_changes(tmp_path):
    target = installation(tmp_path)
    (target / "_manifest.json").write_text('{"id":"some.other.plugin"}')
    with pytest.raises(ValueError, match="ID"):
        install(target)
    assert not (target / ".update-backups").exists()
