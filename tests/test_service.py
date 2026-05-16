"""Tests for the LaunchAgent service installer.

We don't actually drive `launchctl` from the test suite (it'd mess with the
developer's real LaunchAgents). Just verify the plist we generate is shaped
correctly — that's the failure mode that's hard to catch otherwise (one
wrong key and launchd silently refuses to load).
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from oncall import service


def test_plist_shape():
    p = service._build_plist(
        binary=Path("/usr/local/bin/oncall"),
        extra_path="/usr/local/bin:/usr/bin:/bin",
    )
    assert p["Label"] == "com.oncall.agent"
    assert p["ProgramArguments"] == ["/usr/local/bin/oncall", "api"]
    assert p["RunAtLoad"] is True
    assert p["KeepAlive"] is True
    # Absolute paths for stdout/stderr — launchd resolves relative to /
    assert p["StandardOutPath"].startswith("/")
    assert p["StandardErrorPath"].startswith("/")
    # PATH must be passed explicitly so claude resolves at runtime
    assert "PATH" in p["EnvironmentVariables"]
    assert "/usr/local/bin" in p["EnvironmentVariables"]["PATH"]
    assert "HOME" in p["EnvironmentVariables"]


def test_plist_is_valid_plist_format(tmp_path):
    """plistlib must be able to round-trip the dict we build."""
    p = service._build_plist(
        binary=Path("/usr/local/bin/oncall"),
        extra_path="/usr/bin",
    )
    target = tmp_path / "agent.plist"
    with open(target, "wb") as f:
        plistlib.dump(p, f)
    with open(target, "rb") as f:
        roundtripped = plistlib.load(f)
    assert roundtripped == p


def test_label_matches_plist_filename():
    """launchd matches the file's stem to the Label key. Mismatch silently
    fails to load — guard against accidental rename of one but not the other."""
    assert service.PLIST_PATH.stem == service.LABEL


def test_install_refuses_on_non_macos(monkeypatch, capsys):
    monkeypatch.setattr("sys.platform", "linux")
    with pytest.raises(SystemExit) as ei:
        service.install()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "macOS only" in err


def test_find_oncall_binary_uses_which(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/opt/foo/bin/oncall")
    found = service._find_oncall_binary()
    assert str(found) == "/opt/foo/bin/oncall"


def test_find_oncall_binary_exits_when_missing(monkeypatch, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None)
    # Also point sys.executable at a path where no `oncall` lives.
    monkeypatch.setattr("sys.executable", "/nowhere/python")
    with pytest.raises(SystemExit) as ei:
        service._find_oncall_binary()
    assert ei.value.code == 2
    assert "couldn't find" in capsys.readouterr().err
