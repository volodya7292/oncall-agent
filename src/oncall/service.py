"""Install `oncall api` as a macOS LaunchAgent.

LaunchAgents run as the logged-in user (not root), start at login, and are
restarted by launchd if they crash. That's the right model here — the
orchestrator needs the user's keychain (for `claude` OAuth) and writes into
the user's `~/.oncall/` directory.

The plist just invokes `oncall api`; nothing about the api process changes
when launched this way. Manual `oncall api` still works for foreground
testing — just `oncall service stop` first so port 8765 is free.

Linux/systemd-user support: TODO.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path


LABEL = "com.oncall.agent"
PLIST_PATH = Path("~/Library/LaunchAgents").expanduser() / f"{LABEL}.plist"
LOG_DIR = Path("~/.oncall/logs").expanduser()
STDOUT_LOG = LOG_DIR / "oncall.out.log"
STDERR_LOG = LOG_DIR / "oncall.err.log"


def _domain_target() -> str:
    """Modern launchctl service target: gui/<uid>/<label>."""
    return f"gui/{os.getuid()}/{LABEL}"


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _check_macos() -> None:
    if sys.platform != "darwin":
        print(
            f"`oncall service` currently supports macOS only "
            f"(detected: {sys.platform}). Linux/systemd-user support is a TODO.",
            file=sys.stderr,
        )
        sys.exit(2)


def _find_oncall_binary() -> Path:
    """Resolve the absolute path of the `oncall` entry point. LaunchAgents
    inherit a sparse PATH at boot, so we must hardcode the absolute path."""
    found = shutil.which("oncall")
    if found:
        return Path(found).resolve()
    # Fall back to using the current interpreter's installed entry point.
    interp = Path(sys.executable).resolve()
    candidate = interp.parent / "oncall"
    if candidate.exists():
        return candidate
    print(
        "ERROR: couldn't find `oncall` on PATH. Install with `uv tool install …` first.",
        file=sys.stderr,
    )
    sys.exit(2)


def _build_plist(binary: Path, extra_path: str) -> dict:
    """Build the LaunchAgent plist dictionary.

    `extra_path` is prepended to the PATH the agent sees, so the executor
    subprocess (which inherits this env) can find `claude`."""
    return {
        "Label": LABEL,
        "ProgramArguments": [str(binary), "api"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(STDOUT_LOG),
        "StandardErrorPath": str(STDERR_LOG),
        "WorkingDirectory": str(Path("~/.oncall").expanduser()),
        "EnvironmentVariables": {
            # Keep PATH explicit so `claude` (and friends) resolve at runtime.
            # launchd defaults are minimal; we extend with the install-time PATH.
            "PATH": extra_path,
            "HOME": str(Path.home()),
            # Match terminal locale so unicode in audit logs doesn't choke.
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        },
        # ProcessType=Interactive: long-lived foreground service, not a batch job.
        "ProcessType": "Interactive",
    }


def _launchctl(*args: str, check: bool = False, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *args],
        check=check,
        capture_output=capture,
        text=True,
    )


def _is_loaded() -> bool:
    """True if the LaunchAgent is currently registered with launchd."""
    r = _launchctl("print", _domain_target(), capture=True)
    return r.returncode == 0


def install() -> None:
    _check_macos()
    binary = _find_oncall_binary()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Inherit the user's current PATH for `claude` discovery.
    current_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    plist = _build_plist(binary, current_path)
    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)
    PLIST_PATH.chmod(0o644)
    print(f"Wrote {PLIST_PATH}")

    # If already loaded, replace cleanly. Swallow output — bootout prints
    # "No such process" when nothing was loaded, which is harmless noise.
    _launchctl("bootout", _domain_target(), capture=True)
    r = _launchctl("bootstrap", _domain(), str(PLIST_PATH), capture=True)
    if r.returncode != 0:
        print(f"launchctl bootstrap failed: {r.stderr.strip() or r.stdout.strip()}", file=sys.stderr)
        sys.exit(r.returncode)
    print(f"Loaded service {LABEL} (auto-starts at login, restarts on crash).")
    print(f"  logs: tail -f {STDOUT_LOG} {STDERR_LOG}")
    print(f"  status: oncall service status")
    print(f"  stop:   oncall service stop")


def uninstall() -> None:
    _check_macos()
    if _is_loaded():
        _launchctl("bootout", _domain_target())
        print(f"Unloaded {LABEL}.")
    else:
        print(f"{LABEL} not currently loaded.")
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
        print(f"Removed {PLIST_PATH}.")
    else:
        print(f"{PLIST_PATH} not present.")


def start() -> None:
    _check_macos()
    if not PLIST_PATH.exists():
        print("Service not installed. Run `oncall service install` first.", file=sys.stderr)
        sys.exit(2)

    # Two paths:
    #   * already loaded → kickstart -k restarts the process in place.
    #   * not loaded     → bootstrap. Plist has RunAtLoad=True, so bootstrap
    #                      itself starts the process — no kickstart needed.
    #
    # The previous version always called kickstart, which fails with empty
    # stderr if bootstrap silently flopped (e.g. racing with bootout's teardown
    # right after `oncall service stop`). Now we capture+check bootstrap so
    # failures surface, and we only kickstart when there's actually a service
    # to kick.
    if _is_loaded():
        r = _launchctl("kickstart", "-k", _domain_target(), capture=True)
        if r.returncode != 0:
            print(
                f"launchctl kickstart failed (code={r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip() or '<no output>'}",
                file=sys.stderr,
            )
            sys.exit(r.returncode)
        print(f"Restarted {LABEL}.")
        return

    r = _launchctl("bootstrap", _domain(), str(PLIST_PATH), capture=True)
    if r.returncode != 0:
        msg = r.stderr.strip() or r.stdout.strip() or "<no output>"
        # Common race: bootout from a recent `stop` hasn't fully settled and
        # launchctl rejects bootstrap with "Bootstrap failed: 5: Input/output
        # error" or "service already loaded". Try once more after a brief
        # pause — usually the second attempt succeeds.
        import time
        time.sleep(0.5)
        r = _launchctl("bootstrap", _domain(), str(PLIST_PATH), capture=True)
        if r.returncode != 0:
            msg2 = r.stderr.strip() or r.stdout.strip() or "<no output>"
            print(
                f"launchctl bootstrap failed (code={r.returncode}).\n"
                f"  first attempt:  {msg}\n"
                f"  retry attempt:  {msg2}",
                file=sys.stderr,
            )
            sys.exit(r.returncode)
    print(f"Loaded and started {LABEL}.")


def stop() -> None:
    _check_macos()
    if not _is_loaded():
        print(f"{LABEL} not loaded.")
        return
    _launchctl("bootout", _domain_target())
    # bootout returns before launchd fully tears the service down — a
    # subsequent `oncall service start` can race on the unload. Poll
    # _is_loaded() briefly so the caller sees a clean state.
    import time
    for _ in range(20):  # up to ~1s
        if not _is_loaded():
            break
        time.sleep(0.05)
    print(f"Stopped {LABEL}.")


def status() -> None:
    _check_macos()
    if not PLIST_PATH.exists():
        print(f"Service not installed (no plist at {PLIST_PATH}).")
        return
    r = _launchctl("print", _domain_target(), capture=True)
    if r.returncode != 0:
        print(f"Service installed but NOT loaded.\n  install path: {PLIST_PATH}")
        return
    # Distill the useful bits from launchctl print's verbose output.
    out = r.stdout
    keep = [
        line for line in out.splitlines()
        if any(k in line for k in ("state =", "pid =", "last exit", "program ="))
    ]
    print(f"{LABEL}: LOADED")
    for line in keep:
        print(f"  {line.strip()}")
    print(f"  logs: {STDOUT_LOG} (out), {STDERR_LOG} (err)")


def logs(follow: bool = False, lines: int = 100) -> None:
    _check_macos()
    if not STDOUT_LOG.exists() and not STDERR_LOG.exists():
        print(f"No logs yet. Looked at {STDOUT_LOG} and {STDERR_LOG}.")
        return
    args = ["tail", f"-n{lines}"]
    if follow:
        args.append("-f")
    # Both files in one tail so user sees interleaved out+err with file headers.
    files = [str(p) for p in (STDOUT_LOG, STDERR_LOG) if p.exists()]
    os.execvp("tail", ["tail", *args[1:], *files])
