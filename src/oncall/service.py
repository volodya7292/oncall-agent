"""Install oncall processes as macOS LaunchAgents.

LaunchAgents run as the logged-in user (not root), start at login, and are
restarted by launchd if they crash. That's the right model here — both
processes need the user's keychain (for `claude` OAuth) and write into the
user's `~/.oncall/` directory.

Two services, selected by `--worker`:

  * agent  (com.oncall.agent)  → `oncall api`           — the orchestrator.
  * worker (com.oncall.worker) → `oncall laptop-worker` — the laptop-side
        capability worker for cloud-primary deployments. Install this on the
        laptop so it long-polls the server and auto-restarts across reboots,
        crashes, and (re)login.

Manual `oncall api` / `oncall laptop-worker` still work for foreground
testing — just `oncall service stop [--worker]` first.

Linux/systemd-user support: TODO.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


LOG_DIR = Path("~/.oncall/logs").expanduser()


@dataclass(frozen=True)
class ServiceSpec:
    """One installable LaunchAgent: its label, the `oncall` subcommand it
    runs, and where its logs go."""
    label: str
    args: list[str]
    stdout: Path
    stderr: Path
    desc: str


AGENT = ServiceSpec(
    label="com.oncall.agent",
    args=["api"],
    stdout=LOG_DIR / "oncall.out.log",
    stderr=LOG_DIR / "oncall.err.log",
    desc="orchestrator API",
)
WORKER = ServiceSpec(
    label="com.oncall.worker",
    args=["laptop-worker"],
    stdout=LOG_DIR / "worker.out.log",
    stderr=LOG_DIR / "worker.err.log",
    desc="laptop capability worker",
)


def _plist_path(spec: ServiceSpec) -> Path:
    return Path("~/Library/LaunchAgents").expanduser() / f"{spec.label}.plist"


def _domain_target(spec: ServiceSpec) -> str:
    """Modern launchctl service target: gui/<uid>/<label>."""
    return f"gui/{os.getuid()}/{spec.label}"


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


def _build_plist(spec: ServiceSpec, binary: Path, extra_path: str) -> dict:
    """Build the LaunchAgent plist dictionary.

    `extra_path` is prepended to the PATH the agent sees, so the executor
    subprocess (which inherits this env) can find `claude`."""
    return {
        "Label": spec.label,
        "ProgramArguments": [str(binary), *spec.args],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(spec.stdout),
        "StandardErrorPath": str(spec.stderr),
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


def _is_loaded(spec: ServiceSpec) -> bool:
    """True if the LaunchAgent is currently registered with launchd."""
    r = _launchctl("print", _domain_target(spec), capture=True)
    return r.returncode == 0


def install(spec: ServiceSpec = AGENT) -> None:
    _check_macos()
    binary = _find_oncall_binary()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    plist_path = _plist_path(spec)
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    # Inherit the user's current PATH for `claude` discovery.
    current_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    plist = _build_plist(spec, binary, current_path)
    with open(plist_path, "wb") as f:
        plistlib.dump(plist, f)
    plist_path.chmod(0o644)
    print(f"Wrote {plist_path}")

    # If already loaded, replace cleanly. Swallow output — bootout prints
    # "No such process" when nothing was loaded, which is harmless noise.
    _launchctl("bootout", _domain_target(spec), capture=True)
    r = _launchctl("bootstrap", _domain(), str(plist_path), capture=True)
    if r.returncode != 0:
        print(f"launchctl bootstrap failed: {r.stderr.strip() or r.stdout.strip()}", file=sys.stderr)
        sys.exit(r.returncode)
    print(f"Loaded service {spec.label} ({spec.desc}; auto-starts at login, restarts on crash).")
    print(f"  logs: tail -f {spec.stdout} {spec.stderr}")
    flag = " --worker" if spec is WORKER else ""
    print(f"  status: oncall service status{flag}")
    print(f"  stop:   oncall service stop{flag}")


def uninstall(spec: ServiceSpec = AGENT) -> None:
    _check_macos()
    plist_path = _plist_path(spec)
    if _is_loaded(spec):
        _launchctl("bootout", _domain_target(spec))
        print(f"Unloaded {spec.label}.")
    else:
        print(f"{spec.label} not currently loaded.")
    if plist_path.exists():
        plist_path.unlink()
        print(f"Removed {plist_path}.")
    else:
        print(f"{plist_path} not present.")


def start(spec: ServiceSpec = AGENT) -> None:
    _check_macos()
    plist_path = _plist_path(spec)
    if not plist_path.exists():
        flag = " --worker" if spec is WORKER else ""
        print(f"Service not installed. Run `oncall service install{flag}` first.", file=sys.stderr)
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
    if _is_loaded(spec):
        r = _launchctl("kickstart", "-k", _domain_target(spec), capture=True)
        if r.returncode != 0:
            print(
                f"launchctl kickstart failed (code={r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip() or '<no output>'}",
                file=sys.stderr,
            )
            sys.exit(r.returncode)
        print(f"Restarted {spec.label}.")
        return

    r = _launchctl("bootstrap", _domain(), str(plist_path), capture=True)
    if r.returncode != 0:
        msg = r.stderr.strip() or r.stdout.strip() or "<no output>"
        # Common race: bootout from a recent `stop` hasn't fully settled and
        # launchctl rejects bootstrap with "Bootstrap failed: 5: Input/output
        # error" or "service already loaded". Try once more after a brief
        # pause — usually the second attempt succeeds.
        import time
        time.sleep(0.5)
        r = _launchctl("bootstrap", _domain(), str(plist_path), capture=True)
        if r.returncode != 0:
            msg2 = r.stderr.strip() or r.stdout.strip() or "<no output>"
            print(
                f"launchctl bootstrap failed (code={r.returncode}).\n"
                f"  first attempt:  {msg}\n"
                f"  retry attempt:  {msg2}",
                file=sys.stderr,
            )
            sys.exit(r.returncode)
    print(f"Loaded and started {spec.label}.")


def stop(spec: ServiceSpec = AGENT) -> None:
    _check_macos()
    if not _is_loaded(spec):
        print(f"{spec.label} not loaded.")
        return
    _launchctl("bootout", _domain_target(spec))
    # bootout returns before launchd fully tears the service down — a
    # subsequent `oncall service start` can race on the unload. Poll
    # _is_loaded() briefly so the caller sees a clean state.
    import time
    for _ in range(20):  # up to ~1s
        if not _is_loaded(spec):
            break
        time.sleep(0.05)
    print(f"Stopped {spec.label}.")


def status(spec: ServiceSpec = AGENT) -> None:
    _check_macos()
    plist_path = _plist_path(spec)
    if not plist_path.exists():
        print(f"Service not installed (no plist at {plist_path}).")
        return
    r = _launchctl("print", _domain_target(spec), capture=True)
    if r.returncode != 0:
        print(f"Service installed but NOT loaded.\n  install path: {plist_path}")
        return
    # Distill the useful bits from launchctl print's verbose output.
    out = r.stdout
    keep = [
        line for line in out.splitlines()
        if any(k in line for k in ("state =", "pid =", "last exit", "program ="))
    ]
    print(f"{spec.label}: LOADED")
    for line in keep:
        print(f"  {line.strip()}")
    print(f"  logs: {spec.stdout} (out), {spec.stderr} (err)")


def logs(spec: ServiceSpec = AGENT, follow: bool = False, lines: int = 100) -> None:
    _check_macos()
    if not spec.stdout.exists() and not spec.stderr.exists():
        print(f"No logs yet. Looked at {spec.stdout} and {spec.stderr}.")
        return
    args = ["tail", f"-n{lines}"]
    if follow:
        args.append("-f")
    # Both files in one tail so user sees interleaved out+err with file headers.
    files = [str(p) for p in (spec.stdout, spec.stderr) if p.exists()]
    os.execvp("tail", ["tail", *args[1:], *files])
