"""Process-global latency observability.

A tiny in-memory registry of rolling latency windows, shared across the
`oncall api` daemon's subsystems (operator LLM round-trips, voice TTS
synthesis, …). It exists so `/status` can show live p50/p95 without any
external metrics backend.

Scope + lifetime: this is deliberately process-local and non-persistent.
The daemon is a long-lived singleton, so a module-global registry is
reachable from the operator (LLM), the voice call service (TTS), and the
Telegram `/status` handler alike — all run in the same process. On restart
the windows reset; that's fine, these are "how's it doing right now"
gauges, not historical accounting.

Concurrency: the daemon is single-threaded asyncio, so plain list/deque
appends need no lock. `record()` never awaits and never raises on bad
input — instrumentation must never take down the path it measures.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

# Rolling window size per metric. Percentiles are computed over the last
# N samples; lifetime totals/errors are kept separately and unbounded.
_WINDOW = 128


def _percentile(sorted_samples: list[float], pct: float) -> float:
    """Nearest-rank percentile over an already-sorted list. `pct` in [0, 100]."""
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    # rank in [0, n-1]
    rank = (pct / 100.0) * (len(sorted_samples) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = rank - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac


@dataclass
class LatencyWindow:
    """Rolling latency samples (milliseconds) for one named operation."""

    samples: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    total: int = 0            # lifetime count of successful observations
    errors: int = 0           # lifetime count of failed observations
    last_ms: float = 0.0      # most recent successful sample
    last_at: float = 0.0      # time.monotonic() of the last observation (ok or error)

    def record(self, ms: float, *, ok: bool = True) -> None:
        self.last_at = time.monotonic()
        if ok:
            self.samples.append(ms)
            self.last_ms = ms
            self.total += 1
        else:
            self.errors += 1

    def snapshot(self) -> dict[str, float | int]:
        s = sorted(self.samples)
        return {
            "n": len(s),
            "total": self.total,
            "errors": self.errors,
            "p50": _percentile(s, 50),
            "p95": _percentile(s, 95),
            "last_ms": self.last_ms,
            "max": s[-1] if s else 0.0,
            "age_s": (time.monotonic() - self.last_at) if self.last_at else -1.0,
        }


class LatencyRegistry:
    """Named collection of latency windows. Keys are created on first use."""

    def __init__(self) -> None:
        self._windows: dict[str, LatencyWindow] = {}

    def record(self, name: str, ms: float, *, ok: bool = True) -> None:
        w = self._windows.get(name)
        if w is None:
            w = self._windows[name] = LatencyWindow()
        w.record(ms, ok=ok)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {name: w.snapshot() for name, w in self._windows.items()}


# The one process-global registry. Import and use directly.
LATENCY = LatencyRegistry()


@dataclass
class _Timer:
    name: str
    _t0: float = 0.0
    ok: bool = True

    def __enter__(self) -> "_Timer":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (time.monotonic() - self._t0) * 1000.0
        # An exception propagating through the block counts as an error
        # sample (latency is meaningless, but the error tally isn't).
        LATENCY.record(self.name, elapsed_ms, ok=(exc_type is None) and self.ok)


def timed(name: str) -> _Timer:
    """Context manager that records the block's wall-clock latency under `name`.

    Records an error sample (not a latency sample) if the block raises, so a
    timed-out or failed round-trip still shows up in the error tally without
    polluting the percentiles with its stuck duration.

        with timed("llm"):
            resp = await client.chat(...)
    """
    return _Timer(name)
