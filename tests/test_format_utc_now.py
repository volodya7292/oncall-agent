"""Tests for `format_utc_now`, the per-turn clock injected into the
operator and executor prompts.

Regression context: the operator once told the owner it was "21:05" at
00:33 because it had NO clock in context and fabricated one. This helper
is now the single source of that clock. It is always UTC — deterministic
regardless of the host/container timezone — and must never raise (a crash
here would drop the time note and re-open the fabrication hole). The
operator converts to the owner's local time using their timezone from
memory.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from oncall.models import format_utc_now

# 2026-06-24 00:46 UTC (Wednesday)
_SHAPE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC \(\w+\)$")


def test_shape_is_labelled_utc() -> None:
    s = format_utc_now()
    assert _SHAPE.match(s), s


def test_time_matches_real_utc_within_a_minute() -> None:
    # Pin the correctness claim: the rendered HH:MM is the actual current
    # UTC time (the 21:05-at-00:33 bug was a ~3.5h divergence).
    s = format_utc_now()
    now = datetime.now(timezone.utc)
    hh_mm = re.search(r" (\d{2}:\d{2}) UTC", s).group(1)
    rendered_min = int(hh_mm[:2]) * 60 + int(hh_mm[3:])
    real_min = now.hour * 60 + now.minute
    # Allow one-minute skew for the clock ticking between the two reads
    # (and wrap at midnight).
    assert abs(rendered_min - real_min) <= 1 or abs(rendered_min - real_min) >= 1439
