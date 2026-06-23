"""Tests for `format_local_now`, the per-turn clock injected into the
operator and executor prompts.

Regression context: the operator once told the owner it was "21:05" at
00:33 local because it had NO clock in context and fabricated one. This
helper is now the single source of that clock, so its output must be
unambiguous (explicit zone + UTC offset) and must never raise — a crash
here would drop the time note and re-open the fabrication hole.

Device-independent: every assertion passes an explicit IANA zone, so the
host's local tz never enters the picture.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from oncall.models import format_local_now

# 2026-06-24 00:46 (Wednesday), Europe/Berlin (UTC+02:00)
_SHAPE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} \(\w+\), .+ \(UTC[+-]\d{2}:\d{2}\)$"
)


def test_shape_and_label_for_named_zone() -> None:
    s = format_local_now("Europe/Berlin")
    assert _SHAPE.match(s), s
    # The label is the IANA name we passed, not the abbreviation.
    assert "Europe/Berlin" in s


def test_utc_offset_is_colon_formatted_not_raw_strftime() -> None:
    # The offset is sliced from strftime("%z")'s "+0000"/"+0200" into a
    # colon form. This is the off-by-a-slice boundary worth pinning.
    assert "(UTC+00:00)" in format_local_now("UTC")


def test_time_matches_real_clock_within_a_minute() -> None:
    # Pin the correctness claim: the rendered HH:MM is the actual current
    # time in that zone (the 21:05-at-00:33 bug was a ~3.5h divergence).
    s = format_local_now("UTC")
    now = datetime.now(timezone.utc)
    hh_mm = re.search(r" (\d{2}:\d{2}) ", s).group(1)
    # Allow a one-minute skew for the clock ticking between the two reads.
    rendered_min = int(hh_mm[:2]) * 60 + int(hh_mm[3:])
    real_min = now.hour * 60 + now.minute
    assert abs(rendered_min - real_min) <= 1 or abs(rendered_min - real_min) >= 1439


def test_invalid_zone_falls_back_without_raising() -> None:
    # A bad operator_timezone must degrade to local tz, never crash the turn.
    s = format_local_now("Not/ARealZone")
    assert _SHAPE.match(s), s
