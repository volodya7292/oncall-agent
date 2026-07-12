"""Latency registry: percentile math + the error-vs-latency split.

Both are non-obvious: a raised block must count as an *error* (not a
latency sample with a bogus stuck duration), and percentiles must be
computed over only the successful samples.
"""

from oncall.metrics import LatencyRegistry, timed, LATENCY


def test_percentiles_over_successful_samples_only():
    reg = LatencyRegistry()
    for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        reg.record("op", v)
    snap = reg.snapshot()["op"]
    assert snap["n"] == 10
    assert snap["total"] == 10
    assert snap["errors"] == 0
    assert snap["p50"] == 55.0          # midpoint of 50 and 60
    assert snap["max"] == 100.0
    assert 90.0 <= snap["p95"] <= 100.0


def test_error_sample_does_not_pollute_latency():
    reg = LatencyRegistry()
    reg.record("op", 100)
    reg.record("op", 999_999, ok=False)  # a stuck/timed-out call
    snap = reg.snapshot()["op"]
    assert snap["n"] == 1                 # only the good sample is in the window
    assert snap["errors"] == 1
    assert snap["p95"] == 100.0          # the failure's duration is excluded
    assert snap["last_ms"] == 100.0


def test_timed_records_error_on_raise():
    before = LATENCY.snapshot().get("unit-test-raise", {}).get("errors", 0)
    try:
        with timed("unit-test-raise"):
            raise ValueError("boom")
    except ValueError:
        pass
    after = LATENCY.snapshot()["unit-test-raise"]
    assert after["errors"] == before + 1
    assert after["n"] == 0               # no latency sample recorded for the failure
