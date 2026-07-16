"""Executor prompt rendering invariants.

Both properties here fail SILENTLY in production — no crash, no error, just
a worse prompt — so they get a test rather than a code comment.
"""

from __future__ import annotations

import re

from oncall.config import Paths, Settings
from oncall.result_delivery import EXECUTOR_REPLY_BUDGET_CHARS, MAX_USER_FACING_CHARS
from oncall.supervisor import Supervisor


def _render(tmp_path) -> str:
    settings = Settings(
        oncall_token="t", oncall_db_path=tmp_path / "db.sqlite", ai_gateway_api_key="x",
    )
    sv = Supervisor(db=None, events=None, settings=settings, paths=Paths())
    return sv._render_executor_prompt()


def test_no_placeholder_survives_rendering(tmp_path) -> None:
    """Every `{{...}}` in the prompt .md must have a substitution in
    `_render_executor_prompt`. The two live in different files, so adding a
    placeholder without its replace() ships a literal `{{foo}}` to the model
    — which reads as noise and is invisible short of eyeballing the prompt.
    """
    leftover = re.findall(r"\{\{[^}]+\}\}", _render(tmp_path))
    assert leftover == [], f"placeholder(s) with no substitution: {leftover}"


def test_executor_budget_stays_under_the_delivery_ceiling(tmp_path) -> None:
    """The executor is asked for less than the ceiling on purpose.

    Nothing rewrites the executor's reply anymore (an LLM compressor used to,
    and corrupted attribution doing it — see result_delivery's docstring), so
    `MAX_USER_FACING_CHARS` is now a guillotine, not a compressor: whatever
    the executor overshoots by gets cut off the user's message mid-word. The
    gap is the slack that absorbs a mild overshoot. Raise the budget to meet
    the ceiling and every overshoot becomes user-visible again.
    """
    assert EXECUTOR_REPLY_BUDGET_CHARS < MAX_USER_FACING_CHARS
    assert f"≤{EXECUTOR_REPLY_BUDGET_CHARS} characters" in _render(tmp_path)
