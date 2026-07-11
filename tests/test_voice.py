"""voice.to_voice_text — pure-function transformation tests.

Covers the rules the sanitizer applies before output is handed to a TTS engine.
Each case asserts the spoken form is sensible, not a verbatim equality match
on every byte — TTS forgives small whitespace differences.
"""

from __future__ import annotations

import pytest

from oncall.voice import (
    strip_expression_tag_backticks,
    strip_expression_tags,
    to_voice_text,
)


def test_empty_string():
    assert to_voice_text("") == ""


# ---- strip_expression_tag_backticks ----
# The operator (despite the prompt) keeps wrapping expression tags in markdown
# backticks; the conversational voice path doesn't run to_voice_text, so this
# guards the TTS chokepoint directly.

def test_strip_backticks_both_sides():
    assert strip_expression_tag_backticks("`[laughter]` ok") == "[laughter] ok"


def test_strip_backticks_one_sided():
    assert strip_expression_tag_backticks("ok [sigh]`") == "ok [sigh]"
    assert strip_expression_tag_backticks("`[surprise-ah] wow") == "[surprise-ah] wow"


def test_strip_multiple_tags_in_one_reply():
    got = strip_expression_tag_backticks("сміюся `[laughter]`, зітхаю `[sigh]`")
    assert got == "сміюся [laughter], зітхаю [sigh]"


def test_bare_tags_untouched():
    s = "[laughter] still works [dissatisfaction-hnn]"
    assert strip_expression_tag_backticks(s) == s


def test_unrelated_backticks_preserved():
    # An inline-code span that isn't an expression tag must survive intact.
    s = "run `ls -la` then [confirmation-en]"
    assert strip_expression_tag_backticks(s) == "run `ls -la` then [confirmation-en]"


# ---- strip_expression_tags (text-channel backstop) ----
# Voice-only tags must never reach a TEXT channel (Telegram). The prompt
# forbids them off-call, but the model drifts after a voice call (its own
# tagged in-call turns linger in the shared session history) — this is the
# deterministic strip at the owner-text chokepoint. Regression: owner received
# "[laughter] О так, черга в Cuore di Vetro…" as a text DM after a call ended.

def test_strip_tags_leading_tag_and_emoji():
    got = strip_expression_tags("[laughter] О так, черга — це класика! 😉")
    assert got == "О так, черга — це класика! 😉"


def test_strip_tags_mid_sentence_no_double_space():
    assert strip_expression_tags("haha [laughter] good") == "haha good"


def test_strip_tags_backtick_wrapped():
    assert strip_expression_tags("text with `[sigh]` wrapped") == "text with wrapped"


def test_strip_tags_only_tag_becomes_empty():
    # A reply that is nothing but a tag yields empty → the send chokepoint
    # then suppresses the message entirely (correct: it was voice-only).
    assert strip_expression_tags("[laughter]") == ""


def test_strip_tags_preserves_unknown_brackets():
    # Only the known expression vocabulary is stripped; legit bracketed text
    # in a reply survives.
    s = "a legit [TODO] item and [note] stay"
    assert strip_expression_tags(s) == s


def test_strip_tags_multiple():
    got = strip_expression_tags("multi [laughter] and [sigh] tags [dissatisfaction-hnn] gone")
    assert got == "multi and tags gone"


def test_strip_tags_no_brackets_fast_path():
    assert strip_expression_tags("nothing to do here") == "nothing to do here"


def test_plain_text_unchanged():
    assert to_voice_text("staging is up.") == "staging is up."


def test_bold_and_italic_markers_stripped():
    out = to_voice_text("Status: **green**. _All good_.")
    assert "**" not in out and "_" not in out
    assert "green" in out and "All good" in out


def test_inline_code_backticks_dropped():
    out = to_voice_text("Run `kubectl get pods` to inspect.")
    assert "`" not in out
    assert "kubectl get pods" in out


def test_fenced_code_block_collapsed():
    text = "Here:\n```bash\nls -la\necho hi\n```\nDone."
    out = to_voice_text(text)
    assert "ls -la" not in out, "code block should be collapsed, not spoken"
    assert "code block" in out.lower()
    assert "Done." in out


def test_markdown_link_speaks_text_only():
    out = to_voice_text("See [the dashboard](https://grafana.internal/d/x).")
    assert out == "See the dashboard."


def test_bare_url_becomes_link_token():
    out = to_voice_text("Hit https://staging.example.com/healthz then report back.")
    assert "https://" not in out
    assert "link" in out


def test_bullets_dropped_keeping_content():
    text = "Projects:\n- alpha\n- bravo\n- charlie"
    out = to_voice_text(text)
    assert "- alpha" not in out
    assert "alpha" in out and "bravo" in out and "charlie" in out


def test_numbered_list_markers_dropped():
    text = "Steps:\n1. start\n2. verify\n3. exit"
    out = to_voice_text(text)
    # The "1." marker should not appear — TTS reads it as "one dot start".
    assert "1." not in out
    assert "start" in out and "verify" in out and "exit" in out


def test_heading_markers_stripped():
    out = to_voice_text("## Status\nGreen.")
    assert out.startswith("Status")
    assert "##" not in out


def test_blockquote_marker_stripped():
    out = to_voice_text("> warning incoming\n> from the abyss")
    assert ">" not in out
    assert "warning incoming" in out
    assert "from the abyss" in out


def test_multiple_blank_lines_collapsed():
    out = to_voice_text("first\n\n\n\nsecond")
    # At most one blank line between paragraphs.
    assert "\n\n\n" not in out
    assert "first" in out and "second" in out


def test_language_param_does_not_change_output_today():
    """Language is reserved for future TTS-localized rules. The current
    transformation is language-agnostic; verify passing the param doesn't
    explode and produces identical output to no-param."""
    raw = "**status**: green."
    assert to_voice_text(raw, language="en") == to_voice_text(raw)
    assert to_voice_text(raw, language="ru") == to_voice_text(raw)


def test_idempotent():
    """Running the sanitizer twice must not change the result."""
    raw = "## Status\n- **alpha** is `up`\n- visit https://x/\n\n\n```code```"
    once = to_voice_text(raw)
    twice = to_voice_text(once)
    assert once == twice


def test_realistic_operator_reply():
    raw = (
        "**5 projects** under `~/SoftwareProjects`:\n"
        "- alpha\n"
        "- bravo\n"
        "- charlie\n"
        "- delta\n"
        "- echo\n\n"
        "See [the dashboard](https://x.internal/y) for more."
    )
    out = to_voice_text(raw)
    assert "**" not in out and "`" not in out
    assert "- alpha" not in out
    for name in ["alpha", "bravo", "charlie", "delta", "echo"]:
        assert name in out
    assert "the dashboard" in out
    assert "https://" not in out
