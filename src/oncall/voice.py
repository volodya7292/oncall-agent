"""Spoken-output sanitizer for chat replies.

`to_voice_text(text, language)` returns a string suitable for piping into a
TTS engine: markdown stripped, code fences collapsed to a short summary, URLs
shortened, multi-blank-lines flattened. Pure function — no I/O.

This is intentionally lossy. The full markdown reply is still returned in
`text`; `voice_text` is the "what the speaker should actually say" view for a
phone / voice-assistant client.
"""

from __future__ import annotations

import re


# Triple-backtick fenced code blocks (multi-line). Collapsed to a short marker.
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")
# Inline code: `foo` — keep the contents, drop the backticks.
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
# Bold/italic: **x**, __x__, *x*, _x_. Keep the inner text.
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC_RE = re.compile(r"(?<![\w*])([*_])([^*_\s][^*_]*?)\1(?![\w*])")
# Markdown links [text](url) → just "text".
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Bare URLs → "link".
_URL_RE = re.compile(r"https?://\S+")
# Bullet markers / numbered list markers at line start.
_BULLET_RE = re.compile(r"(?m)^\s*[-*•]\s+")
_NUMBERED_RE = re.compile(r"(?m)^\s*\d+\.\s+")
# Heading markers (#, ##, ###) at line start.
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
# Block-quote markers.
_QUOTE_RE = re.compile(r"(?m)^\s*>\s?")
# Three+ blank lines → one blank line.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# Trailing whitespace per line.
_TRAILING_WS_RE = re.compile(r"[ \t]+(?=\n|$)")


def to_voice_text(text: str, language: str | None = None) -> str:
    """Render `text` as something a TTS engine can speak. `language` is
    currently advisory — the transformation is language-agnostic — but we
    accept it so callers don't have to refactor when language-specific rules
    are added (e.g. number-spelling localization)."""
    del language  # reserved for future language-specific handling
    if not text:
        return ""

    out = text

    # 1. Collapse fenced code blocks to a short spoken stand-in. Done BEFORE
    # inline-code so the inner backticks don't get treated as inline.
    out = _FENCED_CODE_RE.sub("(code block omitted)", out)

    # 2. Links: prefer the link text over the URL. Do this before the bare-URL
    # rule so the URL inside `(...)` isn't separately squashed.
    out = _LINK_RE.sub(r"\1", out)

    # 3. Bare URLs → "link".
    out = _URL_RE.sub("link", out)

    # 4. Inline code: drop the backticks, keep the token.
    out = _INLINE_CODE_RE.sub(r"\1", out)

    # 5. Bold / italic markers — keep the inner text.
    out = _BOLD_RE.sub(r"\2", out)
    out = _ITALIC_RE.sub(r"\2", out)

    # 6. Heading / quote markers at line start.
    out = _HEADING_RE.sub("", out)
    out = _QUOTE_RE.sub("", out)

    # 7. List markers — drop the marker, keep the content (TTS reads "dash"
    # otherwise, which sounds wrong in spoken form).
    out = _BULLET_RE.sub("", out)
    out = _NUMBERED_RE.sub("", out)

    # 8. Whitespace cleanup.
    out = _TRAILING_WS_RE.sub("", out)
    out = _MULTI_BLANK_RE.sub("\n\n", out)
    out = out.strip()

    return out
