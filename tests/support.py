"""Shared test scaffolding."""

from __future__ import annotations

import json
from typing import Any

from oncall.classifier import Classifier


class ScriptedClassifierLLM:
    """Stands in for the classifier's shell-judgment LLM.

    Returns `readonly` when the command contains one of `readonly_markers`,
    `mutating` otherwise — enough for tests that need a Bash verdict without a
    network call, and honest about the boundary: the deterministic
    catastrophic layer runs ahead of this and is not stubbed.
    """

    def __init__(self, readonly_markers: tuple[str, ...] = ()) -> None:
        self.readonly_markers = readonly_markers
        self.calls: list[str] = []

    async def chat(self, *, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        command = str(messages[-1].get("content", ""))
        self.calls.append(command)
        readonly = any(m in command for m in self.readonly_markers)
        return {
            "role": "assistant",
            "content": json.dumps({
                "verdict": "readonly" if readonly else "mutating",
                "blast_radius": "stubbed classifier verdict",
                "reason": "stub",
            }),
            "tool_calls": None,
        }


def stub_classifier(*readonly_markers: str) -> Classifier:
    """A Classifier whose shell judgments are scripted. Commands matching any
    marker come back read-only; everything else is mutating."""
    return Classifier(ScriptedClassifierLLM(readonly_markers), "stub-model")
