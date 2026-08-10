"""A record of which geometric corrections ran on a page, and why.

The preprocessor decides between several dewarp and perspective strategies
through a first-hit-wins cascade whose branches mostly swallow their own
exceptions.  Without a record, a correction that silently declined to run and
one that ran and did nothing are indistinguishable from the outside -- which is
why nothing could test the cascade, and why a page that came out uncorrected
gave no clue about which gate rejected it.

The trace travels with the page: ``page_worker`` puts it in the result dict,
``backend_text_layer`` copies it into ``OcrPage.diagnostics``, and from there it
reaches the ``.bigocr.json`` sidecar and every benchmark record.  So values must
stay JSON scalars, and the whole thing must stay small.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# One page cannot legitimately run more steps than this; a longer list means a
# loop is recording, and an unbounded diagnostics blob would end up in the
# sidecar of every page.
MAX_STEPS = 16

# Reasons a step did not change the image. Free-form strings are still allowed
# for the exception case, but the common ones are named so tests and log
# readers agree on the spelling.
REASON_OK = "ok"
REASON_DISABLED = "disabled"
REASON_BELOW_THRESHOLD = "below_threshold"
REASON_REJECTED_VALIDATION = "rejected_validation"
REASON_NO_CHANGE = "no_change"


@dataclass(slots=True)
class GeometryStep:
    """One correction stage: what it chose, whether it changed anything, why."""

    step: str
    method: str = "none"
    applied: bool = False
    reason: str = REASON_NO_CHANGE
    duration_ms: float = 0.0
    params: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "method": self.method,
            "applied": self.applied,
            "reason": self.reason,
            "duration_ms": round(self.duration_ms, 3),
            "params": {k: round(float(v), 4) for k, v in self.params.items()},
        }


class GeometryTrace:
    """Collects one :class:`GeometryStep` per correction stage of a page."""

    __slots__ = ("steps",)

    def __init__(self) -> None:
        self.steps: list[GeometryStep] = []

    def reset(self) -> None:
        self.steps.clear()

    @contextmanager
    def stage(self, step: str) -> Iterator[GeometryStep]:
        """Time a correction stage and record it.

        The caller fills in ``method``, ``applied``, ``reason`` and ``params``
        on the yielded step.  An escaping exception is recorded as the reason
        and re-raised, so a stage that crashes is still visible in the trace.
        """
        record = GeometryStep(step=step)
        started = time.perf_counter()
        try:
            yield record
        except BaseException as exc:
            record.reason = f"exception:{type(exc).__name__}"
            raise
        finally:
            record.duration_ms = (time.perf_counter() - started) * 1000.0
            if len(self.steps) < MAX_STEPS:
                self.steps.append(record)

    def step_named(self, step: str) -> GeometryStep | None:
        """The most recent record for ``step``, for tests and callers."""
        for record in reversed(self.steps):
            if record.step == step:
                return record
        return None

    @property
    def applied_steps(self) -> list[str]:
        return [record.step for record in self.steps if record.applied]

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [record.as_dict() for record in self.steps],
            "total_ms": round(sum(record.duration_ms for record in self.steps), 3),
        }
