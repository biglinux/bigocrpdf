"""The record of which geometric corrections ran on a page."""

import json

import pytest

from bigocrpdf.services.rapidocr_service.geometry_trace import (
    MAX_STEPS,
    REASON_BELOW_THRESHOLD,
    REASON_OK,
    GeometryStep,
    GeometryTrace,
)


def test_a_stage_records_its_name_and_duration():
    trace = GeometryTrace()

    with trace.stage("dewarp") as step:
        step.method = "probmap"
        step.applied = True
        step.reason = REASON_OK

    (record,) = trace.steps
    assert record.step == "dewarp"
    assert record.method == "probmap"
    assert record.applied is True
    assert record.duration_ms >= 0.0


def test_stages_are_recorded_in_execution_order():
    trace = GeometryTrace()

    for name in ("dewarp", "perspective", "trim_dark_borders", "deskew"):
        with trace.stage(name):
            pass

    assert [record.step for record in trace.steps] == [
        "dewarp",
        "perspective",
        "trim_dark_borders",
        "deskew",
    ]


def test_an_escaping_exception_is_recorded_and_re_raised():
    """A stage that crashes must still appear, or the trace would hide it."""
    trace = GeometryTrace()

    with pytest.raises(ValueError), trace.stage("dewarp"):
        raise ValueError("probmap exploded")

    assert trace.steps[0].reason == "exception:ValueError"
    assert trace.steps[0].applied is False


def test_step_count_is_capped():
    """An unbounded trace would end up in the sidecar of every page."""
    trace = GeometryTrace()

    for index in range(MAX_STEPS + 5):
        with trace.stage(f"step{index}"):
            pass

    assert len(trace.steps) == MAX_STEPS


def test_reset_clears_the_previous_page():
    trace = GeometryTrace()
    with trace.stage("dewarp"):
        pass

    trace.reset()

    assert trace.steps == []


def test_applied_steps_lists_only_the_corrections_that_changed_the_image():
    trace = GeometryTrace()
    with trace.stage("dewarp") as step:
        step.reason = REASON_BELOW_THRESHOLD
    with trace.stage("perspective") as step:
        step.applied = True

    assert trace.applied_steps == ["perspective"]


def test_step_named_returns_the_most_recent_record():
    trace = GeometryTrace()
    with trace.stage("deskew") as step:
        step.method = "first"
    with trace.stage("deskew") as step:
        step.method = "second"

    assert trace.step_named("deskew").method == "second"
    assert trace.step_named("absent") is None


def test_serialised_trace_is_json_safe():
    """It round-trips through the .bigocr.json sidecar, so it must serialise."""
    trace = GeometryTrace()
    with trace.stage("deskew") as step:
        step.method = "rotate_mean"
        step.applied = True
        step.params = {"angle_mean": -1.8335021, "n_baselines": 19}

    payload = trace.as_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["steps"][0]["params"]["angle_mean"] == pytest.approx(-1.8335)
    assert payload["total_ms"] >= 0.0


def test_a_disabled_step_can_be_recorded_without_timing():
    """Disabled stages are appended directly, bypassing the timer."""
    trace = GeometryTrace()

    trace.steps.append(GeometryStep("dewarp", reason="disabled"))

    assert trace.as_dict()["steps"][0]["reason"] == "disabled"
    assert trace.as_dict()["steps"][0]["duration_ms"] == 0.0
