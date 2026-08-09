"""Contracts for throttled UI progress state."""

from bigocrpdf.utils.progress_state import ProgressState


def test_fraction_throttles_small_intermediate_changes() -> None:
    state = ProgressState(fraction=0.5)

    assert state.update_fraction(0.505) is False
    assert state.fraction == 0.5


def test_fraction_never_suppresses_completion_endpoint() -> None:
    state = ProgressState(fraction=0.995)

    assert state.update_fraction(1.0) is True
    assert state.fraction == 1.0


def test_text_can_be_cleared() -> None:
    state = ProgressState(text="50%")

    assert state.update_text("") is True
    assert state.text == ""
