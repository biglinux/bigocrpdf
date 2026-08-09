from unittest.mock import Mock, patch

from bigocrpdf.utils.timer import safe_remove_source


def test_safe_remove_source_destroys_the_resolved_source() -> None:
    source = Mock()
    context = Mock()
    context.find_source_by_id.return_value = source

    with patch("bigocrpdf.utils.timer.GLib.MainContext.default", return_value=context):
        assert safe_remove_source(42) is True

    context.find_source_by_id.assert_called_once_with(42)
    source.destroy.assert_called_once_with()


def test_safe_remove_source_returns_false_when_source_is_missing() -> None:
    context = Mock()
    context.find_source_by_id.return_value = None

    with patch("bigocrpdf.utils.timer.GLib.MainContext.default", return_value=context):
        assert safe_remove_source(42) is False


def test_safe_remove_source_rejects_invalid_ids_without_lookup() -> None:
    with patch("bigocrpdf.utils.timer.GLib.MainContext.default") as default_context:
        assert safe_remove_source(None) is False
        assert safe_remove_source(0) is False
        assert safe_remove_source(-1) is False

    default_context.assert_not_called()


def test_safe_remove_source_handles_out_of_range_ids() -> None:
    context = Mock()
    context.find_source_by_id.side_effect = OverflowError

    with patch("bigocrpdf.utils.timer.GLib.MainContext.default", return_value=context):
        assert safe_remove_source(2**64) is False
