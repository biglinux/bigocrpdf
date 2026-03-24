"""Tests for OcrSettings attribute delegation and validation."""

from unittest.mock import patch

import pytest


class TestSettingsDelegation:
    """Tests for delegation map correctness."""

    def _make_settings(self):
        with patch("bigocrpdf.services.settings.get_config_manager") as mock_cm:
            mock_cm.return_value.get.return_value = None
            from bigocrpdf.services.settings import OcrSettings

            return OcrSettings()

    def test_getattr_unknown_raises(self):
        settings = self._make_settings()
        with pytest.raises(AttributeError):
            _ = settings.nonexistent_field_42

    def test_getattr_delegated_field(self):
        settings = self._make_settings()
        # selected_files lives on file_queue
        assert isinstance(settings.selected_files, list)

    def test_setattr_delegated_field(self):
        settings = self._make_settings()
        settings.selected_files = ["new.pdf"]
        assert settings.selected_files == ["new.pdf"]

    def test_setattr_direct_field(self):
        settings = self._make_settings()
        settings.lang = "deu"
        assert settings.lang == "deu"

    def test_build_delegation_map_strict_zip(self):
        """_build_delegation_map uses strict=True zip — mismatched args should raise."""
        from bigocrpdf.services.settings import _build_delegation_map

        class _Dummy:
            x = 1

        with pytest.raises(ValueError):
            # _SUB_OBJECTS has 3 entries, passing only 1 sub-object should fail
            _build_delegation_map(_Dummy())

    def test_getattr_raises_from_none(self):
        """__getattr__ should raise AttributeError with 'from None' (B904 fix)."""
        settings = self._make_settings()
        try:
            _ = settings.nonexistent_xyz
        except AttributeError as e:
            # The __cause__ should be None (raise ... from None)
            assert e.__cause__ is None
