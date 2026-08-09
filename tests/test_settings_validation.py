"""Behavioral contracts for the flat OcrSettings API."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from bigocrpdf.services import settings as settings_module
from bigocrpdf.services.rapidocr_service.config import (
    DEFAULT_DETECTION_FULL_RESOLUTION,
    DEFAULT_ENABLE_BILEVEL_COMPRESSION,
    DEFAULT_FORCE_BILEVEL_COMPRESSION,
    DEFAULT_PAGE_LAYOUT,
)
from bigocrpdf.services.settings import OcrSettings
from bigocrpdf.utils import config_manager as config_manager_module
from bigocrpdf.utils.config_manager import ConfigManager


@pytest.fixture
def settings_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, ConfigManager]]:
    """Isolate both JSON settings and the legacy selected-file list."""
    config_path = tmp_path / "settings.json"
    selected_file_path = tmp_path / "selected-files"
    monkeypatch.setattr(config_manager_module, "LEGACY_PATHS", {})

    managers = {"current": ConfigManager(str(config_path))}
    monkeypatch.setattr(
        settings_module,
        "get_config_manager",
        lambda: managers["current"],
    )
    monkeypatch.setattr(settings_module, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "SELECTED_FILE_PATH", str(selected_file_path))
    return config_path, managers


def test_settings_round_trip_preserves_flat_owner_values(
    settings_environment: tuple[Path, dict[str, ConfigManager]], tmp_path: Path
) -> None:
    config_path, managers = settings_environment
    output_dir = tmp_path / "output"
    settings = OcrSettings()
    settings.pdf_suffix = "searchable"
    settings.md_include_front_matter = True
    settings.md_open_after_export = True
    settings.page_layout = "two-pages"
    settings.enable_bilevel_compression = True
    settings.force_bilevel_compression = True
    settings.detection_full_resolution = True
    settings.enable_deskew = False
    settings.quick_start_mode = False

    settings.save_settings("english", str(output_dir), save_in_same_folder=True)

    managers["current"] = ConfigManager(str(config_path))
    reloaded = OcrSettings()

    assert reloaded.lang == "english"
    assert reloaded.ocr_language == "english"
    assert reloaded._snapshot_ocr_config().language == "english"
    assert reloaded.destination_folder == str(output_dir)
    assert reloaded.save_in_same_folder is True
    assert reloaded.pdf_suffix == "searchable"
    assert reloaded.md_include_front_matter is True
    assert reloaded.md_open_after_export is True
    assert reloaded.page_layout == "two-pages"
    assert reloaded.enable_bilevel_compression is True
    assert reloaded.force_bilevel_compression is True
    assert reloaded.detection_full_resolution is True
    assert reloaded.enable_deskew is False
    assert reloaded.quick_start_mode is False


def test_first_run_uses_versioned_same_folder_default(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
) -> None:
    settings = OcrSettings()

    assert settings.save_in_same_folder is True


def test_load_settings_refreshes_preferences_written_by_other_process(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
) -> None:
    config_path, _managers = settings_environment
    settings = OcrSettings()
    external = ConfigManager(str(config_path))
    external.set("rapidocr.language", "english")

    settings.load_settings()

    assert settings.ocr_language == "english"
    assert settings._snapshot_ocr_config().language == "english"


def test_reset_to_defaults_reinitializes_owners_and_preserves_saved_queue(
    settings_environment: tuple[Path, dict[str, ConfigManager]], tmp_path: Path
) -> None:
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n")
    settings = OcrSettings()
    assert settings.add_files([str(input_pdf)]) == 1
    settings.pdf_suffix = "custom"
    settings.md_include_front_matter = True
    settings.page_layout = "custom-layout"
    settings.detection_full_resolution = not DEFAULT_DETECTION_FULL_RESOLUTION
    settings.save_settings("english", str(tmp_path), save_in_same_folder=False)

    settings.reset_to_defaults()

    assert settings.selected_files == [str(input_pdf)]
    assert settings.pdf_suffix == "ocr"
    assert settings.save_in_same_folder is True
    assert settings.md_include_front_matter is False
    assert settings.page_layout == DEFAULT_PAGE_LAYOUT
    assert settings.enable_bilevel_compression is DEFAULT_ENABLE_BILEVEL_COMPRESSION
    assert settings.force_bilevel_compression is DEFAULT_FORCE_BILEVEL_COMPRESSION
    assert settings.detection_full_resolution is DEFAULT_DETECTION_FULL_RESOLUTION


def test_selected_files_round_trip_preserves_newline_and_replaces_symlink(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    unusual_file = tmp_path / "line one\nline two.pdf"
    unusual_file.write_bytes(b"%PDF-1.4\n")
    selected_path = Path(settings_module.SELECTED_FILE_PATH)
    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP", encoding="utf-8")
    selected_path.symlink_to(victim)
    settings = OcrSettings()
    settings.selected_files = [str(unusual_file)]

    settings._save_selected_files()

    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert not selected_path.is_symlink()
    reloaded = OcrSettings()
    assert reloaded.selected_files == [str(unusual_file)]


def test_selected_file_list_symlink_is_not_loaded(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    queued = tmp_path / "queued.pdf"
    queued.write_bytes(b"%PDF-1.4\n")
    victim = tmp_path / "victim.json"
    victim.write_text(
        json.dumps({"version": 1, "selected_files": [str(queued)]}),
        encoding="utf-8",
    )
    Path(settings_module.SELECTED_FILE_PATH).symlink_to(victim)

    settings = OcrSettings()

    assert settings.selected_files == []


def test_save_settings_reports_persistence_failure(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    settings = OcrSettings()
    settings._config.save = lambda: False

    with pytest.raises(OSError, match="Could not persist settings"):
        settings.save_settings("english", str(tmp_path))


def test_legacy_selected_file_list_still_loads(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF-1.4\n")
    second.write_bytes(b"%PDF-1.4\n")
    Path(settings_module.SELECTED_FILE_PATH).write_text(
        f"{first}\n{second}\n",
        encoding="utf-8",
    )

    settings = OcrSettings()

    assert settings.selected_files == [str(first), str(second)]


def test_queue_mutations_persist_between_settings_instances(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    replacement = tmp_path / "replacement.pdf"
    for path in (first, second, replacement):
        path.write_bytes(b"%PDF-1.4\n")
    settings = OcrSettings()
    assert settings.add_files([str(first), str(second)]) == 2
    settings.page_ranges = {str(first): (1, 2), str(second): (2, 3)}
    settings.file_modifications = {str(first): {"pages": []}, str(second): {"pages": []}}
    settings.original_file_paths = {str(first): "first.png", str(second): "second.png"}

    assert settings._remove_file(str(first)) is True
    assert str(first) not in settings.page_ranges
    assert str(first) not in settings.file_modifications
    assert str(first) not in settings.original_file_paths
    assert settings._replace_file(str(second), str(replacement)) is True
    assert OcrSettings().selected_files == [str(replacement)]
    assert str(second) not in settings.page_ranges
    assert str(second) not in settings.file_modifications
    assert settings.original_file_paths == {str(replacement): "second.png"}

    assert settings._clear_files() is True
    assert settings.page_ranges == {}
    assert settings.file_modifications == {}
    assert settings.original_file_paths == {}
    assert OcrSettings().selected_files == []


def test_queue_deduplicates_aliases_within_batches_and_persisted_state(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    alias = tmp_path / "alias.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    alias.symlink_to(source)
    settings = OcrSettings()

    assert settings.add_files([str(alias), str(source), str(alias)]) == 1
    assert settings.selected_files == [str(alias)]

    Path(settings_module.SELECTED_FILE_PATH).write_text(
        json.dumps({"version": 1, "selected_files": [str(source), str(alias)]}),
        encoding="utf-8",
    )
    assert OcrSettings().selected_files == [str(source)]


def test_full_processing_reset_clears_and_persists_all_queue_state(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    settings = OcrSettings()
    assert settings.add_files([str(source)]) == 1
    settings.page_ranges[str(source)] = (1, 1)
    settings.file_modifications[str(source)] = {"pages": []}
    settings.original_file_paths[str(source)] = "source.png"

    settings.reset_processing_state(full=True)

    assert settings.selected_files == []
    assert settings.page_ranges == {}
    assert settings.file_modifications == {}
    assert settings.original_file_paths == {}
    assert OcrSettings().selected_files == []


def test_rejected_generated_file_preserves_existing_queue_ownership(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated.pdf"
    generated.write_bytes(b"%PDF-1.4\n")
    settings = OcrSettings()
    assert settings.add_files([str(generated)]) == 1

    assert not settings._add_generated_file(str(generated), "source.png")
    assert generated.exists()
    assert settings.original_file_paths == {}


def test_invalid_generated_file_is_removed_without_publishing_name(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    from bigocrpdf.utils import temp_manager

    generated = tmp_path / "generated.txt"
    generated.write_text("temporary", encoding="utf-8")
    temp_manager.track_file(str(generated))
    settings = OcrSettings()

    assert not settings._add_generated_file(str(generated), "source.png")
    assert not generated.exists()
    assert settings.original_file_paths == {}


def test_invalid_untracked_file_is_preserved_when_generation_is_rejected(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    user_file = tmp_path / "user-not-generated.txt"
    user_file.write_text("keep", encoding="utf-8")
    settings = OcrSettings()

    assert not settings._add_generated_file(str(user_file), "source.png")

    assert user_file.read_text(encoding="utf-8") == "keep"
    assert settings.original_file_paths == {}


def test_add_files_rolls_back_memory_when_queue_save_fails(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    user_file = tmp_path / "user.pdf"
    user_file.write_bytes(b"%PDF-1.4\n")
    settings = OcrSettings()

    with patch(
        "bigocrpdf.utils.durable_writes.write_text_atomically",
        side_effect=OSError("disk full"),
    ):
        assert settings.add_files([str(user_file)]) == 0

    assert settings.selected_files == []
    assert user_file.exists()


def test_legacy_cleanup_hook_preserves_unowned_sidecars(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.pdf"
    output.write_bytes(b"%PDF-1.4\n")
    sidecar_dir = tmp_path / ".temp"
    sidecar_dir.mkdir()
    sidecars = (
        sidecar_dir / "temp_report.txt",
        sidecar_dir / "temp_report-user-data.txt",
        sidecar_dir / "notes.txt",
    )
    for sidecar in sidecars:
        sidecar.write_text("keep", encoding="utf-8")
    settings = OcrSettings()

    settings.cleanup_temp_files(processed_files=[str(output)])

    assert all(sidecar.read_text(encoding="utf-8") == "keep" for sidecar in sidecars)


def test_removing_generated_queue_entry_deletes_only_tracked_input(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    from bigocrpdf.utils import temp_manager

    generated = tmp_path / "bigocr_merge_owned.pdf"
    generated.write_bytes(b"%PDF-1.4\n")
    temp_manager.track_file(str(generated))
    settings = OcrSettings()
    assert settings._add_generated_file(str(generated), str(tmp_path / "source.pdf"))

    assert settings._remove_file(str(generated))

    assert not generated.exists()
    assert settings.original_file_paths == {}


def test_removing_queue_entry_preserves_untracked_file_despite_generated_name(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    user_file = tmp_path / "bigocr_merge_user_document.pdf"
    user_file.write_bytes(b"%PDF-1.4\n")
    settings = OcrSettings()
    assert settings.add_files([str(user_file)]) == 1
    settings.original_file_paths = {str(user_file): str(tmp_path / "source.pdf")}

    assert settings._remove_file(str(user_file))

    assert user_file.read_bytes() == b"%PDF-1.4\n"
    assert settings.original_file_paths == {}


def test_partial_batch_releases_success_and_preserves_pending_generated_input(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    from bigocrpdf.utils import temp_manager

    successful = tmp_path / "successful-generated.pdf"
    pending = tmp_path / "pending-generated.pdf"
    for path in (successful, pending):
        path.write_bytes(b"%PDF-1.4\n")
        temp_manager.track_file(str(path))
    settings = OcrSettings()
    assert settings._add_generated_file(str(successful), "successful.png")
    assert settings._add_generated_file(str(pending), "pending.png")

    assert settings._remove_file(str(successful))

    assert not successful.exists()
    assert pending.exists()
    assert settings.selected_files == [str(pending)]
    assert settings.original_file_paths == {str(pending): "pending.png"}

    assert settings._remove_file(str(pending))


def test_replacing_generated_entry_releases_only_previous_tracked_file(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    from bigocrpdf.utils import temp_manager

    first = tmp_path / "first-generated.pdf"
    replacement = tmp_path / "replacement-generated.pdf"
    for path in (first, replacement):
        path.write_bytes(b"%PDF-1.4\n")
        temp_manager.track_file(str(path))
    settings = OcrSettings()
    assert settings._add_generated_file(str(first), "original.png")

    assert settings._replace_file(str(first), str(replacement))

    assert not first.exists()
    assert replacement.exists()
    assert settings.selected_files == [str(replacement)]
    assert settings.original_file_paths == {str(replacement): "original.png"}

    assert settings._remove_file(str(replacement))


def test_remove_rolls_back_and_preserves_tracked_input_when_queue_save_fails(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    from bigocrpdf.utils import temp_manager

    generated = tmp_path / "generated.pdf"
    generated.write_bytes(b"%PDF-1.4\n")
    temp_manager.track_file(str(generated))
    settings = OcrSettings()
    assert settings._add_generated_file(str(generated), "source.png")
    settings.page_ranges[str(generated)] = (1, 2)
    settings.file_modifications[str(generated)] = {"pages": [{"rotation": 90}]}

    with patch(
        "bigocrpdf.utils.durable_writes.write_text_atomically",
        side_effect=OSError("disk full"),
    ):
        assert not settings._remove_file(str(generated))

    assert settings.selected_files == [str(generated)]
    assert settings.page_ranges == {str(generated): (1, 2)}
    assert settings.file_modifications == {str(generated): {"pages": [{"rotation": 90}]}}
    assert settings.original_file_paths == {str(generated): "source.png"}
    assert generated.exists()

    assert settings._remove_file(str(generated))


def test_replace_rolls_back_and_preserves_both_temps_when_queue_save_fails(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    from bigocrpdf.utils import temp_manager

    current = tmp_path / "current.pdf"
    replacement = tmp_path / "replacement.pdf"
    for path in (current, replacement):
        path.write_bytes(b"%PDF-1.4\n")
        temp_manager.track_file(str(path))
    settings = OcrSettings()
    assert settings._add_generated_file(str(current), "source.png")
    settings.page_ranges[str(current)] = (1, 1)
    settings.file_modifications[str(current)] = {"pages": []}

    with patch(
        "bigocrpdf.utils.durable_writes.write_text_atomically",
        side_effect=OSError("permission denied"),
    ):
        assert not settings._replace_file(str(current), str(replacement))

    assert settings.selected_files == [str(current)]
    assert settings.page_ranges == {str(current): (1, 1)}
    assert settings.file_modifications == {str(current): {"pages": []}}
    assert settings.original_file_paths == {str(current): "source.png"}
    assert current.exists()
    assert replacement.exists()

    assert settings._replace_file(str(current), str(replacement))
    assert settings._remove_file(str(replacement))


def test_move_file_persists_queue_order(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    files = [tmp_path / f"{name}.pdf" for name in ("first", "second", "third")]
    for file_path in files:
        file_path.write_bytes(b"%PDF-1.4\n")
    settings = OcrSettings()
    assert settings.add_files([str(file_path) for file_path in files]) == 3

    assert settings._move_file(0, 2)

    expected = [str(files[1]), str(files[2]), str(files[0])]
    assert settings.selected_files == expected
    persisted = json.loads(Path(settings_module.SELECTED_FILE_PATH).read_text(encoding="utf-8"))
    assert persisted["selected_files"] == expected


def test_move_file_restores_order_when_queue_save_fails(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    files = [tmp_path / f"{name}.pdf" for name in ("first", "second", "third")]
    for file_path in files:
        file_path.write_bytes(b"%PDF-1.4\n")
    settings = OcrSettings()
    original_order = [str(file_path) for file_path in files]
    assert settings.add_files(original_order) == 3

    with patch(
        "bigocrpdf.utils.durable_writes.write_text_atomically",
        side_effect=OSError("disk full"),
    ):
        assert not settings._move_file(0, 2)

    assert settings.selected_files == original_order


def test_replace_stays_consistent_when_queue_directory_sync_fails_after_replace(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    from bigocrpdf.utils import durable_writes, temp_manager

    current = tmp_path / "current.pdf"
    replacement = tmp_path / "replacement.pdf"
    for path in (current, replacement):
        path.write_bytes(b"%PDF-1.4\n")
        temp_manager.track_file(str(path))
    settings = OcrSettings()
    assert settings._add_generated_file(str(current), "source.png")
    selected_path = Path(settings_module.SELECTED_FILE_PATH)
    original_sync = durable_writes._fsync_directory
    failure_injected = False

    def fail_once_after_publication(directory: Path) -> None:
        nonlocal failure_injected
        persisted = json.loads(selected_path.read_text(encoding="utf-8"))
        if not failure_injected and persisted["selected_files"] == [str(replacement)]:
            failure_injected = True
            raise OSError("simulated directory sync failure")
        original_sync(directory)

    with patch.object(
        durable_writes,
        "_fsync_directory",
        side_effect=fail_once_after_publication,
    ):
        assert not settings._replace_file(str(current), str(replacement))

    assert failure_injected
    assert settings.selected_files == [str(current)]
    assert json.loads(selected_path.read_text(encoding="utf-8"))["selected_files"] == [str(current)]
    assert current.exists()
    assert replacement.exists()

    assert settings._replace_file(str(current), str(replacement))
    assert settings._remove_file(str(replacement))


def test_clear_removes_only_tracked_generated_inputs_after_queue_save(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    from bigocrpdf.utils import temp_manager

    generated = tmp_path / "generated.pdf"
    user_file = tmp_path / "user.pdf"
    for path in (generated, user_file):
        path.write_bytes(b"%PDF-1.4\n")
    temp_manager.track_file(str(generated))
    settings = OcrSettings()
    assert settings.add_files([str(user_file)]) == 1
    assert settings._add_generated_file(str(generated), "source.png")

    assert settings._clear_files()

    assert not generated.exists()
    assert user_file.exists()
    assert settings.selected_files == []
    assert settings.original_file_paths == {}


def test_clear_rolls_back_and_preserves_files_when_queue_save_fails(
    settings_environment: tuple[Path, dict[str, ConfigManager]],
    tmp_path: Path,
) -> None:
    from bigocrpdf.utils import temp_manager

    generated = tmp_path / "generated.pdf"
    user_file = tmp_path / "user.pdf"
    for path in (generated, user_file):
        path.write_bytes(b"%PDF-1.4\n")
    temp_manager.track_file(str(generated))
    settings = OcrSettings()
    assert settings.add_files([str(user_file)]) == 1
    assert settings._add_generated_file(str(generated), "source.png")
    settings.page_ranges[str(generated)] = (1, 1)

    with patch(
        "bigocrpdf.utils.durable_writes.write_text_atomically",
        side_effect=OSError("read-only filesystem"),
    ):
        assert not settings._clear_files()

    assert settings.selected_files == [str(user_file), str(generated)]
    assert settings.page_ranges == {str(generated): (1, 1)}
    assert settings.original_file_paths == {str(generated): "source.png"}
    assert generated.exists()
    assert user_file.exists()

    assert settings._clear_files()
