from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtCore import QLockFile
from PySide6.QtGui import QKeySequence

from fdm.screenshot_settings import (
    AfterCaptureTask,
    CaptureMode,
    CollisionPolicy,
    HotkeyBinding,
    ImageFormat,
    SCREENSHOT_SETTINGS_SCHEMA_VERSION,
    ScreenshotSettings,
    ScreenshotSettingsIO,
    UnsupportedScreenshotSettingsVersion,
)
from fdm.services.screenshot_capture import CaptureRect


def test_defaults_are_safe_and_supply_required_portable_hotkeys() -> None:
    settings = ScreenshotSettings()

    assert settings.enabled is False
    assert settings.background_resident is False
    assert settings.autostart is False
    assert settings.after_capture_tasks == (AfterCaptureTask.SAVE,)
    assert set(settings.hotkeys) >= {
        CaptureMode.REGION,
        CaptureMode.WINDOW,
        CaptureMode.FULL_SCREEN,
        CaptureMode.LAST_REGION,
        CaptureMode.CU5,
    }
    for binding in settings.hotkeys.values():
        parsed = QKeySequence.fromString(
            binding.sequence,
            QKeySequence.SequenceFormat.PortableText,
        )
        assert not parsed.isEmpty()


def test_settings_round_trip_all_domain_values(tmp_path: Path) -> None:
    path = tmp_path / "screenshot-settings.json"
    settings = ScreenshotSettings(
        enabled=True,
        autostart=True,
        output_directory=str(tmp_path / "captures"),
        filename_template="Fiber_{datetime}_{mode}_{counter:03d}",
        image_format=ImageFormat.WEBP,
        png_compression=8,
        jpeg_quality=87,
        webp_quality=83,
        collision_policy=CollisionPolicy.FAIL,
        after_capture_tasks=(AfterCaptureTask.COPY_CLIPBOARD, AfterCaptureTask.SAVE),
        delay_ms=1250,
        include_cursor=True,
        show_editor=True,
        notification=False,
        hotkeys={CaptureMode.CU5: HotkeyBinding("Ctrl+Alt+5")},
        cu5_selector={
            "process": "CU-5.exe",
            "prefer_child": True,
            "last_handle": 123,
        },
        cu5_diagnostics_enabled=True,
        last_region=CaptureRect(-800, 120, 640, 480),
    )

    saved = ScreenshotSettingsIO.save(settings, path)
    loaded = ScreenshotSettingsIO.load(path)

    assert saved == path
    assert loaded.to_dict() == settings.normalized().to_dict()
    assert loaded.last_region == CaptureRect(-800, 120, 640, 480)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCREENSHOT_SETTINGS_SCHEMA_VERSION
    assert payload["last_region"]["coordinate_space"] == "physical_pixels"


def test_from_dict_normalizes_invalid_values_and_legacy_printscreen() -> None:
    settings = ScreenshotSettings.from_dict(
        {
            "enabled": "false",
            "autostart": "yes",
            "output_directory": "",
            "filename_template": "../unsafe/{unknown}",
            "image_format": "jpg",
            "png_compression": 99,
            "jpeg_quality": -20,
            "webp_quality": "invalid",
            "collision_policy": "rename",
            "after_capture_tasks": ["clipboard", "clipboard", "unknown"],
            "delay_ms": 80_000,
            "hotkeys": {"region": {"sequence": "Ctrl+PrintScreen", "enabled": True}},
            "last_region": {"x": 50, "y": 40, "width": -10, "height": -20},
            "cu5_selector": {"score": float("nan"), "valid": [1, object(), 2]},
        }
    )

    assert settings.enabled is False
    assert settings.autostart is True
    assert settings.filename_template == "Screenshot_{date}_{time}"
    assert settings.image_format is ImageFormat.JPEG
    assert settings.png_compression == 9
    assert settings.jpeg_quality == 1
    assert settings.webp_quality == 90
    assert settings.collision_policy is CollisionPolicy.INCREMENT
    assert settings.after_capture_tasks == (AfterCaptureTask.COPY_CLIPBOARD,)
    assert settings.delay_ms == 60_000
    assert settings.hotkeys[CaptureMode.REGION] == HotkeyBinding("Ctrl+Print")
    assert settings.last_region == CaptureRect(40, 20, 10, 20)
    assert settings.cu5_selector == {"valid": [1, 2]}

    assert ScreenshotSettings(
        filename_template="{counter:999999999d}"
    ).normalized().filename_template == "Screenshot_{date}_{time}"


def test_load_recovers_from_corruption_but_protects_future_versions(tmp_path: Path) -> None:
    path = tmp_path / "screenshot-settings.json"
    path.write_text("{broken", encoding="utf-8")
    assert ScreenshotSettingsIO.load(path).to_dict() == ScreenshotSettings().to_dict()

    path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    with pytest.raises(UnsupportedScreenshotSettingsVersion):
        ScreenshotSettingsIO.load(path)


def test_atomic_save_preserves_previous_settings_on_replace_failure(tmp_path: Path) -> None:
    path = tmp_path / "screenshot-settings.json"
    path.write_text('{"schema_version": 1, "enabled": false}', encoding="utf-8")

    with patch("fdm.atomic_io.os.replace", side_effect=OSError("injected failure")):
        with pytest.raises(OSError, match="injected failure"):
            ScreenshotSettingsIO.save(ScreenshotSettings(enabled=True), path)

    assert json.loads(path.read_text(encoding="utf-8"))["enabled"] is False
    assert list(tmp_path.iterdir()) == [path]


def test_locked_update_merges_latest_values_and_protects_future_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "screenshot-settings.json"
    ScreenshotSettingsIO.save(
        ScreenshotSettings(
            enabled=True,
            cu5_selector={"class_name": "cwndforsdk"},
        ),
        path,
    )

    updated = ScreenshotSettingsIO.update(
        lambda persisted: replace(persisted, delay_ms=750),
        path,
    )

    assert updated.enabled is True
    assert updated.delay_ms == 750
    assert updated.cu5_selector == {"class_name": "cwndforsdk"}
    assert not path.with_name(f".{path.name}.lock").exists()

    future_payload = '{"schema_version":999,"future":"keep"}'
    path.write_text(future_payload, encoding="utf-8")
    called = False

    def should_not_run(_persisted: ScreenshotSettings) -> ScreenshotSettings:
        nonlocal called
        called = True
        return ScreenshotSettings()

    with pytest.raises(UnsupportedScreenshotSettingsVersion):
        ScreenshotSettingsIO.update(should_not_run, path)
    assert called is False
    assert path.read_text(encoding="utf-8") == future_payload
    with pytest.raises(UnsupportedScreenshotSettingsVersion):
        ScreenshotSettingsIO.save(ScreenshotSettings(), path)
    assert path.read_text(encoding="utf-8") == future_payload

    replacement = ScreenshotSettingsIO.update(
        lambda _persisted: ScreenshotSettings(enabled=True),
        path,
        allow_unsupported_replace=True,
    )
    assert replacement.enabled is True
    assert ScreenshotSettingsIO.load(path).enabled is True


def test_update_reports_a_live_cross_process_writer_lock(tmp_path: Path) -> None:
    path = tmp_path / "screenshot-settings.json"
    lock_path = path.with_name(f".{path.name}.lock")
    lock = QLockFile(str(lock_path))
    assert lock.tryLock(0)
    try:
        with patch(
            "fdm.screenshot_settings.SCREENSHOT_SETTINGS_LOCK_TIMEOUT_MS",
            5,
        ):
            with pytest.raises(OSError, match="跨进程写入锁"):
                ScreenshotSettingsIO.update(lambda current: current, path)
    finally:
        lock.unlock()
