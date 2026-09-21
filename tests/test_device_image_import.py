from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from dataclasses import replace
import struct
import time
import zipfile

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt, QCoreApplication, QEvent
from PySide6.QtTest import QTest

from fdm.geometry import Line, Point
from fdm.image_processing_models import DisplayTransform, ImageOperationSpec, ImageProcessingRecipe, RasterSemantic
from fdm.models import Calibration, CalibrationPreset, ImageDocument, Measurement, ProjectState
from fdm.services.device_image_io import inspect_source, read_channel, default_selection, DeviceReadCancelled
from fdm.services.raster_io import raster_plane_to_numpy, raster_plane_to_qimage
from fdm.ui.device_import_dialog import DeviceImportDialog, DeviceReadWorker
from fdm.ui.main_window import MainWindow


@pytest.fixture(autouse=True)
def dispose_import_dialogs(desktop_application):
    yield
    for widget in desktop_application.topLevelWidgets():
        if isinstance(widget, DeviceImportDialog):
            widget.reject()
            wait_until(desktop_application, lambda: widget._thread is None)
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def dsx_file(path, *, bad_calibration=False):
    common = "2000000" if bad_calibration else "1000000"
    description = f'''<?xml version="1.0"?><TiffTagDescData><emImageData>Color</emImageData>
    <ImageCommonCalibrationValueX>{common}</ImageCommonCalibrationValueX>
    <ColorImageData><ColorDataPerPixelX>500000</ColorDataPerPixelX><ColorDataPerPixelY>2000000</ColorDataPerPixelY><ColorDataPerPixelZ>0</ColorDataPerPixelZ></ColorImageData>
    <HeightImageData><HeightDataPerPixelX>500000</HeightDataPerPixelX><HeightDataPerPixelY>2000000</HeightDataPerPixelY><HeightDataPerPixelZ>10000</HeightDataPerPixelZ></HeightImageData></TiffTagDescData>'''
    color = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    height = np.array([[0, 1, 300], [1000, 60000, 65535]], dtype=np.uint16)
    with tifffile.TiffWriter(path) as writer:
        writer.write(color, photometric="rgb", description=description, metadata=None)
        writer.write(height, description="HEIGHT", metadata=None)
    return color, height


def oir_bytes(camera=False, user=1, factory_y=1.2):
    kinds = ["green", "blue", "red"] if camera else ["height", "intensity", "invalid"]
    ids = {kind: f"00000000-0000-0000-0000-{index:012d}" for index, kind in enumerate(kinds)}
    spacing = '<length><x>0.125</x><y>0.25</y><z>0.01</z></length><pixelUnit><x>MICRO_METER</x><y>MICRO_METER</y><z>MICRO_METER</z></pixelUnit>'
    if camera:
        channels = '<channel id="color">' + spacing + ''.join(f'<elementChannel id="{ids[k]}"><elementType>{k.upper()}</elementType><depth>1</depth></elementChannel>' for k in kinds) + '</channel>'
    else:
        channels = ''.join(f'<channel id="{ids[k]}">{spacing}<imageDefinition><imageType>{k.upper()}</imageType><depth>2</depth><bitCounts>16</bitCounts></imageDefinition></channel>' for k in kinds)
    y_calibration = f'<y>{factory_y}</y>' if factory_y is not None else ''
    xml = f'''<?xml version="1.0"?><imageProperties><imageInfo><width>3</width><height>2</height><acquireDevice>{'Camera' if camera else 'LSM'}</acquireDevice><phase><group>{channels}</group></phase></imageInfo><acquisition><microscopeConfiguration><pixelCalibration><x>1.1</x>{y_calibration}<z>1</z></pixelCalibration><userPixelCalibration><x>{user}</x><y>1</y><z>1</z></userPixelCalibration></microscopeConfiguration></acquisition></imageProperties>'''.encode()
    blocks = [(0, struct.pack('<I', len(xml)) + xml)]
    arrays = {}
    for k in reversed(kinds):
        array = np.arange(6, dtype=np.uint8 if camera else '<u2').reshape(2, 3) + kinds.index(k) * 10
        arrays[k] = array
        uid = f't001_0_1_{ids[k]}_0'.encode()
        blocks.extend([(3, b'\0' * 8 + struct.pack('<I', len(uid)) + uid), (4, array.tobytes())])
    data = bytearray(96)
    data[:16] = b'OLYMPUSRAWFORMAT'
    offsets = []
    for kind, content in blocks:
        offsets.append(len(data))
        data.extend(struct.pack('<II', len(content), kind) + content)
    index = len(data)
    data.extend(b'\xff' * 4 + b''.join(struct.pack('<Q', off) for off in offsets))
    struct.pack_into('<QQ', data, 32, len(data), index)
    return bytes(data), arrays


def poir_file(path, *, user=1, factory_y=1.2):
    camera, rgb = oir_bytes(True, user, factory_y)
    laser, gray = oir_bytes(False, user, factory_y)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('Camera.oir', camera)
        archive.writestr('LSM.oir', laser)
    return np.stack([rgb[k] for k in ('red', 'green', 'blue')], axis=-1), gray


def wait_until(app, condition, timeout=12):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(2)
    assert condition(), "Qt worker did not finish"


def test_poir_guid_order_factory_xy_and_selective_read(tmp_path, monkeypatch):
    path = tmp_path / 'sample.poir'
    rgb, gray = poir_file(path)
    channels = inspect_source(path)
    assert {c.kind for c in channels} == {'color', 'intensity', 'height'}
    for c in channels:
        assert c.calibration.pixel_size_x == pytest.approx(.1375)
        assert c.calibration.pixel_size_y == pytest.approx(.3)
        expected = rgb if c.kind == 'color' else gray[c.kind]
        np.testing.assert_array_equal(raster_plane_to_numpy(read_channel(c)), expected)
    from fdm.services._olympus_oir import OirReader
    monkeypatch.setattr(OirReader, 'plane', lambda *a: pytest.fail('metadata-only decoded pixels'))
    assert len(inspect_source(path)) == 3
    assert [c.kind for c in default_selection(channels)] == ['intensity']


@pytest.mark.parametrize('fields', [{'user': 1.1}, {'factory_y': None}, {'factory_y': float('nan')}])
def test_unknown_correction_and_source_changed(tmp_path, fields):
    path = tmp_path / 'sample.poir'
    poir_file(path, **fields)
    channel = inspect_source(path)[0]
    assert channel.calibration is None
    assert channel.calibration_issue
    assert read_channel(channel).width == 3
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='已改变'):
        read_channel(channel)
    with pytest.raises(DeviceReadCancelled):
        inspect_source(path, cancelled=lambda: True)


def test_dsx_native_height_and_default_color(tmp_path):
    path = tmp_path / 'a.dsx'
    rgb, height = dsx_file(path)
    channels = inspect_source(path)
    assert [c.kind for c in default_selection(channels)] == ['color']
    np.testing.assert_array_equal(raster_plane_to_numpy(read_channel(channels[0])), rgb)
    np.testing.assert_array_equal(raster_plane_to_numpy(read_channel(channels[1])), height)
    assert channels[1].semantic is RasterSemantic.HEIGHT
    assert channels[1].calibration_evidence['pixel_size_um']['z'] == .01


def test_channel_preference_never_remembers_height():
    from fdm.settings import AppSettings
    settings = AppSettings(device_import_channels=['color', 'height'])
    assert settings.to_dict()['device_import_channels'] == ['color']
    assert AppSettings.from_dict({'device_import_channels': ['height', 'color', {}]}).device_import_channels == ['color']
    assert AppSettings.from_dict({'device_import_channels': ['height']}).device_import_channels == ['intensity']


def test_anisotropic_measurement_sidecar_preset_and_dependency():
    cal = Calibration('device', 2, 'µm', 'test', pixels_per_unit_y=.5)
    for end, expected in [(Point(4, 0), 2), (Point(0, 4), 8), (Point(4, 4), math.sqrt(68))]:
        measurement = Measurement(id='m', image_id='i', fiber_group_id=None, mode='manual', line_px=Line(Point(0, 0), end))
        measurement.recalculate(cal)
        assert measurement.diameter_unit == pytest.approx(expected)
    polyline = Measurement(id='p', image_id='i', fiber_group_id=None, mode='manual', measurement_kind='polyline', polyline_px=[Point(0, 0), Point(4, 0), Point(4, 4)])
    polyline.recalculate(cal)
    assert polyline.diameter_unit == 10
    area = Measurement(id='a', image_id='i', fiber_group_id=None, mode='manual', measurement_kind='area', exact_area_px=15)
    area.recalculate(cal)
    assert area.area_unit == 15
    assert 'pixels_per_unit' not in cal.to_dict()
    assert Calibration.from_dict(cal.to_dict()) == cal
    preset = CalibrationPreset('test', 2, 'µm', pixels_per_unit_y=.5)
    assert CalibrationPreset.from_dict(preset.to_dict()).to_calibration().pixel_size_y == 2
    doc = ImageDocument('i', 'a.png', (3, 2), calibration=cal)
    sig = MainWindow._document_calibration_signature(doc)
    doc.initialize_runtime_state()
    doc.mark_calibration_saved()
    cal.pixels_per_unit_y = 1
    doc.ensure_external_calibration_change_is_dirty()
    assert doc.dirty_flags.calibration_dirty
    assert MainWindow._document_calibration_signature(doc) != sig


def test_lut_preserves_full_precision_roundtrip():
    from fdm.services.raster_io import numpy_to_raster_plane
    table = np.zeros((65536, 3), dtype=np.uint8)
    table[256] = [12, 34, 56]
    table[257] = [87, 65, 43]
    transform = DisplayTransform(black_point=0, white_point=65535, lut_rgb=table.tobytes())
    restored = DisplayTransform.from_dict(transform.to_dict())
    plane = numpy_to_raster_plane(np.array([[256, 257]], dtype=np.uint16))
    image = raster_plane_to_qimage(plane, display_transform=restored)
    assert image.pixelColor(0, 0).getRgb()[:3] == (12, 34, 56)
    assert image.pixelColor(1, 0).getRgb()[:3] == (87, 65, 43)


def test_xy_hole_area_export_statistics_and_horizontal_ruler(tmp_path):
    import csv
    from fdm.models import ProjectState
    from fdm.services.export_service import ExportSelection, ExportService
    from fdm.services.measurement_statistics import MeasurementMetric, MeasurementStatisticsService
    from fdm.settings import AppSettings
    from fdm.ui.rendering import resolve_scale_overlay_value

    document = ImageDocument('xy', 'xy.tif', (20, 20), calibration=Calibration('device', 2, 'µm', 'source', .25))
    document.initialize_runtime_state()
    document.add_measurement(Measurement('vertical', document.id, None, 'manual',
                                        line_px=Line(Point(0, 0), Point(0, 4))))
    rings = [[Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)],
             [Point(2, 2), Point(8, 2), Point(8, 8), Point(2, 8)]]
    document.add_measurement(Measurement('hole', document.id, None, 'manual',
                                        measurement_kind='area', area_rings_px=rings))
    assert [m.display_value() for m in document.measurements] == [16, 128]
    assert [m.display_label(document.calibration) for m in document.measurements] == ['16.0000 µm', '128.0000 µm²']
    assert resolve_scale_overlay_value(document, AppSettings(scale_overlay_length_value=10),
                                       image_to_output_scale=.5) == (10, 'µm', 10)
    outputs = ExportService().export_project(ProjectState(version='test', documents=[document]), tmp_path,
                                            selection=ExportSelection(include_csv=True, include_excel=False, include_scale_json=True))
    with outputs['measurement_details_csv'].open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row['结果']) for row in rows] == [16, 128]
    scale = json.loads(outputs['scale_jsons'][0].read_text())
    assert scale['calibration']['axis_pixels_per_unit'] == {'x': 2, 'y': .25}
    statistics = MeasurementStatisticsService()
    assert statistics.summarize_documents([document], metric=MeasurementMetric.LENGTH)[0].mean == 16
    assert statistics.summarize_documents([document], metric=MeasurementMetric.AREA)[0].mean == 128


def test_rotation_and_resize_calibration():
    from fdm.services.raster_io import numpy_to_raster_plane
    plane = numpy_to_raster_plane(np.zeros((8, 4), dtype=np.uint8))
    doc = ImageDocument('i', 'x', (4, 8), calibration=Calibration('device', 2, 'µm', 'test', .5))
    recipe = ImageProcessingRecipe(operations=(ImageOperationSpec('rotate_90_clockwise', {}), ImageOperationSpec('resize', {'width': 4, 'height': 8})))
    cal = MainWindow._calibration_for_derived_recipe(doc, plane, recipe, clear_non_uniform=False)
    assert cal.pixel_size_x == 4
    assert cal.pixel_size_y == .25
    recipe = ImageProcessingRecipe(operations=(ImageOperationSpec('rotate', {'angle_degrees': 45}),))
    assert MainWindow._calibration_for_derived_recipe(doc, plane, recipe, clear_non_uniform=False) is ...
    assert MainWindow._calibration_for_derived_recipe(doc, plane, recipe, clear_non_uniform=True) is None


def test_xy_sidecar_and_unknown_correction_survive_reopen(tmp_path, desktop_application, monkeypatch):
    from fdm.services.sidecar_io import CalibrationSidecarIO
    path = tmp_path / 'image.tif'
    doc = ImageDocument('i', str(path), (3, 2), calibration=Calibration('device', 2, 'µm', 'source', .5))
    doc.initialize_runtime_state()
    assert CalibrationSidecarIO.save_document(doc).success
    payload = json.loads(Path(str(path) + '.fdm.json').read_text())
    assert payload['version'] == '2'
    assert 'pixels_per_unit' not in payload['calibration']
    restored = ImageDocument('new', str(path), (3, 2))
    assert CalibrationSidecarIO.load_document(restored)
    assert restored.calibration.pixel_size_y == 2

    dsx_path = tmp_path / 'unverified.dsx'
    dsx_file(dsx_path, bad_calibration=True)
    monkeypatch.setattr('fdm.ui.main_window.AppSettingsIO.save', lambda *a, **k: None)
    window = MainWindow()
    window._session_processed_root = tmp_path / 'session'
    window.project.project_default_calibration = Calibration('project_default', 10, 'µm', 'default')
    worker = DeviceReadWorker('import', default_selection(inspect_source(dsx_path)), asset_root=window._session_processed_root)
    worker.itemReady.connect(window._mount_device_channel)
    worker.run()
    try:
        assert window.current_document().calibration is None
        output = tmp_path / 'unverified.fdmproj'
        assert window.project_session_controller.save_project(str(output)).success
        window._reset_workspace()
        assert window.project_session_controller.load_project_from_path(output).success
        wait_until(desktop_application, lambda: not window.is_image_loading())
        assert window.current_document().calibration is None
        assert window.current_document().calibration_load_error
    finally:
        window._reset_workspace()
        window.close()


def test_dialog_height_exclusion_calibration_confirmation_and_import(tmp_path, desktop_application):
    path = tmp_path / 'a.poir'
    poir_file(path)
    dialog = DeviceImportDialog([str(path)], asset_root=tmp_path / 'assets')
    dialog.show()
    wait_until(desktop_application, lambda: dialog._thread is None and bool(dialog._channels))
    assert [c.kind for c in dialog.selected_channels()] == ['intensity']
    dialog._set_common(True)
    assert {c.kind for c in dialog.selected_channels()} == {'intensity', 'color'}
    assert not dialog.advancedToggle.isChecked()
    results = []
    dialog.channelLoaded.connect(results.append)
    dialog._commit()
    wait_until(desktop_application, lambda: dialog._thread is None)
    assert len(results) == 2
    assert all(r.session_path.exists() for r in results)
    assert all(r.channel.kind != 'height' for r in results)
    dialog.close()
    only = DeviceImportDialog([str(path)], asset_root=tmp_path / 'unused', calibration_only=True, calibration_target_size=(3, 2))
    only.show()
    wait_until(desktop_application, lambda: only._thread is None and bool(only._channels))
    assert not only.importButton.isEnabled()
    only.confirmField.setChecked(True)
    assert only.importButton.isEnabled()
    applied = []
    only.calibrationSelected.connect(applied.append)
    only._commit()
    assert len(applied) == 1
    assert not (tmp_path / 'unused').exists()


def test_import_save_reload_without_original(tmp_path, desktop_application, monkeypatch):
    path = tmp_path / 'a.dsx'
    dsx_file(path)
    from fdm.ui.main_window import AppSettingsIO
    monkeypatch.setattr(AppSettingsIO, 'save', lambda *a, **k: None)
    window = MainWindow()
    window._session_processed_root = tmp_path / 'session'
    worker = DeviceReadWorker('import', inspect_source(path), asset_root=window._session_processed_root)
    worker.itemReady.connect(window._mount_device_channel)
    worker.run()
    assert len(window.project.documents) == 2
    assert window.project.documents[1].raster_semantic is RasterSemantic.HEIGHT
    available, unavailable = window._analysis_batch_source_options()
    assert len(available) == 1 and len(unavailable) == 1
    original = {d.id: window._rasters[d.id].sha256() for d in window.project.documents}
    output = tmp_path / 'project.fdmproj'
    try:
        assert window.project_session_controller.save_project(str(output)).success
        payload = json.loads(output.read_text())
        assert payload['project_schema_version'] == 3 and payload['min_reader_version'] == 3
        path.unlink()
        window._reset_workspace()
        assert window.project_session_controller.load_project_from_path(output).success
        wait_until(desktop_application, lambda: not window.is_image_loading())
        assert len(window.project.documents) == 2
        for doc in window.project.documents:
            assert window._rasters[doc.id].sha256() == original[doc.id]
            assert doc.calibration.pixel_size_x == .5
            assert doc.calibration.pixel_size_y == 2
            assert ' · ' in window._document_display_name(doc)
    finally:
        window._reset_workspace()
        window.close()


def test_supplied_samples_against_research_pixels():
    root = Path(__file__).resolve().parents[1]
    source = root / '.tmp/DSX1000、OLS5000样张'
    evidence_file = root / '.tmp/olympus-format-analysis-2026-09-21/extracted/analysis.json'
    if not source.exists() or not evidence_file.exists():
        pytest.skip('local microscope sample corpus unavailable')
    data = json.loads(evidence_file.read_text())
    expected = set()
    def visit(value):
        if isinstance(value, dict):
            if 'raw_sha256' in value:
                expected.add(value['raw_sha256'])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(data)
    channels = []
    for path in sorted(source.rglob('*')):
        if path.suffix.lower() in {'.dsx', '.poir', '.mpoir'}:
            channels.extend(inspect_source(path))
    assert len(default_selection(channels)) == 9
    assert len(default_selection(channels, ('intensity', 'color'))) == 16
    assert len(channels) == 25
    for channel in channels:
        plane = read_channel(channel)
        assert hashlib.sha256(plane.data).hexdigest() in expected
        assert channel.calibration is not None
        expected_xy = ((5.29527841030118, 5.29527841030118) if channel.device == 'DSX1000'
                       else (1.25087160258375, 1.250549112703571) if '反面' in channel.source_path
                       else (.1261242173690775, .12629383971705876))
        assert (channel.calibration.pixel_size_x, channel.calibration.pixel_size_y) == pytest.approx(expected_xy, rel=1e-13)


def test_mpoir_uses_layout_and_never_stitches_points(tmp_path):
    source = tmp_path / 'point.poir'
    poir_file(source)
    path = tmp_path / 'points.mpoir'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('matl.omp2info', '''<properties><group objectId="one"><stitching>false</stitching><area id="a"><image>first.poir</image></area></group><group objectId="two"><stitching>false</stitching><area id="b"><image>second.poir</image></area></group></properties>''')
        archive.writestr('second.poir', source.read_bytes())
        archive.writestr('first.poir', source.read_bytes())
    channels = inspect_source(path)
    assert len(channels) == 6
    selected = default_selection(channels)
    assert [c.dataset_label for c in selected] == ['点位 1', '点位 2']
    assert selected[0].layout['group_id'] == 'one'
    assert selected[0].members[0] == 'first.poir'
    assert all(read_channel(c).width == 3 for c in selected)


def test_cancel_keeps_completed_channel_and_height_export_is_separate(tmp_path):
    path = tmp_path / 'a.dsx'
    _, expected_height = dsx_file(path)
    channels = inspect_source(path)
    worker = DeviceReadWorker('import', channels, asset_root=tmp_path / 'assets')
    completed, finished = [], []
    def on_item(result):
        completed.append(result)
        worker.cancel()
    worker.itemReady.connect(on_item)
    worker.finished.connect(lambda cancelled, errors: finished.append((cancelled, errors)))
    worker.run()
    assert len(completed) == 1 and completed[0].channel.kind == 'color'
    assert finished == [(True, [])]
    assert len(list((tmp_path / 'assets').rglob('*.png'))) == 1
    preview = DeviceReadWorker('preview', [channels[1]], asset_root=tmp_path / 'preview-assets')
    previews = []
    preview.itemReady.connect(previews.append)
    preview.run()
    assert len(previews) == 1 and not previews[0][1].isNull()
    assert not (tmp_path / 'preview-assets').exists()
    destination = tmp_path / 'raw-height.tif'
    exporter = DeviceReadWorker('export', [channels[1]], export_path=destination)
    exports = []
    exporter.itemReady.connect(exports.append)
    exporter.run()
    assert exports == [destination]
    np.testing.assert_array_equal(tifffile.imread(destination), expected_height)
    meta = json.loads(destination.with_suffix('.json').read_text())
    assert meta['calibration_evidence']['pixel_size_um']['z'] == .01
    assert len(completed) == 1


def test_dialog_scan_cancel_and_duplicate_sources(tmp_path, desktop_application, monkeypatch):
    path = tmp_path / 'a.dsx'
    dsx_file(path)
    copied = tmp_path / 'copy.dsx'
    copied.write_bytes(path.read_bytes())
    dialog = DeviceImportDialog([str(path), str(copied)], asset_root=tmp_path / 'assets')
    dialog.show()
    wait_until(desktop_application, lambda: dialog._thread is None and bool(dialog._channels))
    assert len(dialog._channels) == 2
    dialog.advancedToggle.setChecked(True)
    height = dialog.heightTree.topLevelItem(0)
    height.setCheckState(0, Qt.CheckState.Checked)
    assert len(dialog.selected_channels()) == 2
    dialog.advancedToggle.setChecked(False)
    assert len(dialog.selected_channels()) == 1
    dialog.reject()
    assert not (tmp_path / 'assets').exists()

    def wait_for_cancel(path, *, cancelled):
        while not cancelled():
            time.sleep(.002)
        raise DeviceReadCancelled()
    monkeypatch.setattr('fdm.ui.device_import_dialog.inspect_source', wait_for_cancel)
    dialog = DeviceImportDialog([str(path)], asset_root=tmp_path / 'assets')
    dialog.show()
    wait_until(desktop_application, lambda: dialog._thread is not None)
    dialog.reject()
    wait_until(desktop_application, lambda: dialog._thread is None)
    assert not dialog.isVisible()
    assert not (tmp_path / 'assets').exists()


def test_calibration_only_apply_undo_and_custom_lut_save(tmp_path, desktop_application, monkeypatch):
    path = tmp_path / 'a.poir'
    poir_file(path)
    channel = next(c for c in inspect_source(path) if c.kind == 'intensity')
    table = np.arange(768, dtype=np.uint8).tobytes()
    channel = replace(channel, display_transform=DisplayTransform(black_point=0, white_point=65535, lut_rgb=table))
    monkeypatch.setattr('fdm.ui.main_window.AppSettingsIO.save', lambda *a, **k: None)
    window = MainWindow()
    window._session_processed_root = tmp_path / 'session'
    worker = DeviceReadWorker('import', [channel], asset_root=window._session_processed_root)
    worker.itemReady.connect(window._mount_device_channel)
    worker.run()
    document = window.current_document()
    original = document.calibration.to_dict()
    try:
        changed = replace(channel, calibration=Calibration('device', 3, 'µm', 'other', 7))
        window._apply_device_calibration(document.id, changed)
        assert document.calibration.y_pixels_per_unit == 7
        window.undo_current_document()
        assert document.calibration.to_dict() == original
        output = tmp_path / 'x.fdmproj'
        assert window.project_session_controller.save_project(str(output)).success
        window._reset_workspace()
        path.unlink()
        assert window.project_session_controller.load_project_from_path(output).success
        wait_until(desktop_application, lambda: not window.is_image_loading())
        assert window.current_document().display_transform.lut_rgb == table
    finally:
        window._reset_workspace()
        window.close()


def test_partial_failure_retry_focuses_existing_channel(tmp_path, desktop_application, monkeypatch):
    path = tmp_path / 'a.poir'
    poir_file(path)
    dialog = DeviceImportDialog([str(path)], asset_root=tmp_path / 'assets')
    dialog.show()
    wait_until(desktop_application, lambda: dialog._thread is None and bool(dialog._channels))
    dialog._set_common(True)
    decoded, completed, focused = [], [], []

    def decode(channel, **kwargs):
        decoded.append(channel.kind)
        if channel.kind == 'color' and decoded.count('color') == 1:
            raise ValueError('test decode failure')
        return read_channel(channel, **kwargs)

    def register(result):
        completed.append(result)
        dialog.register_imported_document(result.channel.identity, 'document-' + result.channel.kind)

    monkeypatch.setattr('fdm.ui.device_import_dialog.read_channel', decode)
    dialog.channelLoaded.connect(register)
    dialog.focusDocument.connect(focused.append)
    dialog._commit()
    wait_until(desktop_application, lambda: dialog._thread is None)
    assert [item.channel.kind for item in completed] == ['intensity']
    assert 'test decode failure' in dialog.status.text()
    assert dialog.importButton.text() == '加入项目（1 张）'
    dialog._commit()
    wait_until(desktop_application, lambda: dialog._thread is None)
    assert focused == ['document-intensity']
    assert decoded.count('intensity') == 1 and decoded.count('color') == 2
    assert {item.channel.kind for item in completed} == {'color', 'intensity'}
    assert len(list((tmp_path / 'assets/imports').iterdir())) == 2
    assert dialog.imported_common == {'color', 'intensity'}


def test_relocated_source_identity_and_no_duplicate_decode(tmp_path, desktop_application, monkeypatch):
    path = tmp_path / 'original.dsx'
    dsx_file(path)
    original = inspect_source(path)[0]
    relocated = tmp_path / 'relocated.dsx'
    path.rename(relocated)
    dialog = DeviceImportDialog([str(relocated)], asset_root=tmp_path / 'assets',
                                expected_fingerprint=original.source_sha256,
                                existing={original.identity: 'existing-color'})
    dialog.show()
    wait_until(desktop_application, lambda: dialog._thread is None and bool(dialog._channels))
    assert dialog._channels[0].identity == original.identity
    assert any(dialog.tree.topLevelItem(0).child(i).text(3) == '此点位无此通道'
               for i in range(dialog.tree.topLevelItem(0).childCount()))
    monkeypatch.setattr('fdm.ui.device_import_dialog.read_channel', lambda *a, **k: pytest.fail('duplicate decode'))
    focused = []
    dialog.focusDocument.connect(focused.append)
    dialog._commit()
    assert focused == ['existing-color'] and not (tmp_path / 'assets').exists()
    rejected = DeviceImportDialog([str(relocated)], asset_root=tmp_path / 'assets',
                                  expected_fingerprint='different-fingerprint')
    rejected.show()
    wait_until(desktop_application, lambda: rejected._thread is None and bool(rejected._errors))
    assert not rejected._channels and not rejected.importButton.isEnabled()
    assert '指纹不符' in rejected.status.text()
    rejected.reject()


def test_cancel_device_scan_stops_mixed_file_import(tmp_path, desktop_application, monkeypatch):
    monkeypatch.setattr('fdm.ui.main_window.AppSettingsIO.save', lambda *a, **k: None)
    window = MainWindow()
    scanned = []

    def cancel(paths, **kwargs):
        scanned.extend(paths)
        return False

    monkeypatch.setattr(window, '_open_device_sources', cancel)
    monkeypatch.setattr(window, '_prepare_image_load_requests', lambda *a: pytest.fail('cancel must stop ordinary imports too'))
    try:
        path = str(tmp_path / 'a.poir')
        window._open_image_requests([(path, None), (str(tmp_path / 'b.png'), None)], context_label='test')
        assert scanned == [path] and not window.project.documents
    finally:
        window._reset_workspace()
        window.close()
