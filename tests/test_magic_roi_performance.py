from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services import prompt_segmentation as segmentation
from fdm.services.segmentation_performance import log_slow_segmentation
from fdm.services.segmentation_source import ImageSourceVersionCache, _qimage_content_version
from fdm.settings import MagicSegmentToolMode
from fdm.ui.canvas import DocumentCanvas, MagicSegmentOperationMode
from fdm.ui.main_window import MainWindow
from fdm.ui.prompt_segmentation_worker import PromptSegmentationRequest, PromptSegmentationWorker


class CountingService(segmentation.PromptSegmentationService):
    def __init__(self, *, fill=False, **kwargs):
        super().__init__(**kwargs)
        self.local_masks = True
        self.fill = fill
        self.encoded = []
        self.prompts = []

    def _run_encoder(self, image):
        self.encoded.append(image.shape[:2])
        if self._performance:
            self._performance.encoder_calls += 1
        return np.zeros(8, dtype=np.float32), image.shape[:2]

    def _predict_mask_from_embedding(self, embedding, *, positive_points, negative_points):
        self.prompts.append((list(positive_points), list(negative_points)))
        if self._performance:
            self._performance.decoder_calls += 1
        mask = np.zeros(embedding.original_size, dtype=np.uint8)
        if self.fill:
            mask[:] = 1
        elif positive_points:
            p = positive_points[-1]
            cv2.circle(mask, (round(p.x), round(p.y)), 8, 1, -1)
        return mask.astype(bool)


def image(w=2048, h=1536):
    result = QImage(w, h, QImage.Format.Format_RGB32)
    result.fill(0xFFAABBCC)
    return result


def predict(service, img, points=None, **kwargs):
    return service.predict_polygon(
        image=img, cache_key=kwargs.pop('cache_key', 'source'),
        positive_points=points or [Point(1024, 768)], negative_points=kwargs.pop('negative_points', []),
        tool_mode=MagicSegmentToolMode.STANDARD, roi_enabled=True, **kwargs,
    )


def test_roi_crops_pixels_only_on_cache_miss():
    service = CountingService()
    img = image()
    with patch.object(service, '_image_to_rgb_array', wraps=service._image_to_rgb_array) as convert:
        first = predict(service, img)
        second = predict(service, img)
    assert convert.call_count == 1
    converted = convert.call_args.args[0]
    assert (converted.width(), converted.height()) == (256, 256)
    assert first.metadata['segmentation_performance']['encoder_calls'] == 1
    assert second.metadata['segmentation_performance']['encoder_calls'] == 0
    assert second.metadata['segmentation_performance']['cache_hits'] == 1
    assert np.array_equal(first.mask.data, second.mask.data)


def test_workspace_refinement_preserves_crop_and_passes_both_prompt_types():
    service = CountingService()
    img = image()
    first = predict(service, img)
    box = first.metadata['segmentation_crop_box']
    second = predict(service, img, [Point(1024, 768), Point(1032, 768)],
                     negative_points=[Point(1060, 800)], roi_workspace_box=box)
    assert second.metadata['segmentation_crop_box'] == box
    assert len(service.encoded) == 1
    assert len(service.prompts[-1][0]) == 2 and len(service.prompts[-1][1]) == 1
    assert service.prompts[-1][0][-1] == Point(136, 128)


def test_expansion_deduplicates_full_region_and_repeated_request_uses_workspace():
    service = CountingService(fill=True)
    img = image()
    first = predict(service, img)
    assert service.encoded == [(256, 256), (461, 461), (829, 829), (1536, 2048)]
    assert first.metadata['segmentation_fallback_from_roi'] is True
    second = predict(service, img, roi_workspace_box=first.metadata['segmentation_crop_box'])
    perf = second.metadata['segmentation_performance']
    assert perf['encoder_calls'] == 0 and perf['decoder_calls'] == 1
    assert perf['roi_crops'] == [(0, 0, 2048, 1536)]
    assert np.array_equal(first.mask.data, second.mask.data)


@pytest.mark.parametrize('constraint', [(100, 100, 340, 340), (100, 100, 800, 220)])
def test_constraint_expansion_never_repeats_a_crop_or_exceeds_bounds(constraint):
    service = CountingService(fill=True)
    result = predict(service, image(), [Point(200, 180)], roi_constraint_box=constraint)
    crops = result.metadata['segmentation_performance']['roi_crops']
    assert len(crops) == len(set(crops)) <= 4
    assert crops.count(constraint) == 1
    assert all(constraint[0] <= b[0] < b[2] <= constraint[2] and constraint[1] <= b[1] < b[3] <= constraint[3] for b in crops)


def test_empty_full_result_is_not_retried():
    service = CountingService()
    service._predict_mask_from_embedding = lambda entry, **_: np.zeros(entry.original_size, bool)
    result = predict(service, image())
    assert result.mask is None and result.metadata['reason'] == 'roi_unstable'
    assert len(service.encoded) == 4
    assert service.encoded.count((1536, 2048)) == 1


def test_workspace_expands_to_include_new_positive_and_negative_points():
    service = CountingService()
    img = image()
    first = predict(service, img)
    second = predict(service, img, [Point(1024, 768), Point(1300, 768)],
                     negative_points=[Point(1310, 770)], roi_workspace_box=first.metadata['segmentation_crop_box'])
    box = second.metadata['segmentation_crop_box']
    assert box != first.metadata['segmentation_crop_box']
    assert all(segmentation._point_in_crop(p, box) for p in [Point(1024, 768), Point(1300, 768), Point(1310, 770)])
    assert len(service.prompts[-1][0]) == 2 and len(service.prompts[-1][1]) == 1


def test_internal_single_edge_keeps_result_but_is_not_pinned_for_future_prompts():
    service = CountingService()
    def one_edge(entry, **kwargs):
        mask = np.zeros(entry.original_size, bool)
        mask[120:, 120:140] = True
        return mask
    service._predict_mask_from_embedding = one_edge
    result = predict(service, image())
    assert len(service.encoded) == 1  # original one-edge acceptance rule is unchanged
    assert result.mask is not None
    assert result.metadata['segmentation_workspace_reusable'] is False


def test_source_and_constraint_edges_do_not_disable_workspace_reuse():
    mask = np.zeros((256, 256), bool)
    mask[100:140, :150] = True
    assert segmentation._roi_workspace_is_reusable(mask, crop_box=(0, 0, 256, 256), bounds=(0, 0, 2048, 1536))
    assert segmentation._roi_workspace_is_reusable(mask, crop_box=(0, 0, 256, 256), bounds=(0, 0, 256, 256))


def test_byte_budget_evicts_oldest_roi_and_keeps_non_roi_limit():
    service = CountingService(max_cache_bytes=5 * 32, max_cache_entries=2)
    pixels = np.zeros((32, 32, 3), np.uint8)
    for n in range(6):
        service._embedding_for_rgb_array(pixels, cache_key=f'source|roi={n}')
    assert len(service._embedding_cache) == 5
    assert 'source|roi=0' not in service._embedding_cache
    service._embedding_for_rgb_array(pixels, cache_key='source|roi=1')
    service._embedding_for_rgb_array(pixels, cache_key='source|roi=6')
    assert 'source|roi=1' in service._embedding_cache and 'source|roi=2' not in service._embedding_cache
    for n in range(3):
        service._embedding_for_rgb_array(pixels, cache_key=f'full-{n}')
    assert len([k for k in service._embedding_cache if '|roi=' not in k]) == 2
    assert service.embedding_cache_bytes <= 160
    service.clear_cache()
    assert service.embedding_cache_bytes == 0


def test_changed_source_never_reuses_embedding():
    service = CountingService()
    img = image()
    first = predict(service, img)
    result = predict(service, img, cache_key='changed-source', roi_workspace_box=first.metadata['segmentation_crop_box'])
    assert result.metadata['segmentation_performance']['encoder_calls'] == 1


def test_crop_first_rgb_pixels_match_whole_image_conversion():
    rng = np.random.default_rng(82)
    rgba = rng.integers(0, 256, (107, 133, 4), dtype=np.uint8)
    img = QImage(rgba.data, 133, 107, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
    service = CountingService()
    box = (7, 9, 82, 94)
    actual = service._crop_source(img, box).load()
    expected = segmentation.qimage_to_rgb_array(img)[9:94, 7:82]
    np.testing.assert_array_equal(actual, expected)


def test_version_cache_reuses_digest_and_invalidates_on_pixel_change_and_close():
    cache = ImageSourceVersionCache()
    img = image(64, 64)
    with patch('fdm.services.segmentation_source._qimage_content_version', wraps=_qimage_content_version) as digest:
        first = cache.version('doc', img)
        assert cache.version('doc', QImage(img)) == first
        assert digest.call_count == 1
        img.setPixel(2, 3, 0xFFFFFFFF)
        assert cache.version('doc', img) != first
        assert digest.call_count == 2
        cache.discard('doc')
        cache.version('doc', img)
        assert digest.call_count == 3


def test_main_snapshot_refreshes_changed_image_even_during_session():
    img = image(64, 64)
    doc = ImageDocument(id='doc', path='/tmp/roi.png', image_size=(64, 64))
    host = SimpleNamespace(_segmentation_source_sessions={}, _segmentation_image_versions=ImageSourceVersionCache(), _images={'doc': img})
    first = MainWindow._segmentation_source_for_request(host, doc, None, MagicSegmentToolMode.STANDARD)
    img.setPixel(2, 3, 0xFFFFFFFF)
    second = MainWindow._segmentation_source_for_request(host, doc, None, MagicSegmentToolMode.STANDARD)
    assert first.cache_key != second.cache_key


@pytest.fixture
def canvas():
    app = QApplication.instance() or QApplication([])
    c = DocumentCanvas()
    c.set_document(ImageDocument(id='doc', path='/tmp/test.png', image_size=(64, 64)), image(64, 64))
    c.set_tool_mode(MagicSegmentToolMode.STANDARD)
    yield c
    c.close()
    app.processEvents()


def test_canvas_keeps_workspaces_per_stage_and_checks_context(canvas):
    context = ('source', 'edge_sam_3x', 'add', None)
    box = (2, 2, 60, 60)
    canvas._magic_segment.request_id = 1
    canvas.apply_magic_segment_result(1, None, roi_workspace_box=box, roi_workspace_context=context)
    assert canvas.magic_segment_roi_workspace('add', context) == box
    for changed in [('other-source', *context[1:]), (context[0], 'edge_sam', *context[2:]), (*context[:-1], (0, 0, 20, 20))]:
        assert canvas.magic_segment_roi_workspace('add', changed) is None
    assert canvas.magic_segment_roi_workspace('subtract', context) is None
    canvas._magic_segment.pending_stage = MagicSegmentOperationMode.SUBTRACT
    canvas.apply_magic_segment_result(1, None, roi_workspace_box=(3, 3, 20, 20), roi_workspace_context=context)
    canvas._clear_current_magic_subtract_draft()
    assert canvas.magic_segment_roi_workspace('add', context) == box
    assert canvas.magic_segment_roi_workspace('subtract', context) is None
    assert 'roi_workspace_context' not in canvas._magic_segment.primary_debug_payload


def test_new_canvas_session_rejects_old_result_and_does_not_reuse_request_id(canvas):
    canvas._magic_segment.primary_positive_points = [Point(10, 10)]
    old = canvas._begin_magic_segment_request('add')
    canvas.clear_magic_segment_session()
    canvas._magic_segment.primary_positive_points = [Point(10, 10)]
    new = canvas._begin_magic_segment_request('add')
    assert old['session_token'] != new['session_token']
    assert new['request_id'] > old['request_id']
    assert canvas.apply_magic_segment_result(new['request_id'], None, session_token=old['session_token']) is None
    assert canvas._magic_segment.busy


def test_model_invalidation_retains_prompts_but_rejects_old_work(canvas):
    canvas._magic_segment.primary_positive_points = [Point(10, 10)]
    old = canvas._begin_magic_segment_request('add')
    canvas._magic_segment.roi_workspaces['add'] = ((0, 0, 64, 64), ('old-model',))
    canvas.invalidate_magic_segment_model()
    assert canvas._magic_segment.primary_positive_points == [Point(10, 10)]
    assert canvas._magic_segment.roi_workspaces == {}
    assert not canvas._magic_segment.busy
    assert canvas.apply_magic_segment_result(old['request_id'], None, session_token=old['session_token']) is None
    assert canvas._begin_magic_segment_request('add')['request_id'] > old['request_id']


def test_stale_result_does_not_consume_new_request_source(canvas):
    source = object()
    host = SimpleNamespace(_canvases={'doc': canvas}, _prompt_request_sources={('doc', 1): source},
                           _prompt_request_tool_modes={('doc', 1): MagicSegmentToolMode.STANDARD})
    result = segmentation.PromptSegmentationResult(None, [], [], 0, {'session_token': 'expired'})
    MainWindow._apply_prompt_segmentation_succeeded(host, 'doc', 1, result)
    assert host._prompt_request_sources[('doc', 1)] is source


def test_main_roundtrip_reuses_roi_and_model_change_invalidates_pending_work(canvas):
    from fdm.settings import AppSettings
    from fdm.ui.image_loader import ImageLoadRequest
    with patch('fdm.ui.main_window.AppSettingsIO.load', return_value=AppSettings()):
        window = MainWindow()
    try:
        img = image()
        doc = ImageDocument(id='roi-roundtrip', path='/tmp/roi-roundtrip.png', image_size=(2048, 1536))
        doc.initialize_runtime_state()
        window._add_loaded_document(ImageLoadRequest(path=doc.path, document=doc), img)
        target = window._canvases[doc.id]
        target.set_tool_mode(MagicSegmentToolMode.STANDARD)
        window._tool_mode = MagicSegmentToolMode.STANDARD
        window._magic_standard_add_roi_enabled = True
        worker = SimpleNamespace(requested=Mock(), register_request=Mock(return_value=1),
                                 warmupRequested=Mock(), cancel_document=Mock(), clearRequested=Mock())
        window._prompt_seg_worker = worker
        with patch.object(window, '_ensure_prompt_segmentation_worker'), \
             patch('fdm.ui.main_window.resolve_interactive_segmentation_backend', return_value=('edge_sam', None)), \
             patch('fdm.ui.main_window.interactive_segmentation_models_ready', return_value=True):
            target._magic_segment.primary_positive_points = [Point(1024, 768)]
            window._on_canvas_magic_segment_requested(doc.id, target._begin_magic_segment_request('add'))
            first = worker.requested.emit.call_args.args[0]
            assert first.roi_workspace_box is None
            service = CountingService()
            result = predict(service, first.image, first.positive_points, cache_key=first.cache_key)
            result.metadata.update(source_token=first.source_token, session_token=first.session_token,
                                   roi_workspace_context=first.roi_workspace_context, geometry_final=True)
            window._on_prompt_segmentation_succeeded(doc.id, first.request_id, result)
            target._magic_segment.primary_positive_points.append(Point(1032, 768))
            window._on_canvas_magic_segment_requested(doc.id, target._begin_magic_segment_request('add'))
            second = worker.requested.emit.call_args.args[0]
            assert second.roi_workspace_box == result.metadata['segmentation_crop_box']
            old_token = target.magic_segment_session_token()
            settings = window._app_settings.normalized_copy()
            settings.magic_segment_model_variant = 'edge_sam' if settings.magic_segment_model_variant != 'edge_sam' else 'edge_sam_3x'
            window._activate_app_settings(settings)
            assert target.magic_segment_session_token() != old_token
            assert target._magic_segment.roi_workspaces == {}
            assert target._magic_segment.primary_positive_points == [Point(1024, 768), Point(1032, 768)]
            worker.cancel_document.assert_called_with(doc.id)
            worker.warmupRequested.emit.assert_called_with(settings.magic_segment_model_variant)
    finally:
        window._prompt_seg_worker = None
        window._reset_workspace()
        window.close()


def request(worker, generation=0):
    return PromptSegmentationRequest(document_id='doc', image=image(), cache_key='source', request_id=1,
        positive_points=[Point(1024, 768)], negative_points=[], tool_mode=MagicSegmentToolMode.STANDARD,
        active_stage='add', model_variant='edge_sam', roi_enabled=True, generation=generation, session_token='session')


def test_replaced_request_cancels_old_encoder_before_decoding():
    worker = PromptSegmentationWorker()
    service = CountingService()
    worker._services['edge_sam'] = service
    generation = worker.register_request('doc', 1)
    original = service._run_encoder
    def replace(image):
        result = original(image)
        worker.cancel_document('doc')
        worker.register_request('doc', 1)
        return result
    service._run_encoder = replace
    successes, failures = [], []
    worker.succeeded.connect(lambda *args: successes.append(args))
    worker.failed.connect(lambda *args: failures.append(args))
    with patch('fdm.ui.prompt_segmentation_worker.resolve_interactive_segmentation_backend', return_value=('edge_sam', None)):
        worker.infer(request(worker, generation))
    assert successes == failures == []
    assert service.prompts == []
    assert service.last_performance['stop_reason'] == 'cancelled'


def test_slow_cancelled_queue_request_is_summarized_without_initializing_model():
    from time import perf_counter
    worker = PromptSegmentationWorker()
    pending = request(worker, worker.register_request('doc', 1))
    pending.submitted_at = perf_counter() - 0.2
    worker.cancel_document('doc')
    with patch('fdm.ui.prompt_segmentation_worker.log_slow_segmentation') as log:
        worker.infer(pending)
    assert worker._services == {}
    log.assert_called_once()
    performance = log.call_args.args[0]
    assert performance['encoder_calls'] == 0
    assert performance['stages_ms']['queue_ms'] >= 200
    assert performance['stop_reason'] == 'cancelled_before_start'


def test_worker_prewarm_is_idempotent_and_never_runs_encoder(tmp_path):
    encoder, decoder = tmp_path / 'encoder.onnx', tmp_path / 'decoder.onnx'
    encoder.touch(); decoder.touch()
    service = segmentation.PromptSegmentationService(encoder_path=encoder, decoder_path=decoder)
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name='image', type='tensor(float)', shape=[1, 3, 1024, 1024])]
    worker = PromptSegmentationWorker()
    worker._services['edge_sam'] = service
    with patch('fdm.ui.prompt_segmentation_worker.resolve_interactive_segmentation_backend', return_value=('edge_sam', None)), patch('onnxruntime.InferenceSession', return_value=session) as factory:
        worker.warmup('edge_sam')
        worker.warmup('edge_sam')
    assert factory.call_count == 2  # one encoder and one decoder session
    session.run.assert_not_called()


def test_request_performance_is_removed_before_ui_application():
    result = segmentation.PromptSegmentationResult(None, [], [], 0, {'segmentation_performance': {'service_ms': 180, 'total_ms': 200}})
    host = SimpleNamespace(_apply_prompt_segmentation_succeeded=Mock())
    with patch('fdm.ui.main_window.log_slow_segmentation') as log:
        MainWindow._on_prompt_segmentation_succeeded(host, 'doc', 1, result)
    assert 'segmentation_performance' not in result.metadata
    log.assert_called_once()
    assert log.call_args.args[0]['stages_ms']['ui_apply_ms'] >= 0


def test_slow_summary_is_one_valid_json_record():
    with patch('fdm.services.segmentation_performance.append_runtime_log') as log:
        log_slow_segmentation({'service_ms': 50})
        log_slow_segmentation({'service_ms': 101, 'encoder_calls': 2}, document_id='doc', request_id=1)
    log.assert_called_once()
    assert json.loads(log.call_args.args[1])['encoder_calls'] == 2


def test_release_probe_runs_real_shipped_models_when_available():
    from fdm.services.magic_segmentation_self_check import run_magic_segmentation_self_check
    root = Path(__file__).resolve().parents[1]
    if not all((root / 'runtime/segment-anything' / v / f'{v}_encoder.onnx').exists() for v in ('edge_sam', 'edge_sam_3x')):
        pytest.skip('Shipped ONNX model assets are unavailable in this checkout')
    report = run_magic_segmentation_self_check(root)
    assert report['ok'], report
    for model in report['models'].values():
        assert model['mask_equal'] and model['encoder_calls_repeat'] == 0


@pytest.mark.parametrize('enabled,failure', [(False, False), (True, False), (True, True)])
def test_release_check_gates_magic_feature_and_reports_probe_failure(enabled, failure):
    from fdm.release_manifest import run_release_self_check
    manifest = {'ok': True, 'errors': [], 'warnings': [], 'profile': 'core',
                'features': ['magic-segmentation'] if enabled else [], 'dependency_versions': {}}
    with patch('fdm.release_manifest.verify_release_manifest', return_value=manifest), \
         patch('fdm.release_manifest._validate_pe_executable', return_value=(True, '')), \
         patch('fdm.release_manifest._probe_fiber_quick_geometry', return_value={'ok': True}), \
         patch('fdm.release_manifest._probe_overlay_renderer', return_value={'ok': True}), \
         patch.dict('os.environ', {'FDM_SELF_CHECK_EXECUTE': '1'}), \
         patch('fdm.release_manifest._probe_magic_segmentation', side_effect=RuntimeError('missing ONNX runtime') if failure else None, return_value={'ok': True}) as probe:
        report = run_release_self_check(Path('/tmp/package'))
    assert probe.call_count == int(enabled)
    if not enabled:
        assert report['functional_checks']['magic_segmentation'] == 'skipped_feature_disabled'
    else:
        assert report['ok'] is not failure
        if failure:
            assert any('magic segmentation' in e for e in report['errors'])


@pytest.mark.parametrize('state', ['missing', 'skipped', 'incomplete', 'failed', 'valid', 'no_provider', 'reencoded'])
def test_packaging_enforces_real_magic_probe(state, monkeypatch):
    import importlib.util
    import subprocess
    script = Path(__file__).resolve().parents[1] / 'scripts/build_windows_onedir.py'
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location('roi_build_windows_onedir', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = {'ok': True, 'encoder_calls_first': 1, 'encoder_calls_repeat': 0,
             'decoder_calls_repeat': 1, 'mask_equal': True, 'providers': ['CPUExecutionProvider']}
    valid = {'ok': True, 'backend': 'onnxruntime', 'backend_version': 'test',
             'models': {'edge_sam': dict(model), 'edge_sam_3x': dict(model)}}
    magic = {'missing': None, 'skipped': 'skipped_non_windows', 'incomplete': {'ok': True},
             'failed': {'ok': False}}.get(state, valid)
    if state == 'no_provider':
        magic['models']['edge_sam']['providers'] = None
    if state == 'reencoded':
        magic['models']['edge_sam']['encoder_calls_repeat'] = 1
    payload = {'ok': True, 'errors': [], 'features': ['magic-segmentation'], 'functional_checks': {
        'overlay_renderer': {'ok': True, 'worker_stdio_none': True},
        'fiber_quick_geometry': {'ok': True, 'backend': 'skimage_zhang', 'backend_version': 'test', 'geometry_revision': 3, 'compiled_extension': True},
        'magic_segmentation': magic,
    }}
    with patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr='')):
        errors = module.run_packaged_self_check(Path('/tmp/package'))
    assert errors == ([] if state == 'valid' else ['packaged self-check did not pass the magic segmentation ROI/cache probe'])
