#!/usr/bin/env python3
"""Read-only CU acceptance audit: consumes existing JSON; never imports Qt/CV2 or runs workloads.

Default: expect the eight final macOS/Windows UI reports under --report-dir.
Add --soak PATH for each real-workspace endurance report. --ui PATH may be repeated;
use --no-default-ui for parsing smoke fixtures only. Writes only explicitly named
--output-json/--output-md paths (or prints JSON); input artifacts are never changed.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'docs/validation/cu-live-measurement-2026-10-15'
RULES = REPORT_DIR / 'performance-acceptance-rules.md'
OK, FAIL, MISSING, NA = 'PASS', 'FAIL', 'INCOMPLETE', 'NOT_APPLICABLE'
TEARDOWN_HARNESS_SHA256 = 'dc45983e334186d15d9e9f8ecc1e6063246e27d38f30ea943177a880ab2a3f5e'


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check(name, value, expected, operation='le', note=None):
    if value is None:
        status = MISSING
    elif operation == 'eq':
        status = OK if value == expected else FAIL
    elif not number(value):
        status = MISSING
    else:
        status = OK if (value <= expected if operation == 'le' else value >= expected) else FAIL
    out = {'name': name, 'status': status, 'actual': value, 'operator': operation, 'expected': expected}
    if note: out['note'] = note
    return out


def verdict(checks):
    states = {item['status'] for item in checks}
    return FAIL if FAIL in states else MISSING if MISSING in states or not states else OK


def dist(raw):
    raw = raw if isinstance(raw, dict) else {}
    return {k: raw.get(k) for k in ('count', 'observations', 'p50_ms', 'p95_ms', 'p99_ms', 'max_ms', 'exact_max_ms')}


def maximum(d):
    return d.get('exact_max_ms') if number(d.get('exact_max_ms')) else d.get('max_ms')


def percentile(values, q):
    values = sorted(values)
    if not values: return None
    index = (len(values) - 1) * q
    lo = int(index); hi = min(len(values) - 1, lo + 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def raw_distribution_consistent(metric, trials, key):
    values = [trial[key] for trial in trials if number(trial.get(key))]
    if not isinstance(metric, dict): return None
    if metric.get('count') != len(values): return False
    if not values: return all(metric.get(k) is None for k in ('p50_ms', 'p95_ms', 'p99_ms', 'max_ms'))
    wanted = {'p50_ms': percentile(values, .5), 'p95_ms': percentile(values, .95),
              'p99_ms': percentile(values, .99), 'max_ms': max(values)}
    return all(number(metric.get(k)) and math.isclose(metric[k], v, rel_tol=1e-12, abs_tol=1e-8) for k,v in wanted.items())


def metric_checks(metrics, width):
    result95, result99 = (100, 150) if width == 6144 else (50, 100)
    return [check('draft_p95_ms', metrics['draft']['p95_ms'], 16.7),
            check('draft_p99_ms', metrics['draft']['p99_ms'], 33),
            check('result_visible_p95_ms', metrics['result_visible']['p95_ms'], result95),
            check('result_visible_p99_ms', metrics['result_visible']['p99_ms'], result99),
            check('gui_publish_max_ms', maximum(metrics['gui_publish']), 4)]


def platform_checks(env, qt):
    win = str(env.get('platform', '')).lower().startswith('windows')
    return [check('qt_platform_windows', qt, 'windows', 'eq')] if win else []


def environment_key(doc, case):
    env=doc.get('environment', {}); params=doc.get('parameters', {})
    values={'platform':env.get('platform'), 'machine':env.get('machine'),
        'dependencies':env.get('dependencies'), 'opencv_runtime':env.get('opencv_runtime'),
        'opencv_requested_threads':env.get('opencv_requested_threads'),
        'thread_environment':env.get('thread_environment'), 'qt':doc.get('qt_platform'),
        'sources':env.get('source_sha256'), 'native':env.get('native_build_info'),
        'window_width':params.get('window_width'),'window_height':params.get('window_height'),
        'frame_hz':params.get('frame_hz'),'shape':case.get('shape'),'display':case.get('display'),
        'input_sha256':case.get('input_sha256'),'source_pool_sha256':case.get('source_pool_sha256'),
        'unique_marker':case.get('unique_sequence_marker')}
    return hashlib.sha256(json.dumps(values,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_file(path):
    if not path.is_file(): return {'path':str(path),'status':MISSING,'reason':'Expected report does not exist yet.'}, None
    try:
        data=path.read_bytes()
        doc=json.loads(data,parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))
        if not isinstance(doc,dict): raise ValueError('Root must be object')
        return {'path':str(path),'status':'READ','sha256':hashlib.sha256(data).hexdigest()},doc
    except (OSError,ValueError,TypeError) as error:
        return {'path':str(path),'status':MISSING,'reason':str(error)},None


def _legacy_background_completion(doc, case):
    """Classify only the reviewed harness's narrowly identifiable close response.

    The returned automatic_results list is also appended by background() during
    finally's wait_for(), after automatic.close(). The JSON has no event phase;
    this is a source-constrained inference, never a blanket cancel exemption.
    """
    records = case.get('background_results', [])
    sources = doc.get('environment', {}).get('source_sha256', {})
    harness_sha = sources.get('scripts/benchmark_cu_live_ui.py')
    cancelled = [i for i, record in enumerate(records) if record.get('status') == 'cancelled']
    normal = [i for i, record in enumerate(records)
              if record.get('status') in ('ok', 'empty')
              and 'error' in record and record['error'] is None]
    last = records[-1] if records else {}
    conditions = {
        'reviewed_harness_sha256': harness_sha == TEARDOWN_HARNESS_SHA256,
        'exactly_one_cancelled_response': len(cancelled) == 1,
        'cancelled_is_last_response': bool(records) and cancelled == [len(records) - 1],
        'last_response_explicit_error_null': 'error' in last and last['error'] is None,
        'last_response_zero_candidates': last.get('candidates') == 0,
        'prior_ok_or_empty_completion': any(i < len(records) - 1 for i in normal),
    }
    inferred = all(conditions.values())
    errors = [i for i, record in enumerate(records)
              if (record.get('status') not in ('ok', 'success', 'empty') or bool(record.get('error')))
              and not (inferred and i == len(records) - 1)]
    note = ('The reviewed harness never cancels during sampling. Its returned list is mutated by '
            'background() in finally after automatic.close(); the unique final cancelled/null-error '
            'response is inferred to be teardown. Raw JSON does not record event phases or the '
            'worker source hash, so this is a source-constrained inference, not a runtime phase trace. '
            'It is retained as an incomplete background task, never as a successful completion.')
    return {'total_responses': len(records), 'normal_completed': len(normal),
        'status_counts': dict(collections.Counter(record.get('status', 'missing') for record in records)),
        'cancelled_total': len(cancelled), 'teardown_cancel_inferred': int(inferred),
        'unexplained_cancelled': len(cancelled) - int(inferred), 'background_errors': len(errors),
        'error_indices': errors, 'teardown_cancel_index': len(records) - 1 if inferred else None,
        'classification': 'SOURCE_CONSTRAINED_TEARDOWN_INFERENCE' if inferred else
                          'UNCONFIRMED_CANCELLATION' if cancelled else 'NO_CANCELLATION',
        'inference_conditions': conditions, 'reported_harness_sha256': harness_sha,
        'reviewed_harness_sha256': TEARDOWN_HARNESS_SHA256,
        'note': note if inferred else 'Every unmatched cancellation remains a background error; no exception is inferred.'}


def background_completion(doc, case):
    """Use runtime receipt phases when present; never infer over a broken trace."""
    records = case.get('background_results', [])
    warmup = case.get('background_warmup_results', [])
    if ('background_timeline' not in case and 'background_warmup_results' not in case
            and not any('phase' in record or 'received_monotonic' in record for record in records)):
        result = _legacy_background_completion(doc, case)
        result.update(teardown_cancel_explicit=0, phase_trace_valid=None, phase_trace_errors=[])
        return result

    timeline = case.get('background_timeline', {})
    if not isinstance(timeline, dict):
        timeline = {}
    trace_errors = []
    if timeline.get('phase_schema_version') != 1 or timeline.get('clock') != 'time.perf_counter':
        trace_errors.append('Missing or unsupported phase timeline schema/clock.')
    boundaries = [timeline.get(key) for key in ('warmup_started_at', 'formal_started_at', 'teardown_started_at')]
    valid_boundaries = all(number(value) for value in boundaries) and boundaries == sorted(boundaries)
    if not valid_boundaries:
        trace_errors.append('Phase boundaries must be finite, monotonic warmup/formal/teardown times.')
    tagged = [('warmup', index, record) for index, record in enumerate(warmup)]
    tagged += [('background', index, record) for index, record in enumerate(records)]
    seen_tasks = set()
    previous_time = None
    for collection, index, record in tagged:
        phase, received = record.get('phase'), record.get('received_monotonic')
        label = f'{collection}[{index}]'
        allowed = ('warmup',) if collection == 'warmup' else ('formal', 'teardown')
        if phase not in allowed:
            trace_errors.append(label + ': invalid phase for this collection.')
        if not number(received) or (previous_time is not None and received < previous_time):
            trace_errors.append(label + ': receipt timestamp is missing/nonfinite or moves backwards.')
        if number(received):
            previous_time = received
            if valid_boundaries:
                start, formal, teardown = boundaries
                inside = ((phase == 'warmup' and start <= received < formal)
                    or (phase == 'formal' and formal <= received < teardown)
                    or (phase == 'teardown' and received >= teardown))
                if not inside:
                    trace_errors.append(label + ': receipt lies outside its declared phase.')
        task = record.get('task_id')
        if not isinstance(task, str) or not task or task in seen_tasks:
            trace_errors.append(label + ': missing or repeated task identity.')
        if isinstance(task, str):
            seen_tasks.add(task)
    normal = lambda record: record.get('status') in ('ok', 'empty') and 'error' in record and record['error'] is None
    formal_normal = [i for i, record in enumerate(records) if record.get('phase') == 'formal' and normal(record)]
    cancelled = [i for i, record in enumerate(records) if record.get('status') == 'cancelled']
    warmup_cancelled = [i for i, record in enumerate(warmup) if record.get('status') == 'cancelled']
    last = records[-1] if records else {}
    close_task = timeline.get('close_requested_task_id')
    conditions = {
        'valid_phase_trace': not trace_errors,
        'exactly_one_cancelled_response': len(cancelled) + len(warmup_cancelled) == 1,
        'cancelled_is_last_response': bool(records) and cancelled == [len(records) - 1],
        'explicit_teardown_phase': last.get('phase') == 'teardown',
        'close_task_identity_matches': isinstance(close_task, str) and bool(close_task) and last.get('task_id') == close_task,
        'last_response_explicit_error_null': 'error' in last and last['error'] is None,
        'last_response_zero_candidates': last.get('candidates') == 0,
        'prior_formal_ok_or_empty_completion': any(i < len(records) - 1 for i in formal_normal),
    }
    explicit = all(conditions.values())
    errors = [i for i, record in enumerate(records) if not normal(record)
              and not (explicit and i == len(records) - 1)]
    warmup_errors = [i for i, record in enumerate(warmup) if not normal(record)]
    note = ('Phases and timestamps describe GUI receipt, not worker completion. Only the unique final '
            'cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; '
            'it remains incomplete, never a successful background completion. Formal completions exclude '
            'warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.')
    return {'total_responses': len(records) + len(warmup), 'normal_completed': len(formal_normal),
        'warmup_normal_completed': sum(normal(record) for record in warmup),
        'teardown_normal_completed': sum(record.get('phase') == 'teardown' and normal(record) for record in records),
        'status_counts': dict(collections.Counter(record.get('status', 'missing') for _, _, record in tagged)),
        'phase_counts': dict(collections.Counter(record.get('phase', 'missing') for _, _, record in tagged)),
        'cancelled_total': len(cancelled) + len(warmup_cancelled), 'teardown_cancel_inferred': 0,
        'teardown_cancel_explicit': int(explicit),
        'unexplained_cancelled': len(cancelled) + len(warmup_cancelled) - int(explicit),
        'background_errors': len(errors) + len(warmup_errors), 'error_indices': errors,
        'warmup_error_indices': warmup_errors, 'teardown_cancel_index': len(records) - 1 if explicit else None,
        'classification': 'INVALID_PHASE_TRACE' if trace_errors else
                          'EXPLICIT_TEARDOWN' if explicit else
                          'UNCONFIRMED_CANCELLATION' if cancelled or warmup_cancelled else 'NO_CANCELLATION',
        'phase_trace_valid': not trace_errors, 'phase_trace_errors': trace_errors,
        'explicit_conditions': conditions, 'inference_conditions': {}, 'background_timeline': timeline,
        'reported_harness_sha256': doc.get('environment', {}).get('source_sha256', {}).get('scripts/benchmark_cu_live_ui.py'),
        'reviewed_harness_sha256': TEARDOWN_HARNESS_SHA256, 'note': note}


def ui_cases(path, doc):
    output=[]; params=doc.get('parameters',{}); env=doc.get('environment',{}); qt=doc.get('qt_platform')
    for case in doc.get('cases',[]):
        width=case.get('width'); raw=case.get('metrics',{}); trials=case.get('trials',[])
        trials=[trial for trial in trials if not trial.get('warmup',False)]
        successful=[trial for trial in trials if not trial.get('failure') and not trial.get('rejection_reason')
                    and not trial.get('rejections') and number(trial.get('second_click_to_record_paint_ms'))
                    and trial.get('record_id') and number(trial.get('accepted_at'))]
        accepted=sum(bool(trial.get('record_id')) and number(trial.get('accepted_at')) for trial in trials)
        formal_events=sum(len(trial.get('rejections',[])) for trial in trials)
        all_events=sum(case.get('rejections',{}).values())
        mappings={'draft':'first_click_to_draft_paint_ms', 'result_visible':'second_click_to_record_paint_ms',
                  'gui_publish':'gui_publish_ms', 'accepted':'second_click_to_accept_ms',
                  'accept_to_paint':'accept_to_record_paint_ms','compute_to_accept':'compute_done_to_gui_accept_ms'}
        metrics={name:dist(raw.get(key)) for name,key in mappings.items()}
        eligibility=[check('warmup_attempts',params.get('warmup'),5,'eq'),
                     check('formal_attempts_requested',params.get('samples'),30,'eq'),
                     check('all_formal_attempts_recorded',len(trials),params.get('samples'),'eq')]
        checks=metric_checks(metrics,width)+platform_checks(env,qt)
        checks += [check('supported_width',width in (2048,4096,6144),True,'eq'),
                   check('formal_failures',len(trials)-len(successful),0,'eq'),
                   check('all_rejection_events_including_warmup',all_events,0,'eq'),
                   check('input_requested_hz',params.get('frame_hz'),30,'eq'),
                   check('distinct_painted_fps',case.get('painted_frames_per_second'),27,'ge'),
                   check('producer_errors',len(case.get('producer_errors',[])),0,'eq'),
                   check('frame_budget_peak_bytes',case.get('frame_budget_peak_bytes'),512*1024**2)]
        for name,key in mappings.items():
            checks.append(check('raw_trials_match_'+name,raw_distribution_consistent(raw.get(key),trials,key),True,'eq'))
        checks.append(check('gui_samples_cover_accepts',metrics['gui_publish']['count'],accepted,'ge'))
        if case.get('motion_input')=='lsb-noise':
            checks.append(check('unique_sequence_marker',case.get('unique_sequence_marker'),True,'eq'))
        background=case.get('background_automatic') or None
        completion = None
        if background:
            completion = background_completion(doc, case)
            checks.append(check('background_executed',completion['normal_completed'],1,'ge'))
            checks.append(check('background_errors',completion['background_errors'],0,'eq',
                note=completion['note'] if completion['teardown_cancel_inferred'] or completion['teardown_cancel_explicit'] else None))
            if completion['phase_trace_valid'] is not None:
                checks.append(check('background_phase_trace_valid',completion['phase_trace_valid'],True,'eq'))
        output.append({'file':str(path),'platform':env.get('platform'),'qt_platform':qt,'width':width,
            'png':case.get('save_images'),'input':case.get('motion_input'),'background':background,
            'background_completion': completion,
            'display':case.get('display'),
            'environment_key':environment_key(doc,case),
            'formal_eligibility':verdict(eligibility),'formal_eligibility_checks':eligibility,
            'attempts':{'planned_formal':params.get('samples'),'recorded_formal':len(trials),
                'accepted':accepted,'successful_and_painted':len(successful),'failed_or_unfinished':len(trials)-len(successful),
                'success_fraction':f'{len(successful)}/{len(trials)}','formal_rejection_events':formal_events,
                'warmup_or_unassigned_rejection_events':max(0,all_events-formal_events),
                'rejection_reasons_all':case.get('rejections',{}),'failure_reasons':case.get('failures',{})},
            'metrics':metrics,'fps':{'requested':params.get('frame_hz'),
                'generated':case.get('source_generated_fps_during_measurements'), 'painted':case.get('painted_frames_per_second')},
            'checks':checks,'observed_threshold_status':verdict(checks),
            'relative_png_auto':{'status':NA,'reason':'Applies only to PNG enabled plus background automatic.'},
            'source_sha256':env.get('source_sha256',{}),
            'notes':['Success-only latency quantiles do not erase failed attempts.',
                'Synthetic Qt paints do not measure camera or physical scan-out.']})
    return output


def compare_loaded(rows):
    for row in rows:
        if not(row['png'] is True and row['background']): continue
        matches=[candidate for candidate in rows if candidate['png'] is False and candidate['background'] is None
                 and candidate['width']==row['width'] and candidate['input']==row['input']
                 and candidate['environment_key']==row['environment_key']]
        result={'status':MISSING,'p99_relative_is_diagnostic_only':True}
        if len(matches)!=1:
            result['reason']='Need exactly one matching PNG-off/no-auto baseline with identical source/environment/input; found '+str(len(matches))
        else:
            base=matches[0]; result['baseline_file']=base['file']
            b=base['metrics']['result_visible'];current=row['metrics']['result_visible']
            if any(item['attempts']['failed_or_unfinished'] or item['attempts']['formal_rejection_events'] or item['attempts']['warmup_or_unassigned_rejection_events'] for item in (base,row)):
                result.update(status=FAIL,reason='Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.')
            elif not all(number(v) for v in (b['p95_ms'],current['p95_ms'])):
                result['reason']='Missing result-visible P95 samples.'
            else:
                delta=current['p95_ms']-b['p95_ms']; limit=max(.2*b['p95_ms'],10)
                result.update(status=OK if delta<=limit else FAIL, baseline_p95_ms=b['p95_ms'],loaded_p95_ms=current['p95_ms'],
                    p95_delta_ms=delta,p95_allowed_delta_ms=limit,
                    p95_increase_percent=(100*delta/b['p95_ms']) if b['p95_ms'] else None,
                    baseline_p99_ms=b['p99_ms'],loaded_p99_ms=current['p99_ms'],
                    p99_delta_ms=(current['p99_ms']-b['p99_ms']) if number(current['p99_ms']) and number(b['p99_ms']) else None)
            result['baseline_formal_eligibility']=base['formal_eligibility']
            if base['formal_eligibility']!=OK and result['status']==OK:
                result.update(status=MISSING,reason='Numerical comparison available, but baseline is not the fixed 5-warmup/30-attempt formal run.')
        row['relative_png_auto']=result
    for row in rows:
        extra=[] if row['relative_png_auto']['status']==NA else [{'status':row['relative_png_auto']['status']}]
        row['overall_status']=verdict(row['checks']+row['formal_eligibility_checks']+extra)


def soak_case(path,doc):
    params=doc.get('parameters',{}); env=doc.get('environment',{}); counters=doc.get('counters',{})
    width=(doc.get('shape') or [None,None])[1]; raw=doc.get('metrics',{})
    mappings={'draft':'first_click_to_draft_paint','result_visible':'second_click_to_record_paint',
              'gui_publish':'gui_publish_ms','accepted':'second_click_to_accept','scene_to_clear':'scene_input_to_clear',
              'scene_to_clear_paint':'scene_input_to_clear_paint','motion_complete_to_clear':'motion_complete_to_clear'}
    metrics={name:dist(raw.get(key)) for name,key in mappings.items()}
    checks=metric_checks(metrics,width)+platform_checks(env,env.get('qt_platform'))
    checks += [check('at_least_one_hour',doc.get('active_seconds'),3600,'ge'),
        check('at_least_10000_accepted',doc.get('history_count'),10000,'ge'),
        check('real_workspace_counters_present',bool(counters),True,'eq'),
        check('input_requested_hz',params.get('frame_hz'),30,'eq'),
        check('measurement_attempt_hz',params.get('measurement_hz'),4,'eq'),
        check('png_enabled',params.get('save_images'),True,'eq'),
        check('distinct_painted_fps',doc.get('distinct_painted_fps'),27,'ge'),
        check('scene_to_clear_paint_p95_ms',metrics['scene_to_clear_paint']['p95_ms'],250 if width==6144 else 150),
        check('frame_budget_peak_bytes',doc.get('frame_budget_peak_bytes'),512*1024**2),
        check('leases_after_close',doc.get('frame_budget_after_close'),0,'eq')]
    required=('all_records_have_valid_field','frame_budget_within_512_mib','old_epoch_accepted_zero','old_scene_accepted_zero','accepted_event_count_matches_ledger',
        'all_accepted_mapped_to_click_intent','submitted_intents_accounted_for','accepted_paint_outcomes_accounted_for',
        'undo_history_preserves_each_measurement','ledger_snapshot_roundtrip_exact','background_automatic_executed',
        'no_automatic_candidates_entered_ledger','known_width_unchanged','evidence_complete','evidence_files_exist_and_match_pixels',
        'queues_drained_before_deadline','all_frame_leases_released','producer_stopped','unexpected_errors_zero',
        'scheduled_motion_detected','unexpected_field_changes_zero','unexpected_rejections_zero')
    for key in required:checks.append(check('recorded_integrity_'+key,doc.get('checks',{}).get(key),True,'eq'))
    preflight=doc.get('motion_preflight')
    checks.append(check('two_motion_preflight_directions_unstable',
        len(preflight)==2 and all(row.get('stable') is False for row in preflight) if isinstance(preflight,list) else None,True,'eq'))
    quarters=[]
    for quarter in doc.get('latency_by_fifteen_minutes',[]):
        qr=quarter.get('metrics',{})
        qm={name:dist(qr.get(key)) for name,key in mappings.items() if name in ('draft','result_visible','gui_publish','accepted')}
        qc=metric_checks(qm,width)
        quarters.append({'start_seconds':quarter.get('start_seconds'),'end_seconds':quarter.get('end_seconds'),
                         'metrics':qm,'checks':qc,'status':verdict(qc)})
    # Exact requested duration defines buckets; a millisecond overrun does not demand a fifth quarter.
    expected_quarters=max(4,math.ceil(params.get('duration_seconds',3600)/900))
    checks.append(check('quarter_count',len(quarters),expected_quarters,'ge'))
    checks += [{'name':'quarter_'+str(i)+'_thresholds','status':q['status']} for i,q in enumerate(quarters)]
    accepted=counters.get('accepted',doc.get('history_count')); submitted=counters.get('submitted');first=counters.get('first_clicks')
    for key in ('old_epoch_accepted','old_scene_accepted','unmapped_accepted','wrong_width',
                'unexpected_field_changes','submitted_rejected_other','draft_lost_without_submission'):
        checks.append(check('raw_counter_'+key,counters.get(key),0,'eq'))
    checks.append(check('accepted_equals_history',accepted,doc.get('history_count'),'eq'))
    terminal=[accepted,counters.get('submitted_motion_cancelled'),counters.get('submitted_rejected_other')]
    checks.append(check('raw_submitted_conservation',sum(terminal) if all(number(n) for n in terminal) else None,submitted,'eq'))
    transition=[counters.get(k) for k in ('transition_inputs','transition_clear_events','transition_completed')]
    checks.append(check('raw_transition_counts_match',len(set(transition))==1 if all(number(n) for n in transition) else None,True,'eq'))
    checks.append(check('raw_unresolved_transitions',len(doc['unresolved_motion_transitions']) if isinstance(doc.get('unresolved_motion_transitions'),list) else None,0,'eq'))
    checks.append(check('raw_error_events',sum(doc['errors'].values()) if isinstance(doc.get('errors'),dict) else None,0,'eq'))
    checks.append(check('raw_saved_evidence_equals_history',doc.get('evidence_states',{}).get('saved'),doc.get('history_count'),'eq'))
    return {'file':str(path),'platform':env.get('platform'),'qt_platform':env.get('qt_platform'),'width':width,
        'overall_status':verdict(checks),'script_passed_flag':doc.get('passed'), 'active_seconds':doc.get('active_seconds'),
        'display':doc.get('display'),
        'attempts':{'scheduled_slots':counters.get('attempt_slots'),'first_clicks':first,'submitted':submitted,
            'accepted':accepted,'submitted_success_fraction':f'{accepted}/{submitted}',
            'first_click_success_fraction':f'{accepted}/{first}',
            'submitted_expected_motion_cancels':counters.get('submitted_motion_cancelled'),
            'submitted_other_rejections':counters.get('submitted_rejected_other'),
            'draft_motion_cancels':counters.get('draft_motion_cancelled'),
            'draft_lost_without_submission':counters.get('draft_lost_without_submission'),
            'expected_motion_rejections':doc.get('expected_motion_rejections'),
            'unexpected_rejections':doc.get('unexpected_rejections'),'skipped':doc.get('skipped_attempts')},
        'metrics':metrics,'fps':{'requested':params.get('frame_hz'),'generated':doc.get('generated_fps'),'painted':doc.get('distinct_painted_fps')},
        'checks':checks,'quarters':quarters,'queue_max':doc.get('queue_max'),
        'memory':{'frame_budget_peak_bytes':doc.get('frame_budget_peak_bytes'),'resources':doc.get('resources')},
        'notes':['Expected movement cancellations remain incomplete attempts, but are separate from erroneous rejections.',
                 'Recorded file/hash/pixel and ledger integrity checks are inspected; this tool does not re-hash PNG files.',
                 'Quarter percentiles may be bounded reservoirs; GUI budget uses exact_max_ms when supplied.',
                 'No PNG+auto relative comparison is inferred from the different soak workload.']}


def fmt(value):return 'missing' if value is None else f'{value:.3f}' if number(value) else str(value)

def markdown(report):
    lines=['# CU performance read-only audit','',
        f"Thresholds are fixed by {Path(report['rules_path']).name}. FAIL/INCOMPLETE cannot be replaced by fast successful samples.",'',
        '| file / size / PNG / input / auto | overall | success / formal attempts | draft P95/P99 | visible P95/P99 | GUI max | paint FPS | PNG+auto P95 relative |',
        '|---|---|---|---|---|---|---|---|']
    for row in report['ui_cases']:
        m=row['metrics'];label=f"{Path(row['file']).name} / {row['width']} / {row['png']} / {row['input']} / {row['background']}"
        lines.append('| '+' | '.join([label,row['overall_status'],row['attempts']['success_fraction'],
            fmt(m['draft']['p95_ms'])+'/'+fmt(m['draft']['p99_ms']),
            fmt(m['result_visible']['p95_ms'])+'/'+fmt(m['result_visible']['p99_ms']),fmt(maximum(m['gui_publish'])),
            fmt(row['fps']['painted']),row['relative_png_auto']['status']])+' |')
    for item in report['files']:
        if item['status']!= 'READ':lines.extend(['',f"- {item['status']}: {item['path']} — {item.get('reason')}"])
    for row in report['ui_cases']:
        bad=[item for item in row['checks']+row['formal_eligibility_checks'] if item['status']!=OK]
        if bad:lines.extend(['',f"{Path(row['file']).name} / {row['width']} / PNG {row['png']}: "+'; '.join(item['name']+'='+item['status'] for item in bad)])
        if row['relative_png_auto']['status'] not in (OK,NA):lines.extend(['','Relative: '+str(row['relative_png_auto'])])
        completion = row.get('background_completion')
        if completion and completion['cancelled_total']:
            lines.extend(['', f"{Path(row['file']).name} / {row['width']} / PNG {row['png']}: background completions={completion['normal_completed']}, cancelled={completion['cancelled_total']}, inferred teardown={completion['teardown_cancel_inferred']}, explicit teardown={completion['teardown_cancel_explicit']}, unexplained cancellations={completion['unexplained_cancelled']}. " + completion['note']])
    for soak in report['soaks']:
        lines.extend(['',f"## Soak {Path(soak['file']).name}: {soak['overall_status']}",
            '',f"Duration {fmt(soak['active_seconds'])} s; accepted/submitted {soak['attempts']['submitted_success_fraction']}; paint FPS {fmt(soak['fps']['painted'])}.",
            '',f"GUI P95/P99/max {fmt(soak['metrics']['gui_publish']['p95_ms'])}/{fmt(soak['metrics']['gui_publish']['p99_ms'])}/{fmt(maximum(soak['metrics']['gui_publish']))} ms; scene→clear paint P95 {fmt(soak['metrics']['scene_to_clear_paint']['p95_ms'])} ms."])
        lines.extend(['', '; '.join(item['name']+'='+item['status'] for item in soak['checks'] if item['status']!=OK) or 'All checked thresholds and recorded integrity fields pass.'])
    lines.extend(['','Missing source files or platform/combination coverage remain unaccepted. See JSON for every denominator, check and 15-minute segment.'])
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report-dir',type=Path,default=REPORT_DIR)
    parser.add_argument('--rules',type=Path,default=RULES)
    parser.add_argument('--ui',type=Path,action='append',default=[])
    parser.add_argument('--soak',type=Path,action='append',default=[])
    parser.add_argument('--no-default-ui',action='store_true')
    parser.add_argument('--output-json',type=Path)
    parser.add_argument('--output-md',type=Path)
    args=parser.parse_args()
    paths=[] if args.no_default_ui else [args.report_dir/f'final-{osname}-ui-{kind}.json' for osname in ('macos','windows') for kind in ('stable','dynamic','joint-stable','joint-dynamic')]
    paths=list(dict.fromkeys([*paths,*args.ui]))
    report={'schema_version':1,'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rules_path':str(args.rules),'rules_sha256':hashlib.sha256(args.rules.read_bytes()).hexdigest() if args.rules.is_file() else None,
        'files':[],'ui_cases':[],'soaks':[],
        'limits':['Only reads supplied result JSON. No test, benchmark, CV2, Qt, subprocess or network is executed.',
                  'No independent verification that benchmarks ran without concurrent host/VM workloads; review run logs.',
                  'Windows ARM64/x64 emulation and physical hardware boundaries still require human contextual review.',
                  'P99 relative increase is diagnostic only; the hard relative condition applies to P95.']}
    for path in paths:
        info,doc=load_file(path);report['files'].append(info)
        if doc:
            if not isinstance(doc.get('cases'),list):info.update(status=MISSING,reason='Not a UI benchmark schema (cases list missing).')
            else:
                report['ui_cases'].extend(ui_cases(path,doc))
                params=doc.get('parameters',{})
                sizes=params.get('sizes',[])
                saving=(False,True) if params.get('evidence')=='both' else (params.get('evidence')=='on',)
                expected={(size,png) for size in sizes for png in saving}
                actual=[(case.get('width'),case.get('save_images')) for case in doc['cases']]
                missing=sorted(expected-set(actual))
                duplicated=[key for key,count in collections.Counter(actual).items() if count>1]
                if not expected or missing or duplicated:
                    info.update(status=MISSING,reason='Incomplete or duplicate case matrix; missing='+str(missing)+' duplicates='+str(duplicated))
    compare_loaded(report['ui_cases'])
    for path in args.soak:
        info,doc=load_file(path);report['files'].append(info)
        if doc:
            if 'latency_by_fifteen_minutes' not in doc or 'counters' not in doc:info.update(status=MISSING,reason='Not the final real-Qt workspace soak schema.')
            else:report['soaks'].append(soak_case(path,doc))
    report['summary']={'ui_cases':len(report['ui_cases']),'ui_status_counts':dict(collections.Counter(row['overall_status'] for row in report['ui_cases'])),
        'soak_status_counts':dict(collections.Counter(row['overall_status'] for row in report['soaks'])),
        'missing_or_invalid_files':sum(row['status']!= 'READ' for row in report['files'])}
    report['summary']['background_algorithms_observed']=sorted({row['background'] for row in report['ui_cases'] if row['background']})
    report['summary']['multiscale_concurrent_coverage']='observed' if 'multiscale' in report['summary']['background_algorithms_observed'] else 'NOT_TESTED; do not infer from traditional'
    report['summary']['real_qt_soak_reports_observed']=len(report['soaks'])
    report['summary']['one_hour_duration_reports_observed']=sum(number(row['active_seconds']) and row['active_seconds']>=3600 for row in report['soaks'])
    encoded=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True);args.output_json.write_text(encoded)
    else:print(encoded,end='')
    if args.output_md:
        args.output_md.parent.mkdir(parents=True,exist_ok=True);args.output_md.write_text(markdown(report))
    if args.output_json:print(json.dumps({'summary':report['summary'],'output_json':str(args.output_json),'output_md':str(args.output_md) if args.output_md else None},allow_nan=False))
    # This is a reader, not a workload runner: failures stay in explicit verdicts.
    return 0
if __name__=='__main__':raise SystemExit(main())
