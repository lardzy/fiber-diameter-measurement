# CU performance read-only audit

Thresholds are fixed by performance-acceptance-rules.md. FAIL/INCOMPLETE cannot be replaced by fast successful samples.

| file / size / PNG / input / auto | overall | success / formal attempts | draft P95/P99 | visible P95/P99 | GUI max | paint FPS | PNG+auto P95 relative |
|---|---|---|---|---|---|---|---|
| final-r4-macos-ui-stable.json / 2048 / False / stable / None | PASS | 30/30 | 9.947/11.001 | 18.170/21.207 | 1.769 | 30.139 | NOT_APPLICABLE |
| final-r4-macos-ui-stable.json / 2048 / True / stable / None | PASS | 30/30 | 1.031/1.043 | 14.632/15.283 | 0.386 | 30.268 | NOT_APPLICABLE |
| final-r4-macos-ui-stable.json / 4096 / False / stable / None | PASS | 30/30 | 7.227/11.788 | 15.443/22.074 | 0.623 | 30.061 | NOT_APPLICABLE |
| final-r4-macos-ui-stable.json / 4096 / True / stable / None | PASS | 30/30 | 1.495/2.699 | 16.512/17.167 | 0.849 | 30.358 | NOT_APPLICABLE |
| final-r4-macos-ui-stable.json / 6144 / False / stable / None | PASS | 30/30 | 8.274/13.112 | 9.763/14.350 | 0.336 | 30.217 | NOT_APPLICABLE |
| final-r4-macos-ui-stable.json / 6144 / True / stable / None | FAIL | 21/30 | 9.820/10.211 | 16.356/17.385 | 1.060 | 30.363 | NOT_APPLICABLE |
| final-r4-macos-ui-dynamic.json / 2048 / False / lsb-noise / None | PASS | 30/30 | 1.243/1.289 | 25.558/27.000 | 0.323 | 30.382 | NOT_APPLICABLE |
| final-r4-macos-ui-dynamic.json / 2048 / True / lsb-noise / None | PASS | 30/30 | 1.135/1.225 | 24.221/24.712 | 0.674 | 30.002 | NOT_APPLICABLE |
| final-r4-macos-ui-dynamic.json / 4096 / False / lsb-noise / None | FAIL | 30/30 | 1.427/1.816 | 1547.807/1826.214 | 0.327 | 30.058 | NOT_APPLICABLE |
| final-r4-macos-ui-dynamic.json / 4096 / True / lsb-noise / None | FAIL | 28/30 | 1.343/1.511 | 951.552/1020.519 | 0.735 | 30.057 | NOT_APPLICABLE |
| final-r4-macos-ui-dynamic.json / 6144 / False / lsb-noise / None | FAIL | 0/30 | 1.370/1.783 | missing/missing | missing | 30.004 | NOT_APPLICABLE |
| final-r4-macos-ui-dynamic.json / 6144 / True / lsb-noise / None | FAIL | 0/30 | 1.276/1.742 | missing/missing | missing | 30.012 | NOT_APPLICABLE |
| final-r4-macos-ui-joint-stable.json / 2048 / True / stable / multiscale | PASS | 30/30 | 2.050/6.666 | 16.089/16.611 | 0.719 | 30.348 | PASS |
| final-r4-macos-ui-joint-stable.json / 4096 / True / stable / multiscale | PASS | 30/30 | 1.248/1.309 | 16.186/16.673 | 0.561 | 30.157 | PASS |
| final-r4-macos-ui-joint-stable.json / 6144 / True / stable / multiscale | FAIL | 21/30 | 10.199/10.364 | 10.057/15.012 | 0.410 | 30.362 | FAIL |
| final-r4-macos-ui-joint-dynamic.json / 2048 / True / lsb-noise / multiscale | PASS | 30/30 | 1.074/1.108 | 18.034/21.036 | 0.326 | 30.531 | PASS |
| final-r4-macos-ui-joint-dynamic.json / 4096 / True / lsb-noise / multiscale | FAIL | 29/30 | 1.376/1.530 | 1104.973/1380.537 | 0.414 | 30.059 | FAIL |
| final-r4-macos-ui-joint-dynamic.json / 6144 / True / lsb-noise / multiscale | FAIL | 0/30 | 1.263/1.657 | missing/missing | missing | 30.006 | FAIL |
| final-r4-windows-ui-stable.json / 2048 / False / stable / None | PASS | 30/30 | 5.572/5.706 | 30.433/30.604 | 0.860 | 30.374 | NOT_APPLICABLE |
| final-r4-windows-ui-stable.json / 2048 / True / stable / None | PASS | 30/30 | 6.723/8.457 | 34.504/39.925 | 1.305 | 30.242 | NOT_APPLICABLE |
| final-r4-windows-ui-stable.json / 4096 / False / stable / None | PASS | 30/30 | 6.153/8.291 | 31.208/35.200 | 0.676 | 30.409 | NOT_APPLICABLE |
| final-r4-windows-ui-stable.json / 4096 / True / stable / None | FAIL | 30/30 | 18.697/19.695 | 104.216/112.693 | 19.980 | 28.840 | NOT_APPLICABLE |
| final-r4-windows-ui-stable.json / 6144 / False / stable / None | FAIL | 30/30 | 19.072/23.272 | 44.987/54.066 | 3.287 | 27.516 | NOT_APPLICABLE |
| final-r4-windows-ui-stable.json / 6144 / True / stable / None | FAIL | 10/30 | 24.931/25.090 | 83.036/100.354 | 4.677 | 23.636 | NOT_APPLICABLE |
| final-r4-windows-ui-dynamic.json / 2048 / False / lsb-noise / None | FAIL | 30/30 | 6.334/6.932 | 51.855/57.393 | 4.162 | 30.100 | NOT_APPLICABLE |
| final-r4-windows-ui-dynamic.json / 2048 / True / lsb-noise / None | PASS | 30/30 | 5.393/6.189 | 41.335/44.748 | 0.940 | 30.266 | NOT_APPLICABLE |
| final-r4-windows-ui-dynamic.json / 4096 / False / lsb-noise / None | FAIL | 12/30 | 5.278/7.478 | 1745.723/1900.616 | 0.691 | 30.005 | NOT_APPLICABLE |
| final-r4-windows-ui-dynamic.json / 4096 / True / lsb-noise / None | FAIL | 16/30 | 5.002/5.397 | 1748.206/1811.678 | 4.001 | 30.031 | NOT_APPLICABLE |
| final-r4-windows-ui-dynamic.json / 6144 / False / lsb-noise / None | FAIL | 0/30 | 6.371/8.388 | missing/missing | missing | 29.349 | NOT_APPLICABLE |
| final-r4-windows-ui-dynamic.json / 6144 / True / lsb-noise / None | FAIL | 0/30 | 6.261/6.945 | missing/missing | missing | 29.374 | NOT_APPLICABLE |
| final-r4-windows-ui-joint-stable.json / 2048 / True / stable / multiscale | PASS | 30/30 | 4.105/4.554 | 28.272/30.599 | 1.313 | 30.288 | PASS |
| final-r4-windows-ui-joint-stable.json / 4096 / True / stable / multiscale | FAIL | 30/30 | 6.550/7.239 | 30.643/31.428 | 2.686 | 30.183 | PASS |
| final-r4-windows-ui-joint-stable.json / 6144 / True / stable / multiscale | FAIL | 7/30 | 17.522/17.685 | 33.736/34.824 | 2.176 | 20.726 | FAIL |
| final-r4-windows-ui-joint-dynamic.json / 2048 / True / lsb-noise / multiscale | FAIL | 30/30 | 7.987/8.663 | 52.820/62.687 | 5.052 | 30.212 | PASS |
| final-r4-windows-ui-joint-dynamic.json / 4096 / True / lsb-noise / multiscale | FAIL | 2/30 | 4.957/6.423 | 1391.212/1420.663 | 0.698 | 30.018 | FAIL |
| final-r4-windows-ui-joint-dynamic.json / 6144 / True / lsb-noise / multiscale | FAIL | 0/30 | 5.080/5.819 | missing/missing | missing | 28.322 | FAIL |

final-r4-macos-ui-stable.json / 6144 / PNG True: formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r4-macos-ui-dynamic.json / 4096 / PNG False: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL

final-r4-macos-ui-dynamic.json / 4096 / PNG True: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r4-macos-ui-dynamic.json / 6144 / PNG False: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r4-macos-ui-dynamic.json / 6144 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r4-macos-ui-joint-stable.json / 4096 / PNG True: background completions=4, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r4-macos-ui-joint-stable.json / 6144 / PNG True: formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r4-macos-ui-stable.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r4-macos-ui-joint-dynamic.json / 2048 / PNG True: background completions=16, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r4-macos-ui-joint-dynamic.json / 4096 / PNG True: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r4-macos-ui-dynamic.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r4-macos-ui-joint-dynamic.json / 4096 / PNG True: background completions=18, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r4-macos-ui-joint-dynamic.json / 6144 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r4-macos-ui-dynamic.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r4-macos-ui-joint-dynamic.json / 6144 / PNG True: background completions=34, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r4-windows-ui-stable.json / 4096 / PNG True: draft_p95_ms=FAIL; result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; gui_publish_max_ms=FAIL

final-r4-windows-ui-stable.json / 6144 / PNG False: draft_p95_ms=FAIL

final-r4-windows-ui-stable.json / 6144 / PNG True: draft_p95_ms=FAIL; gui_publish_max_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL; distinct_painted_fps=FAIL

final-r4-windows-ui-dynamic.json / 2048 / PNG False: result_visible_p95_ms=FAIL; gui_publish_max_ms=FAIL

final-r4-windows-ui-dynamic.json / 4096 / PNG False: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r4-windows-ui-dynamic.json / 4096 / PNG True: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; gui_publish_max_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r4-windows-ui-dynamic.json / 6144 / PNG False: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r4-windows-ui-dynamic.json / 6144 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r4-windows-ui-joint-stable.json / 2048 / PNG True: background completions=3, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r4-windows-ui-joint-stable.json / 4096 / PNG True: background_executed=FAIL; background_errors=FAIL

final-r4-windows-ui-joint-stable.json / 4096 / PNG True: background completions=0, cancelled=1, inferred teardown=0, explicit teardown=0, unexplained cancellations=1. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r4-windows-ui-joint-stable.json / 6144 / PNG True: draft_p95_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL; distinct_painted_fps=FAIL; background_executed=FAIL; background_errors=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r4-windows-ui-stable.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r4-windows-ui-joint-stable.json / 6144 / PNG True: background completions=0, cancelled=1, inferred teardown=0, explicit teardown=0, unexplained cancellations=1. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r4-windows-ui-joint-dynamic.json / 2048 / PNG True: result_visible_p95_ms=FAIL; gui_publish_max_ms=FAIL

final-r4-windows-ui-joint-dynamic.json / 2048 / PNG True: background completions=3, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r4-windows-ui-joint-dynamic.json / 4096 / PNG True: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r4-windows-ui-dynamic.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r4-windows-ui-joint-dynamic.json / 4096 / PNG True: background completions=12, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r4-windows-ui-joint-dynamic.json / 6144 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r4-windows-ui-dynamic.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r4-windows-ui-joint-dynamic.json / 6144 / PNG True: background completions=5, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

Missing source files or platform/combination coverage remain unaccepted. See JSON for every denominator, check and 15-minute segment.
