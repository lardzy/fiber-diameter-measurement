# CU performance read-only audit

Thresholds are fixed by performance-acceptance-rules.md. FAIL/INCOMPLETE cannot be replaced by fast successful samples.

| file / size / PNG / input / auto | overall | success / formal attempts | draft P95/P99 | visible P95/P99 | GUI max | paint FPS | PNG+auto P95 relative |
|---|---|---|---|---|---|---|---|
| final-r3-macos-ui-stable.json / 2048 / False / stable / None | PASS | 30/30 | 1.941/2.092 | 14.809/24.126 | 0.364 | 30.036 | NOT_APPLICABLE |
| final-r3-macos-ui-stable.json / 2048 / True / stable / None | FAIL | 30/30 | 1.294/10.266 | 19.080/19.460 | 6.806 | 30.349 | NOT_APPLICABLE |
| final-r3-macos-ui-stable.json / 4096 / False / stable / None | PASS | 30/30 | 4.373/8.191 | 16.138/17.254 | 2.835 | 30.392 | NOT_APPLICABLE |
| final-r3-macos-ui-stable.json / 4096 / True / stable / None | PASS | 30/30 | 2.821/6.826 | 16.008/16.556 | 0.535 | 30.289 | NOT_APPLICABLE |
| final-r3-macos-ui-stable.json / 6144 / False / stable / None | PASS | 30/30 | 8.102/12.929 | 17.549/20.039 | 0.342 | 30.505 | NOT_APPLICABLE |
| final-r3-macos-ui-stable.json / 6144 / True / stable / None | FAIL | 22/30 | 1.328/1.396 | 16.581/16.860 | 5.953 | 30.012 | NOT_APPLICABLE |
| final-r3-macos-ui-dynamic.json / 2048 / False / lsb-noise / None | PASS | 30/30 | 1.176/1.222 | 26.034/26.514 | 0.298 | 30.340 | NOT_APPLICABLE |
| final-r3-macos-ui-dynamic.json / 2048 / True / lsb-noise / None | PASS | 30/30 | 1.259/1.687 | 20.746/23.624 | 0.591 | 30.353 | NOT_APPLICABLE |
| final-r3-macos-ui-dynamic.json / 4096 / False / lsb-noise / None | FAIL | 29/30 | 1.274/1.542 | 926.792/1183.111 | 0.343 | 30.016 | NOT_APPLICABLE |
| final-r3-macos-ui-dynamic.json / 4096 / True / lsb-noise / None | FAIL | 30/30 | 1.484/2.749 | 817.950/973.574 | 0.420 | 30.068 | NOT_APPLICABLE |
| final-r3-macos-ui-dynamic.json / 6144 / False / lsb-noise / None | FAIL | 0/30 | 1.119/1.130 | missing/missing | missing | 30.003 | NOT_APPLICABLE |
| final-r3-macos-ui-dynamic.json / 6144 / True / lsb-noise / None | FAIL | 0/30 | 1.515/2.397 | missing/missing | missing | 30.012 | NOT_APPLICABLE |
| final-r3-macos-ui-joint-stable.json / 2048 / True / stable / multiscale | PASS | 30/30 | 3.506/8.398 | 15.526/15.699 | 0.953 | 29.997 | PASS |
| final-r3-macos-ui-joint-stable.json / 4096 / True / stable / multiscale | PASS | 30/30 | 1.491/1.694 | 15.631/15.655 | 1.839 | 30.133 | PASS |
| final-r3-macos-ui-joint-stable.json / 6144 / True / stable / multiscale | FAIL | 21/30 | 1.669/1.800 | 16.250/17.477 | 0.377 | 30.201 | FAIL |
| final-r3-macos-ui-joint-dynamic.json / 2048 / True / lsb-noise / multiscale | PASS | 30/30 | 1.059/1.085 | 23.077/24.323 | 0.352 | 30.040 | PASS |
| final-r3-macos-ui-joint-dynamic.json / 4096 / True / lsb-noise / multiscale | FAIL | 29/30 | 1.966/7.901 | 1070.340/1565.961 | 0.441 | 30.045 | FAIL |
| final-r3-macos-ui-joint-dynamic.json / 6144 / True / lsb-noise / multiscale | FAIL | 0/30 | 1.182/1.426 | missing/missing | missing | 30.017 | FAIL |
| final-r3-windows-ui-stable.json / 2048 / False / stable / None | PASS | 30/30 | 6.181/8.125 | 28.643/29.886 | 0.854 | 30.001 | NOT_APPLICABLE |
| final-r3-windows-ui-stable.json / 2048 / True / stable / None | PASS | 30/30 | 7.433/8.228 | 27.271/29.925 | 2.285 | 30.307 | NOT_APPLICABLE |
| final-r3-windows-ui-stable.json / 4096 / False / stable / None | PASS | 30/30 | 8.100/9.777 | 28.815/36.179 | 0.725 | 30.101 | NOT_APPLICABLE |
| final-r3-windows-ui-stable.json / 4096 / True / stable / None | FAIL | 30/30 | 24.157/28.396 | 79.980/92.505 | 12.703 | 28.781 | NOT_APPLICABLE |
| final-r3-windows-ui-stable.json / 6144 / False / stable / None | PASS | 30/30 | 15.429/20.091 | 50.360/67.315 | 3.923 | 27.878 | NOT_APPLICABLE |
| final-r3-windows-ui-stable.json / 6144 / True / stable / None | FAIL | 7/30 | 13.450/13.642 | 53.856/56.718 | 2.769 | 22.007 | NOT_APPLICABLE |
| final-r3-windows-ui-dynamic.json / 2048 / False / lsb-noise / None | FAIL | 30/30 | 5.683/6.141 | 137.632/157.238 | 0.736 | 30.125 | NOT_APPLICABLE |
| final-r3-windows-ui-dynamic.json / 2048 / True / lsb-noise / None | FAIL | 30/30 | 5.092/7.021 | 154.715/200.418 | 4.100 | 30.270 | NOT_APPLICABLE |
| final-r3-windows-ui-dynamic.json / 4096 / False / lsb-noise / None | FAIL | 0/30 | 4.326/5.829 | missing/missing | missing | 30.018 | NOT_APPLICABLE |
| final-r3-windows-ui-dynamic.json / 4096 / True / lsb-noise / None | FAIL | 0/30 | 4.302/7.848 | missing/missing | missing | 30.001 | NOT_APPLICABLE |
| final-r3-windows-ui-dynamic.json / 6144 / False / lsb-noise / None | FAIL | 0/30 | 8.424/9.867 | missing/missing | missing | 29.516 | NOT_APPLICABLE |
| final-r3-windows-ui-dynamic.json / 6144 / True / lsb-noise / None | FAIL | 0/30 | 8.736/9.683 | missing/missing | missing | 29.442 | NOT_APPLICABLE |
| final-r3-windows-ui-joint-stable.json / 2048 / True / stable / multiscale | PASS | 30/30 | 4.662/7.589 | 26.847/28.276 | 1.665 | 30.186 | PASS |
| final-r3-windows-ui-joint-stable.json / 4096 / True / stable / multiscale | FAIL | 30/30 | 6.833/8.564 | 29.655/31.650 | 1.130 | 30.153 | PASS |
| final-r3-windows-ui-joint-stable.json / 6144 / True / stable / multiscale | FAIL | 8/30 | 7.730/8.575 | 70.898/82.761 | 19.666 | 21.960 | FAIL |
| final-r3-windows-ui-joint-dynamic.json / 2048 / True / lsb-noise / multiscale | FAIL | 30/30 | 5.372/5.430 | 147.949/175.954 | 1.529 | 30.167 | PASS |
| final-r3-windows-ui-joint-dynamic.json / 4096 / True / lsb-noise / multiscale | FAIL | 0/30 | 5.310/6.178 | missing/missing | missing | 30.023 | FAIL |
| final-r3-windows-ui-joint-dynamic.json / 6144 / True / lsb-noise / multiscale | FAIL | 0/30 | 9.889/10.802 | missing/missing | missing | 28.583 | FAIL |

final-r3-macos-ui-stable.json / 2048 / PNG True: gui_publish_max_ms=FAIL

final-r3-macos-ui-stable.json / 6144 / PNG True: gui_publish_max_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r3-macos-ui-dynamic.json / 4096 / PNG False: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r3-macos-ui-dynamic.json / 4096 / PNG True: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL

final-r3-macos-ui-dynamic.json / 6144 / PNG False: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r3-macos-ui-dynamic.json / 6144 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r3-macos-ui-joint-stable.json / 2048 / PNG True: background completions=16, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-macos-ui-joint-stable.json / 4096 / PNG True: background completions=4, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-macos-ui-joint-stable.json / 6144 / PNG True: formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r3-macos-ui-stable.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r3-macos-ui-joint-dynamic.json / 2048 / PNG True: background completions=16, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-macos-ui-joint-dynamic.json / 4096 / PNG True: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r3-macos-ui-dynamic.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r3-macos-ui-joint-dynamic.json / 4096 / PNG True: background completions=21, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-macos-ui-joint-dynamic.json / 6144 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r3-macos-ui-dynamic.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r3-macos-ui-joint-dynamic.json / 6144 / PNG True: background completions=34, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-windows-ui-stable.json / 4096 / PNG True: draft_p95_ms=FAIL; result_visible_p95_ms=FAIL; gui_publish_max_ms=FAIL

final-r3-windows-ui-stable.json / 6144 / PNG True: formal_failures=FAIL; all_rejection_events_including_warmup=FAIL; distinct_painted_fps=FAIL

final-r3-windows-ui-dynamic.json / 2048 / PNG False: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL

final-r3-windows-ui-dynamic.json / 2048 / PNG True: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; gui_publish_max_ms=FAIL

final-r3-windows-ui-dynamic.json / 4096 / PNG False: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r3-windows-ui-dynamic.json / 4096 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r3-windows-ui-dynamic.json / 6144 / PNG False: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r3-windows-ui-dynamic.json / 6144 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

final-r3-windows-ui-joint-stable.json / 2048 / PNG True: background completions=3, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-windows-ui-joint-stable.json / 4096 / PNG True: background_executed=FAIL; background_errors=FAIL

final-r3-windows-ui-joint-stable.json / 4096 / PNG True: background completions=0, cancelled=1, inferred teardown=0, explicit teardown=0, unexplained cancellations=1. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-windows-ui-joint-stable.json / 6144 / PNG True: gui_publish_max_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL; distinct_painted_fps=FAIL; background_executed=FAIL; background_errors=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r3-windows-ui-stable.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r3-windows-ui-joint-stable.json / 6144 / PNG True: background completions=0, cancelled=1, inferred teardown=0, explicit teardown=0, unexplained cancellations=1. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-windows-ui-joint-dynamic.json / 2048 / PNG True: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL

final-r3-windows-ui-joint-dynamic.json / 2048 / PNG True: background completions=4, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-windows-ui-joint-dynamic.json / 4096 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r3-windows-ui-dynamic.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r3-windows-ui-joint-dynamic.json / 4096 / PNG True: background completions=13, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

final-r3-windows-ui-joint-dynamic.json / 6144 / PNG True: result_visible_p95_ms=INCOMPLETE; result_visible_p99_ms=INCOMPLETE; gui_publish_max_ms=INCOMPLETE; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Relative: {'status': 'FAIL', 'p99_relative_is_diagnostic_only': True, 'baseline_file': 'docs/validation/cu-live-measurement-2026-10-15/final-r3-windows-ui-dynamic.json', 'reason': 'Rejected/unfinished baseline or loaded attempts: success-only relative quantiles cannot qualify.', 'baseline_formal_eligibility': 'PASS'}

final-r3-windows-ui-joint-dynamic.json / 6144 / PNG True: background completions=5, cancelled=1, inferred teardown=0, explicit teardown=1, unexplained cancellations=0. Phases and timestamps describe GUI receipt, not worker completion. Only the unique final cancelled/null-error/zero-candidate response matching the task closed at teardown is exempt; it remains incomplete, never a successful background completion. Formal completions exclude warmup and teardown. Invalid or partial phase traces cannot use the legacy SHA inference.

Missing source files or platform/combination coverage remain unaccepted. See JSON for every denominator, check and 15-minute segment.
