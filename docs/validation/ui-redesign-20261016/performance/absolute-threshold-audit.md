# CU performance read-only audit

Thresholds are fixed by performance-acceptance-rules.md. FAIL/INCOMPLETE cannot be replaced by fast successful samples.

| file / size / PNG / input / auto | overall | success / formal attempts | draft P95/P99 | visible P95/P99 | GUI max | paint FPS | PNG+auto P95 relative |
|---|---|---|---|---|---|---|---|
| cu-ui-stable.json / 2048 / False / stable / None | PASS | 30/30 | 2.793/3.477 | 8.379/8.805 | 0.609 | 30.339 | NOT_APPLICABLE |
| cu-ui-stable.json / 2048 / True / stable / None | PASS | 30/30 | 2.725/2.934 | 10.072/10.136 | 0.915 | 30.533 | NOT_APPLICABLE |
| cu-ui-stable.json / 4096 / False / stable / None | PASS | 30/30 | 2.753/3.007 | 9.167/12.927 | 1.916 | 30.187 | NOT_APPLICABLE |
| cu-ui-stable.json / 4096 / True / stable / None | PASS | 30/30 | 1.711/1.730 | 7.571/10.428 | 0.559 | 30.375 | NOT_APPLICABLE |
| cu-ui-dynamic.json / 2048 / False / lsb-noise / None | PASS | 30/30 | 1.290/1.464 | 26.913/27.420 | 0.331 | 30.245 | NOT_APPLICABLE |
| cu-ui-dynamic.json / 2048 / True / lsb-noise / None | PASS | 30/30 | 1.419/2.324 | 24.448/24.988 | 0.419 | 30.371 | NOT_APPLICABLE |
| cu-ui-dynamic.json / 4096 / False / lsb-noise / None | FAIL | 30/30 | 1.434/1.552 | 565.010/806.072 | 0.441 | 30.094 | NOT_APPLICABLE |
| cu-ui-dynamic.json / 4096 / True / lsb-noise / None | FAIL | 28/30 | 1.309/1.362 | 1047.813/1397.293 | 0.509 | 30.095 | NOT_APPLICABLE |

cu-ui-dynamic.json / 4096 / PNG False: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL

cu-ui-dynamic.json / 4096 / PNG True: result_visible_p95_ms=FAIL; result_visible_p99_ms=FAIL; formal_failures=FAIL; all_rejection_events_including_warmup=FAIL

Missing source files or platform/combination coverage remain unaccepted. See JSON for every denominator, check and 15-minute segment.
