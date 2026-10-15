# CU performance read-only audit

Thresholds are fixed by acceptance-rules.md. FAIL/INCOMPLETE cannot be replaced by fast successful samples.

| file / size / PNG / input / auto | overall | success / formal attempts | draft P95/P99 | visible P95/P99 | GUI max | paint FPS | PNG+auto P95 relative |
|---|---|---|---|---|---|---|---|
| ui-stable.json / 2048 / False / stable / None | FAIL | 30/30 | 5.231/5.666 | 23.006/25.415 | 4.076 | 30.181 | NOT_APPLICABLE |
| ui-stable.json / 2048 / True / stable / None | PASS | 30/30 | 4.738/5.289 | 22.747/24.373 | 2.949 | 30.432 | NOT_APPLICABLE |
| ui-stable.json / 4096 / False / stable / None | PASS | 30/30 | 5.278/10.145 | 24.541/30.543 | 0.836 | 30.288 | NOT_APPLICABLE |
| ui-stable.json / 4096 / True / stable / None | FAIL | 30/30 | 5.157/5.290 | 22.941/24.431 | 5.495 | 30.226 | NOT_APPLICABLE |
| ui-dynamic.json / 2048 / False / lsb-noise / None | PASS | 30/30 | 4.893/5.261 | 23.110/24.498 | 0.910 | 30.169 | NOT_APPLICABLE |
| ui-dynamic.json / 2048 / True / lsb-noise / None | PASS | 30/30 | 4.849/5.127 | 22.955/24.435 | 1.654 | 30.121 | NOT_APPLICABLE |
| ui-dynamic.json / 4096 / False / lsb-noise / None | PASS | 30/30 | 6.553/9.848 | 26.265/39.444 | 1.052 | 30.056 | NOT_APPLICABLE |
| ui-dynamic.json / 4096 / True / lsb-noise / None | FAIL | 30/30 | 10.475/12.903 | 28.670/30.250 | 4.717 | 30.065 | NOT_APPLICABLE |
| ui-joint-stable.json / 2048 / False / stable / multiscale | PASS | 30/30 | 4.828/4.867 | 22.487/25.327 | 0.990 | 30.331 | NOT_APPLICABLE |
| ui-joint-stable.json / 2048 / True / stable / multiscale | FAIL | 30/30 | 5.026/10.302 | 27.188/31.589 | 16.910 | 30.467 | PASS |
| ui-joint-stable.json / 4096 / False / stable / multiscale | FAIL | 30/30 | 12.399/12.688 | 37.422/40.205 | 5.914 | 29.707 | NOT_APPLICABLE |
| ui-joint-stable.json / 4096 / True / stable / multiscale | FAIL | 30/30 | 5.647/5.905 | 26.264/26.396 | 6.084 | 30.411 | PASS |
| ui-joint-dynamic.json / 2048 / False / lsb-noise / multiscale | PASS | 30/30 | 4.928/9.845 | 24.384/31.349 | 0.842 | 30.078 | NOT_APPLICABLE |
| ui-joint-dynamic.json / 2048 / True / lsb-noise / multiscale | PASS | 30/30 | 4.570/4.764 | 23.896/24.496 | 1.086 | 30.101 | PASS |
| ui-joint-dynamic.json / 4096 / False / lsb-noise / multiscale | PASS | 30/30 | 8.655/9.103 | 25.990/27.203 | 1.139 | 30.037 | NOT_APPLICABLE |
| ui-joint-dynamic.json / 4096 / True / lsb-noise / multiscale | FAIL | 30/30 | 12.713/18.488 | 34.082/37.056 | 8.289 | 27.361 | PASS |
| ui-stable.json / 2048 / False / stable / None | PASS | 30/30 | 7.905/8.024 | 7.848/7.893 | 0.466 | 30.034 | NOT_APPLICABLE |
| ui-stable.json / 2048 / True / stable / None | PASS | 30/30 | 2.413/3.782 | 8.639/9.533 | 0.520 | 30.226 | NOT_APPLICABLE |
| ui-stable.json / 4096 / False / stable / None | PASS | 30/30 | 4.761/5.421 | 7.844/8.108 | 0.362 | 30.320 | NOT_APPLICABLE |
| ui-stable.json / 4096 / True / stable / None | PASS | 30/30 | 1.524/1.572 | 7.309/7.744 | 0.429 | 30.239 | NOT_APPLICABLE |
| ui-dynamic.json / 2048 / False / lsb-noise / None | PASS | 30/30 | 1.633/1.773 | 6.852/7.070 | 0.364 | 30.363 | NOT_APPLICABLE |
| ui-dynamic.json / 2048 / True / lsb-noise / None | PASS | 30/30 | 1.560/1.691 | 7.887/8.084 | 0.385 | 30.123 | NOT_APPLICABLE |
| ui-dynamic.json / 4096 / False / lsb-noise / None | PASS | 30/30 | 1.556/2.001 | 6.023/6.161 | 0.258 | 30.276 | NOT_APPLICABLE |
| ui-dynamic.json / 4096 / True / lsb-noise / None | PASS | 30/30 | 1.389/1.731 | 7.123/7.288 | 0.387 | 30.348 | NOT_APPLICABLE |
| ui-joint-stable.json / 2048 / False / stable / multiscale | PASS | 30/30 | 1.371/1.395 | 6.364/6.510 | 0.350 | 30.142 | NOT_APPLICABLE |
| ui-joint-stable.json / 2048 / True / stable / multiscale | PASS | 30/30 | 1.323/1.783 | 6.776/7.081 | 0.815 | 30.407 | PASS |
| ui-joint-stable.json / 4096 / False / stable / multiscale | PASS | 30/30 | 1.454/3.075 | 6.030/6.255 | 0.342 | 30.546 | NOT_APPLICABLE |
| ui-joint-stable.json / 4096 / True / stable / multiscale | PASS | 30/30 | 1.363/1.713 | 7.165/8.863 | 0.674 | 30.127 | PASS |
| ui-joint-dynamic.json / 2048 / False / lsb-noise / multiscale | PASS | 30/30 | 1.670/2.311 | 6.880/7.665 | 0.270 | 30.122 | NOT_APPLICABLE |
| ui-joint-dynamic.json / 2048 / True / lsb-noise / multiscale | PASS | 30/30 | 1.284/1.763 | 7.057/7.640 | 0.416 | 30.417 | PASS |
| ui-joint-dynamic.json / 4096 / False / lsb-noise / multiscale | PASS | 30/30 | 1.639/1.701 | 6.159/6.333 | 0.285 | 30.310 | NOT_APPLICABLE |
| ui-joint-dynamic.json / 4096 / True / lsb-noise / multiscale | PASS | 30/30 | 1.502/4.937 | 6.773/6.916 | 0.573 | 29.944 | PASS |

ui-stable.json / 2048 / PNG False: gui_publish_max_ms=FAIL

ui-stable.json / 4096 / PNG True: gui_publish_max_ms=FAIL

ui-dynamic.json / 4096 / PNG True: gui_publish_max_ms=FAIL

ui-joint-stable.json / 2048 / PNG True: gui_publish_max_ms=FAIL

ui-joint-stable.json / 4096 / PNG False: gui_publish_max_ms=FAIL

ui-joint-stable.json / 4096 / PNG True: gui_publish_max_ms=FAIL

ui-joint-dynamic.json / 4096 / PNG True: gui_publish_max_ms=FAIL

## Soak workspace-4k-short.json: FAIL

Duration 60.002 s; accepted/submitted 236/236; paint FPS 29.949.

GUI P95/P99/max 1.333/2.997/5.712 ms; scene→clear paint P95 76.800 ms.

gui_publish_max_ms=FAIL; at_least_one_hour=FAIL; at_least_10000_accepted=FAIL; quarter_count=FAIL; quarter_0_thresholds=FAIL

## Soak workspace-4k-short.json: FAIL

Duration 60.000 s; accepted/submitted 238/238; paint FPS 29.933.

GUI P95/P99/max 0.498/0.570/15.191 ms; scene→clear paint P95 64.374 ms.

gui_publish_max_ms=FAIL; at_least_one_hour=FAIL; at_least_10000_accepted=FAIL; quarter_count=FAIL; quarter_0_thresholds=FAIL

Missing source files or platform/combination coverage remain unaccepted. See JSON for every denominator, check and 15-minute segment.
