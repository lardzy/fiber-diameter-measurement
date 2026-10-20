# Candidate alignment profiling — research only

2026-10-20. No production files, package contents, thresholds, resolution or presentation budgets were changed.

## Outcome

Do **not** adopt this Python reference-FFT cache prototype: although numerically equivalent on the tested inputs, it is slower in the target Windows AMD64-on-ARM virtual machine. Switching OpenCV to one thread in an isolated profiling process did not help the existing algorithm either. No global product thread setting was changed.

| Environment / 100 Windows iterations | Existing P50 / P95 | Cached P50 / P95 |
|---|---:|---:|
| Windows, 4 threads | 15.755 / 17.221 ms | 19.102 / 20.711 ms |
| Windows, 1 thread | 15.815 / 17.025 ms | 19.781 / 21.773 ms |
| macOS, default threads, 30 iterations | 6.623 / 7.163 ms | 5.805 / 6.628 ms |

The Windows 4-thread existing stage P50 values were: gray conversion 1.328 ms; standard deviation 0.839; normalization 1.344; NCC 0.738; gradient energy 3.536; phase correlation 7.823. The cached implementation's packed-spectrum magnitude alone took 4.626 ms; its NumPy float64 intermediates erased the saved reference DFT time. Stage percentiles are not additive.

The isolated profile is not the gallery/JPEG concurrency acceptance test. That test remains failed: its existing implementation alignment P95 was 36.427 ms, GUI-consumed verification P95 43.943 ms, and candidate availability was about 4.4%. A faster isolated result does not clear that failure.

## Numerical equivalence

`profile_alignment.py` caches only the immutable reference spectrum. Every current image is still fully transformed and checked; no identical-image shortcut or subsampling exists. Packing, padding, magnitude double intermediates, spectrum division epsilon, FFT shift, 5×5 centroid and response normalization follow the [OpenCV 4.13 implementation](https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/phasecorr.cpp).

- 148 synthetic cases across four sizes (including odd/padded dimensions), independent noise, subpixel translations, exposure, blur and unrelated images matched decisions, shift and response exactly on both platforms.
- `optical-equivalence.json` adds 94 cases based on two existing BGR optical fixtures with independent noise, random subpixel motion, exposure and blur. All returned values matched exactly on macOS. These are not Microview raw captures.
- This evidence does not prove universal floating-point equivalence or replace budget/lifecycle integration tests. The prototype has no production frame-budget integration and is not proposed for shipping.

Archived results are `windows-profile-thread4.json`, `windows-profile-thread1.json`, `macos-profile-and-equivalence.json`, and `optical-equivalence.json`. The research-only prototype remains outside production under `.tmp/round5-alignment-optimization/profile_alignment.py`; it was not applied or included in the installer. Its SHA256 is recorded in `archive-manifest.json` and the optical equivalence result.

```sh
.venv/bin/python .tmp/round5-alignment-optimization/profile_alignment.py --iterations 100 --threads 4 --output /tmp/profile-thread4.json
.venv/bin/python .tmp/round5-alignment-optimization/profile_alignment.py --iterations 100 --threads 1 --output /tmp/profile-thread1.json
```
