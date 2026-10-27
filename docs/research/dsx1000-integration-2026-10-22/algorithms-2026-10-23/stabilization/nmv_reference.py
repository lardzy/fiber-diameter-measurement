"""Independent research reconstruction of DSX live NMV; never loads vendor code.

The FFT and least-squares libraries are substitutes, not bit-exact vendor oracles.
`stable_ratio=True` is explicitly an analyst improvement, not shipped behavior.
See implementation-spec.md for evidence, sign conventions, and limitations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import leastsq

FFT_SIZES = json.loads((Path(__file__).parent / "fft-sizes.json").read_text())["sizes"]


@dataclass(frozen=True)
class Settings:
    divider: int = 8
    use_lpf: bool = True
    cutoff: float = 0.5
    subpixel: bool = True
    reliability_threshold: float = 0.35
    stable_ratio: bool = False


def block_convert(src: np.ndarray, divider: int) -> np.ndarray:
    if src.dtype not in (np.uint8, np.uint16):
        raise ValueError("Native input accepts unsigned 8/16-bit channels")
    d = max(1, int(divider))
    a = src[..., None] if src.ndim == 2 else src
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError("Native accepts 1, 3 or 4 channels; alpha is ignored")
    h, w = a.shape[0] // d, a.shape[1] // d
    if not h or not w:
        raise ValueError("Zero destination dimensions")
    # Sum RGB, average the spatial block; do not divide by channel count.
    a = a[: h*d, : w*d, :min(3, a.shape[2])].astype(np.float32)
    a = a.reshape(h, d, w, d, a.shape[2])
    return np.ascontiguousarray(a.sum(axis=(1, 3, 4)) / np.float32(d*d))


def preferred_size(n: int) -> int:
    if n >= 65536:
        raise ValueError("Native size selector rejects n >= 65536")
    return next(v for v in FFT_SIZES if v >= n)


def _native_clamp(v: float) -> float:
    # Branch-equivalent clamp: NaN also follows the <= -1 branch.
    if v >= 1.0:
        return 1.0
    if v > -1.0:
        return v
    return -1.0


def _ratio(t: np.ndarray, c: float, n: int, stable: bool) -> np.ndarray:
    if stable:
        # Analytic limit at t=0; consciously differs from native direct sin/sin.
        return c * n * np.sinc(c*t) / np.sinc(t/n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sin(np.pi*c*t) / np.sin(np.pi*t/n)


def fit_peak(corr: np.ndarray, x: int, y: int, cutoff: float, stable: bool) -> dict:
    h, w = corr.shape
    if not (0 < x < w-1 and 0 < y < h-1):
        return {"status": -4, "peak": [float(x), float(y)], "reason": "peak_border"}
    z = corr[y-1:y+2, x-1:x+2].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = -(z[1, 0]-z[1, 2]) / (2*z[1, 0]-4*z[1, 1]+2*z[1, 2])
        dy = -(z[0, 1]-z[2, 1]) / (2*z[0, 1]-4*z[1, 1]+2*z[2, 1])
    yy, xx = np.mgrid[-1:2, -1:2]
    c = min(1., max(0., cutoff))
    initial = np.array([dx, dy, 1.0])

    def residual(p):
        return (z - _ratio(yy+p[1], c, h, stable) *
                _ratio(xx+p[0], c, w, stable)).ravel()

    # Same 3 parameters, third unused, forward differences and numeric controls.
    # scipy MINPACK is a replacement solver; no bit-exactness claim.
    p, _, info, message, code = leastsq(
        residual, initial, full_output=True, ftol=np.sqrt(np.finfo(float).eps),
        xtol=np.sqrt(np.finfo(float).eps), gtol=0., maxfev=400,
        epsfcn=0., factor=100.)
    ox, oy = _native_clamp(float(p[0])), _native_clamp(float(p[1]))
    return {"status": 0, "peak": [x-ox, y-oy], "initial": initial.tolist(),
            "fitted": p.tolist(), "solver_code": int(code),
            "nfev": int(info["nfev"]), "residual_finite": bool(np.isfinite(info["fvec"]).all()),
            "solver_message": message}


def estimate(previous: np.ndarray, current: np.ndarray, settings=Settings(), *,
             return_surface=False) -> dict:
    a, b = block_convert(previous, settings.divider), block_convert(current, settings.divider)
    if a.shape != b.shape:
        raise ValueError("Frame dimensions differ")
    h, w = a.shape
    if min(h, w) < 2:
        raise ValueError("Reference prototype excludes 1D Hann; native edge case unresolved")
    H, W = preferred_size(h), preferred_size(w)
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    a = np.pad(a*hann, ((0,H-h),(0,W-w)))
    b = np.pad(b*hann, ((0,H-h),(0,W-w)))
    A, B = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT), cv2.dft(b, flags=cv2.DFT_COMPLEX_OUTPUT)
    re = A[...,0]*B[...,0] + A[...,1]*B[...,1]
    im = A[...,1]*B[...,0] - A[...,0]*B[...,1]
    den = np.sqrt(re*re+im*im) + np.finfo(np.float32).eps
    spectrum = np.stack((re/den, im/den), axis=-1)
    if settings.use_lpf:
        c = min(1., max(.1, settings.cutoff))
        fx, fy = int(W*c*.5), int(H*c*.5)
        spectrum[:, fx:W-fx] = 0
        spectrum[fy:H-fy, :] = 0
    corr = cv2.idft(spectrum, flags=cv2.DFT_REAL_OUTPUT)
    # Preferred-size table contains only even values: quadrant swap == fftshift.
    corr = np.fft.fftshift(corr)
    _, maximum, _, (x, y) = cv2.minMaxLoc(corr)
    fit = (fit_peak(corr,x,y,settings.cutoff,settings.stable_ratio) if settings.subpixel
           else {"status":0,"peak":[float(x),float(y)]})
    if fit["status"]:
        native = [0.,0.]
        reliability = 0.
    else:
        native = [(W*.5-fit["peak"][0])*max(1,settings.divider),
                  (H*.5-fit["peak"][1])*max(1,settings.divider)]
        reliability = float(np.divide(maximum, settings.cutoff**2*W*H))
    result = {"status":fit["status"], "native_shift":native,
              "managed_shift":[-native[0],-native[1]], "reliability":reliability,
              "integer_peak":[x,y], "peak_value":maximum, "downsample_shape":[h,w],
              "fft_shape":[H,W], "fit":fit}
    if return_surface:
        result["surface"] = corr
    return result


def filter_gate(result: dict, threshold=.35) -> dict:
    x,y = result["managed_shift"]
    reason = ("stage_shift_gt20" if abs(x)>20 or abs(y)>20 else
              "blur_shift_gt15" if abs(x)>15 or abs(y)>15 else
              "reliability_le_threshold" if result["reliability"]<=threshold else "accepted")
    reset = reason != "accepted"
    return {"reset":reset,"reason":reason,
            "shift":[0.,0.] if reset else [math.floor(x*1000)/1000,math.floor(y*1000)/1000]}


class LiveDisplayModel:
    """Live-only metadata/display model. Does not alter image pixels or stage coordinates."""
    def __init__(self, settings=Settings(), buffer_x=20.):
        self.settings, self.buffer_x = settings, buffer_x
        self.previous = None
        self.base_requested = True
        self.acc = np.zeros(2)

    def request_base(self):
        self.base_requested = True

    def feed(self, frame: np.ndarray, display_size=None) -> dict:
        newbase = self.base_requested or self.previous is None or self.previous.shape != frame.shape
        if newbase:
            self.previous = frame.copy()
            self.base_requested = False
            result = None
            gate = {"reset":True,"reason":"new_base","shift":[0.,0.]}
        else:
            result = estimate(self.previous,frame,self.settings)
            # Native call success advances reference even when managed gate rejects it.
            if result["status"] == 0:
                self.previous = frame.copy()
            gate = filter_gate(result,self.settings.reliability_threshold)
        h,w = frame.shape[:2]
        limits = [self.buffer_x,self.buffer_x*h/w]
        for i in range(2):
            candidate = self.acc[i]+gate["shift"][i]
            self.acc[i] = 0. if gate["reset"] or abs(candidate)>limits[i] else candidate
        size = (w,h) if display_size is None else display_size
        return {"estimate":result,"gate":gate,"accumulated_pixels":self.acc.tolist(),
                "display_offset":[self.acc[0]*size[0]/w,self.acc[1]*size[1]/h],
                "pixels_modified":False}
