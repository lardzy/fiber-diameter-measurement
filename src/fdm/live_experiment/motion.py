"""CU two-stage field detector, with owned buffers and unsigned Sobel fix.

The reference revision is for numerical regression only. Production never
reuses CU image headers or worker threads. FFT is ordinary cross correlation,
not phase correlation.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .frames import FrameHandle

MOTION_REVISION = "cu6-field-unsigned-v1"
_GRAY_NORMALIZATION = (np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)


@dataclass(frozen=True, slots=True)
class MotionResult:
    frame_id: str
    sequence: int
    stable: bool
    reason: str
    centroid_delta: tuple[int, int] = (0, 0)
    translation: tuple[int, int] = (0, 0)
    changed_fraction: float = 0.0


def reduced_gray(pixels: np.ndarray, factor: int = 3) -> np.ndarray:
    height, width = pixels.shape[:2]
    if pixels.ndim == 3:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2GRAY if pixels.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    small = cv2.resize(pixels, (max(1, width // factor), max(1, height // factor)),
                       interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(small)


def dark_centroid(gray: np.ndarray) -> tuple[int, int]:
    weights = 255 - gray
    # Reductions avoid full-frame float64 x/y grids.
    columns = weights.sum(axis=0, dtype=np.int64)
    rows = weights.sum(axis=1, dtype=np.int64)
    total = int(columns.sum())
    if total == 0:
        # Recorded CU output, with an explicit zero-mass branch replacing its
        # undefined division/conversion rather than inheriting that defect.
        return (-1, -1)
    return (int(columns @ np.arange(gray.shape[1], dtype=np.int64)) // total,
            int(rows @ np.arange(gray.shape[0], dtype=np.int64)) // total)


def gray_spectrum(gray: np.ndarray) -> np.ndarray:
    # The reference stores gray / 255 in a float32 complex matrix before DFT.
    # The lookup has exactly the same double-division then float32 rounding as
    # the reference, without a full-size float64 temporary on every frame.
    normalized = _GRAY_NORMALIZATION[gray]
    return cv2.dft(normalized, flags=cv2.DFT_COMPLEX_OUTPUT)


def correlation_shift(reference_fft: np.ndarray, current: np.ndarray) -> tuple[int, int]:
    spectrum = gray_spectrum(current)
    product = cv2.mulSpectrums(spectrum, reference_fft, flags=0, conjB=True)
    correlated = cv2.idft(product, flags=cv2.DFT_SCALE)
    from fdm._native import require_native
    return tuple(require_native().cu_motion_peak(correlated))


def correlation_peak_reference(correlated: np.ndarray) -> tuple[int, int]:
    """Independent NumPy reference for native peak-order regression."""
    # cvGet2D promoted each float32 component to double before squaring.
    real, imaginary = correlated[:, :, 0].astype(np.float64), correlated[:, :, 1].astype(np.float64)
    magnitude = real * real + imaginary * imaginary
    # Preserve the two legacy scans' deterministic tie order without creating
    # native threads: first maximum in each column parity; odd wins an equal
    # positive maximum. All-zero correlation keeps the initialized origin.
    best_value, y, x = 0.0, 0, 0
    for parity in (0, 1):
        part = magnitude[:, parity::2]
        if part.size == 0:
            continue
        row, column = np.unravel_index(int(part.argmax()), part.shape)
        value = float(part[row, column])
        if value > 0 and value >= best_value:
            best_value, y, x = value, int(row), int(column * 2 + parity)
    if x > correlated.shape[1] // 2:
        x -= correlated.shape[1]
    if y > correlated.shape[0] // 2:
        y -= correlated.shape[0]
    return x, y


def compare_geometry(reference: np.ndarray, current: np.ndarray, *, centroid_limit=2,
                     translation_limit=25, reference_fft=None, reference_centroid=None):
    """First CU comparison stage on already reduced gray images.

    This small pure helper also permits replay of the recorded 37-case probes
    without executing any original DLL or EXE.
    """
    if reference.shape != current.shape or reference.ndim != 2 or not reference.size:
        raise ValueError("视场比较需要同尺寸、非空的二维灰度图。")
    if reference.dtype != np.uint8 or current.dtype != np.uint8:
        raise TypeError("视场比较灰度图必须为 uint8。")
    center_a = dark_centroid(reference) if reference_centroid is None else reference_centroid
    center_b = dark_centroid(current)
    delta = (center_b[0] - center_a[0], center_b[1] - center_a[1])
    if max(abs(delta[0]), abs(delta[1])) > centroid_limit:
        return "centroid", delta, (0, 0)
    shift = correlation_shift(gray_spectrum(reference) if reference_fft is None else reference_fft, current)
    return ("translation" if max(abs(shift[0]), abs(shift[1])) > translation_limit else "stable"), delta, shift


def sobel_fraction(previous: np.ndarray, current: np.ndarray, *, legacy_signed=False) -> float:
    difference = cv2.absdiff(previous, current)
    # For uint8 D in [0,d], each 3x3 Sobel direction has magnitude <= 4d.
    # Replicated borders preserve that bound: d<=8 gives dx+dy<=64<71,
    # hence exactly zero changed pixels in both unsigned and legacy modes.
    # Other dtypes/empty inputs retain the original OpenCV behavior.
    if difference is not None and difference.dtype == np.uint8 and difference.size and int(difference.max()) <= 8:
        return 0.0
    dx = cv2.convertScaleAbs(cv2.Sobel(difference, cv2.CV_16S, 1, 0, ksize=3,
                                     borderType=cv2.BORDER_REPLICATE))
    dy = cv2.convertScaleAbs(cv2.Sobel(difference, cv2.CV_16S, 0, 1, ksize=3,
                                     borderType=cv2.BORDER_REPLICATE))
    edges = cv2.add(dx, dy)
    changed = edges > 70
    if legacy_signed:
        changed &= edges <= 127
    return float(np.count_nonzero(changed)) / edges.size


class FieldMotionDetector:
    def __init__(self, *, centroid_limit=2, translation_limit=25,
                 fraction_limit=0.04138815884775365, legacy_signed=False):
        self.centroid_limit = centroid_limit
        self.translation_limit = translation_limit
        self.fraction_limit = fraction_limit
        self.legacy_signed = legacy_signed
        self.reference = None
        self.previous = None
        self.reference_fft = None
        self.reference_centroid = (0, 0)
        self.source_key = None
        self._geometry_pixels = None
        self._geometry_result = None

    def reset(self) -> None:
        self.reference = self.previous = self.reference_fft = self.source_key = None
        self._geometry_pixels = self._geometry_result = None

    def set_reference(self, frame: FrameHandle) -> None:
        gray = reduced_gray(frame.pixels)
        self.set_reference_reduced(gray, (frame.source_id, frame.generation, frame.size))

    def set_reference_reduced(self, gray: np.ndarray, source_key) -> None:
        self.reference = gray
        self.previous = gray
        self.reference_centroid = dark_centroid(gray)
        self.reference_fft = gray_spectrum(gray)
        self.source_key = source_key
        self._geometry_pixels = self._geometry_result = None

    @property
    def retained_bytes(self) -> int:
        arrays = {id(array): array for array in (self.reference, self.previous, self.reference_fft,
                  self._geometry_pixels) if array is not None}
        return sum(array.nbytes for array in arrays.values())

    def compare(self, frame: FrameHandle) -> MotionResult:
        key = (frame.source_id, frame.generation, frame.size)
        return self.compare_reduced(reduced_gray(frame.pixels), key, frame.frame_id, frame.sequence)

    def compare_reduced(self, current: np.ndarray, key, frame_id: str, sequence: int) -> MotionResult:
        """Compare an owned small gray image after its full frame is released."""
        if self.reference is None or self.source_key != key:
            self.set_reference_reduced(current, key)
            return MotionResult(frame_id, sequence, True, "reference")
        # Exact equality proves that this frame has the already computed
        # geometry against the unchanged reference. A timestamp or hash is not
        # sufficient. Keep the original Sobel comparison against `previous`:
        # the first appearance rejection and its next stable frame differ.
        if (self._geometry_pixels is not None
                and np.array_equal(current, self._geometry_pixels)):
            reason, delta, shift = self._geometry_result
        else:
            reason, delta, shift = compare_geometry(self.reference, current,
                centroid_limit=self.centroid_limit, translation_limit=self.translation_limit,
                reference_fft=self.reference_fft, reference_centroid=self.reference_centroid)
            self._geometry_pixels = current
            self._geometry_result = reason, delta, shift
        if reason != "stable":
            return MotionResult(frame_id, sequence, False, reason, delta, shift)
        fraction = sobel_fraction(self.previous, current, legacy_signed=self.legacy_signed)
        self.previous = current
        return MotionResult(frame_id, sequence, fraction <= self.fraction_limit,
                            "stable" if fraction <= self.fraction_limit else "appearance", delta, shift, fraction)
