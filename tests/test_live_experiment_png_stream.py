import hashlib
import io
import struct
import weakref
import zlib

import cv2
import numpy as np
import pytest

from fdm.live_experiment import png_stream
from fdm.live_experiment.png_stream import png_workspace_bytes, write_png_stream


def decode(data):
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("height,width", [(1, 1), (1, 17), (23, 1), (31, 43)])
def test_png_stream_round_trip_preserves_pixels_and_alpha(channels, height, width):
    shape = (height, width) if channels == 1 else (height, width, channels)
    pixels = np.random.default_rng(121).integers(0, 256, shape, dtype=np.uint8)
    original = pixels.copy()
    pixels.flags.writeable = False
    target = io.BytesIO()
    result = write_png_stream(pixels, target)
    data = target.getvalue()
    np.testing.assert_array_equal(decode(data), original)
    np.testing.assert_array_equal(pixels, original)
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert result.size_bytes == len(data)
    assert result.workspace_bytes == png_workspace_bytes(width, channels)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_png_stream_borrows_flipped_sliced_readonly_rows(channels):
    shape = (40, 60) if channels == 1 else (40, 60, channels)
    owner = np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)
    pixels = owner[::-2, ::-3]
    pixels.flags.writeable = False
    target = io.BytesIO()
    write_png_stream(pixels, target)
    np.testing.assert_array_equal(decode(target.getvalue()), pixels)


def test_png_stream_chunks_have_bounded_idat_payload_valid_crc_and_one_zlib_stream():
    pixels = np.random.default_rng(5).integers(0, 256, (256, 256, 4), dtype=np.uint8)
    target = io.BytesIO()
    write_png_stream(pixels, target)
    data = target.getvalue()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, kinds, compressed = 8, [], bytearray()
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        checksum = struct.unpack_from(">I", data, offset + 8 + length)[0]
        assert checksum == zlib.crc32(payload, zlib.crc32(kind)) & 0xffffffff
        kinds.append(kind)
        if kind == b"IDAT":
            assert 0 < length <= png_stream.IDAT_CHUNK_BYTES
            compressed.extend(payload)
        offset += length + 12
    assert kinds[0] == b"IHDR" and kinds[-1] == b"IEND"
    assert set(kinds[1:-1]) == {b"IDAT"}
    assert offset == len(data)
    inflater = zlib.decompressobj()
    scanlines = inflater.decompress(compressed) + inflater.flush()
    assert inflater.eof and not inflater.unused_data
    assert len(scanlines) == 256 * (256 * 4 + 1)
    assert all(scanlines[row * (256 * 4 + 1)] == 1 for row in range(256))
    np.testing.assert_array_equal(decode(data), pixels)


def test_png_stream_bounds_each_compressor_input_even_for_a_very_wide_row(monkeypatch):
    factory = zlib.compressobj
    lengths = []

    class Compressor:
        def __init__(self, *args, **kwargs):
            self.inner = factory(*args, **kwargs)

        def compress(self, data):
            lengths.append(len(data))
            return self.inner.compress(data)

        def flush(self, *args):
            return self.inner.flush(*args)

    monkeypatch.setattr(png_stream.zlib, "compressobj", Compressor)
    pixels = np.random.default_rng(3).integers(0, 256, (2, 25000, 3), dtype=np.uint8)
    target = io.BytesIO()
    write_png_stream(pixels, target)
    assert max(lengths) <= png_stream.COMPRESS_INPUT_BYTES
    assert len(lengths) > pixels.shape[0]
    np.testing.assert_array_equal(decode(target.getvalue()), pixels)


@pytest.mark.parametrize("reply", [None, 0, 3, True])
def test_png_stream_rejects_short_or_invalid_writes(reply):
    class ShortWriter:
        def write(self, data):
            return reply

    with pytest.raises(OSError, match="写入不完整"):
        write_png_stream(np.zeros((2, 2), np.uint8), ShortWriter())


def test_png_stream_io_error_does_not_retain_full_input_in_its_traceback():
    class FailedWriter:
        def write(self, data):
            raise OSError("disk unavailable")

    pixels = np.zeros((10, 20, 3), np.uint8)
    reference = weakref.ref(pixels)
    try:
        write_png_stream(pixels, FailedWriter())
    except OSError as error:
        failure = error
    else:
        pytest.fail("write failure was not propagated")
    del pixels
    # The test frame has no pixel owner; keeping the exception is deliberate.
    assert failure.__traceback__ is not None
    assert reference() is None


def test_png_stream_pre_and_mid_encode_cancellation_then_retry():
    pixels = np.zeros((12, 40, 3), np.uint8)
    for cancel_at in (1, 4):
        calls = 0
        target = io.BytesIO()

        def check():
            nonlocal calls
            calls += 1
            if calls == cancel_at:
                raise RuntimeError("cancelled")

        with pytest.raises(RuntimeError, match="cancelled"):
            write_png_stream(pixels, target, checkpoint=check)
        if cancel_at == 1:
            assert target.getvalue() == b""
    target = io.BytesIO()
    write_png_stream(pixels, target)
    np.testing.assert_array_equal(decode(target.getvalue()), pixels)


def test_png_stream_unicode_file_path(tmp_path):
    pixels = np.random.default_rng(22).integers(0, 256, (13, 23, 4), dtype=np.uint8)
    path = tmp_path / "测量原图-样本α.png"
    with path.open("wb", buffering=0) as target:
        result = write_png_stream(pixels, target)
    data = path.read_bytes()
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    np.testing.assert_array_equal(decode(data), pixels)


@pytest.mark.parametrize("pixels", [np.empty((0, 2), np.uint8), np.empty((2, 0), np.uint8),
    np.empty((2, 3), np.float32), np.empty((2, 3, 1), np.uint8), np.empty((2, 3, 2), np.uint8)])
def test_png_stream_invalid_input_is_rejected_before_writing(pixels):
    target = io.BytesIO()
    with pytest.raises(ValueError):
        write_png_stream(pixels, target)
    assert target.getvalue() == b""


def test_png_workspace_is_two_rows_plus_fixed_codec_allowance():
    assert png_workspace_bytes(6144, 3) == 2 * 1024**2 + 2 * 6144 * 3 + 1
    for width, channels in ((True, 3), (0, 3), (1, 2), (1, False)):
        with pytest.raises(ValueError):
            png_workspace_bytes(width, channels)
