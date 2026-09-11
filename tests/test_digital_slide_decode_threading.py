"""Qt image-plugin locks must not depend on Python virtual device callbacks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtGui import QImage
from shiboken6 import createdByPython

from fdm.services import digital_slide_store as slide_module
from fdm.services.digital_slide_store import (
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
    image_bytes_to_qimage,
    qimage_to_image_bytes,
)


@pytest.mark.parametrize("codec", ["png", "jpeg"])
@pytest.mark.parametrize("scaled", [False, True])
def test_native_and_lod_decoders_only_pass_cpp_owned_devices_to_qt(
    tmp_path, monkeypatch, codec, scaled
):
    source = QImage(80, 60, QImage.Format.Format_RGB32)
    source.fill(0xFF305070)
    original_reader = slide_module.QImageReader
    devices = []

    def checked_reader(device, *args):
        # Calling a Python-created device's virtual methods from the decoder
        # needs the GIL while Qt holds its plugin mutex. A C++ device has no
        # such callback, even though PySide exposes a wrapper for the caller.
        assert not createdByPython(device)
        assert device.isOpen()
        devices.append(device)
        return original_reader(device, *args)

    monkeypatch.setattr(slide_module, "QImageReader", checked_reader)
    if scaled:
        store = DigitalSlideStore.create(
            tmp_path / "decode.fdmslide",
            DigitalSlideManifest(1, 80, 60, 80, 60, [0]),
        )
        try:
            store.write_tile(
                DigitalSlideTile(z_index=0, x=0, y=0, width=80, height=60),
                source,
                codec=codec,
            )
            tile_id = store.list_tile_descriptors(z_index=0)[0].tile_id
            result = store.read_tile_image_scaled(tile_id, width=40, height=30)
            assert (result.width(), result.height()) == (40, 30)
        finally:
            store.close()
    else:
        payload = qimage_to_image_bytes(source, codec=codec)
        result = image_bytes_to_qimage(payload, codec=codec)
        assert result.size() == source.size()
    assert not result.isNull()
    assert devices


def test_background_decoding_and_gui_image_loading_complete_without_deadlock(tmp_path):
    # Isolate the native lock-order regression: a failure must time out this
    # child, never freeze the rest of the desktop regression suite.
    script = tmp_path / "concurrent_decode.py"
    script.write_text(
        """
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from fdm.services.digital_slide_store import image_bytes_to_qimage, qimage_to_image_bytes

app = QApplication([])
image = QImage(128, 128, QImage.Format.Format_RGB32)
image.fill(0xff3070a0)
payload = qimage_to_image_bytes(image)
barrier = Barrier(3)

def decode():
    barrier.wait()
    for _ in range(600):
        assert not image_bytes_to_qimage(payload).isNull()

with ThreadPoolExecutor(max_workers=2) as pool:
    jobs = [pool.submit(decode) for _ in range(2)]
    barrier.wait()
    for _ in range(600):
        # This GIL-holding Qt path also acquires the image-plugin mutex, like
        # native icon loading during a GUI paint or stylesheet refresh.
        target = QImage()
        assert target.loadFromData(payload)
        app.processEvents()
    for job in jobs:
        job.result()
print("completed")
""",
        encoding="utf-8",
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = str(Path(slide_module.__file__).resolve().parents[2])
    completed = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert "completed" in completed.stdout
