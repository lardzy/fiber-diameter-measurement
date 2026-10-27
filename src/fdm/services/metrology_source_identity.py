"""Metadata-only identity for a registered slide, read off the GUI thread."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import Event, Thread
from typing import Callable, Iterable

from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore, DigitalSlideTileDescriptor


def professional_slide_source_revision(manifest: DigitalSlideManifest, descriptors: Iterable[DigitalSlideTileDescriptor], *, ordered: bool = False) -> str:
    """Hash registration/position identity, deliberately excluding image bytes.

    Tile order and the path of a portable copy are irrelevant. This fingerprint
    identifies geometry and acquisition metadata, not pixel-content equality.
    """
    rows = ((tile.tile_id, tile.z_index, tile.x, tile.y, tile.width, tile.height,
             tile.stage_x, tile.stage_y, tile.focus_z) for tile in descriptors)
    if not ordered:
        rows = iter(sorted(rows, key=lambda row: (row[1], row[3], row[2], row[0])))
    payload = json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(payload.encode("utf-8"))
    # The worker streams one focus plane at a time in store order (z/y/x/id),
    # avoiding a second all-slide descriptor table or a large JSON buffer.
    for row in rows:
        digest.update(b"\n")
        digest.update(json.dumps(row, separators=(",", ":"), allow_nan=False).encode("ascii"))
    return "fdmslide-geometry-v1:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProfessionalSourceRevisionResult:
    generation: int
    document_id: str
    revision: str | None = None
    error: str | None = None


class ProfessionalSourceRevisionWorker:
    """One cancellable, thread-affine, read-only SQLite snapshot."""

    def __init__(self, path: str | Path, manifest: DigitalSlideManifest, *, generation: int,
                 document_id: str, callback: Callable[[ProfessionalSourceRevisionResult], None]) -> None:
        self._path = Path(path)
        self._manifest = json.loads(json.dumps(manifest.to_dict(), ensure_ascii=False, allow_nan=False))
        self._generation = generation
        self._document_id = document_id
        self._callback = callback
        self._cancel = Event()
        self._thread = Thread(target=self._run, name=f"fdm-metrology-source-{document_id}", daemon=True)
        self._thread.start()

    def close(self, *, timeout: float = 0.2) -> None:
        self._cancel.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        store = DigitalSlideStore(self._path)
        result = None
        try:
            if self._cancel.is_set():
                return
            store.open_read_only()
            connection = store._connection()  # Thread-owned; no GUI SQLite access.
            connection.set_progress_handler(lambda: int(self._cancel.is_set()), 1000)
            connection.execute("BEGIN")
            manifest = store.read_manifest()
            if manifest.to_dict() != self._manifest:
                raise ValueError("切片来源在挂载期间发生变化，请重新打开后再测量")
            def descriptors():
                for focus_index in range(len(manifest.focus_levels)):
                    if self._cancel.is_set():
                        raise RuntimeError("cancelled")
                    for tile in store.list_tile_descriptors(z_index=focus_index):
                        if self._cancel.is_set():
                            raise RuntimeError("cancelled")
                        yield tile
            if not self._cancel.is_set():
                revision = professional_slide_source_revision(manifest, descriptors(), ordered=True)
                result = ProfessionalSourceRevisionResult(self._generation, self._document_id, revision)
        except Exception as error:
            if not self._cancel.is_set():
                result = ProfessionalSourceRevisionResult(self._generation, self._document_id, error=f"无法核对切片来源：{error}")
        finally:
            store.close()
        if result is not None and not self._cancel.is_set():
            self._callback(result)
