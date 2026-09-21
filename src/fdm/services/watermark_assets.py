"""Immutable PNG assets shared by documents and their undo history."""
from __future__ import annotations

import hashlib
from pathlib import Path

from fdm.atomic_io import atomic_replace_file, staged_path_for
from fdm.watermark import WatermarkSpec


def verify_logo(data: bytes, digest: str) -> None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("水印 Logo 数据损坏或内容标识不匹配，请重新选择图片")


def hydrate_watermark_assets(documents, asset_root: Path) -> None:
    loaded: dict[str, bytes] = {}
    for document in documents:
        spec = document.watermark
        if spec is None or spec.kind != "logo" or not spec.logo_sha256 or document.is_digital_slide():
            continue
        try:
            if spec.logo_sha256 not in loaded:
                data = (asset_root / spec.asset_path).read_bytes()
                verify_logo(data, spec.logo_sha256)
                loaded[spec.logo_sha256] = data
            document.watermark_assets[spec.logo_sha256] = loaded[spec.logo_sha256]
        except (OSError, ValueError) as exc:
            document.watermark_asset_error = f"水印 Logo 无法加载：{exc}"


def stage_watermark_assets(payloads, documents, asset_root: Path, created: list[Path]) -> None:
    """Stage content-addressed files before the caller commits project JSON.

    Only new files are appended to ``created``; a failed save must never remove
    an asset already referenced by the previous project. Runtime PNG bytes stay
    on the document so undo remains usable after Save As or an external delete.
    """
    sources = {document.id: document for document in documents}
    for payload in payloads:
        spec = WatermarkSpec.from_dict(payload.get("watermark"))
        if spec is None or spec.kind != "logo" or not spec.logo_sha256 or payload.get("document_kind") == "digital_slide":
            continue
        destination = asset_root / spec.asset_path
        if destination.exists():
            verify_logo(destination.read_bytes(), spec.logo_sha256)
            continue
        document = sources.get(str(payload.get("id")))
        data = document.watermark_assets.get(spec.logo_sha256) if document is not None else None
        if data is None:
            raise ValueError(f"缺少水印 Logo：{payload.get('path', '')}；请重新选择 Logo 后保存")
        verify_logo(data, spec.logo_sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with staged_path_for(destination, suffix=".png") as temporary:
            temporary.write_bytes(data)
            atomic_replace_file(temporary, destination)
        created.append(destination)
