"""Read-only Olympus DSX / POIR / MPOIR ingestion.

Descriptors contain no pixels. ZIP inflation may still be needed to seek OIR
metadata. Only selected channels are decoded; originals are never modified.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterator
import hashlib
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import tifffile

from fdm.image_processing_models import DisplayTransform, RasterSemantic
from fdm.models import Calibration
from fdm.raster import RasterPlane
from fdm.services.raster_io import numpy_to_raster_plane
from fdm.services._olympus_oir import OirReader, find, value, number

DEVICE_SUFFIXES = frozenset({".dsx", ".poir", ".mpoir"})
CHANNEL_LABELS = {"intensity": "激光强度", "color": "彩图", "height": "高度图"}
COMMON_CHANNELS = ("intensity", "color")


class DeviceReadCancelled(Exception):
    pass


def cancellation_check(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise DeviceReadCancelled()


def source_fingerprint(path: str | Path, cancelled=None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            cancellation_check(cancelled)
            digest.update(chunk)
    cancellation_check(cancelled)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeviceChannel:
    source_path: str
    source_sha256: str
    dataset_id: str
    dataset_label: str
    kind: str
    width: int
    height: int
    members: tuple[str, ...] = ()
    channel_ids: tuple[str, ...] = ()
    page_index: int | None = None
    calibration: Calibration | None = None
    calibration_issue: str = ""
    calibration_evidence: dict | None = None
    display_transform: DisplayTransform | None = None
    device: str = ""
    layout: dict | None = None

    @property
    def identity(self) -> str:
        parts = (self.source_sha256, self.dataset_id, self.kind, *self.members,
                 *self.channel_ids, str(self.page_index))
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    @property
    def display_name(self) -> str:
        parts = [Path(self.source_path).stem]
        if self.dataset_label:
            parts.append(self.dataset_label)
        return " · ".join([*parts, CHANNEL_LABELS[self.kind]])

    @property
    def semantic(self) -> RasterSemantic:
        return {"color": RasterSemantic.COLOR, "intensity": RasterSemantic.INTENSITY,
                "height": RasterSemantic.HEIGHT}[self.kind]

    def source_metadata(self) -> dict:
        return {
            "version": 1, "source_path": self.source_path,
            "source_sha256": self.source_sha256, "identity": self.identity,
            "dataset_id": self.dataset_id, "dataset_label": self.dataset_label,
            "kind": self.kind, "members": list(self.members),
            "channel_ids": list(self.channel_ids), "page_index": self.page_index,
            "device": self.device, "display_name": self.display_name,
            "dimensions": [self.width, self.height], "layout": self.layout,
            "calibration_evidence": self.calibration_evidence,
            "calibration_issue": self.calibration_issue,
            "original_display_transform": self.display_transform.to_dict() if self.display_transform else None,
        }


def default_selection(channels: list[DeviceChannel], preferred=("intensity",)) -> list[DeviceChannel]:
    groups: dict[tuple[str, str], list[DeviceChannel]] = {}
    for channel in channels:
        groups.setdefault((channel.source_sha256, channel.dataset_id), []).append(channel)
    selected = []
    for group in groups.values():
        choices = [c for c in group if c.kind in COMMON_CHANNELS and c.kind in preferred]
        if not choices:
            choices = [c for c in group if c.kind == "intensity"] or [c for c in group if c.kind == "color"]
        selected.extend(choices)
    return selected


def _calibration(x, y, label, issue=""):
    if issue:
        return None, issue
    try:
        return Calibration(mode="device", pixels_per_unit=1.0 / float(x),
                           pixels_per_unit_y=1.0 / float(y), unit="µm", source_label=label), ""
    except (TypeError, ValueError, ZeroDivisionError):
        return None, "标定缺失或无效，需重新标定"


@contextmanager
def _open_oir(path: str, members: tuple[str, ...]) -> Iterator[object]:
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(path))
        for member in members[:-1]:
            stream = stack.enter_context(archive.open(member))
            archive = stack.enter_context(zipfile.ZipFile(stream))
        yield stack.enter_context(archive.open(members[-1]))


def _oir_channels(path, fingerprint, dataset, label, members, layout, cancelled):
    with _open_oir(path, members) as stream:
        reader = OirReader(stream, lambda: cancellation_check(cancelled))
        if reader.width <= 0 or reader.height <= 0:
            raise ValueError("无效的 OIR 图像尺寸")
        channels = reader.channels
        if str(reader.device).lower() == "camera":
            by_kind = {c["kind"]: c for c in channels}
            if set(by_kind) != {"red", "green", "blue"} or len(channels) != 3:
                raise ValueError("不支持的 Camera 通道结构")
            channel_groups = [("color", [by_kind[k] for k in ("red", "green", "blue")])]
        elif str(reader.device).lower() == "lsm":
            channel_groups = [(c["kind"], [c]) for c in channels if c["kind"] in CHANNEL_LABELS]
        else:
            raise ValueError(f"不支持的 OIR 设备类型：{reader.device}")
        result = []
        for kind, components in channel_groups:
            for component in components:
                reader.pixel_layout(component)
            evidence = components[0]["calibration"]
            issue = ""
            if evidence["missing_factory_calibration"]:
                issue = "缺少工厂标定系数，需核验"
            if evidence["nonidentity_user_calibration_unverified"]:
                issue = "非默认用户标定系数尚未验证"
            if any(evidence["units"][axis] != "MICRO_METER" for axis in ("x", "y")):
                issue = "标定单位尚未支持"
            if any(c["calibration"] != evidence for c in components):
                issue = "RGB 分量标定不一致"
            scale = evidence["factory_corrected_um"]
            calibration, issue = _calibration(scale["x"], scale["y"], "OLS 内置标定（含工厂校正）", issue)
            transform = None
            guid = components[0]["guid"]
            if kind != "color" and guid in reader.luts:
                lut = np.frombuffer(bytes.fromhex(value(reader.luts[guid], "data")), dtype=np.uint8).reshape(-1, 4)
                low, high = reader.scale_ranges.get(guid, (0, len(lut) - 1))
                transform = DisplayTransform(black_point=low, white_point=high, lut_rgb=lut[:, :3].tobytes())
            result.append(DeviceChannel(
                path, fingerprint, dataset, label, kind, reader.width, reader.height,
                members=members, channel_ids=tuple(c["guid"] for c in components),
                calibration=calibration, calibration_issue=issue,
                calibration_evidence=evidence, display_transform=transform,
                device="OLS5000", layout=layout,
            ))
        return result


def _inspect_dsx(path, fingerprint, cancelled):
    result = []
    with tifffile.TiffFile(path) as tiff:
        root = ET.fromstring(tiff.pages[0].description)
        issue = ""
        correction = {a: value(root, f"ImageCommonCalibrationValue{a.upper()}") for a in "xyz"}
        if any(v is not None and number(v) != 1000000 for v in correction.values()):
            issue = "非默认 DSX 公共标定系数尚未验证"
        for i, page in enumerate(tiff.pages):
            cancellation_check(cancelled)
            kind = "color" if i == 0 and value(root, "emImageData") == "Color" else page.description.lower()
            if kind not in {"color", "height"}:
                continue
            orientation = page.tags.get("Orientation")
            if orientation is not None and orientation.value != 1:
                raise ValueError("尚不支持此 DSX 图像方向")
            if (kind == "color" and (len(page.shape) != 3 or page.shape[2] != 3 or page.dtype != np.uint8)) or (
                kind == "height" and (len(page.shape) != 2 or page.dtype != np.uint16)
            ):
                raise ValueError("尚不支持此 DSX 像素布局")
            prefix = kind.title()
            raw = {a: value(root, f"{prefix}ImageData/{prefix}DataPerPixel{a.upper()}") for a in "xyz"}
            scale = {a: number(v) / 1e6 if number(v) is not None else None for a, v in raw.items()}
            cal, problem = _calibration(scale["x"], scale["y"], "DSX 图层内置标定", issue)
            result.append(DeviceChannel(path, fingerprint, "main", "", kind,
                int(page.shape[1]), int(page.shape[0]), page_index=i, calibration=cal,
                calibration_issue=problem, device="DSX1000",
                calibration_evidence={"raw_pm": raw, "pixel_size_um": scale, "common_calibration": correction}))
    return result


def inspect_source(path: str | Path, *, cancelled=None) -> list[DeviceChannel]:
    path = str(Path(path).expanduser().resolve())
    suffix = Path(path).suffix.lower()
    if suffix not in DEVICE_SUFFIXES:
        raise ValueError("不支持的设备文件格式")
    fingerprint = source_fingerprint(path, cancelled)
    if suffix == ".dsx":
        result = _inspect_dsx(path, fingerprint, cancelled)
    else:
        result = []
        with zipfile.ZipFile(path) as archive:
            datasets = [((), "main", "", None)]
            if suffix == ".mpoir":
                if "matl.omp2info" not in archive.namelist():
                    raise ValueError("MPOIR 缺少点位布局")
                root = ET.fromstring(archive.read("matl.omp2info"))
                datasets = []
                for index, group in enumerate(root.findall("./{*}group"), 1):
                    if value(group, "stitching") != "false":
                        raise ValueError("尚不支持 MPOIR 内的拼接点位布局")
                    for area in group.findall("./{*}area"):
                        member = value(area, "image")
                        if not member or Path(member).suffix.lower() != ".poir":
                            continue
                        if member not in archive.namelist():
                            raise ValueError(f"缺少点位文件：{member}")
                        coordinates = find(group, "regionInfo/coordinates")
                        datasets.append(((member,), member, f"点位 {index}", {
                            "group_id": group.attrib.get("objectId"),
                            "area_id": area.attrib.get("id"),
                            "coordinates_nm": dict(coordinates.attrib) if coordinates is not None else {},
                            "stitching": value(group, "stitching"),
                        }))
            for prefix, dataset, label, layout in datasets:
                cancellation_check(cancelled)
                with ExitStack() as stack:
                    nested = archive
                    if prefix:
                        stream = stack.enter_context(archive.open(prefix[0]))
                        nested = stack.enter_context(zipfile.ZipFile(stream))
                    names = [n for n in nested.namelist() if Path(n).suffix.lower() == ".oir"]
                for name in names:
                    result.extend(_oir_channels(path, fingerprint, dataset, label, (*prefix, name), layout, cancelled))
    if not result:
        raise ValueError("文件中没有受支持的测量图像")
    if len({c.identity for c in result}) != len(result):
        raise ValueError("文件包含重复通道身份")
    return result


def read_calibration(channel: DeviceChannel) -> tuple[Calibration | None, dict]:
    return (channel.calibration.clone() if channel.calibration else None,
            dict(channel.calibration_evidence or {}))


def read_channel(channel: DeviceChannel, *, cancelled=None, verify_source=True) -> RasterPlane:
    cancellation_check(cancelled)
    if verify_source and source_fingerprint(channel.source_path, cancelled) != channel.source_sha256:
        raise ValueError("设备文件已改变，请重新选择通道")
    if channel.page_index is not None:
        with tifffile.TiffFile(channel.source_path) as tiff:
            array = tiff.pages[channel.page_index].asarray()
    else:
        with _open_oir(channel.source_path, channel.members) as stream:
            reader = OirReader(stream, lambda: cancellation_check(cancelled))
            by_guid = {c["guid"]: c for c in reader.channels}
            planes = [reader.plane(by_guid[guid]) for guid in channel.channel_ids]
            array = np.stack(planes, axis=-1) if channel.kind == "color" else planes[0]
    cancellation_check(cancelled)
    plane = numpy_to_raster_plane(array)
    if (plane.width, plane.height) != (channel.width, channel.height):
        raise ValueError("图像尺寸与通道清单不一致")
    return plane
