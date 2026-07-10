from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Signal


AXIS_X = "00"
AXIS_Y = "01"
AXIS_Z = "02"
DIR_NEG = "00"
DIR_POS = "01"

FTDI_VID = 0x0403
FTDI_PID = 0x6001

try:
    import serial
    import serial.tools.list_ports

    _SERIAL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on optional runtime package
    serial = None
    _SERIAL_IMPORT_ERROR = exc


@dataclass(frozen=True, slots=True)
class MotionPortInfo:
    device: str
    description: str = ""
    manufacturer: str = ""
    serial_number: str = ""
    hwid: str = ""
    vid: int | None = None
    pid: int | None = None
    is_ftdi_motion: bool = False

    def display_label(self) -> str:
        tags: list[str] = []
        if self.is_ftdi_motion:
            tags.append("FTDI 0403:6001")
        if self.serial_number:
            tags.append(f"SN {self.serial_number}")
        if self.manufacturer:
            tags.append(self.manufacturer)
        suffix = " | ".join(tags)
        if suffix:
            return f"{self.device} - {self.description} ({suffix})"
        return f"{self.device} - {self.description}" if self.description else self.device


@dataclass(frozen=True, slots=True)
class MotionShutdownResult:
    reason: str
    closed: bool
    was_enabled: bool
    error: str | None = None


def build_motion_frame(axis: str, steps: int, direction: str) -> bytes:
    axis = str(axis)
    direction = str(direction)
    if axis not in {AXIS_X, AXIS_Y, AXIS_Z}:
        raise ValueError(f"invalid axis: {axis}")
    if direction not in {DIR_NEG, DIR_POS}:
        raise ValueError(f"invalid direction: {direction}")
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if steps > 0xFFFFFFFF:
        raise ValueError("steps too large")
    return (
        bytes.fromhex("AA55")
        + int(axis, 16).to_bytes(2, byteorder="big")
        + steps.to_bytes(4, byteorder="big")
        + bytes.fromhex("000000")
        + int(direction, 16).to_bytes(1, byteorder="big")
    )


def signed_delta(steps: int, direction: str) -> int:
    return int(steps) if direction == DIR_POS else -int(steps)


def direction_for_delta(delta: int) -> str:
    return DIR_POS if int(delta) >= 0 else DIR_NEG


def axis_name(axis: str) -> str:
    return {AXIS_X: "X", AXIS_Y: "Y", AXIS_Z: "Z"}.get(axis, axis)


def list_motion_ports() -> list[MotionPortInfo]:
    if serial is None:
        return []
    ports: list[MotionPortInfo] = []
    for info in serial.tools.list_ports.comports():
        vid = getattr(info, "vid", None)
        pid = getattr(info, "pid", None)
        ports.append(
            MotionPortInfo(
                device=str(getattr(info, "device", "") or ""),
                description=str(getattr(info, "description", "") or ""),
                manufacturer=str(getattr(info, "manufacturer", "") or ""),
                serial_number=str(getattr(info, "serial_number", "") or ""),
                hwid=str(getattr(info, "hwid", "") or ""),
                vid=vid,
                pid=pid,
                is_ftdi_motion=vid == FTDI_VID and pid == FTDI_PID,
            )
        )
    ports.sort(key=lambda item: (not item.is_ftdi_motion, item.device))
    return ports


def preferred_motion_port(ports: list[MotionPortInfo]) -> MotionPortInfo | None:
    for item in ports:
        if item.is_ftdi_motion:
            return item
    return ports[0] if ports else None


class MotionController(QObject):
    statusChanged = Signal(str)
    positionChanged = Signal(object)

    def __init__(self, *, port: str = "", baudrate: int = 256000, timeout: float = 0.2, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.enabled = False
        self.relative_pos: dict[str, int] = {AXIS_X: 0, AXIS_Y: 0, AXIS_Z: 0}
        self.soft_limits: dict[str, int] = {AXIS_X: 50000, AXIS_Y: 50000, AXIS_Z: 5000}
        self._serial: Any | None = None

    def serial_available(self) -> bool:
        return serial is not None

    def serial_import_error(self) -> str:
        return str(_SERIAL_IMPORT_ERROR or "")

    def close(self) -> None:
        if self._serial is not None:
            serial_handle = self._serial
            serial_handle.close()
            self._serial = None
            self.statusChanged.emit("串口已关闭")

    def shutdown(self, reason: str = "shutdown") -> MotionShutdownResult:
        was_enabled = bool(self.enabled)
        self.enabled = False
        error: str | None = None
        try:
            self.close()
        except Exception as exc:  # noqa: BLE001 - retain the handle so shutdown can be retried
            error = str(exc)
            self.statusChanged.emit(f"电机串口关闭失败: {error}")
        self.statusChanged.emit(f"电机输出已关闭 ({reason})")
        return MotionShutdownResult(
            reason=str(reason),
            closed=self._serial is None,
            was_enabled=was_enabled,
            error=error,
        )

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._ensure_open()
            self.enabled = True
            self.statusChanged.emit("电机输出已启用")
        else:
            self.enabled = False
            self.close()
            self.statusChanged.emit("电机输出已禁用")

    def set_soft_limit(self, axis: str, value: int) -> None:
        self.soft_limits[axis] = max(0, int(value))

    def reset_relative_zero(self, axes: set[str] | tuple[str, ...] | list[str] | None = None) -> None:
        if axes is None:
            axes = (AXIS_X, AXIS_Y, AXIS_Z)
        for axis in axes:
            if axis in {AXIS_X, AXIS_Y, AXIS_Z}:
                self.relative_pos[axis] = 0
        self.positionChanged.emit(dict(self.relative_pos))

    def check_available(self) -> tuple[bool, str]:
        if serial is None:
            return False, f"pyserial 未安装: {_SERIAL_IMPORT_ERROR}"
        if not self.port:
            return False, "未选择串口"
        if self._serial is not None and self._serial.is_open:
            return True, "串口已由本程序打开"
        try:
            probe = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            probe.close()
            return True, "串口可打开"
        except Exception as exc:
            return False, f"串口打开失败: {exc}"

    def move(self, axis: str, steps: int, direction: str, *, label: str = "") -> bool:
        frame = build_motion_frame(axis, steps, direction)
        delta = signed_delta(steps, direction)
        target = int(self.relative_pos.get(axis, 0)) + delta
        limit = int(self.soft_limits.get(axis, 0))
        if limit > 0 and abs(target) > limit:
            self.statusChanged.emit(f"已拦截软限位: {axis_name(axis)} 当前={self.relative_pos.get(axis, 0)} 目标={target} 限位=+/-{limit}")
            return False
        if not self.enabled:
            self.statusChanged.emit(f"DRY-RUN {label or axis_name(axis)} steps={steps} frame={frame.hex(' ').upper()}")
            return False
        self._ensure_open()
        written = self._serial.write(frame)
        if written != len(frame):
            raise IOError(f"串口写入不完整: {written}/{len(frame)} bytes")
        self.relative_pos[axis] = target
        self.positionChanged.emit(dict(self.relative_pos))
        self.statusChanged.emit(f"已发送 {axis_name(axis)} {delta:+d} steps")
        return True

    def move_to(self, axis: str, target: int, *, label: str = "") -> bool:
        current = int(self.relative_pos.get(axis, 0))
        delta = int(target) - current
        if delta == 0:
            return True
        return self.move(axis, abs(delta), direction_for_delta(delta), label=label)

    def _ensure_open(self) -> None:
        if serial is None:
            raise RuntimeError(f"pyserial 未安装: {_SERIAL_IMPORT_ERROR}")
        if not self.port:
            raise ValueError("未选择串口")
        if self._serial is not None and self._serial.is_open:
            return
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self.statusChanged.emit(f"串口已打开: {self.port} @ {self.baudrate}")
