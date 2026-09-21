"""Length units offered by calibration tools and their physical scale.

Keep the historical ``um`` storage token. Resolving a legacy spelling is for
display/conversion only; it must not rewrite saved calibrations or signatures.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LengthUnit:
    code: str
    symbol: str
    name: str
    millimeters: float


DEFAULT_LENGTH_UNIT = "um"
LENGTH_UNITS = (
    LengthUnit("nm", "nm", "纳米", 1e-6),
    LengthUnit("um", "μm", "微米", 1e-3),
    LengthUnit("mm", "mm", "毫米", 1.0),
    LengthUnit("cm", "cm", "厘米", 10.0),
    LengthUnit("m", "m", "米", 1000.0),
)
_UNITS_BY_CODE = {unit.code: unit for unit in LENGTH_UNITS}
_MICROMETER_ALIASES = {"µm": "um", "μm": "um"}


def resolve_length_unit(code: str) -> LengthUnit | None:
    return _UNITS_BY_CODE.get(_MICROMETER_ALIASES.get(code, code))


def millimeters_per_unit(code: str) -> float | None:
    """Return None for legacy custom units whose physical scale is unknown."""
    unit = resolve_length_unit(code)
    return unit.millimeters if unit is not None else None
