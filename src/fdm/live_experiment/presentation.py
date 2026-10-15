"""Stable presentation conventions shared by the category board and canvas."""

# CU6 CMsgAdmin2 constructor 0x00542610, COLORREF slots +0x44..+0x64.
# Convert Windows 0x00BBGGRR to RGB. Keep category numbers as a second cue:
# color alone is insufficient for identifying a measurement or material.
CU_CATEGORY_COLORS = (
    "#d60000", "#50ef16", "#163bed", "#f3cc0d", "#0e9611",
    "#00ccff", "#b252e8", "#fb44b8", "#fc7d00",
)


def category_color(session, operator_id, direction, material_id):
    operator = session.operators.get(operator_id)
    materials = operator.materials.get(direction, ()) if operator else ()
    index = next((index for index, material in enumerate(materials) if material.id == material_id), None)
    return CU_CATEGORY_COLORS[index] if index is not None and index < len(CU_CATEGORY_COLORS) else "#a6b4c5"


def category_number(session, operator_id, direction, material_id):
    operator = session.operators.get(operator_id)
    materials = operator.materials.get(direction, ()) if operator else ()
    return next((index + 1 for index, material in enumerate(materials) if material.id == material_id), None)
