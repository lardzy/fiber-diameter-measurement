"""Position-only acquisition anchors in logical coordinates, snapped to pixels."""

ALIGNMENT_FACTORS = {
    "top_left": (0.0, 0.0), "top": (0.5, 0.0), "top_right": (1.0, 0.0),
    "left": (0.0, 0.5), "center": (0.5, 0.5), "right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0), "bottom": (0.5, 1.0), "bottom_right": (1.0, 1.0),
}
ALIGNMENTS = ("free", *ALIGNMENT_FACTORS)


def validate_alignment(alignment):
    if alignment not in ALIGNMENTS:
        raise ValueError("未知画面对齐方向")
    return alignment


def aligned_offset(alignment, viewport, image_extent, dpr=1.0):
    """The same edge formula covers blank margins and oversized images."""
    factors = ALIGNMENT_FACTORS[alignment]
    return tuple(round((view - image) * factor * dpr) / dpr
                 for view, image, factor in zip(viewport, image_extent, factors))
