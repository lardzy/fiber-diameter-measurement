from __future__ import annotations

from pathlib import Path
from typing import Any


MODEL_METADATA_VERSION = 1

LABEL_ALIASES: dict[str, str] = {
    "粘": "粘纤",
    "莱": "莱赛尔",
    "莫": "莫代尔",
}

# class_names is the actual class-index order used by the trusted weight file.
# It intentionally differs from the display name for the two historically
# reversed binary models.
MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model_name": "粘纤-莱赛尔",
        "model_file": "b_v1_1.3.pth",
        "class_names": ("莱赛尔", "粘纤"),
        "sha256": "8d02316e1fdeacebee09143c20232bacc0c775dc529d6c612375a7d8f5cd65f0",
    },
    {
        "model_name": "棉-粘纤",
        "model_file": "b_cv_1.3.pth",
        "class_names": ("棉", "粘纤"),
        "sha256": "c0a12a2526d58cb0192133e7847381aabefcc31df75faec6812b80b1880edde1",
    },
    {
        "model_name": "棉-莱赛尔",
        "model_file": "b_c1_1.3.pth",
        "class_names": ("莱赛尔", "棉"),
        "sha256": "81d49d18b9fe34c690bf2cdf1508f126d80391cdbfbb1ebeb9d5105efee70a94",
    },
    {
        "model_name": "棉-莫代尔",
        "model_file": "b_cm_1.3.pth",
        "class_names": ("棉", "莫代尔"),
        "sha256": "7d3b550f6b1bedab56bc4d1ef06d74ac01d7886507f59309936110b8b0a87c4a",
    },
    {
        "model_name": "棉-再生纤维素纤维",
        "model_file": "b_cc_1.3.pth",
        "class_names": ("棉", "再生纤维素纤维"),
        "sha256": "a1dbfac0c399def813652ec81f2794a84885b4679a76240ee3e3b9c25888506f",
    },
    {
        "model_name": "棉-粘-莱-莫",
        "model_file": "b_cvlm_1.3.pth",
        "class_names": ("棉", "粘纤", "莱赛尔", "莫代尔"),
        "sha256": "a9e2b45926174f3e9662b524f13a866a7adf049719e24b9455395657ddf3074c",
    },
)


def normalize_label(label: str) -> str:
    token = str(label or "").strip()
    if not token:
        return "未分类"
    return LABEL_ALIASES.get(token, token)


def parse_model_classes(model_name: str) -> list[str]:
    classes: list[str] = []
    for item in str(model_name or "").split("-"):
        normalized = normalize_label(item)
        if normalized not in classes:
            classes.append(normalized)
    return classes or ["未分类"]


def find_model_spec(*, model_file: str = "", model_name: str = "") -> dict[str, Any] | None:
    file_key = Path(str(model_file or "").strip()).name.casefold()
    name_key = str(model_name or "").replace(" ", "").strip().casefold()
    for spec in MODEL_SPECS:
        if file_key and str(spec["model_file"]).casefold() == file_key:
            return spec
    for spec in MODEL_SPECS:
        candidate = str(spec["model_name"]).replace(" ", "").strip().casefold()
        if name_key and candidate == name_key:
            return spec
    return None


def resolve_model_classes(*, model_name: str, model_file: str) -> tuple[tuple[str, ...], bool]:
    spec = (
        find_model_spec(model_file=model_file)
        if str(model_file or "").strip()
        else find_model_spec(model_name=model_name)
    )
    if spec is not None:
        return tuple(str(item) for item in spec["class_names"]), True
    return tuple(parse_model_classes(model_name)), False
