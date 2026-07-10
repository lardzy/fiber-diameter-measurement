from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fdm.atomic_io import _atomic_write, atomic_copy_file, atomic_write_bytes, atomic_write_json


def test_production_json_serialization_explicitly_rejects_non_finite_values() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = [
        *(project_root / "src" / "fdm").rglob("*.py"),
        *(project_root / "runtime" / "area-infer" / "app").rglob("*.py"),
        *(project_root / "scripts").glob("*.py"),
    ]
    missing_contract: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            and node.func.attr in {"dump", "dumps"}
        ):
            allow_nan = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "allow_nan"),
                None,
            )
            if not (
                isinstance(allow_nan, ast.Constant)
                and allow_nan.value is False
            ):
                relative = path.relative_to(project_root).as_posix()
                missing_contract.append(f"{relative}:{call.lineno}")

    assert missing_contract == []


def test_atomic_write_replaces_complete_file(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"old")

    result = atomic_write_bytes(target, b"new payload")

    assert result == target
    assert target.read_bytes() == b"new payload"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_preserves_original_when_replace_fails(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"old payload")

    with patch("fdm.atomic_io.os.replace", side_effect=OSError("injected replace failure")):
        with pytest.raises(OSError, match="injected replace failure"):
            atomic_write_bytes(target, b"new payload")

    assert target.read_bytes() == b"old payload"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_preserves_original_when_writer_or_flush_fails(tmp_path: Path) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old payload")

    def failing_writer(stream) -> None:
        stream.write(b"partial")
        raise OSError("injected writer failure")

    with pytest.raises(OSError, match="writer failure"):
        _atomic_write(target, failing_writer)
    assert target.read_bytes() == b"old payload"
    assert list(tmp_path.iterdir()) == [target]

    class FlushFailureStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def write(self, payload: bytes) -> int:
            return len(payload)

        def flush(self) -> None:
            raise OSError("injected flush failure")

        def fileno(self) -> int:
            return -1

    with patch("fdm.atomic_io.Path.open", return_value=FlushFailureStream()):
        with pytest.raises(OSError, match="flush failure"):
            atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old payload"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_preserves_original_when_file_fsync_fails(tmp_path: Path) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old payload")

    with patch("fdm.atomic_io.os.fsync", side_effect=OSError("injected fsync failure")):
        with pytest.raises(OSError, match="fsync failure"):
            atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old payload"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_json_rejects_non_finite_values_before_replacing(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"valid": true}', encoding="utf-8")

    with pytest.raises(ValueError):
        atomic_write_json(target, {"invalid": float("nan")})

    assert json.loads(target.read_text(encoding="utf-8")) == {"valid": True}
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_copy_preserves_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "import.json"
    target = tmp_path / "current" / "settings.json"
    source.write_bytes(b'{\n  "theme": "dark"\n}')

    result = atomic_copy_file(source, target)

    assert result == target
    assert target.read_bytes() == source.read_bytes()
    assert list(target.parent.iterdir()) == [target]
