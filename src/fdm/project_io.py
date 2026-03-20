from __future__ import annotations

from pathlib import Path
import json

from fdm.models import ProjectState


class ProjectIO:
    """Read and write lightweight project files."""

    @staticmethod
    def save(project: ProjectState, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = project.to_dict()
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def load(path: str | Path) -> ProjectState:
        input_path = Path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return ProjectState.from_dict(payload)
