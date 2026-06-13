from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.geometry import Line, Point
from fdm.models import ImageDocument, Measurement, ProjectGroupTemplate, ProjectState, new_id
from fdm.services.group_manager import GroupManager


PALETTE = ["#1F7A8C", "#E07A5F", "#81B29A"]
PROJECT_VERSION = "test"


def _document(path: str) -> ImageDocument:
    document = ImageDocument(id=new_id("image"), path=path, image_size=(320, 240))
    document.initialize_runtime_state()
    return document


def _measurement(document: ImageDocument, group_id: str | None) -> Measurement:
    return Measurement(
        id=new_id("meas"),
        image_id=document.id,
        fiber_group_id=group_id,
        mode="manual",
        line_px=Line(Point(10, 10), Point(90, 10)),
    )


class GroupManagerTests(unittest.TestCase):
    def test_group_rows_include_current_and_project_counts(self) -> None:
        current = _document("/tmp/group_current.png")
        cotton = current.create_group(color="#1F7A8C", label="棉")
        current.set_active_group(cotton.id)
        current.add_measurement(_measurement(current, cotton.id))

        other = _document("/tmp/group_other.png")
        other_cotton = other.create_group(color="#E07A5F", label="棉")
        other.add_measurement(_measurement(other, other_cotton.id))
        other.set_active_group(None)
        other.add_measurement(_measurement(other, None))

        manager = GroupManager(ProjectState(version=PROJECT_VERSION, documents=[current, other]), color_palette=PALETTE)

        rows = manager.group_rows(current, default_uncategorized_color="#98A2B3")

        self.assertEqual(
            [(row.label, row.current_count, row.project_count, row.selected) for row in rows],
            [("1 棉", 1, 2, True)],
        )
        self.assertEqual(manager.project_uncategorized_measurement_count(current), 1)

    def test_uncategorized_row_uses_project_counts(self) -> None:
        current = _document("/tmp/group_uncategorized_current.png")
        current.set_active_group(None)
        other = _document("/tmp/group_uncategorized_other.png")
        other.add_measurement(_measurement(other, None))
        manager = GroupManager(ProjectState(version=PROJECT_VERSION, documents=[current, other]), color_palette=PALETTE)

        rows = manager.group_rows(current, default_uncategorized_color="#98A2B3")

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].group_id)
        self.assertEqual(rows[0].current_count, 0)
        self.assertEqual(rows[0].project_count, 1)
        self.assertTrue(rows[0].selected)

    def test_ensure_document_named_group_merges_duplicates_and_syncs_color(self) -> None:
        document = _document("/tmp/group_merge.png")
        first = document.create_group(color="#111111", label="棉")
        duplicate = document.create_group(color="#222222", label="棉")
        document.add_measurement(_measurement(document, duplicate.id))
        manager = GroupManager(ProjectState(version=PROJECT_VERSION, documents=[document]), color_palette=PALETTE)

        group, changed = manager.ensure_document_named_group(
            document,
            label="棉",
            color="#E07A5F",
            activate=True,
            sync_color=True,
        )

        self.assertTrue(changed)
        self.assertIsNotNone(group)
        self.assertEqual(group.id, first.id)
        self.assertEqual(group.color, "#e07a5f")
        self.assertEqual(len(document.groups_by_label("棉")), 1)
        self.assertEqual(document.active_group_id, first.id)
        self.assertEqual(len(first.measurement_ids), 1)

    def test_area_inference_colors_prefer_template_then_existing_then_palette(self) -> None:
        document = _document("/tmp/group_colors.png")
        document.create_group(color="#ABCDEF", label="莱赛尔")
        project = ProjectState(
            version=PROJECT_VERSION,
            documents=[document],
            project_group_templates=[ProjectGroupTemplate(label="棉", color="#123456")],
        )
        manager = GroupManager(project, color_palette=PALETTE)

        colors = manager.resolve_area_inference_group_colors(["棉", "莱赛尔", "粘纤"])

        self.assertEqual(colors["棉"], "#123456")
        self.assertEqual(colors["莱赛尔"], "#ABCDEF")
        self.assertEqual(colors["粘纤"], PALETTE[1])


if __name__ == "__main__":
    unittest.main()
