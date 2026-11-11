"""Actual Qt selection controls, draft boundaries and reusable combinations."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt, QModelIndex, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from fdm.live_experiment.controller import LiveExperimentController
from fdm.live_experiment.material_catalog import MaterialCatalog
from fdm.live_experiment.material_combinations import MaterialCombinations
from fdm.live_experiment.models import ExperimentSession, MaterialSnapshot
from fdm.ui.material_catalog import MaterialCatalogDialog, TrialMaterialsDialog
from fdm.ui.material_components import MaterialParameterEditor


def trial():
    value = ExperimentSession(save_images=False)
    value.add_operator("操作员 A")
    value.set_materials(list(MaterialCatalog().materials()[:2]))
    value.establish_history_baseline()
    return value


@pytest.fixture
def picker(tmp_path):
    dialog = TrialMaterialsDialog(trial(), catalog=MaterialCatalog(), combinations_path=tmp_path / "组合.json")
    yield dialog
    dialog.close()
    dialog.deleteLater()


def cell(dialog, key, column=0):
    for row in range(dialog.table.rowCount()):
        if dialog.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == key:
            return dialog.table.item(row, column)
    raise AssertionError(key)


def test_real_checkboxes_three_choices_one_confirmation_and_cancel_is_readonly(picker):
    original = picker.session.to_dict()
    picker.resize(940, 620)
    picker.show()
    QApplication.processEvents()
    for key in ("cotton", "nylon", "alpaca"):
        picker.search.setText(picker.catalog.get(key).name)
        item = cell(picker, key)
        assert not picker.table.isRowHidden(item.row())
        picker.table.scrollToItem(item)
        rectangle = picker.table.visualItemRect(item)
        QTest.mouseClick(picker.table.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(rectangle.left() + 10, rectangle.center().y()))
        QApplication.processEvents()
        assert key in {m.id for m in picker.materials()}
    assert [m.id for m in picker.materials()][-3:] == ["cotton", "nylon", "alpaca"]
    picker.accept()
    assert picker.result() == QDialog.DialogCode.Accepted
    assert picker.session.to_dict() == original  # owner commits the returned draft once


def test_search_aliases_never_reorders_existing_or_replaces_trial_parameters(picker):
    item = replace(picker.catalog.get("cotton"), aliases=("测试别名",))
    picker.catalog.upsert(item)
    picker._refresh()
    picker.search.setText("测试别名")
    assert [picker.table.item(r, 1).text() for r in range(picker.table.rowCount()) if not picker.table.isRowHidden(r)] == ["棉"]
    picker.table.sortItems(2, Qt.SortOrder.DescendingOrder)
    assert [m.id for m in picker.materials()] == ["wool", "cashmere"]
    picker._materials[0] = replace(picker._materials[0], density=1.23456789)
    assert picker.set_selected("wool", True)
    assert picker.materials()[0].density == 1.23456789


def test_same_region_editor_roundtrips_full_precision_and_draft_cancel(picker):
    exact = replace(picker._materials[0], density=1.23456789012345, lower_limit=0.123456789012345,
                    upper_limit=99.9876543210987)
    picker._materials[0] = exact
    picker._refresh(exact.id)
    picker.edit()
    assert picker.right_stack.currentWidget() is picker.parameter_page
    assert picker.editor.material() == exact
    assert picker._back()
    assert picker.materials()[0] == exact
    picker.edit()
    picker.editor.density.setText("1.5")
    assert picker._back()
    assert picker.materials()[0].density == 1.5
    assert picker.session.material("wool").density == 1.31
    picker.reject()
    assert picker.session.material("wool").density == 1.31


def test_invalid_inline_editor_blocks_navigation_but_cancel_remains_available(picker):
    picker.edit()
    picker.editor.lower.setText("40")
    picker.editor.upper.setText("30")
    assert not picker._back()
    assert "下限" in picker.editor.error.text()
    assert not picker.set_selected("cotton", True)
    picker.accept()
    assert picker.result() == QDialog.DialogCode.Rejected
    assert "cotton" not in [m.id for m in picker.materials()]
    picker.reject()


def test_record_pending_locks_are_visible_but_allow_reorder(picker):
    picker.session.add_measurement(20, "wool")
    pending = {"cashmere"}
    picker._in_use = lambda: pending
    picker._refresh()
    for key in ("wool", "cashmere"):
        assert not cell(picker, key).flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert not picker.set_selected(key, False)
    picker.list.setCurrentRow(0)
    picker.move(1)
    assert [m.id for m in picker.materials()] == ["cashmere", "wool"]
    pending.clear()
    picker._refresh_locks()
    assert picker.set_selected("cashmere", False)
    assert not picker.set_selected("wool", False)


def test_late_record_or_external_config_prevents_stale_dialog_acceptance(picker):
    assert picker.set_selected("wool", False)
    picker.session.add_measurement(20, "wool")
    picker.accept()
    assert picker.result() == QDialog.DialogCode.Rejected
    assert "引用" in picker.status.text()
    assert picker.set_selected("wool", True)
    picker.session.set_materials([*picker.session.operators[picker.operator_id].materials["warp"], picker.catalog.get("cotton")])
    picker.accept()
    assert picker.result() == QDialog.DialogCode.Rejected
    assert "后台变化" in picker.status.text()


def test_nine_limit_and_drag_order_by_ids(picker):
    for item in picker.catalog.materials()[2:9]:
        assert picker.set_selected(item.id, True)
    assert not picker.set_selected("camel_hair", True)
    assert len(picker.materials()) == 9
    before = [m.id for m in picker.materials()]
    assert picker.list.model().moveRows(QModelIndex(), 0, 1, QModelIndex(), 4)
    QApplication.processEvents()
    assert [m.id for m in picker.materials()] == [*before[1:4], before[0], *before[4:]]
    assert picker.list.item(0).text().startswith("1 ·")


def test_responsive_tabs_keep_editor_draft_and_footer_accessible(picker):
    picker.resize(940, 620)
    picker.show()
    QApplication.processEvents()
    assert not picker.panes.narrow
    picker.edit()
    picker.editor.density.setText("1.77")
    picker.resize(700, 560)
    QApplication.processEvents()
    assert picker.panes.narrow and picker.panes.tabs.count() == 2
    picker.panes.show_right()
    assert picker.editor.density.text() == "1.77"
    assert picker.rect().contains(picker.buttons.mapTo(picker, QPoint(1, picker.buttons.height() - 1)))
    picker.resize(940, 620)
    QApplication.processEvents()
    assert not picker.panes.narrow and picker.editor.density.text() == "1.77"


def test_combinations_only_store_ids_append_atomically_and_keep_existing_parameters(picker, tmp_path):
    picker._materials[0] = replace(picker._materials[0], density=1.2345)
    assert picker.save_combination("毛绒")
    value = json.loads((tmp_path / "组合.json").read_text())
    assert value == {"version": 1, "combinations": [{"name": "毛绒", "material_ids": ["wool", "cashmere"]}]}
    picker.combinations.set("新增", ["cashmere", "cotton", "nylon"])
    assert picker.add_combination("新增")
    assert [m.id for m in picker.materials()] == ["wool", "cashmere", "cotton", "nylon"]
    assert picker.materials()[0].density == 1.2345
    before = picker.materials()
    picker.combinations.set("缺失", ["goat_hair", "deleted"])
    assert not picker.add_combination("缺失") and picker.materials() == before
    assert "整组未加入" in picker.status.text()
    picker.combinations.set("超额", [m.id for m in picker.catalog.materials()[4:13]])
    assert not picker.add_combination("超额") and picker.materials() == before


def test_library_filters_inline_edit_and_failed_save_keep_draft(tmp_path, monkeypatch):
    path = tmp_path / "库.json"
    dialog = MaterialCatalogDialog(path=path)
    try:
        dialog.filter.setCurrentIndex(dialog.filter.findData("incomplete"))
        rows = [r for r in range(dialog.table.rowCount()) if not dialog.table.isRowHidden(r)]
        assert len(rows) == 3
        assert all("待确认密度" in dialog.table.item(r, 3).text() for r in rows)
        dialog.filter.setCurrentIndex(0)
        dialog.editor.density.setText("1.34567891234")
        monkeypatch.setattr(dialog.catalog, "save", lambda *_: (_ for _ in ()).throw(OSError("只读")))
        dialog.accept()
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert "未保存" in dialog.status.text()
        assert dialog.editor.density.text() == "1.34567891234"
        assert not path.exists()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_unchanged_library_and_trial_do_not_write_or_add_revision(tmp_path, monkeypatch):
    path = tmp_path / "库.json"
    dialog = MaterialCatalogDialog(path=path)
    monkeypatch.setattr(dialog.catalog, "save", lambda *_: pytest.fail("unchanged library wrote"))
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted and not path.exists()
    value = trial()
    revision = value.revision
    value.set_materials(value.operators[value.active_operator_id].materials["warp"])
    assert value.revision == revision and not value.undo()
    dialog.deleteLater()


def test_controller_material_locks_can_be_scoped_by_owner_and_direction():
    first = SimpleNamespace(material_id="wool", operator_id="a", direction="warp")
    second = SimpleNamespace(material_id="cotton", operator_id="b", direction="weft")
    controller = SimpleNamespace(_jobs=[(first, None)], pending_classification=[(second, None)],
                                 _confirmation=None, _candidate_queue=[])
    assert LiveExperimentController.materials_in_use(controller) == {"wool", "cotton"}
    assert LiveExperimentController.materials_in_use(controller, operator_id="a", direction="warp") == {"wool"}
    assert not LiveExperimentController.materials_in_use(controller, operator_id="a", direction="weft")


def test_combo_rename_restart_delete_validation_and_missing_file(tmp_path):
    path = tmp_path / "组合.json"
    data = MaterialCombinations.load(path)
    data.set("组合一", ["wool", "cotton"])
    data.set("组合二", ["nylon"])
    with pytest.raises(ValueError):
        data.set("组合二", ["wool"], previous_name="组合一")
    data.set("毛棉", ["wool", "cotton"], previous_name="组合一")
    data.save(path)
    loaded = MaterialCombinations.load(path)
    assert loaded.get("毛棉")["material_ids"] == ["wool", "cotton"]
    loaded.delete("组合二")
    assert len(loaded.items()) == 1
    with pytest.raises(ValueError):
        MaterialCombinations([{"name": "重复", "material_ids": ["wool"]}] * 2)
    with pytest.raises(ValueError):
        loaded.set("无效", ["wool", "wool"])
