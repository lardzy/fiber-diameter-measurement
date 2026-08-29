from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QDialog

from fdm.ui.dataset_export_dialog import DatasetDocumentOption, DatasetExportDialog


def test_dialog_allows_explicit_negative_only_export_without_category_rows(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    dialog = DatasetExportDialog(
        [DatasetDocumentOption("negative", "无目标样本.png", is_current=True)],
        [],
        initial_directory=tmp_path,
    )

    dialog._validate_and_accept()  # noqa: SLF001

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.options().category_mapping == {}
    dialog.close()
    app.processEvents()


def test_dialog_rejects_checked_category_with_empty_training_name(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    dialog = DatasetExportDialog(
        [DatasetDocumentOption("sample", "样本.png", is_current=True)],
        ["棉"],
        initial_directory=tmp_path,
    )
    dialog._category_table.item(0, 2).setText("")  # noqa: SLF001

    with patch("fdm.ui.dataset_export_dialog.QMessageBox.warning") as warning:
        dialog._validate_and_accept()  # noqa: SLF001

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "训练类别名称为空" in warning.call_args.args[2]
    dialog.close()
    app.processEvents()
