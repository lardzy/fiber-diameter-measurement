from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fdm.models import ImageDocument, ProjectState
from fdm.settings import AppSettings, RawRecordTemplate, resolve_resource_relative_path
from fdm.services.export_service import (
    ExportPlan,
    ExportRenderContext,
    ExportScope,
    ExportSelection,
    ExportService,
    RenderedExport,
)


class ExportHost(Protocol):
    project: ProjectState
    _app_settings: AppSettings
    export_service: ExportService

    def current_document(self) -> ImageDocument | None: ...
    def _create_export_options_dialog(self, preset: ExportSelection): ...
    def _show_export_information(self, title: str, message: str) -> None: ...
    def _show_export_warning(self, title: str, message: str) -> None: ...
    def _preferred_dialog_directory(self, *, recent_dir: str = "") -> Path: ...
    def _select_export_save_path(self, default_path: Path, file_filter: str) -> str: ...
    def _select_export_directory(self, default_dir: Path) -> str: ...
    def _single_export_dialog_filter(self, filename: str) -> str: ...
    def _normalize_dialog_save_path(self, selected_path: str, default_filename: str) -> Path: ...
    def _create_blocking_progress_dialog(self, *, title: str, label_text: str, maximum: int): ...
    def _update_blocking_progress_dialog(
        self,
        progress,
        *,
        completed_steps: int,
        total_steps: int,
        label: str,
        path: Path | None,
    ) -> None: ...
    def _pump_modal_progress_events(self) -> None: ...
    def _close_progress_dialog(self, progress) -> None: ...
    def _render_overlay_image(
        self,
        document: ImageDocument,
        output_path: Path,
        *,
        include_measurements: bool,
        include_scale: bool,
        render_mode: str,
        render_context: ExportRenderContext | None = None,
    ) -> RenderedExport: ...
    def _format_export_failure_message(self, exc: Exception, *, export_path: Path | None) -> str: ...
    def _remember_recent_directory(self, *, setting_name: str, directory: Path, context: str) -> None: ...
    def _save_app_settings(self, *, context: str = "") -> None: ...


class ExportController:
    def __init__(self, host: ExportHost) -> None:
        self._host = host

    def export_results(self, preset: ExportSelection | None = None) -> None:
        host = self._host
        project_session = getattr(host, "project_session_controller", None)
        unresolved_documents = getattr(project_session, "unresolved_documents", None)
        unresolved_count = 0
        if callable(unresolved_documents):
            unresolved_count = len(unresolved_documents())
            if unresolved_count:
                host._show_export_warning(
                    "缺失文档不参与导出",
                    f"项目中有 {unresolved_count} 个未挂载文档；其记录会继续保存在项目中，"
                    "但本次导出默认不包含它们。",
                )
        if not host.project.documents:
            if unresolved_count:
                host._show_export_information(
                    "导出结果",
                    f"当前没有已挂载的可导出图片；{unresolved_count} 个缺失文档记录仍保留在项目中。",
                )
            else:
                host._show_export_information("导出结果", "当前没有可导出的图片。")
            return
        preset = preset or ExportSelection.all_enabled(scope=ExportScope.ALL_OPEN)
        dialog = host._create_export_options_dialog(preset)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        selection = dialog.selection()
        if not selection.any_selected():
            host._show_export_information("导出结果", "请至少选择一种导出内容。")
            return
        raw_record_template = self.prepare_raw_record_template_for_export(selection)
        target_documents = (
            host.project.documents
            if selection.scope == ExportScope.ALL_OPEN
            else ([host.current_document()] if host.current_document() else [])
        )
        target_documents = [document for document in target_documents if document is not None]
        try:
            context_provider = getattr(host, "_export_render_contexts", None)
            render_contexts = (
                context_provider(target_documents, selection.render_mode)
                if callable(context_provider)
                else None
            )
            protected_provider = getattr(host, "_export_protected_source_paths", None)
            protected_source_paths = (
                protected_provider(target_documents)
                if callable(protected_provider)
                else None
            )
            export_plan: ExportPlan = host.export_service.build_plan(
                target_documents,
                selection,
                render_contexts=render_contexts,
                protected_source_paths=protected_source_paths,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            host._show_export_warning("导出设置不可用", str(exc))
            return
        planned_outputs = list(export_plan.files)
        if not planned_outputs:
            host._show_export_information("导出结果", "按当前导出内容设置，没有可生成的文件。")
            return
        default_dir = host._preferred_dialog_directory(recent_dir=host._app_settings.recent_export_dir)
        single_output_path: Path | None = None
        if len(planned_outputs) == 1:
            selected_path = host._select_export_save_path(
                default_dir / planned_outputs[0].filename,
                host._single_export_dialog_filter(planned_outputs[0].filename),
            )
            if not selected_path:
                return
            single_output_path = host._normalize_dialog_save_path(selected_path, planned_outputs[0].filename)
            expected_suffix = Path(planned_outputs[0].filename).suffix.lower()
            if expected_suffix and single_output_path.suffix.lower() != expected_suffix:
                single_output_path = single_output_path.with_suffix(expected_suffix)
            output_dir = str(single_output_path.parent)
        else:
            output_dir = host._select_export_directory(default_dir)
            if not output_dir:
                return

        progress = host._create_blocking_progress_dialog(
            title="导出结果",
            label_text="正在准备导出...",
            maximum=max(1, len(planned_outputs)),
        )
        current_output_path: Path | None = None

        def on_export_progress(completed_steps: int, total_steps: int, label: str, path: Path | None) -> None:
            nonlocal current_output_path
            if path is not None:
                current_output_path = path
            host._update_blocking_progress_dialog(
                progress,
                completed_steps=completed_steps,
                total_steps=total_steps,
                label=label,
                path=path,
            )

        progress.show()
        progress.raise_()
        progress.activateWindow()
        host._pump_modal_progress_events()
        try:
            outputs = host.export_service.export_project(
                host.project,
                output_dir,
                selection=selection,
                documents=target_documents,
                overlay_renderer=host._render_overlay_image,
                export_plan=export_plan,
                single_output_path=single_output_path,
                raw_record_template=raw_record_template,
                category_order_document=host.current_document(),
                progress_callback=on_export_progress,
            )
        except Exception as exc:
            host._close_progress_dialog(progress)
            host._show_export_warning(
                "导出失败",
                host._format_export_failure_message(exc, export_path=current_output_path),
            )
            return
        host._close_progress_dialog(progress)
        if not outputs:
            host._show_export_information("导出结果", "没有生成任何文件。")
            return
        export_root = single_output_path.parent if single_output_path is not None else Path(output_dir)
        host._remember_recent_directory(setting_name="recent_export_dir", directory=export_root, context="导出结果")
        summary_lines = self._format_output_summary(outputs)
        location_text = (
            str(outputs.get("xlsx", single_output_path))
            if single_output_path is not None
            else str(output_dir)
        )
        message = f"结果已导出到:\n{location_text}\n\n" + "\n".join(summary_lines)
        host._show_export_information("导出完成", message)

    def prepare_raw_record_template_for_export(self, selection: ExportSelection) -> RawRecordTemplate | None:
        host = self._host
        selected_path = str(selection.raw_record_template_path or "").strip() if selection.include_excel else ""
        template = self.raw_record_template_for_path(selected_path) if selected_path else None
        if selected_path:
            resolved_path = resolve_resource_relative_path(selected_path)
            if template is None or not resolved_path.exists():
                host._show_export_warning(
                    "原始记录模板",
                    "找不到已选择的原始记录模板，已自动回退到默认 Excel 文档：\n"
                    f"{selected_path}",
                )
                selected_path = ""
                template = None
        selection.raw_record_template_path = selected_path
        if host._app_settings.last_raw_record_template_path != selected_path:
            host._app_settings.last_raw_record_template_path = selected_path
            host._save_app_settings(context="导出结果")
        return template

    def raw_record_template_for_path(self, template_path: str) -> RawRecordTemplate | None:
        token = str(template_path or "").strip()
        if not token:
            return None
        token_key = token.casefold()
        token_resolved = resolve_resource_relative_path(token)
        for template in self._host._app_settings.raw_record_templates:
            candidate = str(template.path or "").strip()
            if candidate.casefold() == token_key:
                return template
            try:
                if resolve_resource_relative_path(candidate).resolve() == token_resolved.resolve():
                    return template
            except OSError:
                continue
        return None

    def _format_output_summary(self, outputs: dict[str, object]) -> list[str]:
        output_labels = {
            "measurement_overlays": "测量叠加图",
            "scale_overlays": "比例尺叠加图",
            "combined_overlays": "测量+比例尺叠加图",
            "scale_jsons": "比例尺 JSON",
            "image_summary_csv": "图片汇总 CSV",
            "fiber_details_csv": "纤维种类汇总 CSV",
            "measurement_details_csv": "测量明细 CSV",
            "xlsx": "Excel 工作簿",
        }
        summary_lines = []
        for key, value in outputs.items():
            label = output_labels.get(key, key)
            if isinstance(value, list):
                summary_lines.append(f"{label}: {len(value)} 个文件")
            else:
                summary_lines.append(f"{label}: {value}")
        return summary_lines
