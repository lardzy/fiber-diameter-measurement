from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from fdm.history import DocumentChangeImpact, DocumentHistoryState
from fdm.models import (
    DirtyDomain,
    FiberGroup,
    ImageDocument,
    ProjectGroupTemplate,
    ProjectState,
    UNCATEGORIZED_LABEL,
    normalize_group_label,
)


_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


@dataclass(frozen=True, slots=True)
class GroupListRow:
    label: str
    color: str
    current_count: int
    project_count: int
    group_id: str | None
    selected: bool


class GroupManager:
    """Pure project/group business rules used by the Qt main window."""

    def __init__(
        self,
        project: ProjectState,
        *,
        color_palette: list[str],
        color_normalizer: Callable[..., str] | None = None,
    ) -> None:
        self.project = project
        self._color_palette = list(color_palette) or ["#1F7A8C"]
        self._color_normalizer = color_normalizer

    def normalize_group_color(self, color_value: str, *, fallback: str = "#1F7A8C") -> str:
        if self._color_normalizer is not None:
            return self._color_normalizer(color_value, fallback=fallback)
        token = str(color_value or "").strip()
        if _HEX_COLOR_RE.match(token):
            if len(token) == 4:
                token = "#" + "".join(char * 2 for char in token[1:])
            return token.lower()
        fallback_token = str(fallback or "").strip()
        if _HEX_COLOR_RE.match(fallback_token):
            if len(fallback_token) == 4:
                fallback_token = "#" + "".join(char * 2 for char in fallback_token[1:])
            return fallback_token.lower()
        return "#1f7a8c"

    def _apply_atomic_document_mutations(
        self,
        *,
        history_label: str,
        mutator: Callable[[ImageDocument], None],
    ) -> bool:
        captures = [
            (document, document.state_stamp, DocumentHistoryState.capture(document))
            for document in self.project.documents
        ]
        after_states: list[tuple[ImageDocument, object, DocumentHistoryState, DocumentHistoryState]] = []
        try:
            for document, before_stamp, before in captures:
                mutator(document)
                document.rebuild_group_memberships()
                after_states.append(
                    (
                        document,
                        before_stamp,
                        before,
                        DocumentHistoryState.capture(document),
                    )
                )
        except Exception:
            for document, before_stamp, before in reversed(captures):
                before.restore(document)
                document.restore_state_stamp(before_stamp)
            raise

        any_changed = False
        for document, before_stamp, before, after in after_states:
            if before == after:
                document.restore_state_stamp(before_stamp)
                continue
            after_stamp = document.advance_state({DirtyDomain.SESSION})
            if document.history is not None:
                document.history.push_delta(
                    history_label,
                    before=before,
                    after=after,
                    before_stamp=before_stamp,
                    after_stamp=after_stamp,
                    impact=DocumentChangeImpact.SESSION,
                )
            any_changed = True
        return any_changed

    def project_group_template_for_label(self, label: str) -> ProjectGroupTemplate | None:
        token = normalize_group_label(label)
        if not token:
            return None
        for template in self.project.project_group_templates:
            if normalize_group_label(template.label) == token:
                return template
        return None

    def next_group_color(self, document: ImageDocument) -> str:
        return self._color_palette[(document.next_group_number() - 1) % len(self._color_palette)]

    def ensure_project_group_template(self, *, label: str, color: str) -> bool:
        token = normalize_group_label(label)
        if not token or self.project_group_template_for_label(token) is not None:
            return False
        self.project.project_group_templates.append(
            ProjectGroupTemplate(label=token, color=self.normalize_group_color(color)),
        )
        return True

    def set_project_group_template_color(self, *, label: str, color: str) -> bool:
        template = self.project_group_template_for_label(label)
        if template is None:
            return False
        normalized_color = self.normalize_group_color(color, fallback=template.color)
        if template.color == normalized_color:
            return False
        template.color = normalized_color
        return True

    def apply_project_group_template_edit(self, *, original_label: str, target_label: str, color: str) -> bool:
        target_token = normalize_group_label(target_label)
        if not target_token:
            return False
        original_token = normalize_group_label(original_label)
        normalized_color = self.normalize_group_color(color)
        original_template = self.project_group_template_for_label(original_token) if original_token else None
        target_template = self.project_group_template_for_label(target_token)
        changed = False
        if original_template is not None:
            if target_template is not None and target_template is not original_template:
                if target_template.color != normalized_color:
                    target_template.color = normalized_color
                    changed = True
                self.project.project_group_templates = [
                    template
                    for template in self.project.project_group_templates
                    if template is not original_template
                ]
                return True
            if normalize_group_label(original_template.label) != target_token:
                original_template.label = target_token
                changed = True
            if original_template.color != normalized_color:
                original_template.color = normalized_color
                changed = True
            return changed
        if target_template is not None:
            if target_template.color != normalized_color:
                target_template.color = normalized_color
                return True
            return False
        self.project.project_group_templates.append(
            ProjectGroupTemplate(label=target_token, color=normalized_color),
        )
        return True

    def ensure_document_named_group(
        self,
        document: ImageDocument,
        *,
        label: str,
        color: str,
        activate: bool,
        sync_color: bool = False,
    ) -> tuple[FiberGroup | None, bool]:
        token = normalize_group_label(label)
        if not token:
            return None, False
        normalized_color = self.normalize_group_color(color)
        changed = False
        matches = document.groups_by_label(token)
        if matches:
            canonical = matches[0]
            for duplicate in matches[1:]:
                if document.merge_group_into(duplicate.id, canonical.id):
                    changed = True
            if sync_color and canonical.color != normalized_color:
                canonical.color = normalized_color
                changed = True
            if activate and document.active_group_id != canonical.id:
                document.set_active_group(canonical.id)
                changed = True
        else:
            active_group_id = document.active_group_id
            canonical = document.create_group(color=normalized_color, label=token)
            if activate or active_group_id is None:
                document.set_active_group(canonical.id)
            elif active_group_id != canonical.id:
                document.set_active_group(active_group_id)
            changed = True
        changed = document.unsuppress_project_group_label(token) or changed
        return canonical, changed

    def apply_project_group_templates_to_document(
        self,
        document: ImageDocument,
        *,
        labels: set[str] | None = None,
    ) -> bool:
        changed = False
        for template in self.project.project_group_templates:
            token = normalize_group_label(template.label)
            if (
                not token
                or (labels is not None and token not in labels)
                or document.is_project_group_label_suppressed(token)
            ):
                continue
            _group, ensured_changed = self.ensure_document_named_group(
                document,
                label=token,
                color=template.color,
                activate=False,
                sync_color=True,
            )
            changed = ensured_changed or changed
        if document.active_group_id is None and document.can_delete_uncategorized_entry():
            changed = document.hide_uncategorized_entry() or changed
        return changed

    def sync_project_group_templates(self, *, history_label: str, labels: set[str] | None = None) -> bool:
        return self._apply_atomic_document_mutations(
            history_label=history_label,
            mutator=lambda document: self.apply_project_group_templates_to_document(
                document,
                labels=labels,
            ),
        )

    def sync_project_group_template_edit_to_documents(
        self,
        *,
        original_label: str,
        target_label: str,
        color: str,
        history_label: str,
    ) -> bool:
        target_token = normalize_group_label(target_label)
        if not target_token:
            return False
        original_token = normalize_group_label(original_label)
        normalized_color = self.normalize_group_color(color)
        def mutate_document(document: ImageDocument) -> None:
            original_group = document.find_group_by_label(original_token) if original_token else None
            target_group = document.find_group_by_label(target_token)
            if original_group is not None and target_group is not None and original_group.id != target_group.id:
                document.merge_group_into(original_group.id, target_group.id)
                target_group = document.find_group_by_label(target_token)
            elif original_group is not None:
                if normalize_group_label(original_group.label) != target_token:
                    original_group.label = target_token
                target_group = original_group
            elif target_group is None:
                target_group, _ensured_changed = self.ensure_document_named_group(
                    document,
                    label=target_token,
                    color=normalized_color,
                    activate=False,
                    sync_color=True,
                )
            if target_group is not None and target_group.color != normalized_color:
                target_group.color = normalized_color
            if original_token and original_token != target_token:
                document.unsuppress_project_group_label(original_token)
            document.unsuppress_project_group_label(target_token)

        return self._apply_atomic_document_mutations(
            history_label=history_label,
            mutator=mutate_document,
        )

    def area_inference_group_color_for_label(self, label: str) -> str:
        token = normalize_group_label(label)
        if not token:
            return self._color_palette[0]
        template = self.project_group_template_for_label(token)
        if template is not None:
            return template.color
        for document in self.project.documents:
            group = document.find_group_by_label(token)
            if group is not None:
                return group.color
        template_count = len(
            [
                template
                for template in self.project.project_group_templates
                if normalize_group_label(template.label)
            ]
        )
        return self._color_palette[template_count % len(self._color_palette)]

    def resolve_area_inference_group_colors(self, labels: list[str]) -> dict[str, str]:
        resolved_colors: dict[str, str] = {}
        template_count = len(
            [
                template
                for template in self.project.project_group_templates
                if normalize_group_label(template.label)
            ]
        )
        fallback_offset = 0
        for label in labels:
            token = normalize_group_label(label)
            if not token or token in resolved_colors:
                continue
            template = self.project_group_template_for_label(token)
            if template is not None:
                resolved_colors[token] = template.color
                continue
            existing_color = None
            for document in self.project.documents:
                group = document.find_group_by_label(token)
                if group is not None:
                    existing_color = group.color
                    break
            if existing_color is not None:
                resolved_colors[token] = existing_color
                continue
            palette_index = (template_count + fallback_offset) % len(self._color_palette)
            resolved_colors[token] = self._color_palette[palette_index]
            fallback_offset += 1
        return resolved_colors

    def documents_for_group_counts(self, current_document: ImageDocument | None) -> list[ImageDocument]:
        documents = list(self.project.documents)
        if current_document is not None and all(document.id != current_document.id for document in documents):
            documents.append(current_document)
        return documents

    def project_measurement_count_for_group_label(
        self,
        label: str,
        current_document: ImageDocument | None = None,
    ) -> int:
        token = normalize_group_label(label)
        total = 0
        for document in self.documents_for_group_counts(current_document):
            if token:
                for group in document.groups_by_label(token):
                    total += len(group.measurement_ids)
            else:
                for group in document.sorted_groups():
                    if not normalize_group_label(group.label):
                        total += len(group.measurement_ids)
        return total

    def project_uncategorized_measurement_count(self, current_document: ImageDocument | None = None) -> int:
        return sum(
            document.uncategorized_measurement_count()
            for document in self.documents_for_group_counts(current_document)
        )

    def group_rows(
        self,
        document: ImageDocument | None,
        *,
        default_uncategorized_color: str,
    ) -> list[GroupListRow]:
        if document is None:
            return []
        rows: list[GroupListRow] = []
        if document.should_show_uncategorized_entry():
            rows.append(
                GroupListRow(
                    label=UNCATEGORIZED_LABEL,
                    color=default_uncategorized_color,
                    current_count=document.uncategorized_measurement_count(),
                    project_count=self.project_uncategorized_measurement_count(document),
                    group_id=None,
                    selected=document.active_group_id is None,
                )
            )
        for group in document.sorted_groups():
            rows.append(
                GroupListRow(
                    label=group.display_name(),
                    color=group.color,
                    current_count=len(group.measurement_ids),
                    project_count=self.project_measurement_count_for_group_label(group.label, document),
                    group_id=group.id,
                    selected=document.active_group_id == group.id,
                )
            )
        return rows
