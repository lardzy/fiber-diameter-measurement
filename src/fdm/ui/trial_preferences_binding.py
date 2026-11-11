"""Application-owned trial preferences shared by shortcuts and Settings."""
from dataclasses import replace

from PySide6.QtCore import QObject, QTimer, Signal

from fdm.settings import AppSettingsIO
from fdm.live_experiment.trial_defaults import content_total_target_default, save_content_total_target_default


class TrialPreferencesBinding(QObject):
    changed = Signal(object)
    saveFailed = Signal(str)

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        if not hasattr(host, "_app_settings"):
            host._app_settings = AppSettingsIO.load()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.flush)
        self._quantity_target = 1500
        try:
            self._quantity_target = content_total_target_default()
        except (OSError, ValueError) as exc:
            message = f"无法读取总根数提醒，暂用 1500 根；原配置未覆盖：{exc}"
            QTimer.singleShot(0, self, lambda: self.saveFailed.emit(message))
        self.apply_quantity_target(refresh=False)

    @property
    def preferences(self):
        return self.host._app_settings.trial_preferences.normalized_copy()

    def update(self, **changes):
        candidate = replace(self.preferences, **changes).normalized_copy()
        if candidate == self.preferences:
            return
        self.host._app_settings = replace(self.host._app_settings, trial_preferences=candidate)
        self.changed.emit(candidate)
        self._timer.start()

    def activated(self):
        """Main Settings already owns persistence for Apply/OK/import."""
        self._timer.stop()
        self.changed.emit(self.preferences)

    def flush(self):
        self._timer.stop()
        try:
            AppSettingsIO.save(self.host._app_settings)
        except (OSError, ValueError) as exc:
            self.saveFailed.emit(f"试验设置已应用，但未能保存：{exc}")

    def prepare_quantity_target_dialog(self, dialog):
        """Refresh the single persisted source before creating a settings draft."""
        self._quantity_target = content_total_target_default()
        self.apply_quantity_target()
        dialog.set_trial_total_target(self._quantity_target)

    def commit_quantity_target_dialog(self, dialog):
        """Publish only after persistence succeeds; Cancel never calls this."""
        if not dialog.trial_total_target_changed():
            return
        target = dialog.trial_total_target()
        save_content_total_target_default(target)
        self._quantity_target = target
        self.apply_quantity_target()
        dialog.set_trial_total_target(target)

    def apply_quantity_target(self, session=None, *, refresh=True):
        """Runtime reminder preference; never mutate project/undo/Excel inputs."""
        sessions = list(getattr(getattr(self.host, "project", None), "live_experiments", ()))
        pages = list(getattr(self.host, "_live_experiment_workspaces", {}).values())
        current = getattr(self.host, "_live_experiment_workspace", None)
        if current is not None and current not in pages:
            pages.append(current)
        sessions.extend(page.session for page in pages)
        if session is not None:
            sessions.append(session)
        for trial in {id(item): item for item in sessions}.values():
            if trial.workspace_options["mode"] == "content":
                trial._content_total_target = self._quantity_target
        if refresh:
            for page in pages:
                if not getattr(page, "_closed", False) and hasattr(page, "measurement_feedback"):
                    page._check_reminders(announce=False)


def trial_preferences_binding(host):
    binding = getattr(host, "_trial_preferences_binding", None)
    if binding is None:
        binding = TrialPreferencesBinding(host)
        host._trial_preferences_binding = binding
        binding.saveFailed.connect(lambda message: host.statusBar().showMessage(message, 10000))
    return binding
