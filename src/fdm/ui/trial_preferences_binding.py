"""Application-owned trial preferences shared by shortcuts and Settings."""
from dataclasses import replace

from PySide6.QtCore import QObject, QTimer, Signal

from fdm.settings import AppSettingsIO


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


def trial_preferences_binding(host):
    binding = getattr(host, "_trial_preferences_binding", None)
    if binding is None:
        binding = TrialPreferencesBinding(host)
        host._trial_preferences_binding = binding
        binding.saveFailed.connect(lambda message: host.statusBar().showMessage(message, 10000))
    return binding
