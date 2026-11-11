"""Final-write consent is scoped to one input revision, not a trial forever."""
from types import SimpleNamespace

from fdm.ui import trial_excel_lifecycle as lifecycle


class Manager:
    def __init__(self, state="error", *, ever=True):
        self.state, self.ever, self.flushed = state, ever, []
        self.detached = []
    def status(self, sid):
        return {"state": self.state, "ever_connected": self.ever, "active": self.state == "synced",
                "pending": False, "unsynced": self.ever, "message": "验证状态"}
    def flush(self, session):
        self.flushed.append(session.revision)
        return self.state == "synced"
    def resync(self, session):
        self.state = "synced"
    def detach(self, session, *, force=False):
        self.detached.append((session.id, session.revision, force))
        return True


def fixture(state="error", ever=True):
    session = SimpleNamespace(id="trial-1", revision=3)
    manager = Manager(state, ever=ever)
    host = SimpleNamespace(_trial_live_excel=manager, _trial_excel_sessions={session.id: session})
    return host, session, manager


def test_force_finalization_does_not_cover_subsequent_measurements(monkeypatch):
    host, session, manager = fixture()
    def accept_force(*args, force=None, **kwargs):
        force()
        return True
    monkeypatch.setattr(lifecycle, "_wait", accept_force)
    assert lifecycle.prepare_trial_excel(host, session)
    assert host._trial_excel_force == {session.id: 3}
    session.revision = 4
    asked = []
    monkeypatch.setattr(lifecycle, "_wait", lambda *a, **kw: asked.append(True) or False)
    assert not lifecycle.prepare_trial_excel(host, session)
    assert asked and not host._trial_excel_force


def test_failed_real_workbook_is_not_treated_as_missing_excel(monkeypatch):
    host, session, manager = fixture()
    asked = []
    monkeypatch.setattr(lifecycle, "_wait", lambda *a, **kw: asked.append(True) or False)
    assert not lifecycle.prepare_trial_excel(host, session)
    assert asked


def test_never_created_workbook_does_not_block_trial_close(monkeypatch):
    host, session, manager = fixture(ever=False)
    monkeypatch.setattr(lifecycle, "_wait", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected dialog")))
    assert lifecycle.prepare_trial_excel(host, session)
    assert host._trial_excel_force[session.id] == session.revision


def test_explicit_workbook_close_is_not_automatically_reopened(monkeypatch):
    host, session, manager = fixture("stopped")
    assert lifecycle.prepare_trial_excel(host, session)
    monkeypatch.setattr(lifecycle, "_wait", lambda parent, msg, attempt, **kw: attempt())
    assert lifecycle.detach_trial_excel(host, session)
    assert manager.detached == [(session.id, 3, True)]
    assert not host._trial_excel_sessions


def test_cancel_one_trial_clears_finalization_choices_for_whole_attempt(monkeypatch):
    host, session, manager = fixture()
    host._trial_excel_force = {"previous": 12}
    monkeypatch.setattr(lifecycle, "prepare_trial_excel", lambda *a: False)
    assert not lifecycle.prepare_all_trial_excel(host)
    assert host._trial_excel_force == {}


def test_committed_detach_does_not_offer_false_cancel(monkeypatch):
    host, session, manager = fixture("synced")
    options = []
    def wait(parent, msg, attempt, **kwargs):
        options.append(kwargs)
        return attempt()
    monkeypatch.setattr(lifecycle, "_wait", wait)
    assert lifecycle.detach_trial_excel(host, session)
    assert options == [{"cancellable": False}]
    assert manager.detached == [(session.id, 3, False)]
