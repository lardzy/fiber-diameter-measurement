import numpy as np
import pytest

from fdm.live_experiment.frames import (
    FrameBudget, FrameBudgetExceeded, FrameHandle, acquisition_reservation,
)


def frame(sequence=0, size=100):
    return FrameHandle.snapshot(np.zeros((1, size), np.uint8), source_id="scratch", sequence=sequence)


def test_serial_writer_shares_max_scratch_but_independent_writers_add():
    budget = FrameBudget(1000)
    value = frame()
    a, sa = budget.acquire_with_scratch(value, group="writer-a", scratch_bytes=200)
    b, sb = budget.acquire_with_scratch(value, group="writer-a", scratch_bytes=300)
    c, sc = budget.acquire_with_scratch(value, group="writer-b", scratch_bytes=150)
    assert budget.used_bytes == 100 + 300 + 150
    budget.release_scratch(sb)
    assert budget.used_bytes == 100 + 200 + 150
    budget.release_scratch(sb)
    budget.release(a)
    budget.release(b)
    assert budget.used_bytes == 100 + 200 + 150
    budget.release_scratch(sa)
    budget.release_scratch(sc)
    budget.release(c)
    assert budget.used_bytes == 0
    assert budget.peak_bytes == 550


def test_scratch_and_frame_admission_is_atomic_and_preserves_future_preview_slots():
    budget = FrameBudget(520)
    value = frame()
    preview = budget.acquire(value, reserve_bytes=20, preview=True)
    raw, scratch = budget.acquire_with_scratch(value, group="writer", scratch_bytes=200,
        reserve_bytes=20, keep_preview_slots=2)
    assert budget.used_bytes == 320
    before = (dict(budget._leases), dict(budget._frames), dict(budget._scratch), budget.peak_bytes)
    with pytest.raises(FrameBudgetExceeded, match="暂停新增"):
        budget.acquire_with_scratch(frame(1), group="writer", scratch_bytes=210,
            reserve_bytes=20, keep_preview_slots=2)
    assert before == (budget._leases, budget._frames, budget._scratch, budget.peak_bytes)
    latest = budget.acquire(frame(2), reserve_bytes=20, preview=True)
    presented = budget.acquire(frame(3), reserve_bytes=20, preview=True)
    assert budget.used_bytes == 520
    budget.release(latest)
    budget.release(presented)
    budget.release(raw)
    budget.release_scratch(scratch)
    budget.release(preview)
    assert budget.used_bytes == 0


def test_ordinary_acquire_and_failed_preview_replace_include_scratch():
    budget = FrameBudget(350)
    old = frame()
    preview = budget.acquire(old, preview=True)
    raw, scratch = budget.acquire_with_scratch(old, group="writer", scratch_bytes=200)
    with pytest.raises(FrameBudgetExceeded):
        budget.acquire(frame(1))
    with pytest.raises(FrameBudgetExceeded):
        budget.replace(preview, frame(2), preview=True)
    assert budget._leases[preview][0] == old.frame_id
    assert budget.used_bytes == 300
    budget.release(raw)
    updated = budget.replace(preview, frame(3), preview=True)
    assert budget.used_bytes == 300
    budget.release(updated)
    assert budget.used_bytes == 200
    budget.release_scratch(scratch)
    assert budget.used_bytes == 0


def test_failed_actual_allocation_rolls_back_new_scratch_ticket():
    budget = FrameBudget(299)
    value = frame()
    old = budget.acquire(value)
    with pytest.raises(FrameBudgetExceeded):
        budget.acquire_with_scratch(value, group="writer", scratch_bytes=200)
    assert budget.used_bytes == 100 and len(budget._leases) == 1 and not budget._scratch
    budget.release(old)


def test_scaled_6k_admission_has_preview_room_but_rejects_a_second_pinned_frame():
    # MiB proportions scaled down to bytes; round the ~2.035 MiB row workspace
    # up to 3. This checks the actual admission path without allocating 6K data.
    budget = FrameBudget(512)
    first = frame(0, size=72)
    raw, scratch = budget.acquire_with_scratch(first, group="writer", scratch_bytes=3,
        reserve_bytes=264, keep_preview_slots=2)
    assert budget.used_bytes == 72 + 264 + 3
    with pytest.raises(FrameBudgetExceeded, match="暂停新增"):
        budget.acquire_with_scratch(frame(1, size=72), group="writer", scratch_bytes=3,
            reserve_bytes=264, keep_preview_slots=2)
    assert len(budget._leases) == len(budget._scratch) == 1
    preview_a = budget.acquire(frame(2, size=72), reserve_bytes=264, preview=True)
    preview_b = budget.acquire(frame(3, size=72), reserve_bytes=264, preview=True)
    assert budget.used_bytes == 483
    for lease in (raw, preview_a, preview_b):
        budget.release(lease)
    budget.release_scratch(scratch)
    assert budget.used_bytes == 0


@pytest.mark.parametrize("raw,expected", [(0, 0), (500, 500), (-1, 100), (True, 100), (1.5, 100)])
def test_acquisition_reservation_keeps_controller_fallback_rules(raw, expected):
    value = frame()
    value.metadata["acquisition_reserved_bytes"] = raw
    assert acquisition_reservation(value) == expected
    static = FrameHandle.snapshot(value.pixels, source_id="static", live=False,
        metadata={"acquisition_reserved_bytes": raw})
    assert acquisition_reservation(static) == 0


@pytest.mark.parametrize("group,size", [("", 100), (None, 100), ("writer", -1), ("writer", True)])
def test_invalid_scratch_admission_changes_no_state(group, size):
    budget = FrameBudget()
    with pytest.raises(ValueError):
        budget.acquire_with_scratch(frame(), group=group, scratch_bytes=size)
    assert budget.used_bytes == 0 and not budget._leases and not budget._scratch
