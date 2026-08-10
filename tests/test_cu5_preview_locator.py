from __future__ import annotations

import pytest

from fdm.platform.windows_window_locator import PhysicalRect, WindowRecord, WindowSnapshot
from fdm.services.cu5_preview_locator import (
    Cu5PreviewAmbiguousError,
    Cu5PreviewLocator,
    Cu5PreviewNotFoundError,
    Cu5PreviewSelector,
    Cu5PreviewUnavailableError,
    locate_cu5_preview,
    rank_cu5_preview_candidates,
)


def _record(
    hwnd: int,
    *,
    parent: int | None,
    root: int = 1,
    ancestors: tuple[int, ...] = (),
    class_name: str,
    rect: PhysicalRect,
    title: str = "",
    process_path: str = r"C:\CU-5\CU-5.exe",
    pid: int = 77,
    visible: bool = True,
    minimized: bool = False,
    cloaked: bool = False,
    control_id: int | None = None,
) -> WindowRecord:
    return WindowRecord(
        hwnd=hwnd,
        parent_hwnd=parent,
        root_hwnd=root,
        ancestor_hwnds=ancestors,
        pid=pid,
        process_path=process_path,
        title=title,
        class_name=class_name,
        control_id=control_id,
        rect=rect,
        visible=visible,
        minimized=minimized,
        cloaked=cloaked,
    )


def _base_records() -> list[WindowRecord]:
    return [
        _record(
            1,
            parent=None,
            class_name="Afx:00400000:b:00010003:00000006",
            title="CU-5 纤维细度仪",
            rect=PhysicalRect(0, 0, 1280, 900),
        ),
        _record(
            2,
            parent=1,
            ancestors=(1,),
            class_name="MDIClient",
            rect=PhysicalRect(80, 80, 1200, 840),
        ),
        _record(
            3,
            parent=2,
            ancestors=(1, 2),
            class_name="Static",
            title="实时预览",
            rect=PhysicalRect(120, 90, 240, 120),
        ),
    ]


def test_exact_sdk_host_wins_over_other_four_by_three_children() -> None:
    records = _base_records()
    records.extend(
        [
            _record(
                4,
                parent=2,
                ancestors=(1, 2),
                class_name="CWndForSDK",
                control_id=1201,
                rect=PhysicalRect(220, 150, 988, 726),
            ),
            _record(
                5,
                parent=2,
                ancestors=(1, 2),
                class_name="AfxWnd42",
                rect=PhysicalRect(350, 220, 990, 700),
            ),
        ]
    )

    match = locate_cu5_preview(records)

    assert match.hwnd == 4
    assert match.rect == PhysicalRect(220, 150, 988, 726)
    assert any("CWndForSDK" in reason for reason in match.reasons)
    assert any("768x576" in reason for reason in match.reasons)


def test_generic_host_can_be_identified_by_resource_afx_mdi_and_geometry() -> None:
    records = _base_records()
    records.append(
        _record(
            8,
            parent=2,
            ancestors=(1, 2),
            class_name="AfxWnd42",
            control_id=900,
            rect=PhysicalRect(220, 150, 988, 726),
        )
    )

    ranked = rank_cu5_preview_candidates(WindowSnapshot.from_records(records))
    match = Cu5PreviewLocator(enumerate_snapshot=lambda: WindowSnapshot.from_records(records)).locate()

    assert ranked[0].record.hwnd == 8
    assert match.hwnd == 8
    assert any("MDIClient" in reason for reason in match.reasons)
    assert any("资源文本" in reason for reason in match.reasons)


def test_nearly_equal_video_hosts_fail_as_ambiguous() -> None:
    records = _base_records()
    for hwnd, left in ((10, 180), (11, 210)):
        records.append(
            _record(
                hwnd,
                parent=2,
                ancestors=(1, 2),
                class_name="CWndForSDK",
                rect=PhysicalRect(left, 150, left + 768, 726),
            )
        )

    with pytest.raises(Cu5PreviewAmbiguousError) as captured:
        locate_cu5_preview(records)

    assert {item.record.hwnd for item in captured.value.candidates[:2]} == {10, 11}


def test_stable_selector_resolves_equal_hosts_without_using_hwnd_or_coordinates() -> None:
    records = _base_records()
    for hwnd, control_id, left in ((10, 1201, 180), (11, 1301, 210)):
        records.append(
            _record(
                hwnd,
                parent=2,
                ancestors=(1, 2),
                class_name="CWndForSDK",
                control_id=control_id,
                rect=PhysicalRect(left, 150, left + 768, 726),
            )
        )

    match = locate_cu5_preview(
        records,
        selector={
            "process_name": "CU-5.exe",
            "class_name": "CWndForSDK",
            "control_id": 1301,
            # Unsafe historical fields must have no effect.
            "hwnd": 10,
            "left": 180,
            "top": 150,
        },
    )

    assert match.hwnd == 11
    assert any("控件 ID" in reason for reason in match.reasons)


def test_match_emits_restart_stable_selector_signature() -> None:
    records = _base_records()
    records.append(
        _record(
            80,
            parent=2,
            ancestors=(1, 2),
            class_name="CWndForSDK",
            control_id=1201,
            rect=PhysicalRect(-400, 150, 368, 726),
        )
    )

    match = locate_cu5_preview(records)
    payload = match.selector.to_dict()

    assert payload == {
        "process_name": "cu-5.exe",
        "class_name": "cwndforsdk",
        "control_id": 1201,
        "size": {"width": 768, "height": 576},
        "ancestor_classes": ["afx", "mdiclient"],
    }
    assert not ({"hwnd", "pid", "x", "y", "left", "top"} & payload.keys())
    assert Cu5PreviewSelector.from_value(
        {**payload, "hwnd": 80, "rect": [-400, 150, 368, 726]}
    ).to_dict() == payload


def test_static_video_child_replaces_dialog_container_and_stale_selector() -> None:
    records = _base_records()[:2]
    records.extend(
        [
            _record(
                40,
                parent=2,
                ancestors=(1, 2),
                class_name="#32770",
                control_id=1400,
                rect=PhysicalRect(100, 100, 1296, 811),
            ),
            _record(
                41,
                parent=40,
                ancestors=(1, 2, 40),
                class_name="Static",
                control_id=1501,
                rect=PhysicalRect(106, 100, 874, 676),
            ),
        ]
    )

    ranked = rank_cu5_preview_candidates(
        records,
        selector={
            "process_name": "cu-5.exe",
            "class_name": "#32770",
            "control_id": 1400,
            "size": {"width": 1196, "height": 711},
        },
    )
    match = locate_cu5_preview(
        records,
        selector={
            "process_name": "cu-5.exe",
            "class_name": "#32770",
            "control_id": 1400,
            "size": {"width": 1196, "height": 711},
        },
    )

    assert [item.record.hwnd for item in ranked] == [41]
    assert match.hwnd == 41
    assert match.rect == PhysicalRect(106, 100, 874, 676)
    assert match.selector is not None
    assert match.selector.class_name == "static"
    assert match.selector.width == 768 and match.selector.height == 576
    assert "#32770" in match.selector.ancestor_classes
    assert any("视频子窗口" in reason for reason in match.reasons)


def test_generic_large_static_control_is_not_treated_as_video_without_context() -> None:
    records = _base_records()[:2]
    records.append(
        _record(
            50,
            parent=2,
            ancestors=(1, 2),
            class_name="Static",
            title="产品图示",
            rect=PhysicalRect(220, 150, 860, 630),
        )
    )

    with pytest.raises(Cu5PreviewNotFoundError):
        locate_cu5_preview(records)


@pytest.mark.parametrize(
    ("minimized", "cloaked", "visible", "message"),
    [
        (True, False, True, "最小化"),
        (False, True, True, "系统隐藏"),
        (False, False, False, "不可见"),
    ],
)
def test_unavailable_cu5_root_fails_explicitly(minimized, cloaked, visible, message) -> None:
    records = _base_records()
    root = records[0]
    records[0] = _record(
        root.hwnd,
        parent=None,
        class_name=root.class_name,
        title=root.title,
        rect=root.rect,
        minimized=minimized,
        cloaked=cloaked,
        visible=visible,
    )

    with pytest.raises(Cu5PreviewUnavailableError, match=message):
        locate_cu5_preview(records)


def test_non_cu5_process_is_not_mistaken_for_preview() -> None:
    records = _base_records()
    records[0] = _record(
        1,
        parent=None,
        class_name="AfxFrame",
        title="普通测量程序",
        process_path=r"C:\Other\measurement.exe",
        rect=PhysicalRect(0, 0, 1280, 900),
    )

    with pytest.raises(Cu5PreviewNotFoundError, match="CU 系列"):
        locate_cu5_preview(records)


def test_locator_exposes_unbiased_adjustment_candidates_after_a_saved_choice() -> None:
    records = _base_records()
    for hwnd, control_id, left in ((10, 1201, 180), (11, 1301, 210)):
        records.append(
            _record(
                hwnd,
                parent=2,
                ancestors=(1, 2),
                class_name="CWndForSDK",
                control_id=control_id,
                rect=PhysicalRect(left, 150, left + 768, 726),
            )
        )
    locator = Cu5PreviewLocator(
        enumerate_snapshot=lambda: WindowSnapshot.from_records(records),
        selector={
            "process_name": "cu-5.exe",
            "class_name": "cwndforsdk",
            "control_id": 1301,
        },
    )

    match, candidates = locator.locate_with_candidates()

    assert match.hwnd == 11
    assert {item.record.hwnd for item in candidates} == {10, 11}
    assert candidates[0].record.hwnd == 11
    assert all(item.selector is not None for item in candidates)
    assert {item.selector.control_id for item in candidates} == {1201, 1301}


@pytest.mark.parametrize(
    ("process_path", "title"),
    [
        (r"C:\CU\CU.exe", "实验引导"),
        (r"C:\CU-6\CU-6.exe", "直径实验"),
        (r"C:\CU6\CU6.exe", "直径实验"),
        (r"C:\Vendor\capture.exe", "CU - 实时预览"),
    ],
)
def test_cu_family_process_or_title_can_reuse_preview_detection(
    process_path: str,
    title: str,
) -> None:
    records = _base_records()
    records[0] = _record(
        1,
        parent=None,
        class_name=records[0].class_name,
        title=title,
        process_path=process_path,
        rect=records[0].rect,
    )
    records.append(
        _record(
            4,
            parent=2,
            ancestors=(1, 2),
            class_name="CWndForSDK",
            process_path=process_path,
            control_id=1201,
            rect=PhysicalRect(220, 150, 988, 726),
        )
    )

    assert locate_cu5_preview(records).hwnd == 4


def test_saved_cu5_selector_does_not_block_cu6_root_discovery() -> None:
    records = _base_records()
    process_path = r"C:\CU-6\CU-6.exe"
    records[0] = _record(
        1,
        parent=None,
        class_name=records[0].class_name,
        title="CU-6 纤维细度仪",
        process_path=process_path,
        rect=records[0].rect,
    )
    records.append(
        _record(
            4,
            parent=2,
            ancestors=(1, 2),
            class_name="AfxWnd42",
            process_path=process_path,
            control_id=2202,
            rect=PhysicalRect(220, 150, 988, 726),
        )
    )

    match = locate_cu5_preview(
        records,
        selector={
            "process_name": "cu-5.exe",
            "class_name": "cwndforsdk",
            "control_id": 1201,
        },
    )

    assert match.hwnd == 4
    assert any("通用 CU" in reason for reason in match.reasons)


@pytest.mark.parametrize(
    ("process_path", "title"),
    [
        (r"C:\Tools\Secure.exe", "Secure Capture"),
        (r"C:\Tools\Cubic.exe", "Cubic Preview"),
        (r"C:\Tools\document.exe", "Document Viewer"),
    ],
)
def test_incidental_cu_letters_do_not_open_the_cu_family_gate(
    process_path: str,
    title: str,
) -> None:
    records = _base_records()
    records[0] = _record(
        1,
        parent=None,
        class_name="AfxFrame",
        title=title,
        process_path=process_path,
        rect=PhysicalRect(0, 0, 1280, 900),
    )

    with pytest.raises(Cu5PreviewNotFoundError):
        locate_cu5_preview(records)
