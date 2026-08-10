from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import types
import unittest
from unittest.mock import Mock, patch
import uuid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtNetwork import QLocalSocket
from PySide6.QtWidgets import QApplication

from fdm.application_launch import (
    ApplicationOpenRequestError,
    MAX_IPC_MESSAGE_BYTES,
    SingleInstanceCoordinator,
    build_application_open_request,
    decode_application_open_request,
    encode_application_open_request,
    parse_application_arguments,
    single_instance_server_name,
)
from fdm import app as fdm_app


class ApplicationLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _drain_deferred_socket_deletes(self) -> None:
        """Settle QLocalSocket.deleteLater() while owners are still alive."""

        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def test_argument_parser_extracts_unicode_paths_and_keeps_qt_arguments(self) -> None:
        with TemporaryDirectory() as tmpdir:
            qt_args, request = parse_application_arguments(
                ["fdm", "--style", "Fusion", "子目录/中文 项目.FDMPROJ"],
                cwd=tmpdir,
            )

        self.assertEqual(qt_args, ["fdm", "--style", "Fusion"])
        self.assertEqual(request.paths[0].name, "中文 项目.FDMPROJ")
        self.assertTrue(request.paths[0].is_absolute())

    def test_request_rejects_mixed_or_multiple_projects_and_deduplicates_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ApplicationOpenRequestError, "不能在同一请求"):
                build_application_open_request(
                    [root / "one.fdmproj", root / "one.fdmslide"],
                    source="test",
                )
            with self.assertRaisesRegex(ApplicationOpenRequestError, "只能打开一个项目"):
                build_application_open_request(
                    [root / "one.fdmproj", root / "two.fdmproj"],
                    source="test",
                )
            request = build_application_open_request(
                [root / "one.fdmslide", root / "one.fdmslide"],
                source="test",
            )
            with self.assertRaisesRegex(ApplicationOpenRequestError, "最多打开"):
                build_application_open_request(
                    [root / f"slide-{index}.fdmslide" for index in range(33)],
                    source="test",
                )
            with self.assertRaisesRegex(ApplicationOpenRequestError, "不支持"):
                build_application_open_request([root / "ordinary.png"], source="test")

            long_path = root.joinpath(*(["较长的目录名称"] * 30), "切片.FDMSLIDE")
            long_request = build_application_open_request([long_path], source="test")

        self.assertEqual(len(request.paths), 1)
        self.assertEqual(long_request.paths[0].suffix, ".FDMSLIDE")
        self.assertGreater(len(str(long_request.paths[0])), 260)

    def test_ipc_envelope_round_trips_and_rejects_unknown_or_relative_payloads(self) -> None:
        with TemporaryDirectory() as tmpdir:
            request = build_application_open_request(
                [Path(tmpdir) / "带 空格.fdmslide"],
                source="test",
                request_id="request-1",
            )
            decoded = decode_application_open_request(
                encode_application_open_request(request).rstrip(b"\n")
            )

        self.assertEqual(decoded, request)
        unknown = json.dumps(
            {
                "version": 1,
                "request_id": "request-2",
                "source": "test",
                "activate": True,
                "paths": [],
                "command": "delete",
            }
        ).encode("utf-8")
        with self.assertRaisesRegex(ApplicationOpenRequestError, "未知字段"):
            decode_application_open_request(unknown)
        relative = json.dumps(
            {
                "version": 1,
                "request_id": "request-3",
                "source": "test",
                "activate": True,
                "paths": ["relative.fdmslide"],
            }
        ).encode("utf-8")
        with self.assertRaisesRegex(ApplicationOpenRequestError, "必须是绝对路径"):
            decode_application_open_request(relative)
        invalid_types = json.dumps(
            {
                "version": 1,
                "request_id": {"unexpected": True},
                "source": "test",
                "activate": True,
                "paths": [],
            }
        ).encode("utf-8")
        with self.assertRaisesRegex(ApplicationOpenRequestError, "必须是字符串"):
            decode_application_open_request(invalid_types)
        with self.assertRaisesRegex(ApplicationOpenRequestError, "超过大小限制"):
            decode_application_open_request(b"x" * (MAX_IPC_MESSAGE_BYTES + 1))

    def test_server_name_is_stable_and_scoped_to_installation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = single_instance_server_name(
                app_data_directory=root / "user",
                executable_path=root / "install-a" / "FiberDiameterMeasurement.exe",
            )
            repeated = single_instance_server_name(
                app_data_directory=root / "user",
                executable_path=root / "install-a" / "FiberDiameterMeasurement.exe",
            )
            other = single_instance_server_name(
                app_data_directory=root / "user",
                executable_path=root / "install-b" / "FiberDiameterMeasurement.exe",
            )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)

    def test_secondary_instance_forwards_request_to_primary(self) -> None:
        server_name = f"fdm-t-{uuid.uuid4().hex[:16]}"
        primary = SingleInstanceCoordinator(server_name)
        secondary = SingleInstanceCoordinator(server_name)
        activation = build_application_open_request([], source="test")
        primary_result = primary.start_or_forward(activation, timeout_ms=100)
        self.assertTrue(primary_result.primary, primary_result.error)
        received = []
        primary.requestReceived.connect(received.append)
        with TemporaryDirectory() as tmpdir:
            request = build_application_open_request(
                [Path(tmpdir) / "转发 切片.fdmslide"],
                source="test",
                request_id="forward-1",
            )
            result = secondary.start_or_forward(request, timeout_ms=500)
            deadline = time.monotonic() + 1.0
            while not received and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)

        try:
            self.assertTrue(result.forwarded)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].request_id, "forward-1")
            self.assertEqual(received[0].source, "ipc")
            self.assertEqual(received[0].paths[0].name, "转发 切片.fdmslide")
        finally:
            secondary.close()
            primary.close()
            self._drain_deferred_socket_deletes()

    def test_secondary_without_paths_requests_activation(self) -> None:
        server_name = f"fdm-t-{uuid.uuid4().hex[:16]}"
        primary = SingleInstanceCoordinator(server_name)
        secondary = SingleInstanceCoordinator(server_name)
        activation = build_application_open_request([], source="test", request_id="activate-1")
        primary_result = primary.start_or_forward(activation, timeout_ms=100)
        self.assertTrue(primary_result.primary, primary_result.error)
        received = []
        primary.requestReceived.connect(received.append)
        try:
            result = secondary.start_or_forward(activation, timeout_ms=500)
            deadline = time.monotonic() + 1.0
            while not received and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)

            self.assertTrue(result.forwarded)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].paths, ())
            self.assertTrue(received[0].activate)
        finally:
            secondary.close()
            primary.close()
            self._drain_deferred_socket_deletes()

    def test_listener_race_retries_forwarding(self) -> None:
        coordinator = SingleInstanceCoordinator(f"fdm-t-{uuid.uuid4().hex[:16]}")
        request = build_application_open_request([], source="test")
        try:
            with (
                patch.object(coordinator, "_forward_request", side_effect=[False, True]) as forward_mock,
                patch.object(coordinator._server, "listen", return_value=False),
            ):
                result = coordinator.start_or_forward(request, timeout_ms=100)

            self.assertTrue(result.forwarded)
            self.assertFalse(result.primary)
            self.assertEqual(forward_mock.call_count, 2)
        finally:
            coordinator.close()

    def test_ownership_lock_prevents_second_server_when_primary_pipe_is_unavailable(self) -> None:
        server_name = f"fdm-t-{uuid.uuid4().hex[:16]}"
        primary = SingleInstanceCoordinator(server_name)
        secondary = SingleInstanceCoordinator(server_name)
        request = build_application_open_request([], source="test")
        primary_result = primary.start_or_forward(request, timeout_ms=100)
        self.assertTrue(primary_result.primary, primary_result.error)
        primary._server.close()
        try:
            result = secondary.start_or_forward(request, timeout_ms=40)

            self.assertFalse(result.primary)
            self.assertFalse(result.forwarded)
            self.assertIn("单实例锁", result.error)
            self.assertFalse(secondary._server.isListening())
        finally:
            secondary.close()
            primary.close()

    def test_primary_buffers_fragmented_ipc_message_until_newline(self) -> None:
        server_name = f"fdm-t-{uuid.uuid4().hex[:16]}"
        primary = SingleInstanceCoordinator(server_name)
        activation = build_application_open_request([], source="test")
        primary_result = primary.start_or_forward(activation, timeout_ms=100)
        self.assertTrue(primary_result.primary, primary_result.error)
        received = []
        primary.requestReceived.connect(received.append)
        with TemporaryDirectory() as tmpdir:
            request = build_application_open_request(
                [Path(tmpdir) / "fragment.fdmslide"],
                source="test",
                request_id="fragment-1",
            )
            payload = encode_application_open_request(request)
            split_at = len(payload) // 2
            socket = QLocalSocket()
            socket.connectToServer(server_name)
            self.assertTrue(socket.waitForConnected(500))
            socket.write(payload[:split_at])
            self.assertTrue(socket.waitForBytesWritten(500))
            for _index in range(5):
                self.app.processEvents()
            self.assertEqual(received, [])

            socket.write(payload[split_at:])
            self.assertTrue(socket.waitForBytesWritten(500))
            deadline = time.monotonic() + 1.0
            while not received and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)

        try:
            self.assertEqual([item.request_id for item in received], ["fragment-1"])
        finally:
            socket.abort()
            primary.close()
            self._drain_deferred_socket_deletes()

    def test_repeated_close_defers_child_destruction_to_coordinator_owner(self) -> None:
        """Closing IPC repeatedly must not strand socket deletion events."""

        for _index in range(12):
            server_name = f"fdm-t-{uuid.uuid4().hex[:16]}"
            primary = SingleInstanceCoordinator(server_name)
            activation = build_application_open_request([], source="test")
            result = primary.start_or_forward(activation, timeout_ms=100)
            self.assertTrue(result.primary, result.error)

            client = QLocalSocket()
            client.connectToServer(server_name)
            self.assertTrue(client.waitForConnected(500))
            deadline = time.monotonic() + 1.0
            while not primary._connections and time.monotonic() < deadline:
                self.app.processEvents()
            self.assertTrue(primary._connections)
            accepted_socket = next(iter(primary._connections.values()))[0]

            primary.close()
            client.abort()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.app.processEvents()
            self.assertTrue(shiboken6.isValid(accepted_socket))

            primary.deleteLater()
            QCoreApplication.sendPostedEvents(primary, QEvent.Type.DeferredDelete)
            self.assertFalse(shiboken6.isValid(accepted_socket))

    def test_microview_helper_keeps_priority_over_single_instance_startup(self) -> None:
        helper_module = types.ModuleType("fdm.microview_helper")
        helper_main = Mock(return_value=7)
        helper_module.main = helper_main

        with patch.dict("sys.modules", {"fdm.microview_helper": helper_module}):
            result = fdm_app.main(
                ["fdm", "--microview-helper", "--input", "sample.fdmproj"]
            )

        self.assertEqual(result, 7)
        helper_main.assert_called_once_with(["--input", "sample.fdmproj"])


if __name__ == "__main__":
    unittest.main()
