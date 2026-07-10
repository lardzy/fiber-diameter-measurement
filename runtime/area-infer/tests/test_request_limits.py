from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from app.request_limits import RequestSizeLimitMiddleware


class RequestLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_length_and_chunked_bodies_are_bounded(self) -> None:
        downstream_bodies: list[bytes] = []

        async def downstream(scope, receive, send) -> None:
            request = await receive()
            downstream_bodies.append(request.get("body") or b"")
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = RequestSizeLimitMiddleware(downstream, max_bytes=4)

        oversized_messages: list[dict[str, object]] = []

        async def unused_receive():
            raise AssertionError("oversized Content-Length must be rejected before reading")

        async def collect_oversized(message):
            oversized_messages.append(message)

        await middleware(
            {
                "type": "http",
                "method": "POST",
                "headers": [(b"content-length", b"5")],
            },
            unused_receive,
            collect_oversized,
        )
        self.assertEqual(oversized_messages[0]["status"], 413)

        chunks = iter(
            [
                {"type": "http.request", "body": b"abc", "more_body": True},
                {"type": "http.request", "body": b"de", "more_body": False},
            ]
        )
        chunked_messages: list[dict[str, object]] = []

        async def receive_chunk():
            return next(chunks)

        async def collect_chunked(message):
            chunked_messages.append(message)

        await middleware(
            {"type": "http", "method": "POST", "headers": []},
            receive_chunk,
            collect_chunked,
        )
        self.assertEqual(chunked_messages[0]["status"], 413)
        self.assertEqual(downstream_bodies, [])

    async def test_body_at_limit_is_replayed_to_downstream(self) -> None:
        received_body = b""

        async def downstream(scope, receive, send) -> None:
            nonlocal received_body
            chunks: list[bytes] = []
            while True:
                request = await receive()
                chunks.append(request.get("body") or b"")
                if not request.get("more_body", False):
                    break
            received_body = b"".join(chunks)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        messages = iter(
            [
                {"type": "http.request", "body": b"ab", "more_body": True},
                {"type": "http.request", "body": b"cd", "more_body": False},
            ]
        )
        sent: list[dict[str, object]] = []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        middleware = RequestSizeLimitMiddleware(downstream, max_bytes=4)
        await middleware({"type": "http", "method": "POST", "headers": []}, receive, send)

        self.assertEqual(received_body, b"abcd")
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(sent[1]["body"], b"ok")

    async def test_protected_endpoint_auth_is_rejected_before_body_read(self) -> None:
        async def downstream(scope, receive, send) -> None:
            raise AssertionError("unauthenticated request must not reach downstream")

        async def receive():
            raise AssertionError("authentication must be checked before buffering the body")

        sent: list[dict[str, object]] = []

        async def send(message):
            sent.append(message)

        middleware = RequestSizeLimitMiddleware(downstream, max_bytes=1024)
        with patch.dict(os.environ, {}, clear=True):
            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/infer",
                    "headers": [(b"content-length", b"1000")],
                },
                receive,
                send,
            )

        self.assertEqual(sent[0]["status"], 503)
        self.assertEqual(json.loads(sent[1]["body"]), {"detail": "api_auth_not_configured"})

    async def test_error_payload_is_json(self) -> None:
        sent: list[dict[str, object]] = []

        async def downstream(scope, receive, send) -> None:
            raise AssertionError("invalid Content-Length must not reach downstream")

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        middleware = RequestSizeLimitMiddleware(downstream, max_bytes=4)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "headers": [(b"content-length", b"invalid")],
            },
            receive,
            send,
        )

        self.assertEqual(sent[0]["status"], 400)
        self.assertEqual(json.loads(sent[1]["body"]), {"detail": "invalid_content_length"})


if __name__ == "__main__":
    unittest.main()
