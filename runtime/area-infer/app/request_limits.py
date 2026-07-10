from __future__ import annotations

import json

from app.security import AuthConfigurationError, AuthenticationError, authenticate


class RequestSizeLimitMiddleware:
    def __init__(self, app, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        if str(scope.get("path") or "").startswith("/v1/"):
            raw_authorization = headers.get(b"authorization", b"")
            try:
                authorization = raw_authorization.decode("latin-1")
                authenticate(authorization)
            except AuthConfigurationError as exc:
                await _send_error(send, status_code=503, detail=str(exc))
                return
            except (AuthenticationError, UnicodeDecodeError) as exc:
                await _send_error(send, status_code=401, detail=str(exc) or "invalid_api_token")
                return
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await _send_error(send, status_code=413, detail="request_too_large")
                    return
            except ValueError:
                await _send_error(send, status_code=400, detail="invalid_content_length")
                return

        received = 0
        body_chunks: list[bytes] = []
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                await _send_error(send, status_code=400, detail="invalid_request_stream")
                return
            chunk = message.get("body") or b""
            received += len(chunk)
            if received > self.max_bytes:
                await _send_error(send, status_code=413, detail="request_too_large")
                return
            body_chunks.append(chunk)
            if not message.get("more_body", False):
                break

        replay_index = 0

        async def replay_receive():
            nonlocal replay_index
            if replay_index >= len(body_chunks):
                return {"type": "http.disconnect"}
            chunk = body_chunks[replay_index]
            replay_index += 1
            return {
                "type": "http.request",
                "body": chunk,
                "more_body": replay_index < len(body_chunks),
            }

        await self.app(scope, replay_receive, send)


async def _send_error(send, *, status_code: int, detail: str) -> None:
    body = json.dumps(
        {"detail": detail},
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": int(status_code),
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
