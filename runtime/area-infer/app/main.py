from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.engine import InferServiceError, engine
from app.request_limits import RequestSizeLimitMiddleware
from app.security import (
    AuthConfigurationError,
    AuthenticationError,
    auth_status,
    authenticate,
)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


MAX_REQUEST_BYTES = _env_int(
    "AREA_MAX_REQUEST_BYTES",
    64 * 1024 * 1024,
    minimum=1024,
    maximum=256 * 1024 * 1024,
)
MAX_CONCURRENT_INFER = _env_int(
    "AREA_MAX_CONCURRENT_INFER",
    1,
    minimum=1,
    maximum=4,
)
_INFER_GATE = threading.BoundedSemaphore(MAX_CONCURRENT_INFER)


app = FastAPI(title="area-infer", version="0.2.0")
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)
logger = logging.getLogger("area-infer")


class WarmupRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=128)
    model_file: str = Field(..., min_length=1, max_length=255)


class InferRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=128)
    model_file: str = Field(..., min_length=1, max_length=255)
    image_bytes_b64: str = Field(..., min_length=1, max_length=MAX_REQUEST_BYTES)
    inference_options: dict[str, Any] | None = None


def require_api_auth(authorization: str | None = Header(default=None)) -> str:
    try:
        return authenticate(authorization)
    except AuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    configured, mode = auth_status()
    if not configured:
        raise HTTPException(status_code=503, detail=mode)
    try:
        payload = engine.readiness()
    except InferServiceError as exc:
        logger.error("ready_failed code=%s message=%s", exc.code, exc.message)
        raise HTTPException(status_code=503, detail=exc.code) from exc
    return {**payload, "auth_mode": mode}


@app.get("/health", dependencies=[Depends(require_api_auth)])
def health() -> dict[str, Any]:
    try:
        return engine.health()
    except InferServiceError as exc:
        logger.error("health_failed code=%s message=%s", exc.code, exc.message)
        raise HTTPException(status_code=503, detail=exc.code) from exc


@app.post("/v1/warmup", dependencies=[Depends(require_api_auth)])
def warmup(payload: WarmupRequest) -> dict[str, Any]:
    if not _INFER_GATE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="infer_capacity_exhausted")
    try:
        return engine.warmup(
            model_name=payload.model_name,
            model_file=payload.model_file,
        )
    except InferServiceError as exc:
        logger.error("warmup_failed code=%s message=%s", exc.code, exc.message)
        status = 400 if exc.code == "infer_model_load_failed" else 503
        raise HTTPException(status_code=status, detail=exc.code) from exc
    finally:
        _INFER_GATE.release()


@app.post("/v1/infer", dependencies=[Depends(require_api_auth)])
def infer(payload: InferRequest) -> dict[str, Any]:
    if not _INFER_GATE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="infer_capacity_exhausted")
    try:
        return engine.infer(
            model_name=payload.model_name,
            model_file=payload.model_file,
            image_bytes_b64=payload.image_bytes_b64,
            inference_options=payload.inference_options,
        )
    except InferServiceError as exc:
        logger.error("infer_failed code=%s message=%s", exc.code, exc.message)
        if exc.code == "infer_timeout":
            status = 504
        elif exc.code == "infer_request_too_large":
            status = 413
        elif exc.code in {"infer_model_load_failed", "infer_bad_response"}:
            status = 400
        else:
            status = 503
        raise HTTPException(status_code=status, detail=exc.code) from exc
    finally:
        _INFER_GATE.release()
