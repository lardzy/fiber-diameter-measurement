from __future__ import annotations

from collections.abc import Mapping
import os
import secrets


class AuthConfigurationError(RuntimeError):
    pass


class AuthenticationError(RuntimeError):
    pass


def _env_bool(environ: Mapping[str, str], name: str, default: bool = False) -> bool:
    token = str(environ.get(name, "")).strip().lower()
    if not token:
        return default
    return token in {"1", "true", "yes", "on"}


def auth_status(environ: Mapping[str, str] | None = None) -> tuple[bool, str]:
    environment = os.environ if environ is None else environ
    token = str(environment.get("AREA_API_TOKEN", "")).strip()
    if token:
        if len(token) < 16:
            return False, "api_token_too_short"
        return True, "token"
    if _env_bool(environment, "AREA_ALLOW_ANONYMOUS_DEV", False):
        return True, "anonymous-dev"
    return False, "api_auth_not_configured"


def authenticate(authorization: str | None, environ: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environ is None else environ
    configured, mode = auth_status(environment)
    if not configured:
        raise AuthConfigurationError(mode)
    if mode == "anonymous-dev":
        return mode
    expected = str(environment.get("AREA_API_TOKEN", "")).strip()
    scheme, _, supplied = str(authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not secrets.compare_digest(supplied, expected):
        raise AuthenticationError("invalid_api_token")
    return "token"
