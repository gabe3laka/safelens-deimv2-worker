"""
worker_security.py -- shared-secret authentication for the SafeLens worker.

Every route except the liveness probes (/health, /ping) requires a shared
secret supplied by the Supabase deimv2-proxy (which already verifies the user's
JWT). Without this, anyone with the RunPod URL could submit frames directly.

  * HTTP: an ASGI middleware checks the secret header on every non-public path.
  * WS  : /ws/vision and /ws/echo authenticate ON CONNECT (header or ?token=).
  * The secret is NEVER in code -- only env WORKER_SHARED_SECRET.
  * Compatibility/testing mode: if WORKER_SHARED_SECRET is unset/empty, auth is
    DISABLED (local tests + backwards-compatible). A warning is logged once.

Comparisons use hmac.compare_digest (constant-time). The secret/header are never
logged (see worker_runtime.redact).
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Optional, Set

log = logging.getLogger("safelens-vision-worker.security")

_DEFAULT_PUBLIC = {"/health", "/ping"}
_warned = False


def shared_secret() -> str:
    return os.getenv("WORKER_SHARED_SECRET", "").strip()


def auth_enabled() -> bool:
    """Auth is on only when a secret is configured (else compat/test mode)."""
    enabled = bool(shared_secret())
    global _warned
    if not enabled and not _warned:
        log.warning("WORKER_SHARED_SECRET not set -- worker auth DISABLED "
                    "(compatibility/testing mode; set it in production).")
        _warned = True
    return enabled


def auth_header_name() -> str:
    return os.getenv("WORKER_AUTH_HEADER", "x-worker-secret").strip().lower()


def public_paths() -> Set[str]:
    paths = set(_DEFAULT_PUBLIC)
    extra = os.getenv("WORKER_PUBLIC_PATHS", "")
    for p in extra.split(","):
        p = p.strip()
        if p:
            paths.add(p)
    return paths


def is_public(path: str) -> bool:
    return path in public_paths()


def _matches(provided: Optional[str]) -> bool:
    secret = shared_secret()
    if not secret:
        return True  # compat/test mode
    if not provided:
        return False
    # Accept "Bearer <secret>" or the raw secret.
    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()
    return hmac.compare_digest(provided, secret)


def check_http(headers) -> bool:
    """True if the request carries a valid secret (or auth is disabled)."""
    if not auth_enabled():
        return True
    name = auth_header_name()
    provided = None
    try:
        provided = headers.get(name) or headers.get("authorization")
    except Exception:  # noqa: BLE001
        provided = None
    return _matches(provided)


def check_ws(websocket) -> bool:
    """Authenticate a WebSocket on connect (header or ?token=/?secret=)."""
    if not auth_enabled():
        return True
    name = auth_header_name()
    provided = None
    try:
        provided = websocket.headers.get(name) or websocket.headers.get("authorization")
        if not provided:
            qp = websocket.query_params
            provided = qp.get("token") or qp.get("secret")
    except Exception:  # noqa: BLE001
        provided = None
    return _matches(provided)


def config() -> dict:
    """Non-sensitive auth snapshot for /debug/state (NEVER includes the secret)."""
    return {
        "auth_enabled": auth_enabled(),
        "auth_header": auth_header_name(),
        "public_paths": sorted(public_paths()),
        "secret_configured": bool(shared_secret()),
    }
