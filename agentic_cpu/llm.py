"""
agentic_cpu/llm.py -- the LLM seam for the CPU agents.

Default provider is `mock`: a deterministic, key-free echo used by tests and CPU
integration. An external provider (e.g. OpenAI-compatible / Anthropic) can be
wired at deploy time via env (CPU_AGENT_LLM_PROVIDER / model + an API key in the
environment, NEVER baked into the image). The agents are written to work fully in
mock mode, so a real key is never required.

This module imports NO GPU deps; httpx (already a worker dep) is imported lazily
only when a real provider is configured.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from . import config

log = logging.getLogger("safelens-vision-worker.agentic.llm")


def available() -> bool:
    """True if a usable LLM backend is configured (mock is always available)."""
    return config.llm_provider() == "mock" or _has_external()


def _has_external() -> bool:
    import os
    provider = config.llm_provider()
    if provider in ("openai", "azure-openai"):
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False


def complete(prompt: str, *, system: Optional[str] = None,
             context: Optional[Dict[str, Any]] = None) -> str:
    """Return a completion string. Mock mode returns a deterministic summary;
    a real provider is called only if configured. Never raises -- on any error it
    degrades to the mock response so the agent still produces a draft."""
    if config.llm_provider() == "mock":
        return _mock_complete(prompt, context)
    try:
        return _external_complete(prompt, system, context)
    except Exception as exc:  # noqa: BLE001
        log.warning("llm: external provider failed (%s); using mock", exc)
        return _mock_complete(prompt, context)


def _mock_complete(prompt: str, context: Optional[Dict[str, Any]]) -> str:
    """Deterministic, weight/key-free 'completion' for tests + CPU integration."""
    return json.dumps({
        "mock": True,
        "provider": "mock",
        "echo_len": len(prompt or ""),
        "context_keys": sorted((context or {}).keys()),
        "note": "mock LLM output; agents build structured drafts deterministically.",
    })


def _external_complete(prompt: str, system: Optional[str],
                       context: Optional[Dict[str, Any]]) -> str:
    """Best-effort OpenAI/Anthropic-compatible call (lazy httpx). Only reached if
    a provider + key are configured; otherwise complete() stays on the mock path."""
    import os
    import httpx
    provider = config.llm_provider()
    model = config.llm_model()
    if provider in ("openai", "azure-openai"):
        key = os.getenv("OPENAI_API_KEY")
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model,
                  "messages": [m for m in (
                      {"role": "system", "content": system} if system else None,
                      {"role": "user", "content": prompt}) if m]},
            timeout=20.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 1024,
                  "system": system or "", "messages": [{"role": "user", "content": prompt}]},
            timeout=20.0,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    raise RuntimeError(f"unsupported llm provider: {provider}")
