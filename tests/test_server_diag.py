"""
tests/test_server_diag.py -- Tests for the DEIMv2 worker diagnostics + model-load.

Run with: pytest tests/test_server_diag.py -v

These tests use FastAPI's TestClient and mocked model loading so they never
download weights, never hit a GPU, and never crash the worker. They validate:

  * GET  /health works without a model loaded
  * GET  /debug/startup?deep=true returns import diagnostics
  * POST /debug/model-load returns a structured result and never raises
  * _load_model() calls from_pretrained with trust_remote_code=True
  * /detect missing-image / invalid-base64 behaviour still works

TestClient-based tests are skipped automatically if httpx / TestClient are not
installed in the environment (the _load_model trust_remote_code test does not
need them and always runs).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

# TestClient requires httpx; degrade gracefully if it is unavailable.
try:
    from fastapi.testclient import TestClient
    import server
    _HAVE_TESTCLIENT = True
    _TESTCLIENT_ERR = None
except Exception as _exc:  # pragma: no cover - environment dependent
    _HAVE_TESTCLIENT = False
    _TESTCLIENT_ERR = f"{type(_exc).__name__}: {_exc}"

_skip_no_client = pytest.mark.skipif(
    not _HAVE_TESTCLIENT,
    reason=f"FastAPI TestClient unavailable: {_TESTCLIENT_ERR}",
)


def _client():
    return TestClient(server.app)


# -- 1. /health works without a model loaded ---------------------------------

@_skip_no_client
class TestHealthNoModel(unittest.TestCase):
    def test_health_returns_200_without_model(self):
        with _client() as client:
            r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertIn("status", body)
        self.assertIn("model_loaded", body)

    def test_ping_alias(self):
        with _client() as client:
            r = client.get("/ping")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])


# -- 2. /debug/startup?deep=true returns import diagnostics -------------------

@_skip_no_client
class TestStartupDiagnostics(unittest.TestCase):
    def test_shallow_startup(self):
        with _client() as client:
            r = client.get("/debug/startup")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertIn("env_names", body)
        # Secrets must never be exposed -- only env NAMES, not values.
        self.assertNotIn("env_values", body)

    def test_deep_startup_has_imports(self):
        with _client() as client:
            r = client.get("/debug/startup", params={"deep": "true"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("diagnostics", body)
        diag = body["diagnostics"]
        self.assertIn("imports", diag)
        self.assertIn("versions", diag)
        for key in ("AutoImageProcessor", "AutoModelForObjectDetection",
                    "timm", "safetensors", "torchvision"):
            self.assertIn(key, diag["imports"])

    def test_import_summary_values_are_ok_or_error(self):
        with _client() as client:
            r = client.get("/debug/startup", params={"deep": "true"})
        diag = r.json()["diagnostics"]
        for k, v in diag["imports"].items():
            self.assertIn(v, ("ok", "error"), f"{k} had unexpected status {v}")


# -- 3. POST /debug/model-load is structured + never crashes -----------------

@_skip_no_client
class TestModelLoadRoute(unittest.TestCase):
    def test_model_load_returns_structured(self):
        with _client() as client:
            r = client.post("/debug/model-load")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("ok", body)
        self.assertIn("model_id", body)
        self.assertIn("device", body)
        self.assertIn("transformers_version", body)

    def test_model_load_failure_includes_traceback(self):
        """On load failure the route must return ok=false + a traceback, not 500."""
        def _boom(*a, **k):
            raise RuntimeError("simulated load failure")

        with patch("transformers.AutoImageProcessor.from_pretrained", _boom):
            with _client() as client:
                r = client.post("/debug/model-load")
        self.assertEqual(r.status_code, 200)  # never crashes
        body = r.json()
        self.assertFalse(body["ok"])
        self.assertIn("exception_type", body)
        self.assertIn("traceback", body)

    def test_model_load_success_path(self):
        """When loaders succeed, route reports ok=true without running inference."""
        fake_proc = MagicMock(name="processor")
        fake_model = MagicMock(name="model")
        with patch("transformers.AutoImageProcessor.from_pretrained", return_value=fake_proc), \
             patch("transformers.AutoModelForObjectDetection.from_pretrained", return_value=fake_model):
            with _client() as client:
                r = client.post("/debug/model-load")
        body = r.json()
        self.assertTrue(body["ok"])


# -- 4. _load_model() uses trust_remote_code=True ----------------------------
# This block does NOT require TestClient/httpx and always runs.

class TestTrustRemoteCode(unittest.TestCase):
    def test_load_model_passes_trust_remote_code(self):
        import deimv2_infer
        deimv2_infer._model = None
        deimv2_infer._processor = None

        fake_proc = MagicMock(name="processor")
        fake_model = MagicMock(name="model")
        with patch("transformers.AutoImageProcessor.from_pretrained", return_value=fake_proc) as p_proc, \
             patch("transformers.AutoModelForObjectDetection.from_pretrained", return_value=fake_model) as p_model:
            deimv2_infer._load_model()

        _, proc_kwargs = p_proc.call_args
        _, model_kwargs = p_model.call_args
        self.assertTrue(proc_kwargs.get("trust_remote_code") is True)
        self.assertTrue(model_kwargs.get("trust_remote_code") is True)

        deimv2_infer._model = None
        deimv2_infer._processor = None

    def test_load_model_reraises_and_logs(self):
        import deimv2_infer
        deimv2_infer._model = None
        deimv2_infer._processor = None

        def _boom(*a, **k):
            raise RuntimeError("ModuleNotFoundError-like failure")

        with patch("transformers.AutoImageProcessor.from_pretrained", _boom):
            with self.assertRaises(RuntimeError):
                deimv2_infer._load_model()

        deimv2_infer._model = None
        deimv2_infer._processor = None


# -- 5. /detect missing-image / invalid-base64 behaviour ---------------------

@_skip_no_client
class TestDetectValidation(unittest.TestCase):
    def test_detect_model_not_ready(self):
        """With no model loaded, /detect should report model_not_ready (503)."""
        with server._STATE_LOCK:
            server._STATE["model"] = None
            server._STATE["status"] = "cold"
        with _client() as client:
            r = client.post("/detect", json={"image_b64": "aGVsbG8="})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"], "model_not_ready")

    def test_detect_missing_image_when_ready(self):
        """When ready but image missing -> 400 missing_image_b64."""
        with server._STATE_LOCK:
            server._STATE["model"] = MagicMock()
            server._STATE["status"] = "ready"
        try:
            with _client() as client:
                r = client.post("/detect", json={})
            self.assertEqual(r.status_code, 400)
            self.assertEqual(r.json()["error"], "missing_image_b64")
        finally:
            with server._STATE_LOCK:
                server._STATE["model"] = None
                server._STATE["status"] = "cold"

    def test_detect_invalid_base64_when_ready(self):
        with server._STATE_LOCK:
            server._STATE["model"] = MagicMock()
            server._STATE["status"] = "ready"
        try:
            with _client() as client:
                r = client.post("/detect", json={"image_b64": "!!!not-valid!!!"})
            self.assertEqual(r.status_code, 400)
            self.assertEqual(r.json()["error"], "invalid_base64")
        finally:
            with server._STATE_LOCK:
                server._STATE["model"] = None
                server._STATE["status"] = "cold"


if __name__ == "__main__":
    unittest.main()
