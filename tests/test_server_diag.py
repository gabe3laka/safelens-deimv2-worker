"""
tests/test_server_diag.py -- Tests for the DEIMv2 worker (official loader).

Run with: pytest tests/test_server_diag.py -v

These tests validate the OFFICIAL DEIMv2 loading path (PyTorchModelHubMixin),
not the old transformers Auto-class path. They use FastAPI TestClient + mocks
so they never download weights, never hit a GPU, and never crash the worker:

  * default model id is Intellindust/DEIMv2_DINOv3_S_COCO
  * the old bad id (Intellindust-AI-Lab/DEIMv2-S) is no longer the default
  * the official DEIMv2 loader path is used for DEIMv2 model ids
  * POST /debug/model-load uses the official loader (backend field)
  * GET /debug/startup?deep=true includes official DEIMv2 import diagnostics
  * HF token is never printed / exposed in diagnostics
  * GET /health works without a model loaded
  * POST /detect returns model_not_ready (503) when the model is cold

TestClient-based tests are skipped automatically if httpx / TestClient are not
installed (the loader/default-id tests do not need them and always run).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

OFFICIAL_ID = "Intellindust/DEIMv2_DINOv3_S_COCO"
OLD_BAD_ID = "Intellindust-AI-Lab/DEIMv2-S"

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

# -- 1. Default model id is the official DEIMv2-S id (always runs) ------------

class TestDefaultModelId(unittest.TestCase):
    def test_infer_default_is_official(self):
        import deimv2_infer
        self.assertEqual(deimv2_infer.DEFAULT_MODEL_ID, OFFICIAL_ID)

    def test_loader_default_is_official(self):
        import official_deimv2_loader
        self.assertEqual(official_deimv2_loader.DEFAULT_MODEL_ID, OFFICIAL_ID)

    def test_old_bad_id_not_default_anywhere(self):
        import deimv2_infer, official_deimv2_loader
        self.assertNotEqual(deimv2_infer.DEFAULT_MODEL_ID, OLD_BAD_ID)
        self.assertNotEqual(official_deimv2_loader.DEFAULT_MODEL_ID, OLD_BAD_ID)
        if _HAVE_TESTCLIENT:
            self.assertNotEqual(server.DEFAULT_MODEL_ID, OLD_BAD_ID)

# -- 2. Official loader path is used for DEIMv2 (always runs) -----------------

class TestOfficialLoaderPath(unittest.TestCase):
    def test_load_model_calls_official_loader(self):
        """_load_model() (default backend) routes to the official loader."""
        import deimv2_infer
        deimv2_infer._model = None
        deimv2_infer._processor = None
        deimv2_infer._backend = None

        fake_model = MagicMock(name="deimv2_model")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEIMV2_BACKEND", None)
            with patch("official_deimv2_loader.load_official_deimv2",
                       return_value=(fake_model, "cpu", "DEIMv2")) as p_load:
                deimv2_infer._load_model()
                self.assertTrue(p_load.called)
                self.assertEqual(deimv2_infer._backend, "official-deimv2-hf")

        deimv2_infer._model = None
        deimv2_infer._processor = None
        deimv2_infer._backend = None

    def test_load_model_reraises_on_failure(self):
        import deimv2_infer
        deimv2_infer._model = None
        deimv2_infer._backend = None

        def _boom(*a, **k):
            raise RuntimeError("simulated official load failure")

        os.environ.pop("DEIMV2_BACKEND", None)
        with patch("official_deimv2_loader.load_official_deimv2", _boom):
            with self.assertRaises(RuntimeError):
                deimv2_infer._load_model()

        deimv2_infer._model = None
        deimv2_infer._backend = None

# -- 3. /health works without a model loaded ---------------------------------

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

# -- 4. /debug/startup?deep=true official DEIMv2 import diagnostics -----------

@_skip_no_client
class TestStartupDiagnostics(unittest.TestCase):
    def test_shallow_startup_no_secret_values(self):
        with _client() as client:
            r = client.get("/debug/startup")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertTrue(body["ok"])
            self.assertIn("env_names", body)
            self.assertNotIn("env_values", body)

    def test_deep_startup_has_official_imports(self):
        with _client() as client:
            r = client.get("/debug/startup", params={"deep": "true"})
            self.assertEqual(r.status_code, 200)
            diag = r.json()["diagnostics"]
            self.assertIn("imports", diag)
            self.assertIn("versions", diag)
            # Official DEIMv2 import diagnostics must be present.
            for key in ("engine.backbone", "engine.deim",
                        "engine.deim.postprocessor", "official_deimv2_loader",
                        "PyTorchModelHubMixin"):
                self.assertIn(key, diag["imports"])

    def test_auto_image_processor_is_optional_not_required(self):
        """AutoImageProcessor may appear but is documented as optional."""
        with _client() as client:
            diag = r = client.get("/debug/startup", params={"deep": "true"}).json()["diagnostics"]
            notes = diag.get("notes", {})
            self.assertIn("AutoImageProcessor", notes)
            self.assertIn("optional", notes["AutoImageProcessor"].lower())

    def test_import_summary_values_are_ok_or_error(self):
        with _client() as client:
            diag = client.get("/debug/startup", params={"deep": "true"}).json()["diagnostics"]
            for k, v in diag["imports"].items():
                self.assertIn(v, ("ok", "error"), f"{k} had unexpected status {v}")

# -- 5. /debug/model-load uses the official loader + never crashes ------------

@_skip_no_client
class TestModelLoadRoute(unittest.TestCase):
    def test_model_load_reports_backend_and_model_id(self):
        with _client() as client:
            r = client.post("/debug/model-load")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertIn("ok", body)
            self.assertIn("backend", body)
            self.assertEqual(body["backend"], "official-deimv2-hf")
            self.assertIn("model_id", body)
            self.assertNotEqual(body["model_id"], OLD_BAD_ID)

    def test_model_load_failure_includes_traceback(self):
        def _boom(*a, **k):
            raise RuntimeError("simulated official load failure")
        os.environ.pop("DEIMV2_BACKEND", None)
        with patch("official_deimv2_loader.load_official_deimv2", _boom):
            with _client() as client:
                r = client.post("/debug/model-load")
                self.assertEqual(r.status_code, 200)  # never crashes
                body = r.json()
                self.assertFalse(body["ok"])
                self.assertIn("exception_type", body)
                self.assertIn("traceback", body)

    def test_model_load_success_path(self):
        fake_model = MagicMock(name="deimv2_model")
        os.environ.pop("DEIMV2_BACKEND", None)
        with patch("official_deimv2_loader.load_official_deimv2",
                   return_value=(fake_model, "cpu", "DEIMv2")):
            with _client() as client:
                body = client.post("/debug/model-load").json()
                self.assertTrue(body["ok"])
                self.assertEqual(body.get("model_class"), "DEIMv2")

# -- 6. HF token is never printed / exposed ----------------------------------

class TestNoTokenLeak(unittest.TestCase):
    def test_startup_never_exposes_token_value(self):
        if not _HAVE_TESTCLIENT:
            self.skipTest("TestClient unavailable")
        secret = "hf_THISISASECRETTOKENVALUE123456"
        with patch.dict(os.environ, {"HF_TOKEN": secret}):
            with _client() as client:
                text = client.get("/debug/startup", params={"deep": "true"}).text
                self.assertNotIn(secret, text)
                # env NAME may appear, value must not.
                self.assertIn("HF_TOKEN", client.get("/debug/startup").json()["env_names"])

    def test_loader_hf_token_helper_reads_env(self):
        import official_deimv2_loader
        with patch.dict(os.environ, {"HF_TOKEN": "hf_abc"}):
            self.assertEqual(official_deimv2_loader._hf_token(), "hf_abc")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(official_deimv2_loader._hf_token())

# -- 7. /detect returns model_not_ready (503) when cold ----------------------

@_skip_no_client
class TestDetectValidation(unittest.TestCase):
    def test_detect_model_not_ready(self):
        with server._STATE_LOCK:
            server._STATE["model"] = None
            server._STATE["status"] = "cold"
        with _client() as client:
            r = client.post("/detect", json={"image_b64": "aGVsbG8="})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["error"], "model_not_ready")

    def test_detect_missing_image_when_ready(self):
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
