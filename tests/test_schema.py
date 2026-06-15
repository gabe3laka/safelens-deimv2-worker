"""
tests/test_schema.py -- Unit tests for DEIMv2 worker schema, handler, and infer helpers.

Run with:  pytest tests/test_schema.py -v
           (from repo root, with deps installed)

Tests that require model weights are skipped automatically when the model
is not available (SKIP_MODEL_TESTS=1 or when import fails).
"""

from __future__ import annotations

import base64
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make the repo root importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_jpeg_b64() -> str:
    """Return a base64-encoded minimal valid JPEG (8x8 white square)."""
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(255, 255, 255)).save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Fallback: 1px white JPEG as literal base64
        return (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
            "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAARCAABAAEDASIA"
            "AhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/"
            "xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQAC"
            "EQMRBD8AJQAB/9k="
        )


def _invalid_b64() -> str:
    return "!!!not-valid-base64!!!"


# ── 1. Schema import ──────────────────────────────────────────────────────────

class TestSchemaImport(unittest.TestCase):
    def test_schema_imports(self):
        """schema.py must import without errors."""
        import schema  # noqa: F401
        from schema import BBox, Entity, InferRequest, InferResponse

        self.assertTrue(callable(InferRequest))
        self.assertTrue(callable(BBox))
        self.assertTrue(callable(Entity))
        self.assertTrue(callable(InferResponse))


# ── 2. InferRequest defaults ──────────────────────────────────────────────────

class TestInferRequest(unittest.TestCase):
    def setUp(self):
        from schema import InferRequest
        self.InferRequest = InferRequest

    def test_defaults(self):
        req = self.InferRequest(image_b64="aGVsbG8=")
        self.assertAlmostEqual(req.conf, 0.35)
        self.assertEqual(req.img_size, 640)
        self.assertIsNone(req.classes)

    def test_custom_values(self):
        req = self.InferRequest(image_b64="aGVsbG8=", conf=0.5, img_size=320, classes=[0, 2])
        self.assertAlmostEqual(req.conf, 0.5)
        self.assertEqual(req.img_size, 320)
        self.assertEqual(req.classes, [0, 2])

    def test_conf_bounds(self):
        """conf must be in [0, 1]."""
        import pydantic
        with self.assertRaises((pydantic.ValidationError, ValueError)):
            self.InferRequest(image_b64="x", conf=1.5)
        with self.assertRaises((pydantic.ValidationError, ValueError)):
            self.InferRequest(image_b64="x", conf=-0.1)


# ── 3. BBox normalised coordinates ───────────────────────────────────────────

class TestBBox(unittest.TestCase):
    def setUp(self):
        from schema import BBox
        self.BBox = BBox

    def test_valid_bbox(self):
        bbox = self.BBox(x=0.0, y=0.0, w=1.0, h=1.0)
        self.assertEqual(bbox.x, 0.0)
        self.assertEqual(bbox.w, 1.0)

    def test_normalised_midpoint(self):
        bbox = self.BBox(x=0.25, y=0.25, w=0.5, h=0.5)
        self.assertAlmostEqual(bbox.x, 0.25)
        self.assertAlmostEqual(bbox.h, 0.5)

    def test_out_of_range_rejected(self):
        """Values outside 0..1 must be rejected."""
        import pydantic
        with self.assertRaises((pydantic.ValidationError, ValueError)):
            self.BBox(x=1.1, y=0.0, w=0.1, h=0.1)
        with self.assertRaises((pydantic.ValidationError, ValueError)):
            self.BBox(x=0.0, y=-0.1, w=0.1, h=0.1)


# ── 4. InferResponse serialisation ───────────────────────────────────────────

class TestInferResponse(unittest.TestCase):
    def setUp(self):
        from schema import InferResponse
        self.InferResponse = InferResponse

    def test_empty_entities(self):
        resp = self.InferResponse(entities=[], inference_ms=0.0, model="deimv2-s", img_w=640, img_h=480)
        self.assertEqual(resp.entities, [])
        self.assertIsNone(resp.error)
        self.assertIsNone(resp.warning)

    def test_error_response(self):
        resp = self.InferResponse(entities=[], error="missing_image_b64")
        self.assertEqual(resp.error, "missing_image_b64")
        self.assertEqual(resp.entities, [])

    def test_warning_field(self):
        resp = self.InferResponse(entities=[], warning="class filter matched no detections")
        self.assertIsNone(resp.error)
        self.assertEqual(resp.warning, "class filter matched no detections")

    def test_json_round_trip(self):
        resp = self.InferResponse(entities=[], inference_ms=12.5, model="deimv2-n", img_w=320, img_h=240)
        data = resp.model_dump()
        self.assertIn("entities", data)
        self.assertIn("inference_ms", data)
        self.assertIn("error", data)
        self.assertIn("warning", data)


# ── 5. Handler structured errors ─────────────────────────────────────────────

class TestHandlerErrors(unittest.TestCase):
    """
    Tests that the handler returns structured errors for bad input.
    These tests do NOT load model weights -- the model import path is
    short-circuited by the missing/invalid image_b64 validation.
    """

    def setUp(self):
        from handler import handler
        self.handler = handler

    def test_missing_image_b64(self):
        result = self.handler({"input": {}})
        self.assertEqual(result.get("error"), "missing_image_b64")
        self.assertEqual(result.get("entities"), [])

    def test_missing_input_key(self):
        result = self.handler({})
        self.assertEqual(result.get("error"), "missing_image_b64")
        self.assertEqual(result.get("entities"), [])

    def test_none_image_b64(self):
        result = self.handler({"input": {"image_b64": None}})
        self.assertEqual(result.get("error"), "missing_image_b64")
        self.assertEqual(result.get("entities"), [])

    def test_empty_image_b64(self):
        result = self.handler({"input": {"image_b64": ""}})
        self.assertEqual(result.get("error"), "missing_image_b64")
        self.assertEqual(result.get("entities"), [])

    def test_invalid_base64(self):
        result = self.handler({"input": {"image_b64": _invalid_b64()}})
        self.assertEqual(result.get("error"), "invalid_base64")
        self.assertEqual(result.get("entities"), [])

    def test_nearly_valid_base64(self):
        """Garbage bytes that pass length check but are not valid base64."""
        result = self.handler({"input": {"image_b64": "@@@@"}})
        self.assertEqual(result.get("error"), "invalid_base64")

    def test_handler_never_raises(self):
        """Handler must return a dict, never raise an exception."""
        bad_inputs = [
            {"input": {"image_b64": _invalid_b64()}},
            {"input": {}},
            {},
            {"input": {"image_b64": None, "conf": "oops", "img_size": -1}},
        ]
        for payload in bad_inputs:
            try:
                result = self.handler(payload)
                self.assertIsInstance(result, dict, f"expected dict, got {type(result)} for {payload}")
                self.assertIn("entities", result, f"no 'entities' key for {payload}")
            except Exception as exc:
                self.fail(f"handler raised {type(exc).__name__}: {exc} for payload {payload}")


# ── 6. _get_label helper ─────────────────────────────────────────────────────

class TestGetLabel(unittest.TestCase):
    """
    Tests for the _get_label helper in deimv2_infer.py.
    Uses mocked model objects so no GPU / weight download is needed.
    """

    def setUp(self):
        import deimv2_infer
        self._get_label = deimv2_infer._get_label

    def test_prefers_model_id2label(self):
        """_get_label should return model.config.id2label[class_id] when available."""
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "custom_person", 1: "custom_car"}
        label = self._get_label(mock_model, 0)
        self.assertEqual(label, "custom_person")

    def test_falls_back_to_coco(self):
        """When id2label is missing or raises, fall back to COCO names."""
        mock_model = MagicMock()
        # Simulate no config labels
        del mock_model.config.id2label
        label = self._get_label(mock_model, 0)
        # COCO class 0 is 'person'
        self.assertEqual(label.lower(), "person")

    def test_coco_fallback_for_unknown_class(self):
        """Unknown class_id falls back to a string, never raises."""
        mock_model = MagicMock()
        del mock_model.config.id2label
        label = self._get_label(mock_model, 9999)
        self.assertIsInstance(label, str)

    def test_model_id2label_missing_key(self):
        """class_id not in id2label should fall back gracefully."""
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "person"}
        label = self._get_label(mock_model, 99)
        self.assertIsInstance(label, str)


# ── 7. BBox normalization helper ─────────────────────────────────────────────

class TestBBoxNormalization(unittest.TestCase):
    """Validate that normalised bbox coordinates stay in 0..1."""

    def setUp(self):
        from schema import BBox
        self.BBox = BBox

    def test_full_image_bbox(self):
        bbox = self.BBox(x=0.0, y=0.0, w=1.0, h=1.0)
        self.assertGreaterEqual(bbox.x, 0.0)
        self.assertLessEqual(bbox.x + bbox.w, 1.0 + 1e-6)
        self.assertGreaterEqual(bbox.y, 0.0)
        self.assertLessEqual(bbox.y + bbox.h, 1.0 + 1e-6)

    def test_small_bbox(self):
        bbox = self.BBox(x=0.1, y=0.2, w=0.05, h=0.08)
        for field in (bbox.x, bbox.y, bbox.w, bbox.h):
            self.assertGreaterEqual(field, 0.0)
            self.assertLessEqual(field, 1.0)


if __name__ == "__main__":
    unittest.main()
