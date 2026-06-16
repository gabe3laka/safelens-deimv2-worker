"""
ultralytics_loader.py -- generic Ultralytics detector adapter (A1.3).

Routes the generic YOLO family (YOLO11 / YOLO26 / YOLOE) through the existing
``yolo26_loader`` machinery -- the same Ultralytics ``YOLO()`` loading, lazy
task models, and 0..1 normalization -- so ``VISION_BACKEND=ultralytics`` works
while ``yolo26_loader`` remains the backward-compatible implementation.

This is the minimal "generic adapter layer" from the upgrade plan: a thin,
import-light wrapper (no torch at module top) that delegates at call time, so
``yolo26_loader`` can still be monkeypatched in tests and a fuller
``detectors/`` package can replace the internals later without changing callers.

The active detector model id resolves the generic ``YOLO_DET_MODEL_ID`` first,
then legacy ``YOLO26_DET_MODEL_ID`` / ``YOLO26_MODEL_ID`` (see
``yolo26_loader._model_id`` / ``config_resolver.resolve_detector_model_id``).
"""

from __future__ import annotations

from typing import Any, List, Optional


def load(tasks: Optional[List[str]] = None, device: Optional[str] = None) -> Any:
    """Warmup the active YOLO/Ultralytics detector (delegates to yolo26_loader)."""
    import yolo26_loader
    return yolo26_loader.load(tasks=tasks, device=device)


def infer(pil_img: Any, conf: float, class_filter: Optional[List[int]] = None,
          tasks: Optional[List[str]] = None, img_size: Optional[int] = None,
          iou: Optional[float] = None, max_det: Optional[int] = None) -> Any:
    """Run inference through the active YOLO/Ultralytics detector."""
    import yolo26_loader
    return yolo26_loader.infer(
        pil_img, conf, class_filter=class_filter, tasks=tasks,
        img_size=img_size, iou=iou, max_det=max_det)


def crop_analysis(pil_img: Any, conf: float, mode: str = "build", **kwargs: Any) -> Any:
    import yolo26_loader
    return yolo26_loader.crop_analysis(pil_img, conf, mode=mode, **kwargs)


def status() -> Any:
    import yolo26_loader
    return yolo26_loader.status()


def is_ready() -> bool:
    import yolo26_loader
    return yolo26_loader.is_ready()


def mode_tasks(mode: str = "live") -> List[str]:
    import yolo26_loader
    return yolo26_loader.mode_tasks(mode)


def active_model_id() -> str:
    """Resolved active detector model id (generic-first, legacy fallback)."""
    import config_resolver
    return config_resolver.resolve_detector_model_id()
