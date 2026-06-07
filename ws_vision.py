"""
ws_vision.py -- FluxRT-style streaming WebSocket route (/ws/vision).

This module adds a *real* low-latency streaming endpoint on top of the existing
worker. It does NOT load or reload any model: it reuses the already
warmed/loaded backend (EdgeCrafter by default) through the worker's existing
warmup + inference path, which is injected by server.py via register_ws_vision().

Why a separate module
---------------------
* Keeps server.py's existing HTTP routes (/health, /ping, /debug/*, /warmup,
  /detect) and the Phase-0 /ws/echo probe untouched.
* Stays import-light (no torch / no model imports at module top) so importing it
  never breaks the live server's boot, exactly like server.py / vision_backend.py.
* Dependencies (state lookup, warmup trigger, inference call, backend/tasks/gpu
  accessors) are injected, so the streaming logic is unit-testable on CPU with
  mock inference -- no GPU required.

Protocol (JSON text frames)
----------------------------
Client -> server:
    {"type":"frame","camera_id":"browser-test","frame_id":123,
     "sent_at":1710000000000,"frame_b64":"<jpeg-base64>"}
    {"type":"ping"}                      (optional keepalive)

Server -> client:
    {"type":"connected"}                  immediately on accept
    {"type":"warming"}                    if the model is cold (warmup triggered)
    {"type":"ready","backend":"edgecrafter","tasks":["det","pose"]}
    {"type":"vision","camera_id":...,"frame_id":...,"backend":"edgecrafter",
     "tasks":["det","pose"],"entities":[...],"poses":[...],"model":"EdgeCrafter",
     "inference_ms":123,"img_w":640,"img_h":480,"received_at":...,
     "completed_at":...,"end_to_end_latency_ms":220}
    {"type":"metrics", ...}               periodic stream stats
    {"type":"error","error":"<code>", ...}
    {"type":"pong"}

Backpressure: latest-frame-wins. Only the most recent frame is kept pending; a
newer frame overwrites (and drops) an unprocessed one. This keeps latency low on
a single GPU instead of building an unbounded backlog. Inference runs in a
worker thread (asyncio.to_thread) and is serialized process-wide by _INFER_LOCK
so concurrent connections never thrash the GPU.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

log = logging.getLogger("safelens-vision-worker.ws")

# Process-wide serialization of GPU inference across all live connections.
_INFER_LOCK = asyncio.Lock()

# Live session registry + last-emitted metrics, surfaced by GET /debug/stream.
_SESSIONS: "List[_VisionStreamSession]" = []
_GLOBAL: Dict[str, Any] = {
    "last_metrics": None,
    "totals": {"received_frames": 0, "processed_frames": 0, "dropped_frames": 0},
}


# -- Small helpers ------------------------------------------------------------

def _now_ms() -> int:
    """Wall-clock epoch milliseconds (matches the client's sent_at units)."""
    return int(time.time() * 1000)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _parse_json(raw: str) -> Optional[Any]:
    """Parse a JSON string, returning None on failure (never raises)."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def validate_frame(msg: Any) -> "tuple[bool, Optional[str]]":
    """Validate a decoded 'frame' message.

    Returns (ok, error_code). error_code is one of:
      'invalid_frame'        -- not a JSON object
      'invalid_frame_type'   -- type field is not 'frame'
      'missing_frame_b64'    -- frame_b64 missing / not a non-empty string
      'invalid_base64'       -- frame_b64 is not valid base64
    """
    if not isinstance(msg, dict):
        return False, "invalid_frame"
    if msg.get("type") != "frame":
        return False, "invalid_frame_type"
    b64 = msg.get("frame_b64")
    if not b64 or not isinstance(b64, str):
        return False, "missing_frame_b64"
    try:
        base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return False, "invalid_base64"
    return True, None


class _LatestFrameSlot:
    """A single-slot mailbox implementing latest-frame-wins backpressure.

    Holds at most one pending frame. put() overwrites any unprocessed frame and
    reports whether it dropped one. get() awaits the next frame and clears the
    slot. This bounds the backlog to one frame regardless of inbound rate.
    """

    def __init__(self) -> None:
        self._frame: Optional[Dict[str, Any]] = None
        self._event = asyncio.Event()

    def put(self, frame: Dict[str, Any]) -> bool:
        """Store frame; return True if it replaced (dropped) an unprocessed one."""
        replaced = self._frame is not None
        self._frame = frame
        self._event.set()
        return replaced

    async def get(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Await and return the latest frame, or None on timeout."""
        try:
            if timeout is not None:
                await asyncio.wait_for(self._event.wait(), timeout)
            else:
                await self._event.wait()
        except asyncio.TimeoutError:
            return None
        self._event.clear()
        frame, self._frame = self._frame, None
        return frame

    def depth(self) -> int:
        """Number of frames waiting to be processed (0 or 1)."""
        return 1 if self._frame is not None else 0


class _RateWindow:
    """Rolling event-rate (fps) over a fixed time window, readable at any time."""

    def __init__(self, window_s: float = 5.0) -> None:
        self.window_s = window_s
        self._events: "deque[float]" = deque()
        self._created = time.monotonic()

    def mark(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        self._events.append(now)
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def fps(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.monotonic()
        self._trim(now)
        # Clamp the window to the session age so we do not under-report early on.
        elapsed = min(self.window_s, max(now - self._created, 1e-6))
        return round(len(self._events) / elapsed, 2)


class _Avg:
    """Rolling mean over the most recent N samples."""

    def __init__(self, maxlen: int = 50) -> None:
        self._dq: "deque[float]" = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self._dq.append(float(value))

    def avg(self) -> float:
        return round(sum(self._dq) / len(self._dq), 2) if self._dq else 0.0


# -- Per-connection streaming session -----------------------------------------

class _VisionStreamSession:
    """Owns the receive / inference / metrics loops for one WebSocket client."""

    def __init__(
        self,
        websocket: Optional[WebSocket],
        *,
        get_state: Callable[[], Dict[str, Any]],
        trigger_warmup: Callable[[], None],
        run_inference: Callable[..., Any],
        get_backend: Callable[[], str],
        get_tasks: Callable[[], List[str]],
        get_gpu_device: Callable[[], Optional[str]],
        default_conf: float,
        default_img_size: int,
        metrics_interval_s: float,
        warmup_timeout_s: float,
        warmup_poll_s: float,
    ) -> None:
        self.ws = websocket
        self.get_state = get_state
        self.trigger_warmup = trigger_warmup
        self.run_inference = run_inference
        self.get_backend = get_backend
        self.get_tasks = get_tasks
        self.get_gpu_device = get_gpu_device
        self.default_conf = default_conf
        self.default_img_size = default_img_size
        self.metrics_interval_s = metrics_interval_s
        self.warmup_timeout_s = warmup_timeout_s
        self.warmup_poll_s = warmup_poll_s

        self.closed = asyncio.Event()
        self.slot = _LatestFrameSlot()
        self.camera_id: Optional[str] = None
        self.ready_sent = False

        # Metrics counters / windows.
        self.received_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.errors = 0
        self._recv_rate = _RateWindow()
        self._proc_rate = _RateWindow()
        self._infer_avg = _Avg()
        self._e2e_avg = _Avg()

    # -- safe send ------------------------------------------------------------

    async def safe_send(self, msg: Dict[str, Any]) -> bool:
        """Send JSON, swallowing errors and marking the session closed on failure."""
        if self.closed.is_set() or self.ws is None:
            return False
        try:
            await self.ws.send_json(msg)
            return True
        except Exception:  # noqa: BLE001 -- client gone / socket closed
            self.closed.set()
            return False

    # -- model readiness ------------------------------------------------------

    def _model_ready(self) -> bool:
        try:
            state = self.get_state() or {}
        except Exception:  # noqa: BLE001
            return False
        return bool(state.get("model_loaded")) or state.get("status") == "ready"

    def _safe_tasks(self) -> List[str]:
        try:
            return list(self.get_tasks() or [])
        except Exception:  # noqa: BLE001
            return []

    def _safe_backend(self) -> str:
        try:
            return self.get_backend()
        except Exception:  # noqa: BLE001
            return "unknown"

    def _safe_gpu(self) -> Optional[str]:
        try:
            return self.get_gpu_device()
        except Exception:  # noqa: BLE001
            return None

    # -- ready / warmup watcher ----------------------------------------------

    async def ready_watcher(self) -> None:
        """Send 'ready' when the model is loaded; otherwise warm it up first.

        Reuses the worker's existing warmup path (trigger_warmup) -- it does not
        load anything itself.
        """
        if self._model_ready():
            await self._send_ready()
            return

        await self.safe_send({"type": "warming"})
        try:
            self.trigger_warmup()
        except Exception as exc:  # noqa: BLE001
            log.warning("ws/vision: trigger_warmup failed: %s", exc)

        deadline = time.monotonic() + self.warmup_timeout_s
        while not self.closed.is_set():
            try:
                state = self.get_state() or {}
            except Exception:  # noqa: BLE001
                state = {}
            if state.get("status") == "ready" or state.get("model_loaded"):
                await self._send_ready()
                return
            if state.get("status") == "error":
                await self.safe_send({
                    "type": "error",
                    "error": "model_load_failed",
                    "detail": state.get("error"),
                })
                return
            if time.monotonic() > deadline:
                await self.safe_send({"type": "error", "error": "warmup_timeout"})
                return
            await asyncio.sleep(self.warmup_poll_s)

    async def _send_ready(self) -> None:
        if self.ready_sent:
            return
        self.ready_sent = True
        await self.safe_send({
            "type": "ready",
            "backend": self._safe_backend(),
            "tasks": self._safe_tasks(),
        })

    # -- receive loop ---------------------------------------------------------

    async def recv_loop(self) -> None:
        """Read client messages until disconnect. Never raises on bad input."""
        try:
            while not self.closed.is_set():
                try:
                    raw = await self.ws.receive_text()
                except WebSocketDisconnect:
                    break
                except RuntimeError:
                    # e.g. a non-text frame or socket already closed.
                    break
                await self._handle_raw(raw)
        finally:
            self.closed.set()

    async def _handle_raw(self, raw: str) -> None:
        msg = _parse_json(raw)
        if msg is None:
            await self.safe_send({"type": "error", "error": "invalid_json"})
            return
        mtype = msg.get("type") if isinstance(msg, dict) else None
        if mtype == "frame":
            ok, err = validate_frame(msg)
            if not ok:
                self.errors += 1
                await self.safe_send({
                    "type": "error", "error": err,
                    "camera_id": msg.get("camera_id") if isinstance(msg, dict) else None,
                    "frame_id": msg.get("frame_id") if isinstance(msg, dict) else None,
                })
                return
            msg["_received_at_ms"] = _now_ms()
            msg["_received_mono"] = time.monotonic()
            self.camera_id = msg.get("camera_id", self.camera_id)
            self.received_frames += 1
            self._recv_rate.mark()
            if self.slot.put(msg):
                self.dropped_frames += 1  # latest-frame-wins dropped a stale frame
        elif mtype == "ping":
            await self.safe_send({"type": "pong"})
        # Unknown but well-formed messages are ignored (lenient, never crashes).

    # -- inference loop -------------------------------------------------------

    async def infer_loop(self) -> None:
        """Consume the latest frame and emit a 'vision' result for each."""
        while not self.closed.is_set():
            frame = await self.slot.get(timeout=0.5)
            if frame is None:
                continue
            if not self._model_ready():
                # Model not ready yet -- discard rather than queue.
                self.dropped_frames += 1
                continue
            await self._process_frame(frame)

    async def _process_frame(self, frame: Dict[str, Any]) -> None:
        received_at = frame.get("_received_at_ms", _now_ms())
        received_mono = frame.get("_received_mono", time.monotonic())
        conf = self._frame_float(frame, "conf", self.default_conf)
        img_size = self._frame_int(frame, "img_size", self.default_img_size)
        class_filter = frame.get("classes")
        try:
            async with _INFER_LOCK:
                result = await asyncio.to_thread(
                    self.run_inference,
                    image_b64=frame["frame_b64"],
                    conf=conf,
                    img_size=img_size,
                    class_filter=class_filter,
                )
        except Exception as exc:  # noqa: BLE001 -- one frame must never kill the stream
            self.errors += 1
            log.warning("ws/vision: inference failed: %s", exc)
            await self.safe_send({
                "type": "error",
                "error": "inference_failed: " + type(exc).__name__ + ": " + str(exc),
                "camera_id": frame.get("camera_id"),
                "frame_id": frame.get("frame_id"),
            })
            return

        completed_at = _now_ms()
        e2e_ms = round((time.monotonic() - received_mono) * 1000.0, 2)
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)

        self.processed_frames += 1
        self._proc_rate.mark()
        self._infer_avg.add(data.get("inference_ms") or 0.0)
        self._e2e_avg.add(e2e_ms)

        await self.safe_send({
            "type": "vision",
            "camera_id": frame.get("camera_id"),
            "frame_id": frame.get("frame_id"),
            "backend": data.get("backend"),
            "tasks": data.get("tasks"),
            "entities": data.get("entities", []),
            "poses": data.get("poses", []),
            "model": data.get("model"),
            "inference_ms": data.get("inference_ms"),
            "img_w": data.get("img_w"),
            "img_h": data.get("img_h"),
            "received_at": received_at,
            "completed_at": completed_at,
            "end_to_end_latency_ms": e2e_ms,
        })

    @staticmethod
    def _frame_float(frame: Dict[str, Any], key: str, default: float) -> float:
        try:
            return float(frame[key]) if frame.get(key) is not None else default
        except (KeyError, TypeError, ValueError):
            return default

    @staticmethod
    def _frame_int(frame: Dict[str, Any], key: str, default: int) -> int:
        try:
            return int(frame[key]) if frame.get(key) is not None else default
        except (KeyError, TypeError, ValueError):
            return default

    # -- metrics --------------------------------------------------------------

    def metrics_snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        return {
            "type": "metrics",
            "camera_id": self.camera_id,
            "received_fps": self._recv_rate.fps(now),
            "processed_fps": self._proc_rate.fps(now),
            "received_frames": self.received_frames,
            "processed_frames": self.processed_frames,
            "dropped_frames": self.dropped_frames,
            "errors": self.errors,
            "avg_inference_ms": self._infer_avg.avg(),
            "avg_end_to_end_latency_ms": self._e2e_avg.avg(),
            "current_queue_depth": self.slot.depth(),
            "model_ready": self._model_ready(),
            "backend": self._safe_backend(),
            "tasks": self._safe_tasks(),
            "gpu_device": self._safe_gpu(),
        }

    async def metrics_loop(self) -> None:
        """Periodically emit a 'metrics' message and update the global snapshot."""
        while not self.closed.is_set():
            await asyncio.sleep(self.metrics_interval_s)
            if self.closed.is_set():
                break
            snap = self.metrics_snapshot()
            _GLOBAL["last_metrics"] = snap
            await self.safe_send(snap)

    # -- orchestration --------------------------------------------------------

    async def run(self) -> None:
        """Accept, announce, then run the recv/infer/metrics loops together."""
        await self.ws.accept()
        await self.safe_send({"type": "connected"})
        _SESSIONS.append(self)
        bg = [
            asyncio.create_task(self.ready_watcher()),
            asyncio.create_task(self.infer_loop()),
            asyncio.create_task(self.metrics_loop()),
        ]
        try:
            await self.recv_loop()
        finally:
            self.closed.set()
            for t in bg:
                t.cancel()
            await asyncio.gather(*bg, return_exceptions=True)
            self._retire()

    def _retire(self) -> None:
        totals = _GLOBAL["totals"]
        totals["received_frames"] += self.received_frames
        totals["processed_frames"] += self.processed_frames
        totals["dropped_frames"] += self.dropped_frames
        try:
            _SESSIONS.remove(self)
        except ValueError:
            pass


# -- Registration -------------------------------------------------------------

def _default_get_state() -> Dict[str, Any]:
    return {"status": "ready", "model_loaded": True}


def _default_tasks() -> List[str]:
    raw = os.getenv("EDGECRAFTER_TASKS", "det,pose")
    out = [t.strip().lower() for t in raw.split(",") if t.strip().lower() in ("det", "pose")]
    return out or ["det"]


def _default_backend() -> str:
    return os.getenv("VISION_BACKEND", "edgecrafter").strip().lower()


def _require_run_inference(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("ws_vision: run_inference dependency was not provided")


def register_ws_vision(
    app: Any,
    *,
    get_state: Callable[[], Dict[str, Any]] = _default_get_state,
    trigger_warmup: Callable[[], None] = lambda: None,
    run_inference: Callable[..., Any] = _require_run_inference,
    get_backend: Callable[[], str] = _default_backend,
    get_tasks: Callable[[], List[str]] = _default_tasks,
    get_gpu_device: Callable[[], Optional[str]] = lambda: None,
    metrics_interval_s: Optional[float] = None,
    warmup_timeout_s: Optional[float] = None,
    warmup_poll_s: Optional[float] = None,
) -> None:
    """Register the /ws/vision WebSocket route and GET /debug/stream on `app`.

    All model interaction is injected:
      get_state        -> dict with at least 'status' / 'model_loaded'
      trigger_warmup   -> start the worker's existing background warmup
      run_inference    -> run_inference(image_b64=, conf=, img_size=, class_filter=)
                          returning an InferResponse-like object (.model_dump()).
      get_backend/get_tasks/get_gpu_device -> metrics + ready-message metadata.
    """
    metrics_interval_s = (
        metrics_interval_s if metrics_interval_s is not None
        else _env_float("WS_METRICS_INTERVAL_S", 2.0)
    )
    warmup_timeout_s = (
        warmup_timeout_s if warmup_timeout_s is not None
        else _env_float("WARMUP_TIMEOUT_S", 600.0)
    )
    warmup_poll_s = (
        warmup_poll_s if warmup_poll_s is not None
        else _env_float("WS_WARMUP_POLL_S", 0.5)
    )
    default_conf = _env_float("EDGECRAFTER_CONF", 0.25)
    try:
        default_img_size = int(_env_float("EDGECRAFTER_IMG_SIZE", 640))
    except (TypeError, ValueError):
        default_img_size = 640

    @app.websocket("/ws/vision")
    async def ws_vision(websocket: WebSocket):  # noqa: D401
        session = _VisionStreamSession(
            websocket,
            get_state=get_state,
            trigger_warmup=trigger_warmup,
            run_inference=run_inference,
            get_backend=get_backend,
            get_tasks=get_tasks,
            get_gpu_device=get_gpu_device,
            default_conf=default_conf,
            default_img_size=default_img_size,
            metrics_interval_s=metrics_interval_s,
            warmup_timeout_s=warmup_timeout_s,
            warmup_poll_s=warmup_poll_s,
        )
        try:
            await session.run()
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 -- never let a socket crash the worker
            log.warning("ws/vision: unexpected error: %s", exc)
            session.closed.set()

    @app.get("/debug/stream")
    async def debug_stream():
        # Prefer a live snapshot from the most recent active session.
        snapshot = None
        if _SESSIONS:
            try:
                snapshot = _SESSIONS[-1].metrics_snapshot()
            except Exception:  # noqa: BLE001
                snapshot = None
        return JSONResponse({
            "ok": True,
            "active_connections": len(_SESSIONS),
            "metrics": snapshot or _GLOBAL["last_metrics"],
            "totals": dict(_GLOBAL["totals"]),
        })
