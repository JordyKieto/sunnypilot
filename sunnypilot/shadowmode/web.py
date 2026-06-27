#!/usr/bin/env python3
import json
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog

STATIC_DIR = Path(BASEDIR) / "sunnypilot" / "shadowmode" / "static"
MODEL_PATH = Path(tempfile.gettempdir()) / "shadowmode_model.onnx"
LIVE_REFRESH_S = 0.2
UI_REFRESH_S = 1.0

try:
  import tinygrad.nn.onnx as tinygrad_onnx
except Exception:  # pragma: no cover
  tinygrad_onnx = None

try:
  from tinygrad.tensor import Tensor
except Exception:  # pragma: no cover
  Tensor = None

try:
  import cereal.messaging as messaging
except Exception:  # pragma: no cover
  messaging = None

try:
  from msgq.visionipc import VisionIpcClient, VisionStreamType
except Exception:  # pragma: no cover
  VisionIpcClient = None
  VisionStreamType = None


def _decode_controls(outputs: dict[str, np.ndarray]) -> dict[str, Any]:
  decoded: dict[str, Any] = {}
  pedal_logits = outputs.get("pedal_state_logits")
  if pedal_logits is not None:
    state = int(np.argmax(np.asarray(pedal_logits), axis=-1).reshape(-1)[0])
    state_name = {0: "idle", 1: "throttle", 2: "brake"}.get(state, f"state_{state}")
    decoded["pedal_state"] = {"id": state, "name": state_name}
  if "throttle_magnitude" in outputs:
    decoded["throttle"] = float(np.asarray(outputs["throttle_magnitude"]).reshape(-1)[0])
  if "brake_magnitude" in outputs:
    decoded["brake"] = float(np.asarray(outputs["brake_magnitude"]).reshape(-1)[0])
  if "steering" in outputs:
    decoded["steering"] = float(np.asarray(outputs["steering"]).reshape(-1)[0])
  if "vego" in outputs:
    decoded["vego"] = float(np.asarray(outputs["vego"]).reshape(-1)[0])
  if "delta_v" in outputs:
    decoded["delta_v"] = float(np.asarray(outputs["delta_v"]).reshape(-1)[0])
  return decoded


@dataclass
class LiveSample:
  image_latent: np.ndarray = field(default_factory=lambda: np.zeros((1, 51, 128), dtype=np.float32))
  telemetry: np.ndarray = field(default_factory=lambda: np.zeros((1, 51, 1), dtype=np.float32))
  control_history: np.ndarray = field(default_factory=lambda: np.zeros((1, 51, 4), dtype=np.float32))
  actual: dict[str, Any] = field(default_factory=dict)
  timestamp: float = 0.0


class LiveSampler(threading.Thread):
  daemon = True

  def __init__(self) -> None:
    super().__init__(name="shadowmode-live-sampler")
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._sm = None
    self._latest = self._make_stub_sample()

  def current(self) -> LiveSample:
    with self._lock:
      return self._latest

  def stop(self) -> None:
    self._stop.set()

  def _update_latest(self, sample: LiveSample) -> None:
    with self._lock:
      self._latest = sample

  def _init_streams(self) -> None:
    if messaging is None or VisionIpcClient is None:
      return
    if self._sm is None:
      self._sm = messaging.SubMaster(["carState", "roadCameraState", "liveCalibration", "deviceState", "carControl", "liveDelay"])
    self._vipc_stream = VisionStreamType.VISION_STREAM_ROAD if hasattr(VisionStreamType, "VISION_STREAM_ROAD") else None
    self._vipc_client = None

  @staticmethod
  def _to_float(value: Any, default: float = 0.0) -> float:
    try:
      return float(value)
    except Exception:
      return default

  def _make_sample(self) -> LiveSample:
    car_state = self._sm["carState"] if self._sm is not None and "carState" in self._sm else None
    car_control = self._sm["carControl"] if self._sm is not None and "carControl" in self._sm else None
    speed = self._to_float(getattr(car_state, "vEgo", 0.0) if car_state is not None else 0.0)
    gas = self._to_float(getattr(car_state, "gas", 0.0) if car_state is not None else 0.0)
    brake = self._to_float(getattr(car_state, "brake", 0.0) if car_state is not None else 0.0)
    steer = self._to_float(getattr(car_control, "steer", 0.0) if car_control is not None else 0.0)

    image_latent = np.zeros((1, 51, 128), dtype=np.float32)
    telemetry = np.zeros((1, 51, 1), dtype=np.float32)
    telemetry[0, :, 0] = speed
    control_history = np.zeros((1, 51, 4), dtype=np.float32)
    control_history[0, :, 0] = gas
    control_history[0, :, 1] = brake
    control_history[0, :, 2] = steer
    control_history[0, :, 3] = speed

    return LiveSample(
      image_latent=image_latent,
      telemetry=telemetry,
      control_history=control_history,
      actual={
        "speed": speed,
        "throttle": gas,
        "brake": brake,
        "steering": steer,
        "source": "device" if self._sm is not None else "stub",
      },
      timestamp=time.time(),
    )

  def _make_stub_sample(self) -> LiveSample:
    now = time.time()
    image_latent = np.zeros((1, 51, 128), dtype=np.float32)
    telemetry = np.zeros((1, 51, 1), dtype=np.float32)
    telemetry[0, :, 0] = 0.0
    control_history = np.zeros((1, 51, 4), dtype=np.float32)
    control_history[0, :, 3] = 0.0
    return LiveSample(
      image_latent=image_latent,
      telemetry=telemetry,
      control_history=control_history,
      actual={"speed": 0.0, "throttle": 0.0, "brake": 0.0, "steering": 0.0, "source": "stub"},
      timestamp=now,
    )

  def run(self) -> None:
    self._init_streams()
    while not self._stop.is_set():
      try:
        if self._sm is not None:
          self._sm.update(0)
          self._update_latest(self._make_sample())
        else:
          self._update_latest(self._make_stub_sample())
      except Exception as e:  # pragma: no cover
        cloudlog.exception("shadowmode live sampler failed: %s", e)
      time.sleep(LIVE_REFRESH_S)


class TinygradOnnxSession:
  def __init__(self, onnx_path: Path) -> None:
    if tinygrad_onnx is None or Tensor is None:
      raise RuntimeError("tinygrad is not available on this system")
    self.onnx_path = Path(onnx_path)
    self.model = self._load_model(self.onnx_path)
    self.model_type = type(self.model).__name__
    self.inputs_meta = self._read_meta("inputs")
    self.outputs_meta = self._read_meta("outputs")
    self.ready = True

  def _load_model(self, onnx_path: Path) -> Any:
    candidates = [
      "load_onnx",
      "build_onnx",
      "OnnxRunner",
      "ONNXRunner",
      "Model",
      "load",
    ]
    for name in candidates:
      obj = getattr(tinygrad_onnx, name, None)
      if obj is None:
        continue
      try:
        if callable(obj):
          try:
            return obj(str(onnx_path))
          except TypeError:
            return obj(onnx_path)
      except Exception:
        continue
    raise RuntimeError("No supported Tinygrad ONNX loader found in tinygrad.nn.onnx")

  def _read_meta(self, kind: str) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = []
    if kind == "inputs" and hasattr(self.model, "graph_inputs"):
      for name, spec in self.model.graph_inputs.items():
        shape = [None if isinstance(d, str) else int(d) for d in getattr(spec, "shape", [])]
        meta.append({"name": name, "shape": shape})
    elif kind == "outputs" and hasattr(self.model, "graph_outputs"):
      for name in self.model.graph_outputs:
        meta.append({"name": name, "shape": []})
    elif hasattr(self.model, "captured"):
      expected = getattr(self.model.captured, "expected_names", [])
      info = getattr(self.model.captured, "expected_input_info", [])
      if kind == "inputs":
        for idx, name in enumerate(expected):
          shape = []
          if idx < len(info):
            shape = [int(d) if isinstance(d, int) and d > 0 else -1 for d in info[idx][1]] if len(info[idx]) > 1 else []
          meta.append({"name": name, "shape": shape})
    if not meta and kind == "inputs":
      meta.extend([
        {"name": "image_latent", "shape": [None, 51, 128]},
        {"name": "telemetry", "shape": [None, 51, 1]},
        {"name": "control_history", "shape": [None, 51, 4]},
      ])
    elif not meta and kind == "outputs":
      meta.extend([
        {"name": "pedal_state_logits", "shape": [None, 3]},
        {"name": "throttle_magnitude", "shape": [None, 1]},
        {"name": "brake_magnitude", "shape": [None, 1]},
        {"name": "steering", "shape": [None, 1]},
        {"name": "vego", "shape": [None, 1]},
        {"name": "delta_v", "shape": [None, 1]},
      ])
    return meta

  @staticmethod
  def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "contiguous"):
      value = value.contiguous().realize()
    if hasattr(value, "numpy"):
      return np.asarray(value.numpy())
    if hasattr(value, "uop") and hasattr(value.uop, "base") and hasattr(value.uop.base, "buffer"):
      return np.asarray(value.uop.base.buffer.numpy())
    return np.asarray(value)

  def _normalize_outputs(self, result: Any, output_names: list[str]) -> dict[str, np.ndarray]:
    if isinstance(result, dict):
      arrays = {name: self._to_numpy(value) for name, value in result.items()}
      expected = ["pedal_state_logits", "throttle_magnitude", "brake_magnitude", "steering", "vego", "delta_v"]
      if not any(name in arrays for name in expected) and len(arrays) == len(expected):
        return {expected[i]: value for i, value in enumerate(arrays.values())}
      return arrays

    if isinstance(result, (list, tuple)):
      arrays = [self._to_numpy(value) for value in result]
      return {output_names[i] if i < len(output_names) else f"output_{i}": arr for i, arr in enumerate(arrays)}

    arr = self._to_numpy(result)
    if arr.ndim >= 2 and arr.shape[-1] == len(output_names):
      return {name: arr[..., i] for i, name in enumerate(output_names)}
    return {"raw": arr}

  def run(self, sample: LiveSample) -> dict[str, Any]:
    if not hasattr(self.model, "__call__"):
      raise RuntimeError("Tinygrad ONNX session is not callable")

    inputs = {}
    for item in self.inputs_meta:
      name = item["name"]
      if name == "image_latent":
        value = sample.image_latent
      elif name == "telemetry":
        value = sample.telemetry
      elif name == "control_history":
        value = sample.control_history
      else:
        shape = item.get("shape") or [1]
        shape = [1 if (not isinstance(d, int) or d < 1) else d for d in shape]
        value = np.zeros(shape, dtype=np.float32)
      value = value.astype(np.float32)
      inputs[name] = value

    try:
      result = self.model(inputs)
      call_mode = "dict"
    except Exception as e:
      raise RuntimeError(
        f"Tinygrad ONNX execution failed: {e} | model={self.model_type} | inputs={list(inputs.keys())}"
      ) from e

    output_names = [item["name"] for item in self.outputs_meta] or [
      "pedal_state_logits", "throttle_magnitude", "brake_magnitude", "steering", "vego", "delta_v"
    ]

    outputs = self._normalize_outputs(result, output_names)

    return {
      "ok": True,
      "modelType": self.model_type,
      "actual": sample.actual,
      "predicted": _decode_controls(outputs),
      "outputs": list(outputs.keys()),
      "callMode": call_mode,
      "inputOrder": [item["name"] for item in self.inputs_meta],
      "inputShapes": [list(v.shape) for v in inputs.values()],
    }


SESSION: TinygradOnnxSession | None = None
SESSION_ERROR: str | None = None
LIVE_SAMPLER = LiveSampler()
LIVE_SAMPLER.start()


def _read_upload(body: bytes) -> None:
  MODEL_PATH.write_bytes(body)


def _load_uploaded_model() -> None:
  global SESSION, SESSION_ERROR
  try:
    SESSION = TinygradOnnxSession(MODEL_PATH)
    SESSION_ERROR = None
  except Exception as e:
    SESSION = None
    SESSION_ERROR = str(e)
    raise


def _status() -> dict[str, Any]:
  current = LIVE_SAMPLER.current()
  result = None
  inference_error = None
  if SESSION is not None:
    try:
      result = SESSION.run(current)
    except Exception as e:
      inference_error = str(e)
  return {
    "ok": True,
    "hasModel": SESSION is not None,
    "modelReady": bool(SESSION is not None and getattr(SESSION, "ready", False)),
    "running": bool(SESSION is not None and result is not None and not inference_error),
    "modelPath": str(MODEL_PATH),
    "modelType": SESSION.model_type if SESSION else None,
    "liveTimestamp": current.timestamp,
    "inputs": SESSION.inputs_meta if SESSION else [],
    "outputs": SESSION.outputs_meta if SESSION else [],
    "lastSample": current.actual,
    "actual": current.actual if current else {},
    "predicted": result["predicted"] if result else {},
    "sessionError": SESSION_ERROR,
    "inferenceError": inference_error,
    "message": "model uploaded" if SESSION is not None else "waiting for model upload",
  }


def _run_shadow() -> dict[str, Any]:
  if SESSION is None:
    raise RuntimeError("upload a model first")
  return SESSION.run(LIVE_SAMPLER.current())


class ShadowHandler(SimpleHTTPRequestHandler):
  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

  def log_message(self, fmt: str, *args: Any) -> None:
    cloudlog.info("shadowmode: " + fmt, *args)

  def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def do_GET(self) -> None:
    if self.path == "/shadow/status":
      self._send_json(_status())
      return
    if self.path == "/shadow/run":
      self._send_json(_run_shadow())
      return
    if self.path == "/":
      self.path = "/index.html"
    super().do_GET()

  def do_POST(self) -> None:
    if self.path == "/shadow/upload":
      length = int(self.headers.get("Content-Length", "0"))
      body = self.rfile.read(length)
      _read_upload(body)
      try:
        _load_uploaded_model()
      except Exception as e:
        cloudlog.exception("shadowmode model load failed: %s", e)
        self._send_json({
          "ok": False,
          "error": str(e),
          "modelPath": str(MODEL_PATH),
          "status": _status(),
        }, HTTPStatus.BAD_REQUEST)
        return
      self._send_json({
        "ok": True,
        "message": "model uploaded and loaded",
        "bytes": len(body),
        "modelPath": str(MODEL_PATH),
        "status": _status(),
      })
      return
    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main() -> None:
  server = ThreadingHTTPServer(("0.0.0.0", 5051), ShadowHandler)
  cloudlog.info("shadowmode listening on 0.0.0.0:5051")
  server.serve_forever()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.info("shadowmode stopped")
