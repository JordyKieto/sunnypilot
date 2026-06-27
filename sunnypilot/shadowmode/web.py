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
  image_latent: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float32))
  telemetry: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float32))
  control_history: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float32))
  actual: dict[str, Any] = field(default_factory=dict)
  timestamp: float = 0.0


class LiveSampler(threading.Thread):
  daemon = True

  def __init__(self) -> None:
    super().__init__(name="shadowmode-live-sampler")
    self._lock = threading.Lock()
    self._latest = LiveSample()
    self._stop = threading.Event()
    self._sm = None

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

  def run(self) -> None:
    self._init_streams()
    while not self._stop.is_set():
      try:
        if self._sm is not None:
          self._sm.update(0)
          car_state = self._sm["carState"] if "carState" in self._sm else None
          car_control = self._sm["carControl"] if "carControl" in self._sm else None
          sample = LiveSample(
            image_latent=np.zeros((1, 1, 1), dtype=np.float32),
            telemetry=np.zeros((1, 1, 1), dtype=np.float32),
            control_history=np.zeros((1, 1, 1), dtype=np.float32),
            actual={
              "throttle": float(getattr(car_state, "gas", 0.0) or 0.0) if car_state is not None else 0.0,
              "brake": float(getattr(car_state, "brake", 0.0) or 0.0) if car_state is not None else 0.0,
              "steering": float(getattr(car_control, "steer", 0.0) or 0.0) if car_control is not None else 0.0,
              "source": "device",
            },
            timestamp=time.time(),
          )
          self._update_latest(sample)
      except Exception as e:  # pragma: no cover
        cloudlog.exception("shadowmode live sampler failed: %s", e)
      time.sleep(LIVE_REFRESH_S)


class TinygradOnnxSession:
  def __init__(self, onnx_path: Path) -> None:
    if tinygrad_onnx is None or Tensor is None:
      raise RuntimeError("tinygrad is not available on this system")
    self.onnx_path = Path(onnx_path)
    self.model = self._load_model(self.onnx_path)
    self.inputs_meta = self._read_meta("inputs")
    self.outputs_meta = self._read_meta("outputs")

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
    if hasattr(self.model, "captured"):
      expected = getattr(self.model.captured, "expected_names", [])
      info = getattr(self.model.captured, "expected_input_info", [])
      if kind == "inputs":
        for idx, name in enumerate(expected):
          shape = []
          if idx < len(info):
            shape = [int(d) if isinstance(d, int) and d > 0 else -1 for d in info[idx][1]] if len(info[idx]) > 1 else []
          meta.append({"name": name, "shape": shape})
    return meta

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
      inputs[name] = value.astype(np.float32)

    try:
      if hasattr(self.model, "inputs"):
        for k, v in inputs.items():
          self.model.inputs[k] = Tensor(v, device="NPY").realize()
        result = self.model(**self.model.inputs) if callable(self.model) else self.model
      else:
        tensor_inputs = {k: Tensor(v, device="NPY").realize() for k, v in inputs.items()}
        result = self.model(**tensor_inputs)
    except Exception as e:
      raise RuntimeError(f"Tinygrad ONNX execution failed: {e}") from e

    if hasattr(result, "contiguous"):
      result = result.contiguous().realize().numpy()
    elif isinstance(result, (list, tuple)):
      result = [np.asarray(x) for x in result]
    else:
      result = np.asarray(result)

    if isinstance(result, np.ndarray):
      outputs = {"raw": result}
    else:
      outputs = {f"output_{i}": arr for i, arr in enumerate(result)}

    return {
      "ok": True,
      "actual": sample.actual,
      "predicted": _decode_controls(outputs),
    }


SESSION: TinygradOnnxSession | None = None
LIVE_SAMPLER = LiveSampler()
LIVE_SAMPLER.start()


def _read_upload(body: bytes) -> None:
  MODEL_PATH.write_bytes(body)


def _load_uploaded_model() -> None:
  global SESSION
  SESSION = TinygradOnnxSession(MODEL_PATH)


def _status() -> dict[str, Any]:
  current = LIVE_SAMPLER.current()
  result = SESSION.run(current) if SESSION is not None else None
  return {
    "ok": True,
    "hasModel": SESSION is not None,
    "modelPath": str(MODEL_PATH),
    "liveTimestamp": current.timestamp,
    "inputs": SESSION.inputs_meta if SESSION else [],
    "outputs": SESSION.outputs_meta if SESSION else [],
    "actual": current.actual if current else {},
    "predicted": result["predicted"] if result else {},
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
      _read_upload(self.rfile.read(length))
      _load_uploaded_model()
      self._send_json(_status())
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
