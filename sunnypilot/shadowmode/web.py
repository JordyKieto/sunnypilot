#!/usr/bin/env python3
import json
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog

STATIC_DIR = Path(BASEDIR) / "sunnypilot" / "shadowmode" / "static"
MODEL_PATH = Path(tempfile.gettempdir()) / "shadowmode_model.onnx"
VAE_PATH = Path(tempfile.gettempdir()) / "shadowmode_vae.onnx"
LIVE_REFRESH_S = 0.2
UI_REFRESH_S = 1.0
INFERENCE_REFRESH_S = 0.2
STACK_SIZE = 51
VAE_LATENT_DIM = 128
CONTROL_HISTORY_DIM = 4
STEERING_ANGLE_SCALE_DEG = 540.0
STEERING_RATE_SCALE_DEG = 540.0

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
  image_latent: np.ndarray = field(default_factory=lambda: np.zeros((1, STACK_SIZE, VAE_LATENT_DIM), dtype=np.float32))
  telemetry: np.ndarray = field(default_factory=lambda: np.zeros((1, STACK_SIZE, 1), dtype=np.float32))
  control_history: np.ndarray = field(default_factory=lambda: np.zeros((1, STACK_SIZE, CONTROL_HISTORY_DIM), dtype=np.float32))
  actual: dict[str, Any] = field(default_factory=dict)
  image: np.ndarray | None = None
  timestamp: float = 0.0


class LiveSampler(threading.Thread):
  daemon = True

  def __init__(self) -> None:
    super().__init__(name="shadowmode-live-sampler")
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._sm = None
    self._image_latents = deque(maxlen=STACK_SIZE)
    self._telemetry = deque(maxlen=STACK_SIZE)
    self._control_history = deque(maxlen=STACK_SIZE)
    self._images = deque(maxlen=STACK_SIZE)
    self._last_steering_angle = 0.0
    self._last_steering_time = 0.0
    self._vipc_client = None
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
    if messaging is None:
      return
    if self._sm is None:
      self._sm = messaging.SubMaster(["carState", "roadCameraState", "liveCalibration", "deviceState", "carControl", "liveDelay"])
    if VisionIpcClient is None or VisionStreamType is None or self._vipc_client is not None:
      return
    streams = VisionIpcClient.available_streams("camerad", block=False)
    stream = VisionStreamType.VISION_STREAM_ROAD
    if stream not in streams and hasattr(VisionStreamType, "VISION_STREAM_WIDE_ROAD"):
      stream = VisionStreamType.VISION_STREAM_WIDE_ROAD
    if stream in streams:
      self._vipc_client = VisionIpcClient("camerad", stream, True)
      if not self._vipc_client.connect(False):
        self._vipc_client = None

  @staticmethod
  def _to_float(value: Any, default: float = 0.0) -> float:
    try:
      return float(value)
    except Exception:
      return default

  @staticmethod
  def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))

  @staticmethod
  def _nested_attr(obj: Any, path: str, default: Any = 0.0) -> Any:
    cur = obj
    for part in path.split("."):
      cur = getattr(cur, part, None)
      if cur is None:
        return default
    return cur

  @staticmethod
  def _stack_history(history: deque, shape: tuple[int, ...]) -> np.ndarray:
    if not history:
      values = [np.zeros(shape, dtype=np.float32)] * STACK_SIZE
    else:
      pad = [history[0]] * (STACK_SIZE - len(history))
      values = pad + list(history)
    return np.asarray(values, dtype=np.float32).reshape((1, STACK_SIZE, *shape))

  @staticmethod
  def _resize_rgb_nearest(image: np.ndarray, width: int = 160, height: int = 96) -> np.ndarray:
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int32)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int32)
    return image[y_idx][:, x_idx].astype(np.uint8)

  @staticmethod
  def _yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    ul = np.repeat(np.repeat(u, 2).reshape(u.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
    vl = np.repeat(np.repeat(v, 2).reshape(v.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
    yuv = np.dstack((y, ul, vl)).astype(np.int16)
    yuv[:, :, 1:] -= 128
    matrix = np.array([
      [1.00000,  1.00000, 1.00000],
      [0.00000, -0.39465, 2.03211],
      [1.13983, -0.58060, 0.00000],
    ])
    return np.dot(yuv, matrix).clip(0, 255).astype(np.uint8)

  def _read_camera_image(self) -> np.ndarray | None:
    if self._vipc_client is None:
      return None
    buf = self._vipc_client.recv()
    if buf is None:
      return None
    uv_height = ((buf.height // 2) + 15) // 16 * 16
    uv_plane_size = buf.stride * uv_height
    y = np.array(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
    uv_data = buf.data[buf.uv_offset:buf.uv_offset + uv_plane_size]
    u = np.array(uv_data[::2], dtype=np.uint8).reshape((-1, buf.stride // 2))[:buf.height // 2, :buf.width // 2]
    v = np.array(uv_data[1::2], dtype=np.uint8).reshape((-1, buf.stride // 2))[:buf.height // 2, :buf.width // 2]
    return self._resize_rgb_nearest(self._yuv_to_rgb(y, u, v))

  def _steering_rate(self, car_state: Any, steering_norm: float, now: float) -> float:
    raw_rate = self._nested_attr(car_state, "steeringRateDeg", None)
    if raw_rate is not None:
      return self._clip(self._to_float(raw_rate) / STEERING_RATE_SCALE_DEG, -1.0, 1.0)
    if self._last_steering_time <= 0.0:
      self._last_steering_angle = steering_norm
      self._last_steering_time = now
      return 0.0
    dt = max(now - self._last_steering_time, 1e-3)
    rate = self._clip((steering_norm - self._last_steering_angle) / dt, -1.0, 1.0)
    self._last_steering_angle = steering_norm
    self._last_steering_time = now
    return rate

  def _make_sample(self) -> LiveSample:
    now = time.time()
    car_state = self._sm["carState"] if self._sm is not None and "carState" in self._sm else None
    car_control = self._sm["carControl"] if self._sm is not None and "carControl" in self._sm else None
    speed = self._clip(self._to_float(getattr(car_state, "vEgo", 0.0) if car_state is not None else 0.0) / 30.0, 0.0, 2.0)
    gas = self._clip(self._to_float(getattr(car_state, "gas", 0.0) if car_state is not None else 0.0), 0.0, 1.0)
    brake = self._clip(self._to_float(getattr(car_state, "brake", 0.0) if car_state is not None else 0.0), 0.0, 1.0)
    steering_angle = self._to_float(getattr(car_state, "steeringAngleDeg", 0.0) if car_state is not None else 0.0)
    steering = self._clip(steering_angle / STEERING_ANGLE_SCALE_DEG, -1.0, 1.0)
    steering_rate = self._steering_rate(car_state, steering, now)

    image = self._read_camera_image()
    if image is not None:
      self._images.append(image)
    self._telemetry.append(np.asarray([speed], dtype=np.float32))
    self._control_history.append(np.asarray([gas, brake, steering, steering_rate], dtype=np.float32))

    image_latent = self._stack_history(self._image_latents, (VAE_LATENT_DIM,))
    telemetry = self._stack_history(self._telemetry, (1,))
    control_history = self._stack_history(self._control_history, (CONTROL_HISTORY_DIM,))

    return LiveSample(
      image_latent=image_latent,
      telemetry=telemetry,
      control_history=control_history,
      image=image,
      actual={
        "speed": speed * 30.0,
        "throttle": gas,
        "brake": brake,
        "steering": steering,
        "steering_rate": steering_rate,
        "command_accel": self._to_float(self._nested_attr(car_control, "actuators.accel", 0.0)),
        "command_torque": self._to_float(self._nested_attr(car_control, "actuators.torque", 0.0)),
        "source": "device" if self._sm is not None else "stub",
      },
      timestamp=now,
    )

  def _make_stub_sample(self) -> LiveSample:
    now = time.time()
    image_latent = np.zeros((1, STACK_SIZE, VAE_LATENT_DIM), dtype=np.float32)
    telemetry = np.zeros((1, STACK_SIZE, 1), dtype=np.float32)
    telemetry[0, :, 0] = 0.0
    control_history = np.zeros((1, STACK_SIZE, CONTROL_HISTORY_DIM), dtype=np.float32)
    control_history[0, :, 3] = 0.0
    return LiveSample(
      image_latent=image_latent,
      telemetry=telemetry,
      control_history=control_history,
      actual={"speed": 0.0, "throttle": 0.0, "brake": 0.0, "steering": 0.0, "source": "stub"},
      image=None,
      timestamp=now,
    )

  def push_image_latent(self, latent: np.ndarray) -> LiveSample:
    latent = np.asarray(latent, dtype=np.float32).reshape((VAE_LATENT_DIM,))
    with self._lock:
      self._image_latents.append(latent)
      image_latent = self._stack_history(self._image_latents, (VAE_LATENT_DIM,))
      self._latest = replace(self._latest, image_latent=image_latent)
      return self._latest

  def run(self) -> None:
    while not self._stop.is_set():
      try:
        self._init_streams()
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

  def run_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not hasattr(self.model, "__call__"):
      raise RuntimeError("Tinygrad ONNX session is not callable")
    result = self.model(inputs)
    output_names = [item["name"] for item in self.outputs_meta]
    return self._normalize_outputs(result, output_names)


MODEL_REVISION = 0
VAE_REVISION = 0
MODEL_UPLOAD: dict[str, Any] = {"bytes": 0, "time": 0.0, "revision": 0}
VAE_UPLOAD: dict[str, Any] = {"bytes": 0, "time": 0.0, "revision": 0}
LIVE_SAMPLER = LiveSampler()
LIVE_SAMPLER.start()


class ShadowInferenceWorker(threading.Thread):
  daemon = True

  def __init__(self) -> None:
    super().__init__(name="shadowmode-inference")
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._started_once = False
    self._enabled = False
    self._model_rev_seen = -1
    self._vae_rev_seen = -1
    self._session: TinygradOnnxSession | None = None
    self._vae_session: TinygradOnnxSession | None = None
    self._state: dict[str, Any] = {
      "enabled": False,
      "running": False,
      "lastPolicyRunTime": 0.0,
      "lastVaeRunTime": 0.0,
      "lastPolicyDurationMs": 0.0,
      "lastVaeDurationMs": 0.0,
      "policyLoadTime": 0.0,
      "vaeLoadTime": 0.0,
      "policyLoadedRevision": 0,
      "vaeLoadedRevision": 0,
      "policyFailedRevision": 0,
      "vaeFailedRevision": 0,
      "policyRunCount": 0,
      "vaeRunCount": 0,
      "policyErrorCount": 0,
      "vaeErrorCount": 0,
      "lastPolicyError": None,
      "lastVaeError": None,
      "workerError": None,
      "workerHeartbeatTime": 0.0,
      "workerLoopCount": 0,
      "lastInference": None,
      "lastVaeRuntime": {"ran": False},
      "hasModel": False,
      "modelReady": False,
      "hasVae": False,
      "vaeReady": False,
      "modelType": None,
      "vaeType": None,
      "inputs": [],
      "outputs": [],
      "vaeInputs": [],
      "vaeOutputs": [],
    }

  def start_inference(self) -> None:
    with self._lock:
      self._enabled = True
      self._state["enabled"] = True
    cloudlog.info("shadowmode inference started")

  def stop_inference(self) -> None:
    with self._lock:
      self._enabled = False
      self._state["enabled"] = False
      self._state["running"] = False
    cloudlog.info("shadowmode inference stopped")

  def snapshot(self) -> dict[str, Any]:
    with self._lock:
      return dict(self._state)

  def _set_state(self, **values: Any) -> None:
    with self._lock:
      self._state.update(values)

  def _load_changed_models(self) -> None:
    global MODEL_REVISION, VAE_REVISION
    if self._model_rev_seen != MODEL_REVISION:
      self._model_rev_seen = MODEL_REVISION
      if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        try:
          self._session = TinygradOnnxSession(MODEL_PATH)
          self._set_state(
            hasModel=True,
            modelReady=True,
            modelType=self._session.model_type,
            inputs=self._session.inputs_meta,
            outputs=self._session.outputs_meta,
            lastPolicyError=None,
            policyLoadTime=time.time(),
            policyLoadedRevision=self._model_rev_seen,
          )
          cloudlog.info("shadowmode policy model loaded: %s", MODEL_PATH)
        except Exception as e:
          self._session = None
          self._set_state(
            hasModel=False,
            modelReady=False,
            modelType=None,
            inputs=[],
            outputs=[],
            lastPolicyError=str(e),
            policyFailedRevision=self._model_rev_seen,
            policyErrorCount=self._state.get("policyErrorCount", 0) + 1,
          )
          cloudlog.exception("shadowmode policy model load failed: %s", e)

    if self._vae_rev_seen != VAE_REVISION:
      self._vae_rev_seen = VAE_REVISION
      if VAE_PATH.exists() and VAE_PATH.stat().st_size > 0:
        try:
          self._vae_session = TinygradOnnxSession(VAE_PATH)
          self._set_state(
            hasVae=True,
            vaeReady=True,
            vaeType=self._vae_session.model_type,
            vaeInputs=self._vae_session.inputs_meta,
            vaeOutputs=self._vae_session.outputs_meta,
            lastVaeError=None,
            vaeLoadTime=time.time(),
            vaeLoadedRevision=self._vae_rev_seen,
          )
          cloudlog.info("shadowmode vae model loaded: %s", VAE_PATH)
        except Exception as e:
          self._vae_session = None
          self._set_state(
            hasVae=False,
            vaeReady=False,
            vaeType=None,
            vaeInputs=[],
            vaeOutputs=[],
            lastVaeError=str(e),
            vaeFailedRevision=self._vae_rev_seen,
            vaeErrorCount=self._state.get("vaeErrorCount", 0) + 1,
          )
          cloudlog.exception("shadowmode vae model load failed: %s", e)

  def _vae_input_name(self) -> str:
    if self._vae_session is None or not self._vae_session.inputs_meta:
      return "image"
    for item in self._vae_session.inputs_meta:
      name = item["name"]
      if "image" in name.lower() or "input" in name.lower():
        return name
    return self._vae_session.inputs_meta[0]["name"]

  @staticmethod
  def _select_vae_latent(outputs: dict[str, np.ndarray]) -> np.ndarray:
    if not outputs:
      raise RuntimeError("VAE produced no outputs")
    for key in outputs:
      if "image_latent" in key or "z_mean" in key or "mean" in key:
        return np.asarray(outputs[key], dtype=np.float32).reshape(-1)[-VAE_LATENT_DIM:]
    first = next(iter(outputs.values()))
    return np.asarray(first, dtype=np.float32).reshape(-1)[-VAE_LATENT_DIM:]

  def _encode_live_image(self, sample: LiveSample) -> tuple[LiveSample, dict[str, Any]]:
    if self._vae_session is None:
      return sample, {"ran": False, "reason": "vae not loaded"}
    if sample.image is None:
      return sample, {"ran": False, "reason": "no live rgb frame"}
    image_batch = sample.image.reshape((1, 96, 160, 3)).astype(np.uint8)
    outputs = self._vae_session.run_inputs({self._vae_input_name(): image_batch})
    latent = self._select_vae_latent(outputs)
    updated = LIVE_SAMPLER.push_image_latent(latent)
    return updated, {
      "ran": True,
      "outputs": list(outputs.keys()),
      "latentShape": list(latent.shape),
      "imageShape": list(image_batch.shape),
    }

  def _run_once(self) -> None:
    sample = LIVE_SAMPLER.current()
    vae_runtime = {"ran": False}
    try:
      vae_start = time.monotonic()
      sample, vae_runtime = self._encode_live_image(sample)
      if vae_runtime.get("ran"):
        self._set_state(
          lastVaeRunTime=time.time(),
          lastVaeDurationMs=(time.monotonic() - vae_start) * 1000.0,
          vaeRunCount=self._state.get("vaeRunCount", 0) + 1,
          lastVaeError=None,
        )
    except Exception as e:
      vae_runtime = {"ran": False, "error": str(e)}
      self._set_state(
        lastVaeError=str(e),
        vaeErrorCount=self._state.get("vaeErrorCount", 0) + 1,
      )

    if self._session is None:
      self._set_state(running=False, lastVaeRuntime=vae_runtime)
      return

    try:
      policy_start = time.monotonic()
      result = self._session.run(sample)
      now = time.time()
      self._set_state(
        running=True,
        lastPolicyRunTime=now,
        lastPolicyDurationMs=(time.monotonic() - policy_start) * 1000.0,
        policyRunCount=self._state.get("policyRunCount", 0) + 1,
        lastPolicyError=None,
        lastInference=result,
        lastVaeRuntime=vae_runtime,
      )
    except Exception as e:
      self._set_state(
        running=False,
        lastPolicyError=str(e),
        policyErrorCount=self._state.get("policyErrorCount", 0) + 1,
        lastVaeRuntime=vae_runtime,
      )
      cloudlog.exception("shadowmode policy inference failed: %s", e)

  def run(self) -> None:
    while not self._stop.is_set():
      try:
        self._load_changed_models()
        with self._lock:
          enabled = self._enabled
        if enabled:
          self._run_once()
        else:
          self._set_state(running=False)
        self._set_state(
          workerError=None,
          workerHeartbeatTime=time.time(),
          workerLoopCount=self._state.get("workerLoopCount", 0) + 1,
        )
      except Exception as e:
        self._set_state(
          running=False,
          workerError=str(e),
          workerHeartbeatTime=time.time(),
          workerLoopCount=self._state.get("workerLoopCount", 0) + 1,
        )
        cloudlog.exception("shadowmode inference worker loop failed: %s", e)
      time.sleep(INFERENCE_REFRESH_S)


INFERENCE_WORKER = ShadowInferenceWorker()


def _ensure_inference_worker_started() -> None:
  global INFERENCE_WORKER
  if INFERENCE_WORKER.is_alive():
    return
  if INFERENCE_WORKER._started_once:
    cloudlog.error("shadowmode inference worker was dead; creating replacement worker")
    replacement = ShadowInferenceWorker()
    replacement._enabled = INFERENCE_WORKER.snapshot().get("enabled", False)
    replacement._state["enabled"] = replacement._enabled
    INFERENCE_WORKER = replacement
  INFERENCE_WORKER._started_once = True
  INFERENCE_WORKER.start()
  cloudlog.info("shadowmode inference worker thread started")


def _read_upload(body: bytes, path: Path) -> None:
  path.write_bytes(body)


def _status() -> dict[str, Any]:
  _ensure_inference_worker_started()
  current = LIVE_SAMPLER.current()
  worker = INFERENCE_WORKER.snapshot()
  latest = worker.get("lastInference") or {}
  model_uploaded = MODEL_UPLOAD["revision"] > 0
  vae_uploaded = VAE_UPLOAD["revision"] > 0
  policy_loaded_revision = worker["policyLoadedRevision"]
  vae_loaded_revision = worker["vaeLoadedRevision"]
  policy_failed_revision = worker["policyFailedRevision"]
  vae_failed_revision = worker["vaeFailedRevision"]
  policy_state = "loaded" if policy_loaded_revision == MODEL_REVISION and MODEL_REVISION > 0 else (
    "load_failed" if policy_failed_revision == MODEL_REVISION and MODEL_REVISION > 0 else (
      "loading" if model_uploaded else "not_uploaded"
    )
  )
  vae_state = "loaded" if vae_loaded_revision == VAE_REVISION and VAE_REVISION > 0 else (
    "load_failed" if vae_failed_revision == VAE_REVISION and VAE_REVISION > 0 else (
      "loading" if vae_uploaded else "not_uploaded"
    )
  )
  return {
    "ok": True,
    "inferenceEnabled": worker["enabled"],
    "running": worker["running"],
    "workerAlive": INFERENCE_WORKER.is_alive(),
    "workerError": worker["workerError"],
    "workerHeartbeatTime": worker["workerHeartbeatTime"],
    "workerLoopCount": worker["workerLoopCount"],
    "hasModelFile": MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0,
    "hasVaeFile": VAE_PATH.exists() and VAE_PATH.stat().st_size > 0,
    "policyState": policy_state,
    "vaeState": vae_state,
    "modelUpload": dict(MODEL_UPLOAD),
    "vaeUpload": dict(VAE_UPLOAD),
    "hasModel": worker["hasModel"],
    "modelReady": worker["modelReady"],
    "hasVae": worker["hasVae"],
    "vaeReady": worker["vaeReady"],
    "modelPath": str(MODEL_PATH),
    "vaePath": str(VAE_PATH),
    "modelType": worker["modelType"],
    "vaeType": worker["vaeType"],
    "liveTimestamp": current.timestamp,
    "inputs": worker["inputs"],
    "outputs": worker["outputs"],
    "vaeInputs": worker["vaeInputs"],
    "vaeOutputs": worker["vaeOutputs"],
    "vaeRuntime": worker["lastVaeRuntime"],
    "lastPolicyRunTime": worker["lastPolicyRunTime"],
    "lastVaeRunTime": worker["lastVaeRunTime"],
    "lastPolicyDurationMs": worker["lastPolicyDurationMs"],
    "lastVaeDurationMs": worker["lastVaeDurationMs"],
    "policyLoadTime": worker["policyLoadTime"],
    "vaeLoadTime": worker["vaeLoadTime"],
    "modelRevision": MODEL_REVISION,
    "vaeRevision": VAE_REVISION,
    "policyLoadedRevision": policy_loaded_revision,
    "vaeLoadedRevision": vae_loaded_revision,
    "policyFailedRevision": policy_failed_revision,
    "vaeFailedRevision": vae_failed_revision,
    "policyRunCount": worker["policyRunCount"],
    "vaeRunCount": worker["vaeRunCount"],
    "policyErrorCount": worker["policyErrorCount"],
    "vaeErrorCount": worker["vaeErrorCount"],
    "lastPolicyError": worker["lastPolicyError"],
    "lastVaeError": worker["lastVaeError"],
    "latestInference": latest,
    "lastSample": current.actual,
    "actual": current.actual if current else {},
    "predicted": latest.get("predicted", {}),
    "sessionError": worker["lastPolicyError"],
    "vaeError": worker["lastVaeError"],
    "inferenceError": worker["lastPolicyError"],
    "message": "inference running" if worker["running"] else ("inference stopped" if not worker["enabled"] else "waiting for successful inference"),
  }


def _run_shadow() -> dict[str, Any]:
  return _status()


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
      _ensure_inference_worker_started()
      global MODEL_REVISION, MODEL_UPLOAD
      length = int(self.headers.get("Content-Length", "0"))
      body = self.rfile.read(length)
      _read_upload(body, MODEL_PATH)
      MODEL_REVISION += 1
      MODEL_UPLOAD = {"bytes": len(body), "time": time.time(), "revision": MODEL_REVISION}
      cloudlog.info("shadowmode policy upload received: bytes=%d revision=%d path=%s", len(body), MODEL_REVISION, MODEL_PATH)
      self._send_json({
        "ok": True,
        "message": "policy uploaded; backend worker is loading it",
        "bytes": len(body),
        "modelRevision": MODEL_REVISION,
        "upload": dict(MODEL_UPLOAD),
        "modelPath": str(MODEL_PATH),
        "status": _status(),
      })
      return
    if self.path == "/shadow/upload_vae":
      _ensure_inference_worker_started()
      global VAE_REVISION, VAE_UPLOAD
      length = int(self.headers.get("Content-Length", "0"))
      body = self.rfile.read(length)
      _read_upload(body, VAE_PATH)
      VAE_REVISION += 1
      VAE_UPLOAD = {"bytes": len(body), "time": time.time(), "revision": VAE_REVISION}
      cloudlog.info("shadowmode vae upload received: bytes=%d revision=%d path=%s", len(body), VAE_REVISION, VAE_PATH)
      self._send_json({
        "ok": True,
        "message": "vae uploaded; backend worker is loading it",
        "bytes": len(body),
        "vaeRevision": VAE_REVISION,
        "upload": dict(VAE_UPLOAD),
        "vaePath": str(VAE_PATH),
        "status": _status(),
      })
      return
    if self.path == "/shadow/start":
      _ensure_inference_worker_started()
      INFERENCE_WORKER.start_inference()
      self._send_json({"ok": True, "message": "inference started", "status": _status()})
      return
    if self.path == "/shadow/stop":
      _ensure_inference_worker_started()
      INFERENCE_WORKER.stop_inference()
      self._send_json({"ok": True, "message": "inference stopped", "status": _status()})
      return
    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main() -> None:
  _ensure_inference_worker_started()
  # Tinygrad's ONNX runner can keep SQLite state that is tied to the thread that
  # created it, so handle upload and inference requests on the same server thread.
  server = HTTPServer(("0.0.0.0", 5051), ShadowHandler)
  cloudlog.info("shadowmode listening on 0.0.0.0:5051")
  server.serve_forever()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.info("shadowmode stopped")
