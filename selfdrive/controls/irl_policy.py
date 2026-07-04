import contextlib
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog


STACK_SIZE = 51
VAE_LATENT_DIM = 128
CONTROL_HISTORY_DIM = 4
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 160
SPEED_SCALE_MS = 30.0
STEERING_ANGLE_SCALE_DEG = 540.0
STEERING_RATE_SCALE_DEG = 720.0
POLICY_HZ = 20.0
POLICY_STALE_S = 1.0
ACCEL_MIN = -3.5
ACCEL_MAX = 2.0

POLICY_PATH = Path(BASEDIR) / "artifacts" / "irl_policy" / "policy_epoch_0350.onnx"
VAE_PATH = Path(BASEDIR) / "artifacts" / "irl_policy" / "encoder_epoch_0900.onnx"
DECISION_LOG_PATH = Path(BASEDIR) / "artifacts" / "irl_policy" / "actuator_decisions.jsonl"
DECISION_LOG_HZ = 10.0


try:
  from tinygrad import Tensor
  from tinygrad.nn.onnx import OnnxRunner
  import tinygrad.helpers as tinygrad_helpers
except Exception as e:  # pragma: no cover
  Tensor = None
  OnnxRunner = None
  tinygrad_helpers = None
  TINYGRAD_IMPORT_ERROR = str(e)
else:
  TINYGRAD_IMPORT_ERROR = None

try:
  from msgq.visionipc import VisionIpcClient, VisionStreamType
except Exception as e:  # pragma: no cover
  VisionIpcClient = None
  VisionStreamType = None
  VISIONIPC_IMPORT_ERROR = str(e)
else:
  VISIONIPC_IMPORT_ERROR = None


def _patch_tinygrad_cache_for_threads() -> None:
  if tinygrad_helpers is None or getattr(tinygrad_helpers, "_irl_policy_thread_cache_patch", False):
    return

  thread_local = __import__("threading").local()

  def db_connection():
    conn = getattr(thread_local, "db_connection", None)
    if conn is None:
      cache_db = tinygrad_helpers.CACHEDB
      os.makedirs(cache_db.rsplit(os.sep, 1)[0], exist_ok=True)
      conn = tinygrad_helpers.sqlite3.connect(cache_db, timeout=60, isolation_level="IMMEDIATE")
      with contextlib.suppress(tinygrad_helpers.sqlite3.OperationalError):
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
      thread_local.db_connection = conn
    return conn

  old_conn = getattr(tinygrad_helpers, "_db_connection", None)
  if old_conn is not None:
    with contextlib.suppress(Exception):
      old_conn.close()
  tinygrad_helpers._db_connection = None
  tinygrad_helpers.db_connection = db_connection
  tinygrad_helpers._irl_policy_thread_cache_patch = True


def _as_numpy(value: Any) -> np.ndarray:
  if hasattr(value, "contiguous"):
    value = value.contiguous().realize()
  if hasattr(value, "numpy"):
    return np.asarray(value.numpy())
  if hasattr(value, "uop") and hasattr(value.uop, "base") and hasattr(value.uop.base, "buffer"):
    return np.asarray(value.uop.base.buffer.numpy())
  return np.asarray(value)


class _OnnxSession:
  def __init__(self, path: Path) -> None:
    if OnnxRunner is None or Tensor is None:
      raise RuntimeError(f"tinygrad unavailable: {TINYGRAD_IMPORT_ERROR}")
    if not path.exists():
      raise FileNotFoundError(path)
    _patch_tinygrad_cache_for_threads()
    self.path = path
    self.runner = OnnxRunner(str(path))
    self.input_names = self._input_names()
    self.output_names = self._output_names()

  def _input_names(self) -> list[str]:
    if hasattr(self.runner, "graph_inputs"):
      return list(self.runner.graph_inputs.keys())
    captured = getattr(self.runner, "captured", None)
    names = list(getattr(captured, "expected_names", []) or [])
    return names

  def _output_names(self) -> list[str]:
    if hasattr(self.runner, "graph_outputs"):
      return list(self.runner.graph_outputs)
    return []

  def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = self.runner(inputs)
    if isinstance(result, dict):
      return {name: _as_numpy(value) for name, value in result.items()}
    if isinstance(result, (list, tuple)):
      names = self.output_names or [f"output_{i}" for i in range(len(result))]
      return {names[i] if i < len(names) else f"output_{i}": _as_numpy(value) for i, value in enumerate(result)}
    return {"output": _as_numpy(result)}


class IrlPolicyController:
  def __init__(self) -> None:
    self.enabled = os.getenv("IRL_POLICY_ENABLED", "1") not in ("0", "false", "False")
    self._vae: _OnnxSession | None = None
    self._policy: _OnnxSession | None = None
    self._vipc_client = None
    self._last_run_t = 0.0
    self._last_prediction_t = 0.0
    self._last_prediction: dict[str, Any] | None = None
    self._last_error = ""
    self._image_latents: deque[np.ndarray] = deque(maxlen=STACK_SIZE)
    self._telemetry: deque[np.ndarray] = deque(maxlen=STACK_SIZE)
    self._control_history: deque[np.ndarray] = deque(maxlen=STACK_SIZE)
    self._last_decision_log_t = 0.0

  @property
  def last_error(self) -> str:
    return self._last_error

  def _ensure_loaded(self) -> bool:
    if not self.enabled:
      return False
    if self._vae is not None and self._policy is not None:
      return True
    try:
      self._vae = _OnnxSession(VAE_PATH)
      self._policy = _OnnxSession(POLICY_PATH)
      cloudlog.warning("IRL policy loaded policy=%s vae=%s", POLICY_PATH, VAE_PATH)
      return True
    except Exception as e:
      self._last_error = f"load failed: {e}"
      cloudlog.exception("IRL policy load failed")
      return False

  def _ensure_camera(self) -> bool:
    if self._vipc_client is not None:
      return True
    if VisionIpcClient is None or VisionStreamType is None:
      self._last_error = f"VisionIPC unavailable: {VISIONIPC_IMPORT_ERROR}"
      return False
    streams = VisionIpcClient.available_streams("camerad", block=False)
    stream = None
    for candidate in (VisionStreamType.VISION_STREAM_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD):
      if candidate in streams:
        stream = candidate
        break
    if stream is None:
      self._last_error = "no road camera stream"
      return False
    client = VisionIpcClient("camerad", stream, True)
    if not client.connect(False):
      self._last_error = f"camera connect failed stream={stream}"
      return False
    self._vipc_client = client
    cloudlog.warning("IRL policy connected camera stream=%s size=%sx%s", stream, client.width, client.height)
    return True

  @staticmethod
  def _resize_rgb_nearest(image: np.ndarray) -> np.ndarray:
    y_idx = np.linspace(0, image.shape[0] - 1, IMAGE_HEIGHT).astype(np.int32)
    x_idx = np.linspace(0, image.shape[1] - 1, IMAGE_WIDTH).astype(np.int32)
    return image[y_idx][:, x_idx].astype(np.uint8)

  @staticmethod
  def _yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    ul = np.repeat(np.repeat(u, 2).reshape(u.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
    vl = np.repeat(np.repeat(v, 2).reshape(v.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
    yuv = np.dstack((y, ul, vl)).astype(np.int16)
    yuv[:, :, 1:] -= 128
    matrix = np.array([
      [1.00000, 1.00000, 1.00000],
      [0.00000, -0.39465, 2.03211],
      [1.13983, -0.58060, 0.00000],
    ])
    return np.dot(yuv, matrix).clip(0, 255).astype(np.uint8)

  def _read_camera_image(self) -> np.ndarray | None:
    if not self._ensure_camera():
      return None
    buf = self._vipc_client.recv(timeout_ms=0)
    if buf is None:
      self._last_error = "no camera frame"
      return None
    uv_height = ((buf.height // 2) + 15) // 16 * 16
    uv_plane_size = buf.stride * uv_height
    y = np.array(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
    uv_data = buf.data[buf.uv_offset:buf.uv_offset + uv_plane_size]
    u = np.array(uv_data[::2], dtype=np.uint8).reshape((-1, buf.stride // 2))[:buf.height // 2, :buf.width // 2]
    v = np.array(uv_data[1::2], dtype=np.uint8).reshape((-1, buf.stride // 2))[:buf.height // 2, :buf.width // 2]
    return self._resize_rgb_nearest(self._yuv_to_rgb(y, u, v))

  @staticmethod
  def _stack(history: deque[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if not history:
      values = [np.zeros(shape, dtype=np.float32)] * STACK_SIZE
    else:
      values = [history[0]] * (STACK_SIZE - len(history)) + list(history)
    return np.asarray(values, dtype=np.float32).reshape((1, STACK_SIZE, *shape))

  @staticmethod
  def _select_latent(outputs: dict[str, np.ndarray]) -> np.ndarray:
    for key, value in outputs.items():
      if "image_latent" in key or "z_mean" in key or "mean" in key:
        return np.asarray(value, dtype=np.float32).reshape(-1)[-VAE_LATENT_DIM:]
    first = next(iter(outputs.values()))
    return np.asarray(first, dtype=np.float32).reshape(-1)[-VAE_LATENT_DIM:]

  @staticmethod
  def _decode_policy(outputs: dict[str, np.ndarray]) -> dict[str, Any]:
    if "policy_output" in outputs:
      policy = np.asarray(outputs["policy_output"], dtype=np.float32).reshape(-1)
    elif "output" in outputs:
      policy = np.asarray(outputs["output"], dtype=np.float32).reshape(-1)
    else:
      parts = []
      for name in ("pedal_state_logits", "throttle_magnitude", "brake_magnitude", "steering", "vego", "delta_v"):
        if name in outputs:
          parts.extend(np.asarray(outputs[name], dtype=np.float32).reshape(-1).tolist())
      policy = np.asarray(parts, dtype=np.float32)
    if policy.size < 8:
      raise RuntimeError(f"policy output too small: {policy.shape}")

    state = int(np.argmax(policy[0:3]))
    throttle = float(np.clip(policy[3], 0.0, 1.0)) if state == 1 else 0.0
    brake = float(np.clip(policy[4], 0.0, 1.0)) if state == 2 else 0.0
    steering = float(np.clip(policy[5], -1.0, 1.0))
    accel = float(np.clip(throttle * ACCEL_MAX - brake * abs(ACCEL_MIN), ACCEL_MIN, ACCEL_MAX))
    return {
      "state": state,
      "throttle": throttle,
      "brake": brake,
      "steering": steering,
      "accel": accel,
      "vego": float(policy[6]),
      "delta_v": float(policy[7]),
    }

  def _update_history(self, CS: Any) -> None:
    v_ego = float(getattr(CS, "vEgo", 0.0))
    gas = float(np.clip(getattr(CS, "gas", 0.0), 0.0, 1.0))
    brake = 1.0 if bool(getattr(CS, "brakePressed", False)) else 0.0
    steering = float(np.clip(getattr(CS, "steeringAngleDeg", 0.0) / STEERING_ANGLE_SCALE_DEG, -1.0, 1.0))
    steering_rate = float(np.clip(getattr(CS, "steeringRateDeg", 0.0) / STEERING_RATE_SCALE_DEG, -1.0, 1.0))
    self._telemetry.append(np.asarray([np.clip(v_ego / SPEED_SCALE_MS, 0.0, 2.0)], dtype=np.float32))
    self._control_history.append(np.asarray([gas, brake, steering, steering_rate], dtype=np.float32))

  def update(self, CS: Any) -> dict[str, Any] | None:
    now = time.monotonic()
    self._update_history(CS)
    if self._last_prediction is not None and now - self._last_prediction_t <= POLICY_STALE_S:
      return self._last_prediction
    if now - self._last_run_t < 1.0 / POLICY_HZ:
      return self._last_prediction
    self._last_run_t = now

    try:
      if not self._ensure_loaded():
        return self._last_prediction
      image = self._read_camera_image()
      if image is None:
        return self._last_prediction
      vae_input_name = self._vae.input_names[0] if self._vae.input_names else "image"
      latent = self._select_latent(self._vae.run({vae_input_name: image.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.uint8)}))
      self._image_latents.append(latent.astype(np.float32))

      inputs = {
        "image_latent": self._stack(self._image_latents, (VAE_LATENT_DIM,)),
        "telemetry": self._stack(self._telemetry, (1,)),
        "control_history": self._stack(self._control_history, (CONTROL_HISTORY_DIM,)),
      }
      policy_inputs = {name: inputs[name] for name in (self._policy.input_names or inputs.keys()) if name in inputs}
      prediction = self._decode_policy(self._policy.run(policy_inputs))
      completed_t = time.monotonic()
      prediction["age"] = 0.0
      prediction["inferenceDuration"] = completed_t - now
      self._last_prediction = prediction
      self._last_prediction_t = completed_t
      self._last_error = ""
      return prediction
    except Exception as e:
      self._last_error = str(e)
      cloudlog.exception("IRL policy update failed")
      return self._last_prediction

  def _log_decision(
    self,
    source: str,
    reason: str,
    CC: Any,
    CS: Any,
    stock_accel: float,
    stock_torque: float,
    prediction: dict[str, Any] | None = None,
  ) -> None:
    now = time.monotonic()
    if now - self._last_decision_log_t < 1.0 / DECISION_LOG_HZ:
      return
    self._last_decision_log_t = now

    row = {
      "monoTime": now,
      "wallTime": time.time(),
      "source": source,
      "reason": reason,
      "enabled": bool(getattr(CC, "enabled", False)),
      "latActive": bool(getattr(CC, "latActive", False)),
      "longActive": bool(getattr(CC, "longActive", False)),
      "vEgo": float(getattr(CS, "vEgo", 0.0)),
      "stockAccel": float(stock_accel),
      "stockTorque": float(stock_torque),
      "finalAccel": float(getattr(CC.actuators, "accel", 0.0)),
      "finalTorque": float(getattr(CC.actuators, "torque", 0.0)),
      "policyAge": None,
      "policyInferenceDuration": None,
      "policy": prediction,
      "lastError": self._last_error,
    }
    if prediction is not None and "age" in prediction:
      row["policyAge"] = float(prediction["age"])
    if prediction is not None and "inferenceDuration" in prediction:
      row["policyInferenceDuration"] = float(prediction["inferenceDuration"])

    try:
      DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
      with open(DECISION_LOG_PATH, "a") as file:
        file.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception:
      cloudlog.exception("IRL policy decision log write failed")

  def apply(self, CC: Any, CS: Any) -> bool:
    stock_accel = float(getattr(CC.actuators, "accel", 0.0))
    stock_torque = float(getattr(CC.actuators, "torque", 0.0))

    control_active = bool(CC.longActive or CC.latActive)
    if not self.enabled or not control_active:
      if control_active:
        self._log_decision("stock_fallback", "policy_disabled", CC, CS, stock_accel, stock_torque)
      return False

    prediction = self.update(CS)
    if prediction is None:
      self._log_decision("stock_fallback", "no_policy_prediction", CC, CS, stock_accel, stock_torque)
      return False
    prediction["age"] = time.monotonic() - self._last_prediction_t
    if prediction["age"] > POLICY_STALE_S:
      self._log_decision("stock_fallback", "stale_policy_prediction", CC, CS, stock_accel, stock_torque, prediction)
      return False

    if CC.longActive:
      CC.actuators.accel = float(prediction["accel"])
    if CC.latActive:
      CC.actuators.torque = float(prediction["steering"])
    self._log_decision("policy", "applied", CC, CS, stock_accel, stock_torque, prediction)
    return bool(CC.longActive or CC.latActive)
