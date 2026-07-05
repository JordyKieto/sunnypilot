import contextlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog

try:
  from opendbc.car.gm.values import CarControllerParams
except Exception:  # pragma: no cover
  CarControllerParams = None


STACK_SIZE = 51
VAE_LATENT_DIM = 128
CONTROL_HISTORY_DIM = 4
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 160
SPEED_SCALE_MS = 30.0
STEERING_ANGLE_SCALE_DEG = 540.0
STEERING_RATE_SCALE_DEG = 720.0
CAN_BUS_POWERTRAIN = 0
CAN_ACCELERATOR_PEDAL = 0x1C4
CAN_BRAKE_PEDAL = 0x0BE
CAN_STEERING_ANGLE = 0x1E5
CONTROL_HISTORY_CAN_MAX_AGE_S = 0.25
POLICY_HZ = 20.0
POLICY_STALE_S = 1.0
LIVE_FRAME_STALE_S = 2.0
LIVE_CAMERA_TIMEOUT_MS = 100
ACCEL_MIN = -3.5
ACCEL_MAX = 2.0
WORKER_IDLE_SLEEP_S = 0.005
STEERING_TARGET_SCALE_DEG = 540.0
STEERING_ADAPTER_MAX_CAN_TORQUE = 70.0
STEERING_ADAPTER_ENGAGE_BLEND_S = 0.35

POLICY_PATH = Path(BASEDIR) / "artifacts" / "irl_policy" / "policy_epoch_0350.onnx"
VAE_PATH = Path(BASEDIR) / "artifacts" / "irl_policy" / "encoder_epoch_0900.onnx"
DECISION_LOG_PATH = Path(BASEDIR) / "artifacts" / "irl_policy" / "actuator_decisions.jsonl"
DECISION_LOG_HZ = 10.0


try:
  import cereal.messaging as messaging
except Exception as e:  # pragma: no cover
  messaging = None
  MESSAGING_IMPORT_ERROR = str(e)
else:
  MESSAGING_IMPORT_ERROR = None

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


def extract_motorola_signal(payload, start_bit, bit_length, signed=False):
  value = 0
  bit = int(start_bit)
  for _ in range(int(bit_length)):
    byte_index = bit // 8
    bit_index = bit % 8
    if byte_index >= len(payload):
      raise ValueError("payload too short for CAN signal")
    value = (value << 1) | ((payload[byte_index] >> bit_index) & 1)
    bit = bit + 15 if bit_index == 0 else bit - 1

  if signed and value & (1 << (bit_length - 1)):
    value -= 1 << bit_length
  return value


def decode_powertrain_control_frame(address, payload):
  if address == CAN_ACCELERATOR_PEDAL and len(payload) >= 6:
    return {"throttle": extract_motorola_signal(payload, 47, 8) / 254.0}
  if address == CAN_BRAKE_PEDAL and len(payload) >= 2:
    return {"brake": extract_motorola_signal(payload, 15, 8) / 255.0}
  if address == CAN_STEERING_ANGLE and len(payload) >= 5:
    angle_deg = extract_motorola_signal(payload, 15, 16, signed=True) * 0.0625
    rate_deg_s = extract_motorola_signal(payload, 27, 12, signed=True)
    return {
      "steering": angle_deg / STEERING_ANGLE_SCALE_DEG,
      "steering_rate": rate_deg_s / STEERING_RATE_SCALE_DEG,
    }
  return {}


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
  def __init__(self, CP: Any | None = None) -> None:
    self.enabled = os.getenv("IRL_POLICY_ENABLED", "1") not in ("0", "false", "False")
    self.CP = CP
    self._gm_params = CarControllerParams(CP) if CarControllerParams is not None and CP is not None else None
    self._steer_max = float(getattr(self._gm_params, "STEER_MAX", 300.0) or 300.0)
    self._steer_delta_up = float(getattr(self._gm_params, "STEER_DELTA_UP", 10.0) or 10.0)
    self._steer_delta_down = float(getattr(self._gm_params, "STEER_DELTA_DOWN", 15.0) or 15.0)
    self._max_norm_torque = min(0.24, STEERING_ADAPTER_MAX_CAN_TORQUE / self._steer_max)
    self._max_norm_delta_up = max(0.004, min(0.025, 0.65 * self._steer_delta_up / self._steer_max))
    self._max_norm_delta_down = max(0.006, min(0.035, 0.80 * self._steer_delta_down / self._steer_max))
    self._vae: _OnnxSession | None = None
    self._policy: _OnnxSession | None = None
    self._can_sock = None
    self._vipc_client = None
    self._vipc_stream = None
    self._bad_vipc_streams: set[str] = set()
    self._available_streams: list[str] = []
    self._last_stream_error: str | None = None
    self._last_frame_time = 0.0
    self._last_frame_id = -1
    self._frame_count = 0
    self._stream_connect_count = 0
    self._stream_reconnect_count = 0
    self._last_stream_connect_time = 0.0
    self._last_stream_reconnect_time = 0.0
    self._last_stream_reconnect_reason: str | None = None
    self._last_run_t = 0.0
    self._last_prediction_t = 0.0
    self._last_prediction: dict[str, Any] | None = None
    self._last_error = ""
    self._latest_snapshot: dict[str, float] | None = None
    self._lock = threading.Lock()
    self._worker_started = False
    self._worker_thread: threading.Thread | None = None
    self._image_latents: deque[np.ndarray] = deque(maxlen=STACK_SIZE)
    self._telemetry: deque[np.ndarray] = deque(maxlen=STACK_SIZE)
    self._control_history: deque[np.ndarray] = deque(maxlen=STACK_SIZE)
    self._last_decision_log_t = 0.0
    self._last_adapted_torque = 0.0
    self._last_policy_apply_t = 0.0
    self._policy_lateral_active = False
    self._last_control_history_source = "none"
    self._last_powertrain_controls = {
      "throttle": 0.0,
      "brake": 0.0,
      "steering": 0.0,
      "steering_rate": 0.0,
    }
    self._last_powertrain_control_time = 0.0

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

  def _drop_camera_client(self, reason: str) -> None:
    if self._vipc_client is None:
      return
    if "no first frame" in reason and self._vipc_stream is not None:
      self._bad_vipc_streams.add(str(self._vipc_stream))
    self._last_stream_reconnect_reason = reason
    self._last_stream_reconnect_time = time.time()
    self._stream_reconnect_count += 1
    cloudlog.warning("IRL policy reconnecting camera stream: %s", reason)
    self._vipc_client = None
    self._vipc_stream = None

  def _frame_is_stale(self, now: float) -> bool:
    return self._last_frame_time > 0.0 and now - self._last_frame_time > LIVE_FRAME_STALE_S

  def _ensure_camera(self) -> bool:
    if self._vipc_client is not None:
      return True
    if VisionIpcClient is None or VisionStreamType is None:
      self._last_error = f"VisionIPC unavailable: {VISIONIPC_IMPORT_ERROR}"
      self._last_stream_error = self._last_error
      return False
    streams = VisionIpcClient.available_streams("camerad", block=False)
    self._available_streams = [str(stream) for stream in streams]
    if not streams:
      self._last_stream_error = "camerad has no available VisionIPC streams"
      self._last_error = self._last_stream_error
      return False

    preferred = []
    for candidate in (VisionStreamType.VISION_STREAM_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD):
      if candidate in streams:
        preferred.append(candidate)
    ordered_streams = preferred + [stream for stream in streams if stream not in preferred]
    candidates = [stream for stream in ordered_streams if str(stream) not in self._bad_vipc_streams]
    if not candidates and ordered_streams:
      self._bad_vipc_streams.clear()
      candidates = ordered_streams
    if not candidates:
      self._last_stream_error = f"no usable camera stream; streams={self._available_streams}"
      self._last_error = self._last_stream_error
      return False

    stream = candidates[0]
    client = VisionIpcClient("camerad", stream, True)
    if not client.connect(False):
      self._last_stream_error = f"connect failed stream={stream}"
      self._last_error = self._last_stream_error
      return False
    self._vipc_client = client
    self._vipc_stream = str(stream)
    self._stream_connect_count += 1
    self._last_stream_connect_time = time.time()
    self._last_stream_error = None
    cloudlog.warning(
      "IRL policy connected camera stream=%s size=%sx%s buffer_len=%s",
      self._vipc_stream,
      client.width,
      client.height,
      getattr(client, "buffer_len", None),
    )
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
    buf = self._vipc_client.recv(timeout_ms=LIVE_CAMERA_TIMEOUT_MS)
    if buf is None:
      self._last_stream_error = "VisionIPC recv returned no frame"
      self._last_error = self._last_stream_error
      now = time.time()
      if self._frame_is_stale(now):
        self._drop_camera_client(f"no frame for {now - self._last_frame_time:.1f}s")
      elif self._last_frame_time <= 0.0 and self._last_stream_connect_time > 0.0 and now - self._last_stream_connect_time > LIVE_FRAME_STALE_S:
        self._drop_camera_client(f"no first frame for {now - self._last_stream_connect_time:.1f}s")
      return None
    self._last_frame_time = time.time()
    self._last_frame_id = int(getattr(self._vipc_client, "frame_id", -1))
    self._frame_count += 1
    self._last_stream_error = None
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
    policy = np.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)
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

  @staticmethod
  def _snapshot_from_car_state(CS: Any) -> dict[str, float]:
    return {
      "vEgo": float(getattr(CS, "vEgo", 0.0)),
      "gas": float(getattr(CS, "gas", 0.0)),
      "brakePressed": float(bool(getattr(CS, "brakePressed", False))),
      "steeringAngleDeg": float(getattr(CS, "steeringAngleDeg", 0.0)),
      "steeringRateDeg": float(getattr(CS, "steeringRateDeg", 0.0)),
    }

  def _start_worker(self) -> None:
    if self._worker_started:
      return
    self._worker_started = True
    self._worker_thread = threading.Thread(target=self._worker_loop, name="irl_policy_worker", daemon=True)
    self._worker_thread.start()

  def feed_car_state(self, CS: Any) -> None:
    if not self.enabled:
      return
    snapshot = self._snapshot_from_car_state(CS)
    with self._lock:
      self._latest_snapshot = snapshot
    self._start_worker()

  def _update_powertrain_controls_from_can(self) -> None:
    if messaging is None:
      self._last_error = f"messaging unavailable: {MESSAGING_IMPORT_ERROR}"
      return
    if self._can_sock is None:
      self._can_sock = messaging.sub_sock("can", conflate=False, timeout=0)
    if self._can_sock is None:
      return

    try:
      updated = False
      for msg in messaging.drain_sock(self._can_sock, wait_for_one=False):
        for frame in getattr(msg, "can", []):
          if int(frame.src) != CAN_BUS_POWERTRAIN:
            continue
          decoded = decode_powertrain_control_frame(int(frame.address), bytes(frame.dat))
          if decoded:
            self._last_powertrain_controls.update(decoded)
            updated = True
      if updated:
        self._last_powertrain_control_time = time.time()
    except Exception as e:
      self._last_error = f"CAN control decode failed: {e}"

  def _update_history_from_snapshot(self, snapshot: dict[str, float]) -> None:
    self._update_powertrain_controls_from_can()

    v_ego = snapshot["vEgo"]
    control_age = time.time() - self._last_powertrain_control_time if self._last_powertrain_control_time > 0.0 else float("inf")
    if control_age <= CONTROL_HISTORY_CAN_MAX_AGE_S:
      controls = self._last_powertrain_controls
      gas = float(np.clip(controls["throttle"], 0.0, 1.0))
      brake = float(np.clip(controls["brake"], 0.0, 1.0))
      steering = float(np.clip(controls["steering"], -1.0, 1.0))
      steering_rate = float(np.clip(controls["steering_rate"], -1.0, 1.0))
      self._last_control_history_source = "can_powertrain"
    else:
      gas = float(np.clip(snapshot["gas"], 0.0, 1.0))
      brake = float(np.clip(snapshot["brakePressed"], 0.0, 1.0))
      steering = float(np.clip(snapshot["steeringAngleDeg"] / STEERING_ANGLE_SCALE_DEG, -1.0, 1.0))
      steering_rate = float(np.clip(snapshot["steeringRateDeg"] / STEERING_RATE_SCALE_DEG, -1.0, 1.0))
      self._last_control_history_source = "car_state_fallback"

    self._telemetry.append(np.asarray([np.clip(v_ego / SPEED_SCALE_MS, 0.0, 2.0)], dtype=np.float32))
    self._control_history.append(np.asarray([gas, brake, steering, steering_rate], dtype=np.float32))

  def _run_policy_once(self, snapshot: dict[str, float]) -> None:
    now = time.monotonic()
    if now - self._last_run_t < 1.0 / POLICY_HZ:
      return
    self._last_run_t = now

    try:
      if not self._ensure_loaded():
        return
      self._update_history_from_snapshot(snapshot)
      image = self._read_camera_image()
      if image is None:
        return
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
      with self._lock:
        self._last_prediction = prediction
        self._last_prediction_t = completed_t
        self._last_error = ""
    except Exception as e:
      with self._lock:
        self._last_error = str(e)
      cloudlog.exception("IRL policy update failed")

  def _worker_loop(self) -> None:
    while True:
      with self._lock:
        snapshot = dict(self._latest_snapshot) if self._latest_snapshot is not None else None
      if snapshot is None:
        time.sleep(WORKER_IDLE_SLEEP_S)
        continue
      self._run_policy_once(snapshot)
      time.sleep(WORKER_IDLE_SLEEP_S)

  def get_fresh_prediction(self, CS: Any) -> tuple[dict[str, Any] | None, str]:
    self.feed_car_state(CS)
    if not self.enabled:
      return None, "policy_disabled"
    with self._lock:
      prediction = dict(self._last_prediction) if self._last_prediction is not None else None
      prediction_t = self._last_prediction_t
    if prediction is None:
      return None, "no_policy_prediction"
    prediction["age"] = time.monotonic() - prediction_t
    if prediction["age"] > POLICY_STALE_S:
      return prediction, "stale_policy_prediction"
    return prediction, "fresh"

  def _log_decision(
    self,
    source: str,
    reason: str,
    CC: Any,
    CS: Any,
    stock_accel: float,
    stock_torque: float,
    prediction: dict[str, Any] | None = None,
    adapted_torque: float | None = None,
    target_steering_angle_deg: float | None = None,
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
      "adaptedTorque": adapted_torque,
      "targetSteeringAngleDeg": target_steering_angle_deg,
      "policyAge": None,
      "policyInferenceDuration": None,
      "policy": prediction,
      "lastError": self._last_error,
      "controlHistorySource": self._last_control_history_source,
      "powertrainControlAgeSeconds": None if self._last_powertrain_control_time <= 0.0 else time.time() - self._last_powertrain_control_time,
      "adapter": {
        "steerMax": self._steer_max,
        "maxNormTorque": self._max_norm_torque,
        "maxNormDeltaUp": self._max_norm_delta_up,
        "maxNormDeltaDown": self._max_norm_delta_down,
        "policyLateralActive": self._policy_lateral_active,
      },
      "vision": {
        "availableStreams": list(self._available_streams),
        "vipcConnected": self._vipc_client is not None,
        "vipcStream": self._vipc_stream,
        "lastStreamError": self._last_stream_error,
        "lastFrameAgeSeconds": None if self._last_frame_time <= 0.0 else time.time() - self._last_frame_time,
        "lastFrameId": self._last_frame_id,
        "frameCount": self._frame_count,
        "streamConnectCount": self._stream_connect_count,
        "streamReconnectCount": self._stream_reconnect_count,
        "lastStreamReconnectReason": self._last_stream_reconnect_reason,
      },
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

  def log_decision(
    self,
    source: str,
    reason: str,
    CC: Any,
    CS: Any,
    stock_accel: float,
    stock_torque: float,
    prediction: dict[str, Any] | None = None,
    adapted_torque: float | None = None,
    target_steering_angle_deg: float | None = None,
  ) -> None:
    self._log_decision(
      source,
      reason,
      CC,
      CS,
      stock_accel,
      stock_torque,
      prediction,
      adapted_torque,
      target_steering_angle_deg,
    )

  @staticmethod
  def _interp(value: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    return float(np.interp(value, xs, ys))

  def _decay_toward(self, current: float, target: float) -> float:
    if current > target:
      return max(target, current - self._max_norm_delta_down)
    return min(target, current + self._max_norm_delta_up)

  def _adapt_policy_steering(
    self,
    prediction: dict[str, Any],
    CC: Any,
    CS: Any,
    stock_torque: float,
  ) -> tuple[float | None, float | None, str]:
    if not CC.latActive:
      self._policy_lateral_active = False
      self._last_adapted_torque = 0.0
      return None, None, "lateral_inactive"

    if bool(getattr(CS, "steeringPressed", False)):
      self._policy_lateral_active = False
      self._last_adapted_torque = self._decay_toward(self._last_adapted_torque, 0.0)
      return None, None, "driver_steering_override"

    v_ego = float(getattr(CS, "vEgo", 0.0))
    target_steering_angle_deg = float(np.clip(prediction["steering"], -1.0, 1.0) * STEERING_TARGET_SCALE_DEG)
    angle_error_deg = target_steering_angle_deg - float(getattr(CS, "steeringAngleDeg", 0.0))
    steering_rate_deg_s = float(getattr(CS, "steeringRateDeg", 0.0))

    # Convert desired wheel-angle error into a conservative CAN torque request.
    kp_counts_per_deg = self._interp(v_ego, (0.0, 5.0, 15.0, 30.0), (0.25, 0.55, 0.90, 1.15))
    kd_counts_per_deg_s = self._interp(v_ego, (0.0, 5.0, 15.0, 30.0), (0.015, 0.025, 0.040, 0.055))
    max_can_torque = min(
      STEERING_ADAPTER_MAX_CAN_TORQUE,
      self._interp(v_ego, (0.0, 5.0, 15.0, 30.0), (18.0, 30.0, 52.0, 70.0)),
    )
    desired_can_torque = (
      kp_counts_per_deg * angle_error_deg
      - kd_counts_per_deg_s * steering_rate_deg_s
    )
    desired_can_torque = float(np.clip(desired_can_torque, -max_can_torque, max_can_torque))
    desired_torque = float(np.clip(desired_can_torque / self._steer_max, -self._max_norm_torque, self._max_norm_torque))

    if not self._policy_lateral_active:
      self._last_adapted_torque = float(np.clip(stock_torque, -self._max_norm_torque, self._max_norm_torque))
      self._last_policy_apply_t = time.monotonic()
      self._policy_lateral_active = True

    elapsed = max(0.0, time.monotonic() - self._last_policy_apply_t)
    blend = 1.0 if STEERING_ADAPTER_ENGAGE_BLEND_S <= 0.0 else float(np.clip(elapsed / STEERING_ADAPTER_ENGAGE_BLEND_S, 0.0, 1.0))
    blended_target = (1.0 - blend) * float(np.clip(stock_torque, -self._max_norm_torque, self._max_norm_torque)) + blend * desired_torque

    max_delta = self._max_norm_delta_up if abs(blended_target) > abs(self._last_adapted_torque) else self._max_norm_delta_down
    adapted_torque = float(np.clip(
      blended_target,
      self._last_adapted_torque - max_delta,
      self._last_adapted_torque + max_delta,
    ))
    adapted_torque = float(np.clip(adapted_torque, -self._max_norm_torque, self._max_norm_torque))
    self._last_adapted_torque = adapted_torque
    return adapted_torque, target_steering_angle_deg, "policy_lateral"

  def _decision(
    self,
    applied: bool,
    source: str,
    reason: str,
    prediction: dict[str, Any] | None = None,
    adapted_torque: float | None = None,
    target_steering_angle_deg: float | None = None,
  ) -> dict[str, Any]:
    return {
      "applied": applied,
      "source": source,
      "reason": reason,
      "prediction": prediction,
      "adaptedTorque": adapted_torque,
      "targetSteeringAngleDeg": target_steering_angle_deg,
    }

  def apply_cached(self, CC: Any, CS: Any) -> dict[str, Any]:
    stock_accel = float(getattr(CC.actuators, "accel", 0.0))
    stock_torque = float(getattr(CC.actuators, "torque", 0.0))

    control_active = bool(CC.longActive or CC.latActive)
    prediction, reason = self.get_fresh_prediction(CS)
    if not self.enabled or not control_active:
      self._policy_lateral_active = False
      if not CC.latActive:
        self._last_adapted_torque = 0.0
      if control_active:
        self.log_decision("stock_fallback", "policy_disabled", CC, CS, stock_accel, stock_torque, prediction)
      return self._decision(False, "stock", "policy_disabled", prediction)
    if prediction is None or reason != "fresh":
      self._policy_lateral_active = False
      if CC.latActive:
        self._last_adapted_torque = float(np.clip(
          stock_torque,
          -self._max_norm_torque,
          self._max_norm_torque,
        ))
      self.log_decision("stock_fallback", reason, CC, CS, stock_accel, stock_torque, prediction)
      return self._decision(False, "stock", reason, prediction)

    adapted_torque = None
    target_steering_angle_deg = None
    if CC.latActive:
      adapted_torque, target_steering_angle_deg, lateral_reason = self._adapt_policy_steering(
        prediction,
        CC,
        CS,
        stock_torque,
      )
      if adapted_torque is None:
        self.log_decision("stock_fallback", lateral_reason, CC, CS, stock_accel, stock_torque, prediction)
        return self._decision(False, "stock", lateral_reason, prediction, adapted_torque, target_steering_angle_deg)
      CC.actuators.torque = adapted_torque

    if CC.longActive:
      CC.actuators.accel = float(prediction["accel"])

    self.log_decision(
      "policy",
      "applied",
      CC,
      CS,
      stock_accel,
      stock_torque,
      prediction,
      adapted_torque,
      target_steering_angle_deg,
    )
    return self._decision(
      bool(CC.longActive or CC.latActive),
      "irl_policy",
      "applied",
      prediction,
      adapted_torque,
      target_steering_angle_deg,
    )
