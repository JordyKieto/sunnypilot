#!/usr/bin/env python3
import json
import mimetypes
import os
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
from opendbc.car import structs
from opendbc.car.gm.carcontroller import CarController
from opendbc.car.gm.interface import CarInterface
from opendbc.car.gm.values import CAR
from opendbc.sunnypilot.car.gm.values_ext import GMFlagsSP

STATIC_DIR = Path(BASEDIR) / "sunnypilot" / "shadowmode" / "static"
MODEL_PATH = Path(tempfile.gettempdir()) / "shadowmode_model.onnx"
VAE_PATH = Path(tempfile.gettempdir()) / "shadowmode_vae.onnx"
PERSISTENT_ROOT = Path("/data/openpilot") if Path("/data/openpilot").exists() else Path(BASEDIR)
LOG_DIR = PERSISTENT_ROOT / "sunnypilot" / "shadowmode" / "logs"
LIVE_REFRESH_S = 0.2
UI_REFRESH_S = 1.0
INFERENCE_REFRESH_S = 0.2
LIVE_SAMPLER_STALE_S = 2.0
LIVE_FRAME_STALE_S = 2.0
LIVE_MESSAGING_TIMEOUT_MS = 50
LIVE_CAMERA_TIMEOUT_MS = 100
STACK_SIZE = 51
VAE_LATENT_DIM = 128
CONTROL_HISTORY_DIM = 4
STEERING_ANGLE_SCALE_DEG = 540.0
STEERING_RATE_SCALE_DEG = 540.0
CAN_BUS_POWERTRAIN = 0
CAN_ACCELERATOR_PEDAL = 0x1C4
CAN_BRAKE_PEDAL = 0x0BE
CAN_STEERING_ANGLE = 0x1E5
CAN_SIGNAL_MAX_AGE_SECONDS = 0.05
SHADOWMODE_ENABLE_ACTUATION = os.environ.get("SHADOWMODE_ENABLE_ACTUATION", "0") == "1"
GM_STEER_MAX = 300.0
GM_ACCEL_MIN = -4.0
GM_ACCEL_MAX = 2.0
GM_GAS_LOOKUP_BP = (-0.1, 0.0, GM_ACCEL_MAX)
GM_GAS_LOOKUP_V = (-650.0, 0.0, 1018.0)
GM_BRAKE_LOOKUP_BP = (GM_ACCEL_MIN, -0.1)
GM_BRAKE_LOOKUP_V = (400.0, 0.0)

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
  MESSAGING_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
  messaging = None
  MESSAGING_IMPORT_ERROR = str(e)

try:
  from msgq.visionipc import VisionIpcClient, VisionStreamType
  VISIONIPC_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
  VisionIpcClient = None
  VisionStreamType = None
  VISIONIPC_IMPORT_ERROR = str(e)

try:
  from selfdrive.pandad.pandad_api_impl import can_list_to_can_capnp
  CAN_CAPNP_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
  can_list_to_can_capnp = None
  CAN_CAPNP_IMPORT_ERROR = str(e)


def _decode_controls(outputs: dict[str, np.ndarray]) -> dict[str, Any]:
  if "policy_output" in outputs:
    policy = np.asarray(outputs["policy_output"], dtype=np.float32).reshape(-1)
    if policy.size >= 8:
      outputs = {
        "pedal_state_logits": policy[0:3],
        "throttle_magnitude": policy[3:4],
        "brake_magnitude": policy[4:5],
        "steering": policy[5:6],
        "vego": policy[6:7],
        "delta_v": policy[7:8],
      }
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
      "steering_rate": rate_deg_s / 720.0,
    }
  return {}


def map_shadow_controls_to_gm_can(steering: float | None = None,
                                  throttle_magnitude: float | None = None,
                                  brake_magnitude: float | None = None) -> dict[str, Any]:
  mapped: dict[str, Any] = {
    "enabled": False,
    "actuation_allowed": False,
    "hard_gate": bool(SHADOWMODE_ENABLE_ACTUATION),
  }

  if steering is not None:
    steering_norm = float(np.clip(steering, -1.0, 1.0))
    steering_torque = steering_norm * GM_STEER_MAX
    mapped["steering"] = {
      "input_norm": steering_norm,
      "requested_torque": steering_torque,
      "gm_can": {
        "name": "ASCMLKASteeringCmd",
        "bus": 0,
        "fields": {
          "LKASteeringCmdActive": int(SHADOWMODE_ENABLE_ACTUATION),
          "LKASteeringCmd": int(round(steering_torque)),
        },
      },
    }

  if throttle_magnitude is not None or brake_magnitude is not None:
    throttle = float(np.clip(throttle_magnitude if throttle_magnitude is not None else 0.0, 0.0, 1.0))
    brake = float(np.clip(brake_magnitude if brake_magnitude is not None else 0.0, 0.0, 1.0))
    accel_request = float(np.clip(throttle - brake, GM_ACCEL_MIN, GM_ACCEL_MAX))
    gas_cmd = float(np.interp(accel_request, GM_GAS_LOOKUP_BP, GM_GAS_LOOKUP_V))
    brake_cmd = float(np.interp(accel_request, GM_BRAKE_LOOKUP_BP, GM_BRAKE_LOOKUP_V))
    mapped["longitudinal"] = {
      "throttle_magnitude": throttle,
      "brake_magnitude": brake,
      "requested_accel": accel_request,
      "gm_can": {
        "gas_regen": {
          "name": "ASCMGasRegenCmd",
          "bus": 0,
          "fields": {
            "GasRegenCmdActive": int(SHADOWMODE_ENABLE_ACTUATION),
            "GasRegenCmd": gas_cmd,
          },
        },
        "friction_brake": {
          "name": "EBCMFrictionBrakeCmd",
          "bus": 1,
          "fields": {
            "FrictionBrakeMode": 0x1,
            "FrictionBrakeCmd": -brake_cmd,
          },
        },
      },
    }

  return mapped


class _ShadowGMController:
  def __init__(self) -> None:
    cp = CarInterface.get_non_essential_params(CAR.CHEVROLET_BOLT_EUV)
    cp_sp = CarInterface.get_non_essential_params_sp(cp, CAR.CHEVROLET_BOLT_EUV)
    cp_sp.flags |= int(GMFlagsSP.NON_ACC)
    cp.openpilotLongitudinalControl = True
    self.cp = cp
    self.cp_sp = cp_sp
    self.controller = CarController({}, cp, cp_sp)
    self.frame = 0

  @staticmethod
  def _make_cc(steering: float, throttle_magnitude: float, brake_magnitude: float, actual: dict[str, Any] | None = None):
    cc = structs.CarControl.new_message()
    cc.enabled = True
    cc.latActive = True
    cc.longActive = True
    cc.cruiseControl.cancel = False
    cc.actuators.torque = float(np.clip(steering, -1.0, 1.0))
    cc.actuators.accel = float(np.clip(throttle_magnitude - brake_magnitude, GM_ACCEL_MIN, GM_ACCEL_MAX))
    cc.actuators.gas = float(np.clip(throttle_magnitude, 0.0, 1.0))
    cc.actuators.brake = float(np.clip(brake_magnitude, 0.0, 1.0))
    cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping if cc.actuators.accel <= 0.0 else structs.CarControl.Actuators.LongControlState.pid
    cc.hudControl.visualAlert = structs.CarControl.HUDControl.VisualAlert.none
    cc.hudControl.setSpeed = float(actual.get("cruise_speed", 0.0) if actual else 0.0)
    cc.hudControl.leadDistanceBars = int(actual.get("lead_distance_bars", 0) if actual else 0)
    cc.hudControl.leadVisible = bool(actual.get("lead_visible", False) if actual else False)
    return cc

  @staticmethod
  def _make_cs(actual: dict[str, Any] | None = None):
    actual = actual or {}
    v_ego = float(actual.get("speed", 0.0) or 0.0)
    out = type("Out", (), {
      "steeringTorque": float(actual.get("steering_torque", 0.0) or 0.0),
      "standstill": bool(actual.get("standstill", v_ego < 0.01)),
      "vEgo": v_ego,
    })()
    return type("CS", (), {
      "CP": None,
      "out": out,
      "buttons_counter": int(actual.get("buttons_counter", 0) or 0),
      "pscm_status": {
        "HandsOffSWDetectionMode": int(actual.get("HandsOffSWDetectionMode", 0) or 0),
        "HandsOffSWlDetectionStatus": int(actual.get("HandsOffSWlDetectionStatus", 1) or 1),
        "LKATorqueDeliveredStatus": int(actual.get("LKATorqueDeliveredStatus", 1) or 1),
        "LKADriverAppldTrq": int(actual.get("LKADriverAppldTrq", 0) or 0),
        "LKATorqueDelivered": int(actual.get("LKATorqueDelivered", 0) or 0),
        "LKATotalTorqueDelivered": int(actual.get("LKATotalTorqueDelivered", 0) or 0),
        "RollingCounter": int(actual.get("RollingCounter", 0) or 0),
        "PSCMStatusChecksum": int(actual.get("PSCMStatusChecksum", 0) or 0),
      },
      "loopback_lka_steering_cmd_updated": bool(actual.get("loopback_lka_steering_cmd_updated", False)),
      "loopback_lka_steering_cmd_ts_nanos": int(actual.get("loopback_lka_steering_cmd_ts_nanos", 0) or 0),
      "pt_lka_steering_cmd_counter": int(actual.get("pt_lka_steering_cmd_counter", 0) or 0),
      "cam_lka_steering_cmd_counter": int(actual.get("cam_lka_steering_cmd_counter", 0) or 0),
    })()

  def build(self, steering: float, throttle_magnitude: float, brake_magnitude: float, actual: dict[str, Any] | None = None) -> list[Any]:
    cc = self._make_cc(steering, throttle_magnitude, brake_magnitude, actual)
    cs = self._make_cs(actual)
    _, can_msgs = self.controller.update(cc, self.cp_sp, cs, int(time.monotonic() * 1e9))
    self.frame += 1
    return can_msgs


_SHADOW_GM_CONTROLLER = _ShadowGMController()


@dataclass
class LiveSample:
  image_latent: np.ndarray = field(default_factory=lambda: np.zeros((1, STACK_SIZE, VAE_LATENT_DIM), dtype=np.float32))
  telemetry: np.ndarray = field(default_factory=lambda: np.zeros((1, STACK_SIZE, 1), dtype=np.float32))
  control_history: np.ndarray = field(default_factory=lambda: np.zeros((1, STACK_SIZE, CONTROL_HISTORY_DIM), dtype=np.float32))
  actual: dict[str, Any] = field(default_factory=dict)
  actuator_map: dict[str, Any] = field(default_factory=dict)
  image: np.ndarray | None = None
  timestamp: float = 0.0


class LiveSampler(threading.Thread):
  daemon = True

  def __init__(self) -> None:
    super().__init__(name="shadowmode-live-sampler")
    self._lock = threading.Lock()
    self._stop_event = threading.Event()
    self._started_once = False
    self._sm = None
    self._can_sock = None
    self._image_latents = deque(maxlen=STACK_SIZE)
    self._telemetry = deque(maxlen=STACK_SIZE)
    self._control_history = deque(maxlen=STACK_SIZE)
    self._images = deque(maxlen=STACK_SIZE)
    self._last_steering_angle = 0.0
    self._last_steering_time = 0.0
    self._vipc_client = None
    self._vipc_stream = None
    self._bad_vipc_streams: set[str] = set()
    self._available_streams: list[str] = []
    self._last_error: str | None = None
    self._last_stream_error: str | None = None
    self._last_frame_time = 0.0
    self._last_frame_id = -1
    self._last_loop_time = 0.0
    self._loop_count = 0
    self._frame_count = 0
    self._stream_connect_count = 0
    self._stream_reconnect_count = 0
    self._last_stream_connect_time = 0.0
    self._last_stream_reconnect_time = 0.0
    self._last_stream_reconnect_reason: str | None = None
    self._latest = self._make_stub_sample()
    self._last_powertrain_controls = {
      "throttle": 0.0,
      "brake": 0.0,
      "steering": 0.0,
      "steering_rate": 0.0,
    }
    self._last_powertrain_control_time = 0.0
    self._last_actuator_map: dict[str, Any] = {"enabled": False, "actuation_allowed": False}
    self._actuation_requested = False
    self._sendcan = messaging.pub_sock("sendcan") if (messaging is not None and SHADOWMODE_ENABLE_ACTUATION and can_list_to_can_capnp is not None) else None
    self._shadow_can_frame = 0

  def current(self) -> LiveSample:
    with self._lock:
      return self._latest

  def _update_powertrain_controls_from_can(self) -> None:
    if messaging is None:
      return
    if self._can_sock is None:
      self._can_sock = messaging.sub_sock("can", conflate=False, timeout=0)
    if self._can_sock is None:
      return
    try:
      for msg in messaging.drain_sock(self._can_sock, wait_for_one=False):
        for frame in getattr(msg, "can", []):
          if int(frame.src) != CAN_BUS_POWERTRAIN:
            continue
          decoded = decode_powertrain_control_frame(int(frame.address), bytes(frame.dat))
          if decoded:
            self._last_powertrain_controls.update(decoded)
            self._last_powertrain_control_time = time.time()
    except Exception as e:
      self._last_stream_error = f"can decode failed: {e}"

  def diagnostics(self) -> dict[str, Any]:
    with self._lock:
      sm_seen = {}
      sm_updated = {}
      sm_alive = {}
      sm_valid = {}
      sm_age = {}
      sm_recv_frame = {}
      sm_log_mono_time = {}
      now = time.time()
      now_mono = time.monotonic()
      if self._sm is not None:
        for key in ("carState", "roadCameraState", "carControl", "controlsState", "deviceState"):
          sm_seen[key] = bool(self._sm.seen.get(key, False))
          sm_updated[key] = bool(self._sm.updated.get(key, False))
          sm_alive[key] = bool(self._sm.alive.get(key, False))
          sm_valid[key] = bool(self._sm.valid.get(key, False))
          recv_time = float(self._sm.recv_time.get(key, 0.0) or 0.0)
          sm_age[key] = now_mono - recv_time if recv_time > 0.0 else None
          sm_recv_frame[key] = int(self._sm.recv_frame.get(key, 0) or 0)
          sm_log_mono_time[key] = int(self._sm.logMonoTime.get(key, 0) or 0)
      last_frame_age = now - self._last_frame_time if self._last_frame_time > 0.0 else None
      return {
        "messagingAvailable": messaging is not None,
        "messagingImportError": MESSAGING_IMPORT_ERROR,
        "visionIpcAvailable": VisionIpcClient is not None and VisionStreamType is not None,
        "visionIpcImportError": VISIONIPC_IMPORT_ERROR,
        "subMasterReady": self._sm is not None,
        "availableStreams": list(self._available_streams),
        "vipcConnected": self._vipc_client is not None,
        "vipcStream": self._vipc_stream,
        "vipcWidth": int(getattr(self._vipc_client, "width", 0) or 0) if self._vipc_client is not None else 0,
        "vipcHeight": int(getattr(self._vipc_client, "height", 0) or 0) if self._vipc_client is not None else 0,
        "vipcBufferLen": int(getattr(self._vipc_client, "buffer_len", 0) or 0) if self._vipc_client is not None else 0,
        "lastFrameTime": self._last_frame_time,
        "lastFrameAgeSeconds": last_frame_age,
        "lastFrameId": self._last_frame_id,
        "frameCount": self._frame_count,
        "streamConnectCount": self._stream_connect_count,
        "streamReconnectCount": self._stream_reconnect_count,
        "lastStreamConnectTime": self._last_stream_connect_time,
        "lastStreamReconnectTime": self._last_stream_reconnect_time,
        "lastStreamReconnectReason": self._last_stream_reconnect_reason,
        "lastLoopTime": self._last_loop_time,
        "loopCount": self._loop_count,
        "alive": self.is_alive(),
        "lastError": self._last_error,
        "lastStreamError": self._last_stream_error,
        "seen": sm_seen,
        "updated": sm_updated,
        "aliveServices": sm_alive,
        "validServices": sm_valid,
        "serviceAgeSeconds": sm_age,
        "recvFrame": sm_recv_frame,
        "logMonoTime": sm_log_mono_time,
      }

  def stop(self) -> None:
    self._stop_event.set()

  def _update_latest(self, sample: LiveSample) -> None:
    with self._lock:
      self._latest = sample

  def _drop_camera_client(self, reason: str) -> None:
    if self._vipc_client is None:
      return
    if "no first frame" in reason and self._vipc_stream is not None:
      self._bad_vipc_streams.add(str(self._vipc_stream))
    self._last_stream_reconnect_reason = reason
    self._last_stream_reconnect_time = time.time()
    self._stream_reconnect_count += 1
    cloudlog.warning("shadowmode reconnecting camera stream: %s", reason)
    self._vipc_client = None
    self._vipc_stream = None

  def _frame_is_stale(self, now: float) -> bool:
    return self._last_frame_time > 0.0 and now - self._last_frame_time > LIVE_FRAME_STALE_S

  def _init_streams(self) -> None:
    if messaging is None:
      self._last_stream_error = f"messaging unavailable: {MESSAGING_IMPORT_ERROR}"
      return
    if self._sm is None:
      services = ["carState", "roadCameraState", "liveCalibration", "deviceState", "carControl", "controlsState", "liveDelay"]
      self._sm = messaging.SubMaster(services, poll="carState")
    if VisionIpcClient is None or VisionStreamType is None or self._vipc_client is not None:
      if VisionIpcClient is None or VisionStreamType is None:
        self._last_stream_error = f"VisionIPC unavailable: {VISIONIPC_IMPORT_ERROR}"
      return
    streams = VisionIpcClient.available_streams("camerad", block=False)
    self._available_streams = [str(stream) for stream in streams]
    if not streams:
      self._last_stream_error = "camerad has no available VisionIPC streams"
      return
    preferred = []
    for stream in (VisionStreamType.VISION_STREAM_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD):
      if stream in streams:
        preferred.append(stream)
    ordered_streams = preferred + [stream for stream in streams if stream not in preferred]
    candidates = [stream for stream in ordered_streams if str(stream) not in self._bad_vipc_streams]
    if not candidates and ordered_streams:
      self._bad_vipc_streams.clear()
      candidates = ordered_streams
    if not candidates:
      self._last_stream_error = f"no usable camera stream; streams={self._available_streams}"
      return
    stream = candidates[0]
    client = VisionIpcClient("camerad", stream, True)
    if client.connect(False):
      self._vipc_client = client
      self._vipc_stream = str(stream)
      self._stream_connect_count += 1
      self._last_stream_connect_time = time.time()
      self._last_stream_error = None
      cloudlog.info("shadowmode connected camera stream=%s size=%sx%s buffer_len=%s", self._vipc_stream, client.width, client.height, client.buffer_len)
    else:
      self._last_stream_error = f"connect failed stream={stream}"

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
    buf = self._vipc_client.recv(timeout_ms=LIVE_CAMERA_TIMEOUT_MS)
    if buf is None:
      self._last_stream_error = "VisionIPC recv returned no frame"
      now = time.time()
      if self._frame_is_stale(now):
        self._drop_camera_client(
          f"no frame for {now - self._last_frame_time:.1f}s"
        )
      elif self._last_frame_time <= 0.0 and self._last_stream_connect_time > 0.0 and now - self._last_stream_connect_time > LIVE_FRAME_STALE_S:
        self._drop_camera_client(
          f"no first frame for {now - self._last_stream_connect_time:.1f}s"
        )
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
    now_mono = time.monotonic()
    car_state = self._sm["carState"] if self._sm is not None else None
    car_control = self._sm["carControl"] if self._sm is not None else None
    car_state_age = None
    car_state_seen = False
    car_state_updated = False
    car_state_alive = False
    car_state_valid = False
    if self._sm is not None:
      recv_time = float(self._sm.recv_time.get("carState", 0.0) or 0.0)
      car_state_age = now_mono - recv_time if recv_time > 0.0 else None
      car_state_seen = bool(self._sm.seen.get("carState", False))
      car_state_updated = bool(self._sm.updated.get("carState", False))
      car_state_alive = bool(self._sm.alive.get("carState", False))
      car_state_valid = bool(self._sm.valid.get("carState", False))
    self._update_powertrain_controls_from_can()
    speed = self._clip(self._to_float(getattr(car_state, "vEgo", 0.0) if car_state is not None else 0.0) / 30.0, 0.0, 2.0)
    gas = self._clip(self._last_powertrain_controls["throttle"], 0.0, 1.0)
    brake = self._clip(self._last_powertrain_controls["brake"], 0.0, 1.0)
    steering = self._clip(self._last_powertrain_controls["steering"], -1.0, 1.0)
    steering_rate = self._clip(self._last_powertrain_controls["steering_rate"], -1.0, 1.0)
    actuation_enabled = bool(self._actuation_requested and self._sendcan is not None)
    self._last_actuator_map = map_shadow_controls_to_gm_can(steering=steering, throttle_magnitude=gas, brake_magnitude=brake)
    self._last_actuator_map.update({
      "requested": bool(self._actuation_requested),
      "enabled": actuation_enabled,
      "actuation_allowed": actuation_enabled,
      "ui_enabled": bool(self._actuation_requested),
      "hard_gate": bool(SHADOWMODE_ENABLE_ACTUATION),
      "canTransmit": bool(self._sendcan is not None),
      "reason": "enabled" if actuation_enabled else ("hard gate disabled" if not SHADOWMODE_ENABLE_ACTUATION else ("ui disabled" if not self._actuation_requested else "sendcan unavailable")),
    })
    if actuation_enabled:
      can_msgs = _SHADOW_GM_CONTROLLER.build(steering=steering, throttle_magnitude=gas, brake_magnitude=brake, actual=self._last_powertrain_controls)
      if can_msgs:
        self._sendcan.send(can_list_to_can_capnp(can_msgs, msgtype="sendcan"))

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
        "source": "device" if car_state_seen else ("device_waiting_for_carState" if self._sm is not None else "stub"),
        "car_state_seen": car_state_seen,
        "car_state_updated": car_state_updated,
        "car_state_alive": car_state_alive,
        "car_state_valid": car_state_valid,
        "car_state_age_seconds": car_state_age,
        "powertrain_control_age_seconds": None if self._last_powertrain_control_time <= 0.0 else (now - self._last_powertrain_control_time),
        "has_live_rgb": image is not None,
      },
      actuator_map=self._last_actuator_map,
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
      actuator_map={"enabled": False, "actuation_allowed": False, "requested": False, "hard_gate": bool(SHADOWMODE_ENABLE_ACTUATION)},
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
    while not self._stop_event.is_set():
      try:
        self._last_loop_time = time.time()
        self._loop_count += 1
        self._init_streams()
        if self._sm is not None:
          self._sm.update(LIVE_MESSAGING_TIMEOUT_MS)
          self._update_latest(self._make_sample())
        else:
          self._update_latest(self._make_stub_sample())
        self._last_error = None
      except Exception as e:  # pragma: no cover
        self._last_error = str(e)
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


def _ensure_live_sampler_started() -> None:
  global LIVE_SAMPLER
  diagnostics = LIVE_SAMPLER.diagnostics()
  last_loop_time = float(diagnostics.get("lastLoopTime") or 0.0)
  stale = last_loop_time > 0.0 and time.time() - last_loop_time > LIVE_SAMPLER_STALE_S
  if LIVE_SAMPLER.is_alive() and not stale:
    return
  if stale:
    cloudlog.error("shadowmode live sampler stale for %.1fs; creating replacement sampler", time.time() - last_loop_time)
    LIVE_SAMPLER = LiveSampler()
  elif LIVE_SAMPLER._started_once:
    cloudlog.error("shadowmode live sampler was dead; creating replacement sampler")
    LIVE_SAMPLER = LiveSampler()
  LIVE_SAMPLER._started_once = True
  LIVE_SAMPLER.start()
  cloudlog.info("shadowmode live sampler thread started")


_ensure_live_sampler_started()


class ShadowInferenceWorker(threading.Thread):
  daemon = True

  def __init__(self) -> None:
    super().__init__(name="shadowmode-inference")
    self._lock = threading.Lock()
    self._stop_event = threading.Event()
    self._started_once = False
    self._enabled = False
    self._model_rev_seen = -1
    self._vae_rev_seen = -1
    self._session: TinygradOnnxSession | None = None
    self._vae_session: TinygradOnnxSession | None = None
    self._log_path: Path | None = None
    self._log_count = 0
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
      "sessionLogPath": None,
      "sessionLogCount": 0,
      "sessionLogRecent": [],
      "lastVaeRuntime": {"ran": False},
      "hasModel": False,
      "modelReady": False,
      "hasVae": False,
      "vaeReady": False,
      "modelType": None,
      "vaeType": None,
      "shadowActuationRequested": False,
      "shadowActuationEnabled": False,
      "shadowActuationHardGate": bool(SHADOWMODE_ENABLE_ACTUATION),
      "shadowActuationReason": "ui disabled",
      "inputs": [],
      "outputs": [],
      "vaeInputs": [],
      "vaeOutputs": [],
    }

  def start_inference(self) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / time.strftime("shadowmode_%Y%m%d_%H%M%S.jsonl")
    metadata = {
      "type": "session_start",
      "time": time.time(),
      "modelPath": str(MODEL_PATH),
      "vaePath": str(VAE_PATH),
      "modelRevision": MODEL_REVISION,
      "vaeRevision": VAE_REVISION,
    }
    with log_path.open("w") as file:
      file.write(json.dumps(metadata, sort_keys=True) + "\n")
    with self._lock:
      self._enabled = True
      self._log_path = log_path
      self._log_count = 1
      self._state["enabled"] = True
      self._state["sessionLogPath"] = str(log_path)
      self._state["sessionLogCount"] = 1
      self._state["sessionLogRecent"] = []
    cloudlog.info("shadowmode inference started log=%s", log_path)

  def stop_inference(self) -> None:
    with self._lock:
      self._enabled = False
      self._state["enabled"] = False
      self._state["running"] = False
    cloudlog.info("shadowmode inference stopped log=%s rows=%d", self._log_path, self._log_count)

  def snapshot(self) -> dict[str, Any]:
    with self._lock:
      return dict(self._state)

  def set_actuation_requested(self, requested: bool) -> None:
    with self._lock:
      self._actuation_requested = bool(requested)

  def actuation_requested(self) -> bool:
    with self._lock:
      return bool(getattr(self, "_actuation_requested", False))

  def _set_state(self, **values: Any) -> None:
    with self._lock:
      self._state.update(values)

  def _append_session_log(self, result: dict[str, Any], vae_runtime: dict[str, Any]) -> None:
    with self._lock:
      log_path = self._log_path
    if log_path is None:
      return
    row = {
      "time": time.time(),
      "actual": result.get("actual", {}),
      "predicted": result.get("predicted", {}),
      "vaeRuntime": vae_runtime,
      "modelType": result.get("modelType"),
      "outputs": result.get("outputs", []),
      "inputShapes": result.get("inputShapes", []),
    }
    with log_path.open("a") as file:
      file.write(json.dumps(row, sort_keys=True) + "\n")
    with self._lock:
      self._log_count += 1
      self._state["sessionLogCount"] = self._log_count
      recent = list(self._state.get("sessionLogRecent", []))
      recent.append(row)
      self._state["sessionLogRecent"] = recent[-25:]

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
    _ensure_live_sampler_started()
    if self._vae_session is None:
      return sample, {"ran": False, "reason": "vae not loaded"}
    if sample.image is None:
      camera = LIVE_SAMPLER.diagnostics()
      frame_age = camera.get("lastFrameAgeSeconds")
      reason = "no fresh live rgb frame" if frame_age is not None else "no live rgb frame"
      return sample, {"ran": False, "reason": reason, "frameAgeSeconds": frame_age, "camera": camera}
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
    if self._vae_session is not None and not vae_runtime.get("ran"):
      self._set_state(running=False, lastVaeRuntime=vae_runtime)
      return

    try:
      policy_start = time.monotonic()
      result = self._session.run(sample)
      now = time.time()
      self._append_session_log(result, vae_runtime)
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
    while not self._stop_event.is_set():
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
  _ensure_live_sampler_started()
  _ensure_inference_worker_started()
  current = LIVE_SAMPLER.current()
  live_inputs = LIVE_SAMPLER.diagnostics()
  status_probe_streams = []
  status_probe_error = None
  if VisionIpcClient is not None:
    try:
      status_probe_streams = [str(stream) for stream in VisionIpcClient.available_streams("camerad", block=False)]
    except Exception as e:
      status_probe_error = str(e)
  worker = INFERENCE_WORKER.snapshot()
  latest = worker.get("lastInference") or {}
  actuation_requested = INFERENCE_WORKER.actuation_requested()
  actuation_enabled = bool(actuation_requested and SHADOWMODE_ENABLE_ACTUATION and worker["running"] and worker["enabled"])
  actuation_reason = "enabled" if actuation_enabled else (
    "hard gate disabled" if not SHADOWMODE_ENABLE_ACTUATION else (
      "inference not running" if not worker["running"] else ("ui disabled" if not actuation_requested else "waiting for sendcan")
    )
  )
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
    "liveSamplerAlive": live_inputs["alive"],
    "liveSamplerLoopCount": live_inputs["loopCount"],
    "liveSamplerLastLoopTime": live_inputs["lastLoopTime"],
    "liveSamplerLastError": live_inputs["lastError"],
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
    "shadowActuationRequested": actuation_requested,
    "shadowActuationEnabled": actuation_enabled,
    "shadowActuationHardGate": bool(SHADOWMODE_ENABLE_ACTUATION),
    "shadowActuationReason": actuation_reason,
    "liveTimestamp": current.timestamp,
    "liveInputs": live_inputs,
    "statusProbeAvailableStreams": status_probe_streams,
    "statusProbeStreamError": status_probe_error,
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
    "sessionLogPath": worker["sessionLogPath"],
    "sessionLogCount": worker["sessionLogCount"],
    "sessionLogRecent": worker["sessionLogRecent"],
    "policyErrorCount": worker["policyErrorCount"],
    "vaeErrorCount": worker["vaeErrorCount"],
    "lastPolicyError": worker["lastPolicyError"],
    "lastVaeError": worker["lastVaeError"],
    "latestInference": latest,
    "lastSample": current.actual,
    "actual": current.actual if current else {},
    "actuatorMap": current.actuator_map if current else {},
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

  def _send_file(self, path: Path, download_name: str | None = None) -> None:
    data = path.read_bytes()
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    self.send_header("Content-Length", str(len(data)))
    if download_name:
      self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
    self.end_headers()
    self.wfile.write(data)

  def do_GET(self) -> None:
    if self.path == "/shadow/status":
      try:
        self._send_json(_status())
      except Exception as e:
        cloudlog.exception("shadowmode status failed: %s", e)
        self._send_json({"ok": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
      return
    if self.path == "/shadow/run":
      try:
        self._send_json(_run_shadow())
      except Exception as e:
        cloudlog.exception("shadowmode run failed: %s", e)
        self._send_json({"ok": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
      return
    if self.path == "/shadow/log":
      try:
        log_path = INFERENCE_WORKER.snapshot().get("sessionLogPath")
        if not log_path:
          self._send_json({"ok": False, "error": "no active session log"}, HTTPStatus.NOT_FOUND)
          return
        path = Path(log_path)
        if not path.exists():
          self._send_json({"ok": False, "error": "session log not found"}, HTTPStatus.NOT_FOUND)
          return
        self._send_file(path, download_name=path.name)
      except Exception as e:
        cloudlog.exception("shadowmode log download failed: %s", e)
        self._send_json({"ok": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
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
    if self.path == "/shadow/actuation":
      _ensure_inference_worker_started()
      length = int(self.headers.get("Content-Length", "0"))
      body = self.rfile.read(length) if length > 0 else b"{}"
      try:
        payload = json.loads(body.decode("utf-8") or "{}")
      except Exception:
        payload = {}
      requested = bool(payload.get("enabled", False))
      INFERENCE_WORKER.set_actuation_requested(requested)
      self._send_json({
        "ok": True,
        "message": "shadow actuation updated",
        "requested": requested,
        "status": _status(),
      })
      return
    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


class ShadowHTTPServer(HTTPServer):
  allow_reuse_address = True


def main() -> None:
  _ensure_inference_worker_started()
  # Tinygrad's ONNX runner can keep SQLite state that is tied to the thread that
  # created it, so handle upload and inference requests on the same server thread.
  server = ShadowHTTPServer(("0.0.0.0", 5051), ShadowHandler)
  cloudlog.info("shadowmode listening on 0.0.0.0:5051")
  server.serve_forever()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.info("shadowmode stopped")
