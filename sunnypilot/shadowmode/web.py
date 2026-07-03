#!/usr/bin/env python3
import json
import mimetypes
import os
import contextlib
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
from opendbc.car.gm.carstate import CarState as GMCarState
from opendbc.car.gm.interface import CarInterface
from opendbc.car.gm.values import CAR, CarControllerParams, GMSafetyFlags

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
CAN_BUS_CAMERA = 2
CAN_BUS_LOOPBACK = 128
CAN_ACCELERATOR_PEDAL = 0x1C4
CAN_BRAKE_PEDAL = 0x0BE
CAN_STEERING_ANGLE = 0x1E5
CAN_LKA_STEERING_CMD = 0x180  # ASCMLKASteeringCmd
CAN_PSCM_STATUS = 0x184  # PSCMStatus
CAN_STEERING_BUTTON = 0x1E1  # ASCMSteeringButton
CAN_SIGNAL_MAX_AGE_SECONDS = 0.05
SHADOWMODE_ENABLE_ACTUATION = True
GM_CAMERA_LONG_SAFETY_PARAM = int(
  GMSafetyFlags.HW_CAM.value | GMSafetyFlags.HW_CAM_LONG.value | GMSafetyFlags.EV.value
)
# Derive lookup tables from actual Bolt EUV CarControllerParams (CAMERA_ACC_CAR):
# MAX_GAS=1346, MAX_ACC_REGEN=-540, INACTIVE_REGEN=-500, max_regen_acceleration=0.
_BOLT_EUV_CP = CarInterface.get_non_essential_params(CAR.CHEVROLET_BOLT_EUV)
_BOLT_EUV_PARAMS = CarControllerParams(_BOLT_EUV_CP)
GM_STEER_MAX = float(_BOLT_EUV_PARAMS.STEER_MAX)
GM_ACCEL_MIN = float(_BOLT_EUV_PARAMS.ACCEL_MIN)
GM_ACCEL_MAX = float(_BOLT_EUV_PARAMS.ACCEL_MAX)
GM_GAS_LOOKUP_BP = tuple(float(x) for x in _BOLT_EUV_PARAMS.GAS_LOOKUP_BP)
GM_GAS_LOOKUP_V = tuple(float(x) for x in _BOLT_EUV_PARAMS.GAS_LOOKUP_V)
GM_BRAKE_LOOKUP_BP = tuple(float(x) for x in _BOLT_EUV_PARAMS.BRAKE_LOOKUP_BP)
GM_BRAKE_LOOKUP_V = tuple(float(x) for x in _BOLT_EUV_PARAMS.BRAKE_LOOKUP_V)

if os.getenv("DEBUG") and not os.getenv("DEBUG").isdigit():
  os.environ["DEBUG"] = "0"

try:
  from tinygrad.nn.onnx import OnnxRunner
except Exception:  # pragma: no cover
  OnnxRunner = None

try:
  from tinygrad.tensor import Tensor
except Exception:  # pragma: no cover
  Tensor = None

try:
  import tinygrad.helpers as tinygrad_helpers
except Exception:  # pragma: no cover
  tinygrad_helpers = None

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


def _patch_tinygrad_cache_for_threads() -> None:
  if tinygrad_helpers is None or getattr(tinygrad_helpers, "_shadowmode_thread_cache_patch", False):
    return

  thread_local = threading.local()

  def db_connection():
    conn = getattr(thread_local, "db_connection", None)
    if conn is None:
      cache_db = tinygrad_helpers.CACHEDB
      os.makedirs(cache_db.rsplit(os.sep, 1)[0], exist_ok=True)
      conn = tinygrad_helpers.sqlite3.connect(
        cache_db,
        timeout=60,
        isolation_level="IMMEDIATE",
      )
      with contextlib.suppress(tinygrad_helpers.sqlite3.OperationalError):
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
      if getattr(tinygrad_helpers, "DEBUG", 0) >= 8:
        conn.set_trace_callback(print)
      thread_local.db_connection = conn
    return conn

  old_conn = getattr(tinygrad_helpers, "_db_connection", None)
  if old_conn is not None:
    with contextlib.suppress(Exception):
      old_conn.close()
  tinygrad_helpers._db_connection = None
  tinygrad_helpers.db_connection = db_connection
  tinygrad_helpers._shadowmode_thread_cache_patch = True

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


def _gate_longitudinal_controls(predicted: dict[str, Any]) -> dict[str, Any]:
  gated = dict(predicted)
  state = gated.get("pedal_state", {}).get("id")
  throttle = gated.get("throttle")
  brake = gated.get("brake")

  if throttle is not None:
    throttle = float(throttle)
  if brake is not None:
    brake = float(brake)

  if state == 1:
    if brake is not None:
      brake = 0.0
  elif state == 2:
    if throttle is not None:
      throttle = 0.0
  elif state == 0:
    if throttle is not None:
      throttle = 0.0
    if brake is not None:
      brake = 0.0

  if throttle is not None:
    gated["throttle"] = throttle
  if brake is not None:
    gated["brake"] = brake
  return gated


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


def decode_lka_steering_cmd_counter(payload: bytes) -> int:
  """Extract RollingCounter (bits 5|2@0+) from ASCMLKASteeringCmd."""
  if len(payload) < 1:
    return 0
  return int(extract_motorola_signal(payload, 5, 2))


def decode_pscm_status_frame(payload: bytes) -> dict[str, Any]:
  """Decode all PSCMStatus fields from raw 8-byte CAN payload into physical values."""
  if len(payload) < 8:
    return {}
  return {
    "HandsOffSWDetectionMode": int(extract_motorola_signal(payload, 20, 2)),
    "HandsOffSWlDetectionStatus": int(extract_motorola_signal(payload, 21, 1)),
    "LKATorqueDeliveredStatus": int(extract_motorola_signal(payload, 5, 3)),
    "LKADriverAppldTrq": float(extract_motorola_signal(payload, 50, 11, signed=True)) * 0.01,
    "LKATorqueDelivered": float(extract_motorola_signal(payload, 18, 11, signed=True)) * 0.01,
    "LKATotalTorqueDelivered": float(extract_motorola_signal(payload, 2, 11, signed=True)) * 0.01,
    "RollingCounter": int(extract_motorola_signal(payload, 38, 4)),
    "PSCMStatusChecksum": int(extract_motorola_signal(payload, 33, 10)),
  }


def map_shadow_controls_to_gm_can(steering: float | None = None,
                                  throttle_magnitude: float | None = None,
                                  brake_magnitude: float | None = None) -> dict[str, Any]:
  mapped: dict[str, Any] = {
    "previewed": False,
    "requested": False,
    "allowed": False,
    "transmitting": False,
    "canTransmit": False,
    "hard_gate": bool(SHADOWMODE_ENABLE_ACTUATION),
    "source": "predictedGated",
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
          "bus": 0,
          "fields": {
            "FrictionBrakeMode": 0x1,
            "FrictionBrakeCmd": -brake_cmd,
          },
        },
      },
    }

  mapped["previewed"] = "steering" in mapped or "longitudinal" in mapped
  return mapped


class _ShadowActuators:
  def __init__(self, torque: float, accel: float, long_control_state: Any) -> None:
    self.torque = float(torque)
    self.accel = float(accel)
    self.longControlState = long_control_state
    self.gas = 0.0
    self.brake = 0.0
    self.steeringAngleDeg = 0.0
    self.speed = 0.0
    self.curvature = 0.0
    self.torqueOutputCan = 0.0

  def as_builder(self) -> Any:
    actuators = structs.CarControl.Actuators.new_message()
    actuators.torque = self.torque
    actuators.accel = self.accel
    actuators.longControlState = self.longControlState
    actuators.gas = self.gas
    actuators.brake = self.brake
    actuators.steeringAngleDeg = self.steeringAngleDeg
    actuators.speed = self.speed
    actuators.curvature = self.curvature
    actuators.torqueOutputCan = self.torqueOutputCan
    return actuators


class _ShadowCarControl:
  def __init__(self, torque: float, accel: float, active: bool) -> None:
    long_state = structs.CarControl.Actuators.LongControlState
    self.enabled = bool(active)
    self.latActive = bool(active)
    self.longActive = bool(active)
    self.actuators = _ShadowActuators(
      torque,
      accel,
      long_state.stopping if accel < -0.01 else long_state.pid,
    )
    self.hudControl = structs.CarControl.HUDControl.new_message()
    self.cruiseControl = structs.CarControl.CruiseControl.new_message()


class _ShadowGMController:
  def __init__(self) -> None:
    cp = _BOLT_EUV_CP
    cp_sp = CarInterface.get_non_essential_params_sp(cp, CAR.CHEVROLET_BOLT_EUV)
    cp.openpilotLongitudinalControl = True
    cp.pcmCruise = False
    if cp.safetyConfigs:
      cp.safetyConfigs[0].safetyParam |= int(GMSafetyFlags.HW_CAM_LONG.value)
    self.cp = cp
    self.cp_sp = cp_sp
    self.controller = CarController({}, cp, cp_sp)
    self.frame = 0
    # Persistent real CarState — hydrated from live data on each build() call.
    # Avoids the synthetic duck-typed object and keeps counter state across frames.
    self._live_cs = GMCarState(cp, cp_sp)
    # Seed pscm_status so create_pscm_status() can always access it before CAN data arrives.
    self._live_cs.pscm_status = {
      "HandsOffSWDetectionMode": 0,
      "HandsOffSWlDetectionStatus": 1,
      "LKATorqueDeliveredStatus": 1,
      "LKADriverAppldTrq": 0.0,
      "LKATorqueDelivered": 0.0,
      "LKATotalTorqueDelivered": 0.0,
      "RollingCounter": 0,
      "PSCMStatusChecksum": 0,
    }

  def build(self, cc_cereal: Any, cs_live: dict[str, Any], car_state_cereal: Any = None) -> list[Any]:
    """
    cc_cereal:        real structs.CarControl reader from self._sm["carControl"]
    cs_live:          dict of live-decoded CAN/cereal state for GM-specific CS fields
    car_state_cereal: real structs.CarState reader from self._sm["carState"]
    """
    if cc_cereal is None:
      # No live CarControl yet — emit nothing rather than sending garbage.
      return []

    # Bind real carState output struct — controller reads vEgo, standstill, steeringTorque from here.
    if car_state_cereal is not None:
      self._live_cs.out = car_state_cereal

    # Hydrate GM-specific fields from live CAN tracking.
    pscm = cs_live.get("pscm_status")
    if pscm:
      self._live_cs.pscm_status = pscm
    self._live_cs.loopback_lka_steering_cmd_updated = bool(cs_live.get("loopback_lka_steering_cmd_updated", False))
    self._live_cs.loopback_lka_steering_cmd_ts_nanos = int(cs_live.get("loopback_lka_steering_cmd_ts_nanos", 0) or 0)
    self._live_cs.pt_lka_steering_cmd_counter = int(cs_live.get("pt_lka_steering_cmd_counter", 0) or 0)
    self._live_cs.cam_lka_steering_cmd_counter = int(cs_live.get("cam_lka_steering_cmd_counter", 0) or 0)
    self._live_cs.buttons_counter = int(cs_live.get("buttons_counter", 0) or 0)

    _, can_msgs = self.controller.update(cc_cereal, self.cp_sp, self._live_cs, int(time.monotonic() * 1e9))
    self.frame += 1
    return can_msgs


_SHADOW_GM_CONTROLLER = _ShadowGMController()


@dataclass
class LiveSample:
  image_latent: np.ndarray = field(default_factory=lambda: np.zeros((1, STACK_SIZE, VAE_LATENT_DIM), dtype=np.float32))
  telemetry: np.ndarray = field(default_factory=lambda: np.zeros((1, STACK_SIZE, 1), dtype=np.float32))
  control_history: np.ndarray = field(
    default_factory=lambda: np.zeros((1, STACK_SIZE, CONTROL_HISTORY_DIM), dtype=np.float32))
  actual: dict[str, Any] = field(default_factory=dict)
  gm_live_state: dict[str, Any] = field(default_factory=dict)
  panda_safety: dict[str, Any] = field(default_factory=dict)
  car_control: Any = None
  car_state: Any = None
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
    self._last_pscm_status: dict[str, Any] = {
      "HandsOffSWDetectionMode": 0,
      "HandsOffSWlDetectionStatus": 1,
      "LKATorqueDeliveredStatus": 1,
      "LKADriverAppldTrq": 0.0,
      "LKATorqueDelivered": 0.0,
      "LKATotalTorqueDelivered": 0.0,
      "RollingCounter": 0,
      "PSCMStatusChecksum": 0,
    }
    self._loopback_lka_steering_cmd_updated = False
    self._loopback_lka_steering_cmd_ts_nanos = 0
    self._pt_lka_steering_cmd_counter = 0
    self._cam_lka_steering_cmd_counter = 0
    self._buttons_counter = 0
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
    self._loopback_lka_steering_cmd_updated = False
    try:
      now_nanos = int(time.monotonic() * 1e9)
      for msg in messaging.drain_sock(self._can_sock, wait_for_one=False):
        for frame in getattr(msg, "can", []):
          src = int(frame.src)
          addr = int(frame.address)
          dat = bytes(frame.dat)

          if src == CAN_BUS_POWERTRAIN:
            decoded = decode_powertrain_control_frame(addr, dat)
            if decoded:
              self._last_powertrain_controls.update(decoded)
              self._last_powertrain_control_time = time.time()
            if addr == CAN_PSCM_STATUS:
              pscm = decode_pscm_status_frame(dat)
              if pscm:
                self._last_pscm_status.update(pscm)
            elif addr == CAN_LKA_STEERING_CMD:
              self._pt_lka_steering_cmd_counter = decode_lka_steering_cmd_counter(dat)

          elif src == CAN_BUS_CAMERA:
            if addr == CAN_LKA_STEERING_CMD:
              self._cam_lka_steering_cmd_counter = decode_lka_steering_cmd_counter(dat)

          elif src == CAN_BUS_LOOPBACK:
            if addr == CAN_LKA_STEERING_CMD:
              self._loopback_lka_steering_cmd_updated = True
              self._loopback_lka_steering_cmd_ts_nanos = now_nanos

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
      services = ["carState", "roadCameraState", "liveCalibration", "deviceState", "carControl", "controlsState",
                  "liveDelay", "pandaStates"]
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
      cloudlog.info("shadowmode connected camera stream=%s size=%sx%s buffer_len=%s", self._vipc_stream, client.width,
                    client.height, client.buffer_len)
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
  def _enum_name(value: Any) -> str:
    try:
      return str(value)
    except Exception:
      return ""

  @staticmethod
  def _enum_raw(value: Any) -> int | None:
    raw = getattr(value, "raw", None)
    if raw is not None:
      try:
        return int(raw)
      except Exception:
        pass
    try:
      return int(value)
    except Exception:
      return None

  def _panda_safety_snapshot(self) -> dict[str, Any]:
    if self._sm is None:
      return {"seen": False, "ready": False, "reason": "messaging unavailable"}

    panda_states = self._sm["pandaStates"]
    seen = bool(self._sm.seen.get("pandaStates", False))
    alive = bool(self._sm.alive.get("pandaStates", False))
    valid = bool(self._sm.valid.get("pandaStates", False))
    if not seen or len(panda_states) == 0:
      return {"seen": seen, "alive": alive, "valid": valid, "ready": False, "reason": "no panda state"}

    expected_model = structs.CarParams.SafetyModel.gm
    expected_model_raw = self._enum_raw(expected_model)
    pandas = []
    gm_camera_long_ready = False
    controls_allowed = False
    controls_allowed_lateral = False
    controls_allowed_longitudinal = False

    for panda_state in panda_states:
      safety_model = getattr(panda_state, "safetyModel", None)
      safety_model_raw = self._enum_raw(safety_model)
      safety_model_name = self._enum_name(safety_model)
      safety_param = int(getattr(panda_state, "safetyParam", 0) or 0)
      controls = bool(getattr(panda_state, "controlsAllowed", False))
      controls_lat = bool(getattr(panda_state, "controlsAllowedLateral", controls))
      controls_long = bool(getattr(panda_state, "controlsAllowedLongitudinal", controls))
      safety_model_matches = (
        safety_model_raw == expected_model_raw
        or safety_model_name == "gm"
        or safety_model_name.endswith(".gm")
      )
      camera_long = (
        safety_model_matches
        and (safety_param & GM_CAMERA_LONG_SAFETY_PARAM) == GM_CAMERA_LONG_SAFETY_PARAM
      )
      gm_camera_long_ready = gm_camera_long_ready or camera_long
      controls_allowed = controls_allowed or controls
      controls_allowed_lateral = controls_allowed_lateral or controls_lat
      controls_allowed_longitudinal = controls_allowed_longitudinal or controls_long
      pandas.append({
        "safetyModel": safety_model_name,
        "safetyModelRaw": safety_model_raw,
        "safetyParam": safety_param,
        "safetyParamHex": hex(safety_param),
        "controlsAllowed": controls,
        "controlsAllowedLateral": controls_lat,
        "controlsAllowedLongitudinal": controls_long,
        "gmCameraLong": camera_long,
      })

    if not gm_camera_long_ready:
      reason = "wrong GM safety mode"
    elif not controls_allowed:
      reason = "panda controls not allowed"
    elif not controls_allowed_lateral:
      reason = "panda lateral not allowed"
    elif not controls_allowed_longitudinal:
      reason = "panda longitudinal not allowed"
    elif not alive or not valid:
      reason = "panda state not alive/valid"
    else:
      reason = "ready"

    ready = bool(
      alive
      and valid
      and gm_camera_long_ready
      and controls_allowed
      and controls_allowed_lateral
      and controls_allowed_longitudinal
    )
    return {
      "seen": seen,
      "alive": alive,
      "valid": valid,
      "ready": ready,
      "reason": reason,
      "expectedSafetyModel": "gm",
      "expectedSafetyParamMask": GM_CAMERA_LONG_SAFETY_PARAM,
      "expectedSafetyParamMaskHex": hex(GM_CAMERA_LONG_SAFETY_PARAM),
      "gmCameraLongReady": gm_camera_long_ready,
      "controlsAllowed": controls_allowed,
      "controlsAllowedLateral": controls_allowed_lateral,
      "controlsAllowedLongitudinal": controls_allowed_longitudinal,
      "pandas": pandas,
    }

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
      [1.00000, 1.00000, 1.00000],
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
    panda_safety = self._panda_safety_snapshot()
    self._update_powertrain_controls_from_can()

    # Real vehicle state from carState cereal message
    v_ego = self._to_float(getattr(car_state, "vEgo", 0.0) if car_state is not None else 0.0)
    standstill = bool(getattr(car_state, "standstill", v_ego < 0.01) if car_state is not None else (v_ego < 0.01))
    steering_torque = self._to_float(getattr(car_state, "steeringTorque", 0.0) if car_state is not None else 0.0)
    eps_torque = self._to_float(getattr(car_state, "steeringTorqueEps", 0.0) if car_state is not None else 0.0)

    speed = self._clip(v_ego / 30.0, 0.0, 2.0)
    gas = self._clip(self._last_powertrain_controls["throttle"], 0.0, 1.0)
    brake = self._clip(self._last_powertrain_controls["brake"], 0.0, 1.0)
    steering = self._clip(self._last_powertrain_controls["steering"], -1.0, 1.0)
    steering_rate = self._clip(self._last_powertrain_controls["steering_rate"], -1.0, 1.0)

    # Merge CAN-decoded PSCM status with higher-fidelity cereal torque values
    live_pscm = dict(self._last_pscm_status)
    live_pscm["LKADriverAppldTrq"] = steering_torque
    live_pscm["LKATorqueDelivered"] = eps_torque

    # GM-specific live state passed to build() to hydrate the real GMCarState instance
    cs_actual = {
      "speed": v_ego,
      "standstill": standstill,
      "steering_torque": steering_torque,
      "pscm_status": live_pscm,
      "loopback_lka_steering_cmd_updated": self._loopback_lka_steering_cmd_updated,
      "loopback_lka_steering_cmd_ts_nanos": self._loopback_lka_steering_cmd_ts_nanos,
      "pt_lka_steering_cmd_counter": self._pt_lka_steering_cmd_counter,
      "cam_lka_steering_cmd_counter": self._cam_lka_steering_cmd_counter,
      "buttons_counter": self._buttons_counter,
    }

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
        "speed": v_ego,
        "throttle": gas,
        "brake": brake,
        "steering": steering,
        "steering_rate": steering_rate,
        "steering_torque": steering_torque,
        "eps_torque": eps_torque,
        "standstill": standstill,
        "command_accel": self._to_float(self._nested_attr(car_control, "actuators.accel", 0.0)),
        "command_torque": self._to_float(self._nested_attr(car_control, "actuators.torque", 0.0)),
        "source": "device" if car_state_seen else ("device_waiting_for_carState" if self._sm is not None else "stub"),
        "car_state_seen": car_state_seen,
        "car_state_updated": car_state_updated,
        "car_state_alive": car_state_alive,
        "car_state_valid": car_state_valid,
        "car_state_age_seconds": car_state_age,
        "powertrain_control_age_seconds": None if self._last_powertrain_control_time <= 0.0 else (
                  now - self._last_powertrain_control_time),
        "has_live_rgb": image is not None,
        "pscm_rolling_counter": live_pscm["RollingCounter"],
        "loopback_lka_updated": self._loopback_lka_steering_cmd_updated,
        "cam_lka_counter": self._cam_lka_steering_cmd_counter,
        "pt_lka_counter": self._pt_lka_steering_cmd_counter,
        "panda_safety": panda_safety,
      },
      gm_live_state=cs_actual,
      panda_safety=panda_safety,
      car_control=car_control,
      car_state=car_state,
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
      actual={"speed": 0.0, "throttle": 0.0, "brake": 0.0, "steering": 0.0, "steering_torque": 0.0,
              "eps_torque": 0.0, "standstill": True, "source": "stub"},
      gm_live_state={},
      panda_safety={"seen": False, "ready": False, "reason": "stub sample"},
      car_control=None,
      car_state=None,
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
    if OnnxRunner is None or Tensor is None:
      raise RuntimeError("tinygrad is not available on this system")
    _patch_tinygrad_cache_for_threads()
    self.onnx_path = Path(onnx_path)
    self.model = OnnxRunner(str(self.onnx_path))
    self.model_type = type(self.model).__name__
    self.inputs_meta = self._read_meta("inputs")
    self.outputs_meta = self._read_meta("outputs")
    self.ready = True

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
            shape = [int(d) if isinstance(d, int) and d > 0 else -1 for d in info[idx][1]] if len(
              info[idx]) > 1 else []
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
    cloudlog.error("shadowmode live sampler stale for %.1fs; creating replacement sampler",
                   time.time() - last_loop_time)
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
    self._log_path: Path | None = None
    self._log_count = 0
    self._actuation_requested = False
    self._sendcan = messaging.pub_sock("sendcan") if (
      messaging is not None and SHADOWMODE_ENABLE_ACTUATION and can_list_to_can_capnp is not None
    ) else None
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
      "actuatorLogPath": None,
      "actuatorLogCount": 0,
      "actuatorLogRecent": [],
      "lastVaeRuntime": {"ran": False},
      "hasModel": False,
      "modelReady": False,
      "hasVae": False,
      "vaeReady": False,
      "modelType": None,
      "vaeType": None,
      "shadowActuationRequested": False,
      "shadowActuationAllowed": False,
      "shadowActuationTransmitting": False,
      "shadowActuationHardGate": bool(SHADOWMODE_ENABLE_ACTUATION),
      "shadowActuationReason": "ui disabled",
      "lastActuatorMap": map_shadow_controls_to_gm_can(),
      "inputs": [],
      "outputs": [],
      "vaeInputs": [],
      "vaeOutputs": [],
    }

  def start_inference(self) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / time.strftime("shadowmode_%Y%m%d_%H%M%S.jsonl")
    actuator_log_path = LOG_DIR / time.strftime("shadowmode_actuator_%Y%m%d_%H%M%S.jsonl")
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
    actuator_metadata = dict(metadata)
    actuator_metadata["type"] = "actuator_session_start"
    actuator_metadata["note"] = "Rows are written only while shadow actuation is requested."
    with actuator_log_path.open("w") as file:
      file.write(json.dumps(actuator_metadata, sort_keys=True) + "\n")
    with self._lock:
      self._enabled = True
      self._log_path = log_path
      self._log_count = 1
      self._state["enabled"] = True
      self._state["sessionLogPath"] = str(log_path)
      self._state["sessionLogCount"] = 1
      self._state["sessionLogRecent"] = []
      self._state["actuatorLogPath"] = str(actuator_log_path)
      self._state["actuatorLogCount"] = 1
      self._state["actuatorLogRecent"] = []
    cloudlog.info("shadowmode inference started log=%s actuator_log=%s", log_path, actuator_log_path)

  def stop_inference(self) -> None:
    with self._lock:
      self._enabled = False
      self._state["enabled"] = False
      self._state["running"] = False
      self._state["shadowActuationAllowed"] = False
      self._state["shadowActuationTransmitting"] = False
      self._state["shadowActuationReason"] = "inference stopped"
    cloudlog.info("shadowmode inference stopped log=%s rows=%d", self._log_path, self._log_count)

  def snapshot(self) -> dict[str, Any]:
    with self._lock:
      return dict(self._state)

  def set_actuation_requested(self, requested: bool) -> None:
    with self._lock:
      self._actuation_requested = bool(requested)
      self._state["shadowActuationRequested"] = self._actuation_requested

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
      "predictedRaw": result.get("predicted_raw", result.get("predicted", {})),
      "predictedGated": result.get("predicted_gated", result.get("predicted", {})),
      "actuatorMap": result.get("actuator_map", {}),
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

  @staticmethod
  def _control_direction(value: Any, negative: str, positive: str, neutral: str, deadband: float = 0.01) -> str:
    value = float(value or 0.0)
    if value > deadband:
      return positive
    if value < -deadband:
      return negative
    return neutral

  def _append_actuator_log(self, result: dict[str, Any]) -> None:
    actuator_map = result.get("actuator_map", {})
    if not actuator_map.get("requested"):
      return
    with self._lock:
      log_path = self._state.get("actuatorLogPath")
    if not log_path:
      return

    controls = result.get("predicted_gated", {})
    throttle = float(controls.get("throttle", 0.0) or 0.0)
    brake = float(controls.get("brake", 0.0) or 0.0)
    steering = float(controls.get("steering", 0.0) or 0.0)
    row = {
      "time": time.time(),
      "controls": {
        "pedal_state": controls.get("pedal_state", {}),
        "throttle": throttle,
        "brake": brake,
        "steering": steering,
      },
      "directions": {
        "longitudinal": "throttle" if throttle > 0.01 else ("brake" if brake > 0.01 else "idle"),
        "steering": self._control_direction(steering, "left", "right", "straight"),
      },
      "actuatorMap": actuator_map,
      "actual": result.get("actual", {}),
    }
    path = Path(log_path)
    with path.open("a") as file:
      file.write(json.dumps(row, sort_keys=True) + "\n")
    with self._lock:
      self._state["actuatorLogCount"] = int(self._state.get("actuatorLogCount", 0)) + 1
      recent = list(self._state.get("actuatorLogRecent", []))
      recent.append(row)
      self._state["actuatorLogRecent"] = recent[-25:]

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

  @staticmethod
  def _model_accel_request(predicted_gated: dict[str, Any]) -> float:
    throttle = float(np.clip(predicted_gated.get("throttle", 0.0), 0.0, 1.0))
    brake = float(np.clip(predicted_gated.get("brake", 0.0), 0.0, 1.0))
    return float(np.clip(throttle - brake, GM_ACCEL_MIN, GM_ACCEL_MAX))

  @staticmethod
  def _model_car_control(base_control: Any, predicted_gated: dict[str, Any], active: bool) -> Any:
    steering = float(np.clip(predicted_gated.get("steering", 0.0), -1.0, 1.0))
    accel = ShadowInferenceWorker._model_accel_request(predicted_gated)
    return _ShadowCarControl(steering, accel, active)

  def _actuator_reason(
    self,
    requested: bool,
    allowed: bool,
    previewed: bool,
    live_ready: bool,
    live_reason: str,
    safety_ready: bool,
    safety_reason: str,
  ) -> str:
    if allowed:
      return "allowed"
    if not previewed:
      return "no gated prediction"
    if not SHADOWMODE_ENABLE_ACTUATION:
      return "hard gate disabled"
    if not requested:
      return "ui disabled"
    if self._sendcan is None:
      return "sendcan unavailable"
    if not live_ready:
      return live_reason or "live control unavailable"
    if not safety_ready:
      return safety_reason or "blocked by panda safety"
    return "not allowed"

  def _build_actuator_map(self, sample: LiveSample, predicted_gated: dict[str, Any]) -> dict[str, Any]:
    requested = self.actuation_requested()
    actuator_map = map_shadow_controls_to_gm_can(
      steering=predicted_gated.get("steering"),
      throttle_magnitude=predicted_gated.get("throttle"),
      brake_magnitude=predicted_gated.get("brake"),
    )
    previewed = bool(actuator_map.get("previewed"))
    can_transmit = bool(self._sendcan is not None and can_list_to_can_capnp is not None)
    actual = dict(sample.actual or {})
    car_state_seen = bool(actual.get("car_state_seen", False))
    car_state_alive = bool(actual.get("car_state_alive", False))
    car_state_valid = bool(actual.get("car_state_valid", False))
    car_state_age = actual.get("car_state_age_seconds")
    car_state_fresh = (
      isinstance(car_state_age, (int, float))
      and float(car_state_age) <= LIVE_SAMPLER_STALE_S
    )
    if not car_state_seen:
      live_reason = "waiting for carState"
    elif not car_state_alive:
      live_reason = "carState not alive"
    elif not car_state_valid:
      live_reason = "carState invalid"
    elif not car_state_fresh:
      live_reason = f"carState stale ({float(car_state_age):.1f}s)" if isinstance(car_state_age, (int, float)) else "carState stale"
    elif sample.car_control is None:
      live_reason = "waiting for carControl"
    elif not sample.gm_live_state:
      live_reason = "GM live state unavailable"
    else:
      live_reason = "ready"
    live_ready = bool(
      sample.car_control is not None
      and sample.car_state is not None
      and sample.gm_live_state
      and car_state_alive
      and car_state_valid
      and car_state_fresh
    )
    panda_safety = dict(sample.panda_safety or {})
    safety_ready = bool(panda_safety.get("ready", False))
    safety_reason = str(panda_safety.get("reason", "panda safety unavailable"))
    can_accepted_observed = bool(sample.gm_live_state.get("loopback_lka_steering_cmd_updated", False))
    allowed = bool(
      requested
      and SHADOWMODE_ENABLE_ACTUATION
      and can_transmit
      and previewed
      and live_ready
      and safety_ready
    )
    can_msgs = []
    published_to_sendcan = False

    if allowed:
      model_cc = self._model_car_control(sample.car_control, predicted_gated, active=True)
      can_msgs = _SHADOW_GM_CONTROLLER.build(
        cc_cereal=model_cc,
        cs_live=sample.gm_live_state,
        car_state_cereal=sample.car_state,
      )
      if can_msgs:
        self._sendcan.send(can_list_to_can_capnp(can_msgs, msgtype="sendcan"))
        published_to_sendcan = True

    transmitting = bool(allowed and published_to_sendcan)
    actuator_map.update({
      "previewed": previewed,
      "requested": requested,
      "allowed": allowed,
      "transmitting": transmitting,
      "publishedToSendcan": published_to_sendcan,
      "canAcceptedObserved": can_accepted_observed,
      "canTransmit": can_transmit,
      "liveReady": live_ready,
      "liveReason": live_reason,
      "carStateAlive": car_state_alive,
      "carStateValid": car_state_valid,
      "carStateAgeSeconds": car_state_age,
      "safetyReady": safety_ready,
      "safety": panda_safety,
      "hard_gate": bool(SHADOWMODE_ENABLE_ACTUATION),
      "reason": (
        "transmitting"
        if transmitting
        else self._actuator_reason(
          requested,
          allowed,
          previewed,
          live_ready,
          live_reason,
          safety_ready,
          safety_reason,
        )
      ),
      "canMessageCount": len(can_msgs),
    })
    return actuator_map

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
      self._set_state(
        running=False,
        lastVaeRuntime=vae_runtime,
        shadowActuationAllowed=False,
        shadowActuationTransmitting=False,
        shadowActuationReason="policy not loaded",
      )
      return
    if self._vae_session is not None and not vae_runtime.get("ran"):
      self._set_state(
        running=False,
        lastVaeRuntime=vae_runtime,
        shadowActuationAllowed=False,
        shadowActuationTransmitting=False,
        shadowActuationReason="vae not ready",
      )
      return

    try:
      policy_start = time.monotonic()
      result = self._session.run(sample)
      predicted_raw = result.get("predicted", {})
      predicted_gated = _gate_longitudinal_controls(predicted_raw)
      result["predicted_raw"] = predicted_raw
      result["predicted_gated"] = predicted_gated
      actuator_map = self._build_actuator_map(sample, predicted_gated)
      result["actuator_map"] = actuator_map
      now = time.time()
      self._append_session_log(result, vae_runtime)
      self._append_actuator_log(result)
      self._set_state(
        running=True,
        lastPolicyRunTime=now,
        lastPolicyDurationMs=(time.monotonic() - policy_start) * 1000.0,
        policyRunCount=self._state.get("policyRunCount", 0) + 1,
        lastPolicyError=None,
        lastInference={
          "ok": result.get("ok", False),
          "modelType": result.get("modelType"),
          "actual": result.get("actual", {}),
          "predictedRaw": predicted_raw,
          "predictedGated": predicted_gated,
          "actuatorMap": actuator_map,
          "outputs": result.get("outputs", []),
          "callMode": result.get("callMode"),
          "inputOrder": result.get("inputOrder", []),
          "inputShapes": result.get("inputShapes", []),
        },
        lastActuatorMap=actuator_map,
        shadowActuationRequested=bool(actuator_map.get("requested")),
        shadowActuationAllowed=bool(actuator_map.get("allowed")),
        shadowActuationTransmitting=bool(actuator_map.get("transmitting")),
        shadowActuationReason=str(actuator_map.get("reason", "")),
        lastVaeRuntime=vae_runtime,
      )
    except Exception as e:
      self._set_state(
        running=False,
        lastPolicyError=str(e),
        policyErrorCount=self._state.get("policyErrorCount", 0) + 1,
        lastVaeRuntime=vae_runtime,
        shadowActuationAllowed=False,
        shadowActuationTransmitting=False,
        shadowActuationReason="policy inference failed",
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
          self._set_state(
            running=False,
            shadowActuationAllowed=False,
            shadowActuationTransmitting=False,
            shadowActuationReason="inference stopped",
          )
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
          shadowActuationAllowed=False,
          shadowActuationTransmitting=False,
          shadowActuationReason="worker error",
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
    replacement._actuation_requested = INFERENCE_WORKER.actuation_requested()
    replacement._state["enabled"] = replacement._enabled
    replacement._state["shadowActuationRequested"] = replacement._actuation_requested
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
  actuator_map = worker.get("lastActuatorMap", {})
  actuation_requested = bool(worker.get("shadowActuationRequested", False))
  actuation_allowed = bool(worker.get("shadowActuationAllowed", False))
  actuation_transmitting = bool(worker.get("shadowActuationTransmitting", False))
  actuation_reason = worker.get("shadowActuationReason") or actuator_map.get("reason") or "ui disabled"
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
    "shadowActuationAllowed": actuation_allowed,
    "shadowActuationTransmitting": actuation_transmitting,
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
    "actuatorLogPath": worker["actuatorLogPath"],
    "actuatorLogCount": worker["actuatorLogCount"],
    "actuatorLogRecent": worker["actuatorLogRecent"],
    "policyErrorCount": worker["policyErrorCount"],
    "vaeErrorCount": worker["vaeErrorCount"],
    "lastPolicyError": worker["lastPolicyError"],
    "lastVaeError": worker["lastVaeError"],
    "latestInference": latest,
    "lastSample": current.actual,
    "actual": current.actual if current else {},
    "actuatorMap": actuator_map,
    "predictedRaw": latest.get("predictedRaw", {}),
    "predictedGated": latest.get("predictedGated", {}),
    "sessionError": worker["lastPolicyError"],
    "vaeError": worker["lastVaeError"],
    "inferenceError": worker["lastPolicyError"],
    "message": "inference running" if worker["running"] else (
      "inference stopped" if not worker["enabled"] else "waiting for successful inference"),
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
    if self.path == "/shadow/actuator_log":
      try:
        log_path = INFERENCE_WORKER.snapshot().get("actuatorLogPath")
        if not log_path:
          self._send_json({"ok": False, "error": "no active actuator log"}, HTTPStatus.NOT_FOUND)
          return
        path = Path(log_path)
        if not path.exists():
          self._send_json({"ok": False, "error": "actuator log not found"}, HTTPStatus.NOT_FOUND)
          return
        self._send_file(path, download_name=path.name)
      except Exception as e:
        cloudlog.exception("shadowmode actuator log download failed: %s", e)
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
      cloudlog.info("shadowmode policy upload received: bytes=%d revision=%d path=%s", len(body), MODEL_REVISION,
                    MODEL_PATH)
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
