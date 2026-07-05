#!/usr/bin/env python3
import os
from openpilot.selfdrive.modeld.tinygrad_helpers import MODELS_DIR, set_tinygrad_backend_from_compiled_flags
set_tinygrad_backend_from_compiled_flags()

# FIXME-SP: remove once we bump tg
from openpilot.system.hardware import TICI
os.environ['DEV'] = 'QCOM' if TICI else 'CPU'

USBGPU = "USBGPU" in os.environ
if USBGPU:
  os.environ['DEV'] = 'AMD'
  os.environ['AMD_IFACE'] = 'USB'
from tinygrad.tensor import Tensor
import time
import pickle
import numpy as np
import cereal.messaging as messaging
from cereal import car, log
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value, get_curvature_from_plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.common.file_chunker import read_file_chunked
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.selfdrive.controls.irl_policy import (
  _OnnxSession,
  ACCEL_MAX,
  ACCEL_MIN,
  CONTROL_HISTORY_DIM,
  IMAGE_HEIGHT,
  IMAGE_WIDTH,
  POLICY_PATH as IRL_POLICY_PATH,
  SPEED_SCALE_MS,
  STACK_SIZE,
  STEERING_ANGLE_SCALE_DEG,
  STEERING_RATE_SCALE_DEG,
  VAE_LATENT_DIM,
  VAE_PATH as IRL_ENCODER_PATH,
  IrlPolicyController,
)


PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

VISION_PKL_PATH = MODELS_DIR / 'driving_vision_tinygrad.pkl'
VISION_METADATA_PATH = MODELS_DIR / 'driving_vision_metadata.pkl'
POLICY_PKL_PATH = MODELS_DIR / 'driving_policy_tinygrad.pkl'
POLICY_METADATA_PATH = MODELS_DIR / 'driving_policy_metadata.pkl'

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3

IMG_QUEUE_SHAPE = (6*(ModelConstants.MODEL_RUN_FREQ//ModelConstants.MODEL_CONTEXT_FREQ + 1), 128, 256)
assert IMG_QUEUE_SHAPE[0] == 30
IRL_MODELD_ENABLED = os.getenv("IRL_MODELD_ENABLED", "1") not in ("0", "false", "False")


def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    plan = model_output['plan'][0]
    desired_accel, should_stop = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                     plan[:,Plan.ACCELERATION][:,0],
                                                     ModelConstants.T_IDXS,
                                                     action_t=long_action_t)
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)

    desired_curvature = get_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                plan[:,Plan.ORIENTATION_RATE][:,2],
                                                ModelConstants.T_IDXS,
                                                v_ego,
                                                lat_action_t)
    if v_ego > MIN_LAT_CONTROL_SPEED:
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
    else:
      desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                  desiredAcceleration=float(desired_accel),
                                  shouldStop=bool(should_stop))

class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof

class InputQueues:
  def __init__ (self, model_fps, env_fps, n_frames_input):
    assert env_fps % model_fps == 0
    assert env_fps >= model_fps
    self.model_fps = model_fps
    self.env_fps = env_fps
    self.n_frames_input = n_frames_input

    self.dtypes = {}
    self.shapes = {}
    self.q = {}

  def update_dtypes_and_shapes(self, input_dtypes, input_shapes) -> None:
    self.dtypes.update(input_dtypes)
    if self.env_fps == self.model_fps:
      self.shapes.update(input_shapes)
    else:
      for k in input_shapes:
        shape = list(input_shapes[k])
        if 'img' in k:
          n_channels = shape[1] // self.n_frames_input
          shape[1] = (self.env_fps // self.model_fps + (self.n_frames_input - 1)) * n_channels
        else:
          shape[1] = (self.env_fps // self.model_fps) * shape[1]
        self.shapes[k] = tuple(shape)

  def reset(self) -> None:
    self.q = {k: np.zeros(self.shapes[k], dtype=self.dtypes[k]) for k in self.dtypes.keys()}

  def enqueue(self, inputs:dict[str, np.ndarray]) -> None:
    for k in inputs.keys():
      if inputs[k].dtype != self.dtypes[k]:
        raise ValueError(f'supplied input <{k}({inputs[k].dtype})> has wrong dtype, expected {self.dtypes[k]}')
      input_shape = list(self.shapes[k])
      input_shape[1] = -1
      single_input = inputs[k].reshape(tuple(input_shape))
      sz = single_input.shape[1]
      self.q[k][:,:-sz] = self.q[k][:,sz:]
      self.q[k][:,-sz:] = single_input

  def get(self, *names) -> dict[str, np.ndarray]:
    if self.env_fps == self.model_fps:
      return {k: self.q[k] for k in names}
    else:
      out = {}
      for k in names:
        shape = self.shapes[k]
        if 'img' in k:
          n_channels = shape[1] // (self.env_fps // self.model_fps + (self.n_frames_input - 1))
          out[k] = np.concatenate([self.q[k][:, s:s+n_channels] for s in np.linspace(0, shape[1] - n_channels, self.n_frames_input, dtype=int)], axis=1)
        elif 'pulse' in k:
          # any pulse within interval counts
          out[k] = self.q[k].reshape((shape[0], shape[1] * self.model_fps // self.env_fps, self.env_fps // self.model_fps, -1)).max(axis=2)
        else:
          idxs = np.arange(-1, -shape[1], -self.env_fps // self.model_fps)[::-1]
          out[k] = self.q[k][:, idxs]
      return out

class ModelState(ModelStateBase):
  inputs: dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self):
    ModelStateBase.__init__(self)
    self.LAT_SMOOTH_SECONDS = LAT_SMOOTH_SECONDS
    with open(VISION_METADATA_PATH, 'rb') as f:
      vision_metadata = pickle.load(f)
      self.vision_input_shapes =  vision_metadata['input_shapes']
      self.vision_input_names = list(self.vision_input_shapes.keys())
      self.vision_output_slices = vision_metadata['output_slices']
      vision_output_size = vision_metadata['output_shapes']['outputs'][1]

    with open(POLICY_METADATA_PATH, 'rb') as f:
      policy_metadata = pickle.load(f)
      self.policy_input_shapes =  policy_metadata['input_shapes']
      self.policy_output_slices = policy_metadata['output_slices']
      policy_output_size = policy_metadata['output_shapes']['outputs'][1]

    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

    # policy inputs
    self.numpy_inputs = {k: np.zeros(self.policy_input_shapes[k], dtype=np.float32) for k in self.policy_input_shapes}
    self.full_input_queues = InputQueues(ModelConstants.MODEL_CONTEXT_FREQ, ModelConstants.MODEL_RUN_FREQ, ModelConstants.N_FRAMES)
    for k in ['desire_pulse', 'features_buffer']:
      self.full_input_queues.update_dtypes_and_shapes({k: self.numpy_inputs[k].dtype}, {k: self.numpy_inputs[k].shape})
    self.full_input_queues.reset()

    self.img_queues = {'img': Tensor.zeros(IMG_QUEUE_SHAPE, dtype='uint8').contiguous().realize(),
                       'big_img': Tensor.zeros(IMG_QUEUE_SHAPE, dtype='uint8').contiguous().realize()}
    self.full_frames : dict[str, Tensor] = {}
    self._blob_cache : dict[int, Tensor] = {}
    self.transforms_np = {k: np.zeros((3,3), dtype=np.float32) for k in self.img_queues}
    self.transforms = {k: Tensor(v, device='NPY').realize() for k, v in self.transforms_np.items()}
    self.vision_output = np.zeros(vision_output_size, dtype=np.float32)
    self.policy_inputs = {k: Tensor(v, device='NPY').realize() for k,v in self.numpy_inputs.items()}
    self.policy_output = np.zeros(policy_output_size, dtype=np.float32)
    self.parser = Parser()
    self.frame_buf_params : dict[str, tuple[int, int, int, int]] = {}
    self.update_imgs = None
    self.vision_run = pickle.loads(read_file_chunked(str(VISION_PKL_PATH)))
    self.policy_run = pickle.loads(read_file_chunked(str(POLICY_PKL_PATH)))

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
                inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire_pulse'][0] = 0
    new_desire = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    if self.update_imgs is None:
      for key in bufs.keys():
        w, h = bufs[key].width, bufs[key].height
        self.frame_buf_params[key] = get_nv12_info(w, h)
      warp_path = MODELS_DIR / f'warp_{w}x{h}_tinygrad.pkl'
      with open(warp_path, "rb") as f:
        self.update_imgs = pickle.load(f)

    for key in bufs.keys():
      ptr = bufs[key].data.ctypes.data
      yuv_size = self.frame_buf_params[key][3]
      # There is a ringbuffer of imgs, just cache tensors pointing to all of them
      cache_key = (key, ptr)
      if cache_key not in self._blob_cache:
        self._blob_cache[cache_key] = Tensor.from_blob(ptr, (yuv_size,), dtype='uint8')
      self.full_frames[key] = self._blob_cache[cache_key]
    for key in bufs.keys():
      self.transforms_np[key][:,:] = transforms[key][:,:]

    out = self.update_imgs(self.img_queues['img'], self.full_frames['img'], self.transforms['img'],
                           self.img_queues['big_img'], self.full_frames['big_img'], self.transforms['big_img'])
    vision_inputs = {'img': out[0], 'big_img': out[1]}

    if prepare_only:
      return None

    self.vision_output = self.vision_run(**vision_inputs).contiguous().realize().uop.base.buffer.numpy().flatten()
    vision_outputs_dict = self.parser.parse_vision_outputs(self.slice_outputs(self.vision_output, self.vision_output_slices))

    self.full_input_queues.enqueue({'features_buffer': vision_outputs_dict['hidden_state'], 'desire_pulse': new_desire})
    for k in ['desire_pulse', 'features_buffer']:
      self.numpy_inputs[k][:] = self.full_input_queues.get(k)[k]
    self.numpy_inputs['traffic_convention'][:] = inputs['traffic_convention']

    self.policy_output = self.policy_run(**self.policy_inputs).contiguous().realize().uop.base.buffer.numpy().flatten()
    policy_outputs_dict = self.parser.parse_policy_outputs(self.slice_outputs(self.policy_output, self.policy_output_slices))
    combined_outputs_dict = {**vision_outputs_dict, **policy_outputs_dict}
    if SEND_RAW_PRED:
      combined_outputs_dict['raw_pred'] = np.concatenate([self.vision_output.copy(), self.policy_output.copy()])

    return combined_outputs_dict


class IrlModelState(ModelStateBase):
  vision_input_names = ["img"]

  def __init__(self):
    ModelStateBase.__init__(self)
    self.LAT_SMOOTH_SECONDS = LAT_SMOOTH_SECONDS
    self.encoder = _OnnxSession(IRL_ENCODER_PATH)
    self.policy = _OnnxSession(IRL_POLICY_PATH)
    self.image_latents: list[np.ndarray] = []
    self.telemetry: list[np.ndarray] = []
    self.control_history: list[np.ndarray] = []
    self.last_prediction = {"steering": 0.0, "accel": 0.0, "state": 0}
    cloudlog.warning("IRL modeld loaded encoder=%s policy=%s", IRL_ENCODER_PATH, IRL_POLICY_PATH)

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

  @classmethod
  def _buf_to_rgb(cls, buf: VisionBuf) -> np.ndarray:
    uv_height = ((buf.height // 2) + 15) // 16 * 16
    uv_plane_size = buf.stride * uv_height
    y = np.array(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
    uv_data = buf.data[buf.uv_offset:buf.uv_offset + uv_plane_size]
    u = np.array(uv_data[::2], dtype=np.uint8).reshape((-1, buf.stride // 2))[:buf.height // 2, :buf.width // 2]
    v = np.array(uv_data[1::2], dtype=np.uint8).reshape((-1, buf.stride // 2))[:buf.height // 2, :buf.width // 2]
    return cls._resize_rgb_nearest(cls._yuv_to_rgb(y, u, v))

  @staticmethod
  def _append(history: list[np.ndarray], value: np.ndarray) -> None:
    history.append(value.astype(np.float32))
    del history[:-STACK_SIZE]

  @staticmethod
  def _stack(history: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if not history:
      values = [np.zeros(shape, dtype=np.float32)] * STACK_SIZE
    else:
      values = [history[0]] * (STACK_SIZE - len(history)) + history
    return np.asarray(values, dtype=np.float32).reshape((1, STACK_SIZE, *shape))

  @staticmethod
  def _policy_to_curvature(steering: float, v_ego: float) -> float:
    # The IRL policy's steering output is a normalized steering-wheel target.
    # Convert it to a conservative road curvature proxy for the existing lateral stack.
    target_angle_rad = np.deg2rad(float(np.clip(steering, -1.0, 1.0)) * STEERING_ANGLE_SCALE_DEG)
    steer_ratio = 16.8
    wheelbase = 2.6
    curvature = target_angle_rad / max(steer_ratio * wheelbase, 1e-3)
    max_curvature = 3.0 / max(float(v_ego) ** 2, 1.0)
    return float(np.clip(curvature, -max_curvature, max_curvature))

  @staticmethod
  def _empty_model_output(v_ego: float, accel: float, curvature: float) -> dict[str, np.ndarray]:
    t = np.asarray(ModelConstants.T_IDXS, dtype=np.float32)
    x = np.maximum(0.0, float(v_ego) * t + 0.5 * float(accel) * t * t)
    v = np.maximum(0.0, float(v_ego) + float(accel) * t)
    a = np.full_like(t, float(accel), dtype=np.float32)
    yaw = 0.5 * float(curvature) * max(float(v_ego), 1.0) * t

    plan = np.zeros((1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH), dtype=np.float32)
    plan[0, :, Plan.POSITION] = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    plan[0, :, Plan.VELOCITY] = np.stack([v, np.zeros_like(v), np.zeros_like(v)], axis=1)
    plan[0, :, Plan.ACCELERATION] = np.stack([a, np.zeros_like(a), np.zeros_like(a)], axis=1)
    plan[0, :, Plan.T_FROM_CURRENT_EULER] = np.stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw], axis=1)
    plan[0, :, Plan.ORIENTATION_RATE] = 0.0

    line_x = np.asarray(ModelConstants.X_IDXS, dtype=np.float32)
    lane_lines = np.zeros((1, ModelConstants.NUM_LANE_LINES, ModelConstants.IDX_N, 2), dtype=np.float32)
    for i, y_off in enumerate((-1.8, -0.6, 0.6, 1.8)):
      lane_lines[0, i, :, 0] = y_off
      lane_lines[0, i, :, 1] = 0.0
    road_edges = np.zeros((1, ModelConstants.NUM_ROAD_EDGES, ModelConstants.IDX_N, 2), dtype=np.float32)
    road_edges[0, 0, :, 0] = -3.7
    road_edges[0, 1, :, 0] = 3.7

    return {
      "plan": plan,
      "plan_stds": np.ones_like(plan) * 0.1,
      "lane_lines": lane_lines,
      "lane_lines_stds": np.ones((1, ModelConstants.NUM_LANE_LINES, 1, 1), dtype=np.float32),
      "lane_lines_prob": np.asarray([[0.0, 0.2, 0.0, 0.6, 0.0, 0.6, 0.0, 0.2]], dtype=np.float32),
      "road_edges": road_edges,
      "road_edges_stds": np.ones((1, ModelConstants.NUM_ROAD_EDGES, 1, 1), dtype=np.float32),
      "lead": np.zeros((1, 3, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), dtype=np.float32),
      "lead_stds": np.ones((1, 3, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), dtype=np.float32),
      "lead_prob": np.zeros((1, 3), dtype=np.float32),
      "desire_state": np.zeros((1, ModelConstants.DESIRE_LEN), dtype=np.float32),
      "desire_pred": np.zeros((1, ModelConstants.DESIRE_PRED_LEN, ModelConstants.DESIRE_LEN), dtype=np.float32),
      "meta": np.zeros((1, 55), dtype=np.float32),
      "pose": np.zeros((1, ModelConstants.POSE_WIDTH), dtype=np.float32),
      "pose_stds": np.ones((1, ModelConstants.POSE_WIDTH), dtype=np.float32),
      "wide_from_device_euler": np.zeros((1, ModelConstants.WIDE_FROM_DEVICE_WIDTH), dtype=np.float32),
      "wide_from_device_euler_stds": np.ones((1, ModelConstants.WIDE_FROM_DEVICE_WIDTH), dtype=np.float32),
      "road_transform": np.zeros((1, ModelConstants.POSE_WIDTH), dtype=np.float32),
      "road_transform_stds": np.ones((1, ModelConstants.POSE_WIDTH), dtype=np.float32),
    }

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    if prepare_only:
      return None

    v_ego = float(inputs.get("v_ego", 0.0))
    steering_angle_deg = float(inputs.get("steering_angle_deg", 0.0))
    steering_rate_deg = float(inputs.get("steering_rate_deg", 0.0))
    gas = float(inputs.get("gas", 0.0))
    brake_pressed = float(inputs.get("brake_pressed", 0.0))

    rgb = self._buf_to_rgb(next(iter(bufs.values())))
    encoder_input = rgb.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.uint8)
    encoder_name = self.encoder.input_names[0] if self.encoder.input_names else "image"
    latent = IrlPolicyController._select_latent(self.encoder.run({encoder_name: encoder_input}))
    self._append(self.image_latents, latent.reshape((VAE_LATENT_DIM,)))
    self._append(self.telemetry, np.asarray([np.clip(v_ego / SPEED_SCALE_MS, 0.0, 2.0)], dtype=np.float32))
    self._append(self.control_history, np.asarray([
      np.clip(gas, 0.0, 1.0),
      np.clip(brake_pressed, 0.0, 1.0),
      np.clip(steering_angle_deg / STEERING_ANGLE_SCALE_DEG, -1.0, 1.0),
      np.clip(steering_rate_deg / STEERING_RATE_SCALE_DEG, -1.0, 1.0),
    ], dtype=np.float32))

    policy_inputs_all = {
      "image_latent": self._stack(self.image_latents, (VAE_LATENT_DIM,)),
      "telemetry": self._stack(self.telemetry, (1,)),
      "control_history": self._stack(self.control_history, (CONTROL_HISTORY_DIM,)),
    }
    policy_inputs = {name: policy_inputs_all[name] for name in (self.policy.input_names or policy_inputs_all.keys()) if name in policy_inputs_all}
    prediction = IrlPolicyController._decode_policy(self.policy.run(policy_inputs))
    prediction["accel"] = float(np.clip(prediction["accel"], ACCEL_MIN, ACCEL_MAX))
    self.last_prediction = prediction

    curvature = self._policy_to_curvature(prediction["steering"], v_ego)
    return self._empty_model_output(v_ego, prediction["accel"], curvature)

  def get_action(self, model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action, v_ego: float) -> log.ModelDataV2.Action:
    prediction = self.last_prediction
    desired_curvature = self._policy_to_curvature(float(prediction.get("steering", 0.0)), v_ego)
    desired_accel = float(np.clip(prediction.get("accel", 0.0), ACCEL_MIN, ACCEL_MAX))
    return log.ModelDataV2.Action(
      desiredCurvature=desired_curvature,
      desiredAcceleration=desired_accel,
      shouldStop=bool(v_ego < 0.3 and desired_accel < 0.1),
    )


def main(demo=False):
  cloudlog.warning("modeld init")

  if not USBGPU:
    # USB GPU currently saturates a core so can't do this yet,
    # also need to move the aux USB interrupts for good timings
    config_realtime_process(7, 54)

  st = time.monotonic()
  cloudlog.warning("loading model")
  model = IrlModelState() if IRL_MODELD_ENABLED else ModelState()
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()


  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                         extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    if sm.frame % 60 == 0:
      model.lat_delay = get_lat_delay(params, sm["liveDelay"].lateralDelay)
    lat_delay = model.lat_delay + LAT_SMOOTH_SECONDS
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    inputs:dict[str, np.ndarray] = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
      'v_ego': v_ego,
      'gas': float(sm["carState"].gas),
      'brake_pressed': float(sm["carState"].brakePressed),
      'steering_angle_deg': float(sm["carState"].steeringAngleDeg),
      'steering_rate_deg': float(sm["carState"].steeringRateDeg),
    }

    mt1 = time.perf_counter()
    model_output = model.run(bufs, transforms, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      mdv2sp_send = messaging.new_message('modelDataV2SP')

      frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
      action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
      if hasattr(model, "get_action"):
        action = model.get_action(model_output, prev_action, v_ego)
      else:
        action = get_action_from_model(model_output, prev_action, lat_delay + frame_delay + action_delay, long_delay + frame_delay + action_delay, v_ego)
      prev_action = action
      fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      mdv2sp_send.modelDataV2SP.laneTurnDirection = DH.lane_turn_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
      pm.send('modelDataV2SP', mdv2sp_send)
    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
