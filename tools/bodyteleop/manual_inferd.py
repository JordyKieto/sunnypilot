#!/usr/bin/env python3
import json
import time

import cereal.messaging as messaging
import numpy as np
from cereal import car, log
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
from openpilot.sunnypilot.modeld_v2.fill_model_msg import PublishState, fill_model_msg, fill_pose_msg
from openpilot.sunnypilot.modeld_v2.modeld import FrameMeta, ModelState
from openpilot.sunnypilot.modeld_v2.meta_helper import load_meta_constants
from openpilot.sunnypilot.models.helpers import get_active_bundle
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper


PROCESS_NAME = "tools.bodyteleop.manual_inferd"


def _log_payload(step, message, payload):
  params = Params()
  params.put("ManualInferStatus", json.dumps({
    "step": int(step),
    "message": message,
    "payload": payload,
  }))


def _log_error(step, exc):
  _log_payload(step, "error", {
    "error": str(exc),
    "type": exc.__class__.__name__,
  })


def _main_stream_and_camera():
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = (
        VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and
        VisionStreamType.VISION_STREAM_ROAD in available_streams
      )
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      return main_wide_camera, use_extra_client
    time.sleep(0.1)


def main():
  cloudlog.warning("manual inferd init")
  config_realtime_process(6, 54)

  params = Params()
  params.put_bool("ManualInferMode", True)

  model = ModelState()
  bundle = get_active_bundle(params)
  if bundle is not None:
    cloudlog.info("manual inferd active bundle: %s", bundle.name if hasattr(bundle, "name") else bundle)

  main_wide_camera, use_extra_client = _main_stream_and_camera()
  vipc_client_main_stream = (
    VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  )
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  pm = messaging.PubMaster(["manualInferStatus", "modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"])
  sm = messaging.SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])
  publish_state = PublishState()
  desire_helper = DesireHelper()
  camera_offset_helper = CameraOffsetHelper()

  meta_main = FrameMeta()
  meta_extra = FrameMeta()
  buf_main = None
  buf_extra = None
  last_vipc_frame_id = 0
  run_count = 0
  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  prev_action = log.ModelDataV2.Action()

  try:
    while params.get_bool("ManualInferMode"):
      while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
        buf_main = vipc_client_main.recv()
        meta_main = FrameMeta(vipc_client_main)
        if buf_main is None:
          break

      if buf_main is None:
        continue

      if use_extra_client:
        while True:
          buf_extra = vipc_client_extra.recv()
          meta_extra = FrameMeta(vipc_client_extra)
          if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
            break
        if buf_extra is None:
          continue
      else:
        buf_extra = buf_main
        meta_extra = meta_main

      sm.update(0)
      desire = desire_helper.desire
      is_rhd = sm["driverMonitoringState"].isRHD if sm.seen["driverMonitoringState"] else False
      frame_id = sm["roadCameraState"].frameId if sm.seen["roadCameraState"] else meta_main.frame_id
      v_ego = max(sm["carState"].vEgo, 0.) if sm.seen["carState"] else 0.0

      if sm.frame % 60 == 0:
        model.lat_delay = get_lat_delay(params, sm["liveDelay"].lateralDelay if sm.seen["liveDelay"] else 0.0)
        model.PLANPLUS_CONTROL = params.get("PlanplusControl", return_default=True)
        camera_offset_helper.set_offset(params.get("CameraOffset", return_default=True))
      lat_delay = model.lat_delay + model.LAT_SMOOTH_SECONDS

      if sm.updated["liveCalibration"] and sm.seen["roadCameraState"] and sm.seen["deviceState"]:
        device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
        dc = DEVICE_CAMERAS[(str(sm["deviceState"].deviceType), str(sm["roadCameraState"].sensor))]
        model_transform_main = get_warp_matrix(
          device_from_calib_euler,
          dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics,
          False,
        ).astype(np.float32)
        model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
        model_transform_main, model_transform_extra = camera_offset_helper.update(
          model_transform_main, model_transform_extra, sm, main_wide_camera
        )
        live_calib_seen = True

      traffic_convention = np.zeros(2)
      traffic_convention[int(is_rhd)] = 1

      vec_desire = np.zeros(model.constants.DESIRE_LEN, dtype=np.float32)
      if 0 <= desire < model.constants.DESIRE_LEN:
        vec_desire[desire] = 1

      vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
      frames_dropped = vipc_dropped_frames
      if run_count < 10:
        frames_dropped = 0
      run_count += 1
      frame_drop_ratio = frames_dropped / (1 + frames_dropped)
      prepare_only = vipc_dropped_frames > 0

      bufs = {name: buf_extra if "big" in name else buf_main for name in model.model_runner.vision_input_names}
      transforms = {name: model_transform_extra if "big" in name else model_transform_main for name in model.model_runner.vision_input_names}
      inputs = {
        model.desire_key: vec_desire,
        "traffic_convention": traffic_convention,
      }
      if "lateral_control_params" in model.numpy_inputs:
        inputs["lateral_control_params"] = np.array([v_ego, lat_delay], dtype=np.float32)

      mt1 = time.perf_counter()
      model_output = model.run(bufs, transforms, inputs, prepare_only)
      mt2 = time.perf_counter()
      model_execution_time = mt2 - mt1

      payload = {
        "frame_id": int(meta_main.frame_id),
        "frame_age": int(frame_id - meta_main.frame_id if frame_id > meta_main.frame_id else 0),
        "frame_drop": float(frame_drop_ratio),
        "model_execution_time": float(model_execution_time),
        "live_calib_seen": bool(live_calib_seen),
        "prepare_only": bool(prepare_only),
      }

      if model_output is not None:
        action = model.get_action_from_model(
          model_output,
          prev_action,
          lat_delay + DT_MDL,
          model.LONG_SMOOTH_SECONDS + DT_MDL,
          v_ego,
        )
        prev_action = action

        if sm.seen["carState"]:
          payload["carState"] = {
            "vEgo": float(sm["carState"].vEgo),
            "brakePressed": bool(sm["carState"].brakePressed),
            "gasPressed": bool(sm["carState"].gasPressed),
            "steeringAngleDeg": float(sm["carState"].steeringAngleDeg),
          }
        payload["policy_outputs"] = {
          k: (v.tolist() if hasattr(v, "tolist") else v)
          for k, v in model_output.items()
        }
        payload["action"] = {
          "desiredCurvature": float(action.desiredCurvature),
          "desiredAcceleration": float(action.desiredAcceleration),
          "shouldStop": bool(action.shouldStop),
        }

        modelv2_send = messaging.new_message("modelV2")
        drivingdata_send = messaging.new_message("drivingModelData")
        posenet_send = messaging.new_message("cameraOdometry")
        mdv2sp_send = messaging.new_message("modelDataV2SP")
        fill_model_msg(
          drivingdata_send, modelv2_send, model_output, action,
          publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
          frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen, load_meta_constants()
        )
        fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
        pm.send("modelV2", modelv2_send)
        pm.send("drivingModelData", drivingdata_send)
        pm.send("cameraOdometry", posenet_send)
        pm.send("modelDataV2SP", mdv2sp_send)

      _log_payload(meta_main.frame_id, "running", payload)
      msg = messaging.new_message("manualInferStatus", valid=True)
      msg.manualInferStatus.frameId = int(meta_main.frame_id)
      msg.manualInferStatus.status = json.dumps(payload)
      pm.send("manualInferStatus", msg)
      last_vipc_frame_id = meta_main.frame_id

      if not params.get_bool("ManualInferMode"):
        break
  except Exception as exc:
    cloudlog.exception("manual inferd error")
    _log_error(last_vipc_frame_id, exc)
    raise

  _log_payload(last_vipc_frame_id, "stopped", {})
  params.put_bool("ManualInferMode", False)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("manual inferd got SIGINT")
