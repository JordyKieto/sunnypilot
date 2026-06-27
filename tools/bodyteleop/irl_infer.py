import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import cereal.messaging as messaging

import numpy as np

from openpilot.common.params import Params

LOGGER = logging.getLogger("irl_infer")


def _import_onnxruntime():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])
        import onnxruntime as ort
        return ort


class InferenceEngine:
    def __init__(self, model_path: str, dataset_path: str | None = None):
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._stop = threading.Event()
        self.pm = messaging.PubMaster(['carControl'])
        self._status = {
            "running": False,
            "message": "idle",
            "step": 0,
            "sample_index": None,
            "prediction": None,
            "source": None,
        }
        self._dataset = None
        self._cursor = 0
        self.model = None
        self._backend = None
        self._input_names = None

    def reload_model(self, model_path: str | None = None):
        if model_path is not None:
            self.model_path = Path(model_path)
        self.model = None
        self._backend = None
        self._input_names = None
        return self._load_model()

    def _load_model(self):
        if self.model is not None:
            return self.model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {self.model_path}")
        if self.model_path.suffix.lower() == ".onnx":
            ort = _import_onnxruntime()
            self._backend = "onnxruntime"
            self.model = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self._input_names = [i.name for i in self.model.get_inputs()]
        else:
            import tensorflow as tf

            self._backend = "keras"
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
        return self.model

    def _load_dataset(self):
        if self._dataset is not None:
            return self._dataset
        if self.dataset_path and self.dataset_path.exists():
            with np.load(self.dataset_path, allow_pickle=False) as cache:
                self._dataset = {
                    "image_latents": cache["image_latents"].astype(np.float32),
                    "telemetry": cache["telemetry"].astype(np.float32),
                    "control_history": cache["control_history"].astype(np.float32),
                }
                if "navigation_features" in cache and "route_waypoints" in cache:
                    self._dataset["navigation_features"] = cache["navigation_features"].astype(np.float32)
                    self._dataset["route_waypoints"] = cache["route_waypoints"].astype(np.float32)
            return self._dataset
        return None

    def _build_inputs(self, index: int):
        dataset = self._load_dataset()
        if dataset is None:
            image_latents = np.zeros((1, 51, 128), dtype=np.float32)
            telemetry = np.zeros((1, 51, 1), dtype=np.float32)
            control_history = np.zeros((1, 51, 4), dtype=np.float32)
            inputs = {
                "image_latent": image_latents,
                "telemetry": telemetry,
                "control_history": control_history,
            }
            source = "synthetic_fallback"
        else:
            size = len(dataset["image_latents"])
            idx = int(index % max(size, 1))
            image_latents = np.expand_dims(dataset["image_latents"][idx], axis=0)
            telemetry = np.expand_dims(dataset["telemetry"][idx], axis=0)
            control_history = np.expand_dims(dataset["control_history"][idx], axis=0)
            inputs = {
                "image_latent": image_latents,
                "telemetry": telemetry,
                "control_history": control_history,
            }
            if "navigation_features" in dataset and "route_waypoints" in dataset:
                inputs["navigation_features"] = np.expand_dims(dataset["navigation_features"][idx], axis=0)
                inputs["route_waypoints"] = np.expand_dims(dataset["route_waypoints"][idx], axis=0)
            source = "irl_cache"
        return inputs, source

    def step_once(self):
        model = self._load_model()
        inputs, source = self._build_inputs(self._cursor)
        if self._backend == "onnxruntime":
            feed = {}
            for name in self._input_names or []:
                if name in inputs:
                    feed[name] = inputs[name].astype(np.float32)
            outputs = model.run(None, feed)
            output_names = [o.name for o in model.get_outputs()]
            prediction = {name: np.asarray(value) for name, value in zip(output_names, outputs)}
        else:
            prediction = model(inputs, training=False)
        pedal_logits = np.asarray(prediction["pedal_state_logits"])[0]
        pedal_state = int(np.argmax(pedal_logits))
        throttle = float(np.asarray(prediction["throttle_magnitude"])[0, 0])
        brake = float(np.asarray(prediction["brake_magnitude"])[0, 0])
        steering = float(np.asarray(prediction["steering"])[0, 0])
        vego = float(np.asarray(prediction["vego"])[0, 0])
        delta_v = float(np.asarray(prediction["delta_v"])[0, 0])
        mapped = {
            "chevy_bolt_controls": {
                "accelerator_pedal": throttle if pedal_state == 1 else 0.0,
                "brake_pedal": brake if pedal_state == 2 else 0.0,
                "steering_wheel": steering,
            },
            "policy_outputs": {
                "pedal_state": pedal_state,
                "pedal_state_logits": pedal_logits.tolist(),
                "throttle_magnitude": throttle,
                "brake_magnitude": brake,
                "steering": steering,
                "vego": vego,
                "delta_v": delta_v,
            },
            "source": source,
            "sample_index": int(self._cursor),
        }
        self._cursor += 1
        with self._lock:
            self._status.update({
                "running": self._running,
                "message": "running" if self._running else "stopped",
                "step": self._status["step"] + 1,
                "sample_index": mapped["sample_index"],
                "prediction": mapped,
                "source": source,
            })
        return mapped

    def _loop(self):
        while not self._stop.is_set():
            try:
                result = self.step_once()
                if not Params().get_bool("ManualInferShadowMode"):
                    controls = result.get("chevy_bolt_controls", {})
                    if controls:
                        CC = messaging.new_message('carControl')
                        CC.carControl.actuators.accel = controls.get("accelerator_pedal", 0.0)
                        CC.carControl.actuators.steer = controls.get("steering_wheel", 0.0)
                        CC.carControl.actuators.gas = controls.get("accelerator_pedal", 0.0)
                        CC.carControl.actuators.brake = controls.get("brake_pedal", 0.0)
                        self.pm.send('carControl', CC)
            except Exception as exc:
                with self._lock:
                    self._status.update({"message": f"error: {exc}", "running": False})
                self._running = False
                return
            time.sleep(0.25)

    def start(self):
        if self._running:
            return self.status()
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        with self._lock:
            self._status["running"] = True
            self._status["message"] = "running"
        return self.status()

    def stop(self):
        self._stop.set()
        self._running = False
        # Send a neutral carControl message on stop
        CC = messaging.new_message('carControl')
        CC.carControl.actuators.accel = 0.0
        CC.carControl.actuators.steer = 0.0
        CC.carControl.actuators.gas = 0.0
        CC.carControl.actuators.brake = 0.0
        self.pm.send('carControl', CC)
        with self._lock:
            self._status["running"] = False
            self._status["message"] = "stopped"
        return self.status()

    def status(self):
        with self._lock:
            return dict(self._status)
