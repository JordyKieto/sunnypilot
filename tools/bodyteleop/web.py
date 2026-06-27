import dataclasses
import json
import logging
import os
import ssl
import subprocess
from pathlib import Path
import cereal.messaging as messaging

from aiohttp import web
from aiohttp import ClientSession

from openpilot.common.basedir import BASEDIR
from openpilot.system.webrtc.webrtcd import StreamRequestBody
from openpilot.common.params import Params
from tools.bodyteleop.irl_infer import InferenceEngine

logger = logging.getLogger("bodyteleop")
logging.basicConfig(level=logging.INFO)

TELEOPDIR = f"{BASEDIR}/tools/bodyteleop"
WEBRTCD_HOST, WEBRTCD_PORT = "localhost", 5001
DEFAULT_MODEL_PATH = os.path.join(BASEDIR, "sunnypilot", "models", "irl_policy_bolt_200.onnx")
DEFAULT_DATASET_PATH = os.path.join(BASEDIR, "drift-datasets-irl", "preprocessed_dataset.npz")
ENGINE = InferenceEngine(DEFAULT_MODEL_PATH, DEFAULT_DATASET_PATH)
PM = messaging.PubMaster(['carControl'])
MODEL_UPLOAD_PATH = Path(DEFAULT_MODEL_PATH)


## SSL
def create_ssl_cert(cert_path: str, key_path: str):
  try:
    proc = subprocess.run(f'openssl req -x509 -newkey rsa:4096 -nodes -out {cert_path} -keyout {key_path} \
                          -days 365 -subj "/C=US/ST=California/O=commaai/OU=comma body"',
                          capture_output=True, shell=True)
    proc.check_returncode()
  except subprocess.CalledProcessError as ex:
    raise ValueError(f"Error creating SSL certificate:\n[stdout]\n{proc.stdout.decode()}\n[stderr]\n{proc.stderr.decode()}") from ex


def create_ssl_context():
  cert_path = os.path.join(TELEOPDIR, "cert.pem")
  key_path = os.path.join(TELEOPDIR, "key.pem")
  if not os.path.exists(cert_path) or not os.path.exists(key_path):
    logger.info("Creating certificate...")
    create_ssl_cert(cert_path, key_path)
  else:
    logger.info("Certificate exists!")
  ssl_context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_SERVER)
  ssl_context.load_cert_chain(cert_path, key_path)

  return ssl_context

## ENDPOINTS
async def index(request: 'web.Request'):
  with open(os.path.join(TELEOPDIR, "static", "index.html")) as f:
    content = f.read()
    return web.Response(content_type="text/html", text=content)


async def ping(request: 'web.Request'):
  return web.Response(text="pong")


async def infer_status(request: 'web.Request'):
  status = ENGINE.status()
  status["manual_infer_mode"] = Params().get_bool("ManualInferMode")
  status["manual_infer_shadow_mode"] = Params().get_bool("ManualInferShadowMode")
  status["log"] = Params().get("ManualInferStatus") or ""
  return web.json_response(status)


async def infer_start(request: 'web.Request'):
  data = await request.json()
  shadow_mode = data.get('shadow', True)
  Params().put_bool("ManualInferMode", True)
  Params().put_bool("ManualInferShadowMode", shadow_mode)
  return web.json_response(ENGINE.start())


async def infer_stop(request: 'web.Request'):
  Params().put_bool("ManualInferMode", False)
  Params().put_bool("ManualInferShadowMode", False)
  # Send a neutral carControl message on stop
  CC = messaging.new_message('carControl')
  CC.carControl.actuators.accel = 0.0
  CC.carControl.actuators.steer = 0.0
  CC.carControl.actuators.gas = 0.0
  CC.carControl.actuators.brake = 0.0
  PM.send('carControl', CC)
  return web.json_response(ENGINE.stop())


async def infer_step(request: 'web.Request'):
  result = ENGINE.step_once()
  if not Params().get_bool("ManualInferShadowMode"):
    controls = result.get("chevy_bolt_controls", {})
    if controls:
      CC = messaging.new_message('carControl')
      CC.carControl.actuators.accel = controls.get("accelerator_pedal", 0.0)
      CC.carControl.actuators.steer = controls.get("steering_wheel", 0.0)
      CC.carControl.actuators.gas = controls.get("accelerator_pedal", 0.0)
      CC.carControl.actuators.brake = controls.get("brake_pedal", 0.0)
      PM.send('carControl', CC)
  return web.json_response(result)


async def upload_model(request: 'web.Request'):
  reader = await request.multipart()
  field = await reader.next()
  if field is None or field.name != "model":
    return web.json_response({"ok": False, "error": "missing model upload"}, status=400)

  MODEL_UPLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = MODEL_UPLOAD_PATH.with_suffix(".uploading")
  size = 0
  with open(tmp_path, "wb") as f:
    while True:
      chunk = await field.read_chunk()
      if not chunk:
        break
      size += len(chunk)
      f.write(chunk)
  os.replace(tmp_path, MODEL_UPLOAD_PATH)
  ENGINE.reload_model(str(MODEL_UPLOAD_PATH))
  Params().put("ManualInferStatus", json.dumps({
    "step": 0,
    "message": "model uploaded and ready",
    "payload": {"path": str(MODEL_UPLOAD_PATH), "bytes": size, "status": "ready"},
  }))
  return web.json_response({
    "ok": True,
    "path": str(MODEL_UPLOAD_PATH),
    "bytes": size,
    "message": "upload complete, model reloaded",
  })


async def offer(request: 'web.Request'):
  params = await request.json()
  body = StreamRequestBody(params["sdp"], ["driver"], ["testJoystick"], ["carState"])
  body_json = json.dumps(dataclasses.asdict(body))

  logger.info("Sending offer to webrtcd...")
  webrtcd_url = f"http://{WEBRTCD_HOST}:{WEBRTCD_PORT}/stream"
  async with ClientSession() as session, session.post(webrtcd_url, data=body_json) as resp:
    assert resp.status == 200
    answer = await resp.json()
    return web.json_response(answer)


def main():
  # Enable joystick debug mode
  Params().put_bool("JoystickDebugMode", True)

  # App needs to be HTTPS for WebRTC to work on the browser
  ssl_context = create_ssl_context()

  app = web.Application()
  app.router.add_get("/", index)
  app.router.add_get("/ping", ping, allow_head=True)
  app.router.add_get("/status", infer_status)
  app.router.add_post("/start", infer_start)
  app.router.add_post("/stop", infer_stop)
  app.router.add_post("/step", infer_step)
  app.router.add_post("/model", upload_model)
  app.router.add_post("/offer", offer)
  app.router.add_static('/static', os.path.join(TELEOPDIR, 'static'))
  web.run_app(app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context)


if __name__ == "__main__":
  main()
