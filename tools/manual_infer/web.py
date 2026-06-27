import dataclasses
import json
import logging
import os
import ssl

from aiohttp import web

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params

logger = logging.getLogger("manual_infer")
logging.basicConfig(level=logging.INFO)

APPDIR = f"{BASEDIR}/tools/manual_infer"


def create_ssl_context():
  cert_path = os.path.join(APPDIR, "cert.pem")
  key_path = os.path.join(APPDIR, "key.pem")
  ssl_context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_SERVER)
  if os.path.exists(cert_path) and os.path.exists(key_path):
    ssl_context.load_cert_chain(cert_path, key_path)
  return ssl_context


async def index(request: 'web.Request'):
  with open(os.path.join(APPDIR, "static", "index.html")) as f:
    return web.Response(content_type="text/html", text=f.read())


async def status(request: 'web.Request'):
  p = Params()
  return web.json_response({
    "running": p.get_bool("ManualInferMode"),
    "log": p.get("ManualInferStatus") or "",
  })


async def start(request: 'web.Request'):
  Params().put_bool("ManualInferMode", True)
  return web.json_response({"ok": True})


async def stop(request: 'web.Request'):
  Params().put_bool("ManualInferMode", False)
  return web.json_response({"ok": True})


def main():
  app = web.Application()
  app.router.add_get("/", index)
  app.router.add_get("/status", status)
  app.router.add_post("/start", start)
  app.router.add_post("/stop", stop)
  app.router.add_static("/static", os.path.join(APPDIR, "static"))
  web.run_app(app, access_log=None, host="0.0.0.0", port=5002, ssl_context=create_ssl_context())


if __name__ == "__main__":
  main()
