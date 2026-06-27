#!/usr/bin/env python3
import json
import time

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


def _write_status(step, message, payload=None):
  Params().put("ManualInferStatus", json.dumps({
    "step": int(step),
    "message": message,
    "payload": payload or {},
  }))


def main():
  params = Params()
  params.put_bool("ManualInferMode", True)
  step = 0
  try:
    while params.get_bool("ManualInferMode"):
      _write_status(step, "running", {"note": "manual infer placeholder"})
      step += 1
      time.sleep(1.0)
  except Exception as exc:
    cloudlog.exception("manual infer error")
    _write_status(step, "error", {"error": str(exc), "type": exc.__class__.__name__})
    raise
  finally:
    params.put_bool("ManualInferMode", False)
    _write_status(step, "stopped", {})


if __name__ == "__main__":
  main()
