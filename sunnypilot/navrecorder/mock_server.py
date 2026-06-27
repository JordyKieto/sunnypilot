#!/usr/bin/env python3
import json
import math
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_URL_WAYPOINTS = 8


def _encode_polyline(points: list[tuple[float, float]]) -> str:
  result = []
  prev_lat = 0
  prev_lng = 0
  for lat, lng in points:
    lat_i = int(round(lat * 1e5))
    lng_i = int(round(lng * 1e5))
    for value, prev in ((lat_i, prev_lat), (lng_i, prev_lng)):
      delta = value - prev
      prev = value
      shifted = delta << 1 if delta >= 0 else ~(delta << 1)
      while shifted >= 0x20:
        result.append(chr((0x20 | (shifted & 0x1f)) + 63))
        shifted >>= 5
      result.append(chr(shifted + 63))
      if value == lat_i:
        prev_lat = value
      else:
        prev_lng = value
  return "".join(result)


def _lerp(a: float, b: float, t: float) -> float:
  return a + (b - a) * t


def _route_points(origin: dict[str, float], destination: dict[str, float], count: int = 28) -> list[tuple[float, float]]:
  lat1, lng1 = origin["lat"], origin["lng"]
  lat2, lng2 = destination["lat"], destination["lng"]
  points = []
  for i in range(count):
    t = i / max(count - 1, 1)
    curve = math.sin(t * math.pi) * 0.0015
    lat = _lerp(lat1, lat2, t) + curve
    lng = _lerp(lng1, lng2, t) + curve * 0.7
    points.append((lat, lng))
  return points


def _as_point(value: Any, default: tuple[float, float]) -> dict[str, float]:
  if isinstance(value, dict):
    if isinstance(value.get("latitude"), (int, float)) and isinstance(value.get("longitude"), (int, float)):
      return {"lat": float(value["latitude"]), "lng": float(value["longitude"])}
    if isinstance(value.get("lat"), (int, float)) and isinstance(value.get("lng"), (int, float)):
      return {"lat": float(value["lat"]), "lng": float(value["lng"])}
  return {"lat": default[0], "lng": default[1]}


def _google_maps_route_url(origin: dict[str, float], destination: dict[str, float], points: list[tuple[float, float]]) -> str:
  params = {
    "api": "1",
    "travelmode": "driving",
    "dir_action": "navigate",
    "origin": f"{origin['lat']:.6f},{origin['lng']:.6f}",
    "destination": f"{destination['lat']:.6f},{destination['lng']:.6f}",
  }
  waypoint_texts = [f"{lat:.6f},{lng:.6f}" for lat, lng in points[1:-1][:MAX_URL_WAYPOINTS]]
  if waypoint_texts:
    params["waypoints"] = "|".join(waypoint_texts)
  return "https://www.google.com/maps/dir/?%s" % urlencode(params, safe="|,")


@dataclass
class MockState:
  started_at: float = time.time()
  route_name: str = "mock-route"
  api_key: str = ""
  pending_destination: dict[str, Any] | None = None
  pending_origin: dict[str, Any] | None = None
  pending_polyline: str = ""


STATE = MockState()


def _live_gps() -> dict[str, Any]:
  t = time.time() - STATE.started_at
  lat = 37.7749 + math.sin(t / 20.0) * 0.006
  lng = -122.4194 + math.cos(t / 22.0) * 0.0065
  heading = (t * 12.0) % 360.0
  return {"latitude": lat, "longitude": lng, "bearingDeg": heading}


def _status() -> dict[str, Any]:
  current_gps = _live_gps()
  return {
    "hasApiKey": bool(STATE.api_key),
    "currentRoute": STATE.route_name,
    "hasPendingNavigation": STATE.pending_destination is not None,
    "pendingRoute": STATE.route_name if STATE.pending_destination else None,
    "pendingDestination": STATE.pending_destination,
    "pendingDestinationWaypoint": STATE.pending_destination,
    "pendingOrigin": STATE.pending_origin,
    "pendingEncodedPolyline": STATE.pending_polyline or None,
    "pendingStepCount": 5 if STATE.pending_polyline else 0,
    "lastRecordedRoute": STATE.route_name if STATE.pending_polyline else None,
    "lastDestination": STATE.pending_destination,
    "currentGps": current_gps,
  }


def _preview_route(body: dict[str, Any]) -> dict[str, Any]:
  origin = _as_point(body.get("origin"), (37.7749, -122.4194))
  destination = _as_point(body.get("destination"), (37.7849, -122.4094))
  points = _route_points(origin, destination)
  encoded = _encode_polyline(points)
  return {
    "ok": True,
    "distanceMeters": 1800,
    "duration": "420s",
    "encodedPolyline": encoded,
    "googleMapsUrl": _google_maps_route_url(origin, destination, points),
  }


def _plan_route(body: dict[str, Any]) -> dict[str, Any]:
  origin = _as_point(body.get("origin"), (37.7749, -122.4194))
  destination = _as_point(body.get("destination"), (37.7849, -122.4094))
  points = _route_points(origin, destination)
  encoded = _encode_polyline(points)
  STATE.pending_origin = {"location": {"latLng": {"latitude": origin["lat"], "longitude": origin["lng"]}}}
  STATE.pending_destination = {"location": {"latLng": {"latitude": destination["lat"], "longitude": destination["lng"]}}}
  STATE.pending_polyline = encoded
  return {
    "ok": True,
    "distanceMeters": 1800,
    "duration": "420s",
    "stepCount": 5,
    "speedLimitsIncluded": False,
    "currentRoute": STATE.route_name,
    "origin": STATE.pending_origin,
    "destination": STATE.pending_destination,
    "encodedPolyline": encoded,
    "googleMapsUrl": _google_maps_route_url(origin, destination, points),
    "message": "Mock route planned locally.",
  }


def _clear_pending_navigation() -> dict[str, Any]:
  STATE.pending_destination = None
  STATE.pending_origin = None
  STATE.pending_polyline = ""
  return {"ok": True, "message": "Pending navigation cleared."}


def _save_api_key(body: dict[str, Any]) -> dict[str, Any]:
  api_key = body.get("apiKey", "")
  if not isinstance(api_key, str) or not api_key.strip():
    return {"ok": True, "message": "API key cleared."}
  STATE.api_key = api_key.strip()
  return {"ok": True, "message": "API key saved locally."}


class MockNavRecorderHandler(SimpleHTTPRequestHandler):
  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

  def log_message(self, fmt: str, *args: Any) -> None:
    print(fmt % args)

  def _read_json(self) -> dict[str, Any]:
    length = int(self.headers.get("Content-Length", "0"))
    raw = self.rfile.read(length).decode("utf-8") if length else "{}"
    body = json.loads(raw)
    return body if isinstance(body, dict) else {}

  def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def do_GET(self) -> None:
    path = urlparse(self.path).path
    if path == "/status":
      self._send_json(_status())
      return
    if path == "/maps_api_key":
      self._send_json({"apiKey": STATE.api_key})
      return
    if path == "/":
      self.path = "/index.html"
    super().do_GET()

  def do_POST(self) -> None:
    path = urlparse(self.path).path
    body = self._read_json()
    if path == "/api_key":
      self._send_json(_save_api_key(body))
    elif path == "/preview_route":
      self._send_json(_preview_route(body))
    elif path == "/route":
      self._send_json(_plan_route(body))
    elif path == "/clear_nav":
      self._send_json(_clear_pending_navigation())
    elif path == "/client_log":
      self._send_json({"ok": True})
    else:
      self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main() -> None:
  server = ThreadingHTTPServer(("0.0.0.0", 5050), MockNavRecorderHandler)
  print("mock navrecorder listening on http://0.0.0.0:5050")
  server.serve_forever()


if __name__ == "__main__":
  main()
