#!/usr/bin/env python3
import json
import math
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


STATIC_DIR = f"{BASEDIR}/sunnypilot/navrecorder/static"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
ROADS_SPEED_LIMITS_URL = "https://roads.googleapis.com/v1/speedLimits"
FIELD_MASK = ",".join([
  "routes.duration",
  "routes.distanceMeters",
  "routes.polyline.encodedPolyline",
  "routes.legs.distanceMeters",
  "routes.legs.duration",
  "routes.legs.steps.distanceMeters",
  "routes.legs.steps.staticDuration",
  "routes.legs.steps.polyline.encodedPolyline",
  "routes.legs.steps.navigationInstruction",
  "routes.legs.steps.startLocation",
  "routes.legs.steps.endLocation",
  "routes.travelAdvisory.speedReadingIntervals",
])
MAX_SPEED_LIMIT_POINTS = 100
MAX_URL_WAYPOINTS = 8


class RequestError(Exception):
  def __init__(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
    self.status = status
    self.payload = payload
    super().__init__(payload.get("error", status.phrase))


def _typed_json_param(params: Params, key: str) -> dict[str, Any] | None:
  value = params.get(key)
  return value if isinstance(value, dict) else None


def _string_json_param(params: Params, key: str) -> dict[str, Any] | None:
  raw = params.get(key)
  if not isinstance(raw, str) or not raw:
    return None
  try:
    value = json.loads(raw)
  except json.JSONDecodeError:
    return None
  return value if isinstance(value, dict) else None


def _current_origin(params: Params) -> dict[str, Any] | None:
  for key in ("LastGPSPositionLLK", "LastGPSPosition"):
    pos = _string_json_param(params, key)
    if pos is None:
      continue
    lat = pos.get("latitude")
    lng = pos.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
      cloudlog.info("navrecorder: using current GPS origin from %s", key)
      return {"location": {"latLng": {"latitude": lat, "longitude": lng}}}
  return None


def _waypoint(value: Any) -> dict[str, Any] | None:
  if value is None:
    return None
  if isinstance(value, dict):
    lat = value.get("latitude")
    lng = value.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
      return {"location": {"latLng": {"latitude": lat, "longitude": lng}}}
    address = value.get("address")
    if isinstance(address, str) and address.strip():
      return {"address": address.strip()}
  if isinstance(value, str) and value.strip():
    return {"address": value.strip()}
  return None


def _map_url_location(value: dict[str, Any] | None) -> str | None:
  if not isinstance(value, dict):
    return None
  place_id = value.get("placeId") or value.get("place_id")
  if isinstance(place_id, str) and place_id.strip():
    return place_id.strip()
  lat_lng = value.get("location", {}).get("latLng") if isinstance(value.get("location"), dict) else None
  if isinstance(lat_lng, dict):
    lat = lat_lng.get("latitude")
    lng = lat_lng.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
      return f"{lat:.6f},{lng:.6f}"
  if "latitude" in value and "longitude" in value:
    lat = value.get("latitude")
    lng = value.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
      return f"{lat:.6f},{lng:.6f}"
  return None


def _google_maps_route_url(route_response: dict[str, Any], origin: dict[str, Any], destination: dict[str, Any], steps: list[dict[str, Any]]) -> str:
  params: dict[str, Any] = {"api": "1", "travelmode": "driving", "dir_action": "navigate"}
  origin_text = _map_url_location(origin)
  destination_text = _map_url_location(destination)
  if origin_text:
    params["origin"] = origin_text
  if destination_text:
    params["destination"] = destination_text

  origin_place_id = origin.get("placeId") or origin.get("place_id") if isinstance(origin, dict) else None
  destination_place_id = destination.get("placeId") or destination.get("place_id") if isinstance(destination, dict) else None
  if isinstance(origin_place_id, str) and origin_place_id.strip():
    params["origin_place_id"] = origin_place_id.strip()
  if isinstance(destination_place_id, str) and destination_place_id.strip():
    params["destination_place_id"] = destination_place_id.strip()

  waypoint_texts: list[str] = []
  waypoint_place_ids: list[str] = []
  for step in steps[:MAX_URL_WAYPOINTS]:
    start = step.get("startLocation")
    text = _map_url_location(start if isinstance(start, dict) else None)
    if text:
      waypoint_texts.append(text)
    place_id = start.get("placeId") or start.get("place_id") if isinstance(start, dict) else None
    if isinstance(place_id, str) and place_id.strip():
      waypoint_place_ids.append(place_id.strip())

  if waypoint_texts:
    params["waypoints"] = "|".join(waypoint_texts)
  if waypoint_place_ids and len(waypoint_place_ids) == len(waypoint_texts):
    params["waypoint_place_ids"] = "|".join(waypoint_place_ids)

  return "https://www.google.com/maps/dir/?%s" % urlencode(params, safe="|,")


def _decode_polyline(encoded: str) -> list[tuple[float, float]]:
  points: list[tuple[float, float]] = []
  index = lat = lng = 0

  while index < len(encoded):
    result = shift = 0
    while True:
      b = ord(encoded[index]) - 63
      index += 1
      result |= (b & 0x1f) << shift
      shift += 5
      if b < 0x20:
        break
    lat += ~(result >> 1) if result & 1 else result >> 1

    result = shift = 0
    while True:
      b = ord(encoded[index]) - 63
      index += 1
      result |= (b & 0x1f) << shift
      shift += 5
      if b < 0x20:
        break
    lng += ~(result >> 1) if result & 1 else result >> 1

    points.append((lat / 1e5, lng / 1e5))

  return points


def _sample_points(points: list[tuple[float, float]], max_points: int = MAX_SPEED_LIMIT_POINTS) -> list[tuple[float, float]]:
  if len(points) <= max_points:
    return points
  last = len(points) - 1
  return [points[round(i * last / (max_points - 1))] for i in range(max_points)]


def _route_polyline(route: dict[str, Any]) -> str:
  polyline = route.get("polyline")
  if not isinstance(polyline, dict):
    return ""
  encoded = polyline.get("encodedPolyline")
  return encoded if isinstance(encoded, str) else ""


def _summarize_steps(route: dict[str, Any]) -> list[dict[str, Any]]:
  steps: list[dict[str, Any]] = []
  for leg_index, leg in enumerate(route.get("legs", [])):
    for step_index, step in enumerate(leg.get("steps", [])):
      instruction = step.get("navigationInstruction", {})
      if not isinstance(instruction, dict):
        instruction = {}
      steps.append({
        "legIndex": leg_index,
        "stepIndex": step_index,
        "distanceMeters": step.get("distanceMeters"),
        "staticDuration": step.get("staticDuration"),
        "maneuver": instruction.get("maneuver"),
        "instructions": instruction.get("instructions"),
        "startLocation": step.get("startLocation"),
        "endLocation": step.get("endLocation"),
        "encodedPolyline": (step.get("polyline") or {}).get("encodedPolyline"),
      })
  return steps


def _http_json(request: Request, timeout: int) -> tuple[int, dict[str, Any]]:
  try:
    with urlopen(request, timeout=timeout) as response:
      raw = response.read().decode("utf-8")
      return response.status, json.loads(raw) if raw else {}
  except HTTPError as e:
    raw = e.read().decode("utf-8", "replace")
    try:
      payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
      payload = {"raw": raw}
    return e.code, payload


def _fetch_speed_limits(api_key: str, encoded_polyline: str) -> dict[str, Any] | None:
  if not encoded_polyline:
    cloudlog.warning("navrecorder: skipping speed limits, route polyline missing")
    return None

  points = _sample_points(_decode_polyline(encoded_polyline))
  if not points:
    cloudlog.warning("navrecorder: skipping speed limits, decoded route polyline is empty")
    return None

  path = "|".join(f"{lat:.6f},{lng:.6f}" for lat, lng in points)
  url = f"{ROADS_SPEED_LIMITS_URL}?{urlencode({'path': path, 'units': 'MPH', 'key': api_key})}"
  cloudlog.info("navrecorder: requesting Roads speed limits for %d sampled route points", len(points))
  try:
    status, body = _http_json(Request(url), timeout=20)
    if status >= 400:
      cloudlog.warning("navrecorder: speed limits unavailable http_status=%d; continuing without them", status)
      return None
    speed_limit_count = len(body.get("speedLimits", [])) if isinstance(body.get("speedLimits"), list) else 0
    warning = body.get("warningMessage") or body.get("warning_message")
    cloudlog.info("navrecorder: Roads speed limits succeeded point_count=%d speed_limit_count=%d warning=%s",
                  len(points), speed_limit_count, warning or "none")
    return body
  except (OSError, URLError, TimeoutError):
    cloudlog.warning("navrecorder: speed limits unavailable due to request error; continuing without them", exc_info=True)
    return None


def _compute_route(api_key: str, origin: dict[str, Any], destination: dict[str, Any], avoid: dict[str, Any]) -> dict[str, Any]:
  body = {
    "origin": origin,
    "destination": destination,
    "travelMode": "DRIVE",
    "routingPreference": "TRAFFIC_AWARE",
    "computeAlternativeRoutes": False,
    "routeModifiers": {
      "avoidTolls": bool(avoid.get("tolls", False)),
      "avoidHighways": bool(avoid.get("highways", False)),
      "avoidFerries": bool(avoid.get("ferries", False)),
    },
    "languageCode": "en-US",
    "units": "IMPERIAL",
    "polylineQuality": "HIGH_QUALITY",
  }
  data = json.dumps(body).encode("utf-8")
  request = Request(
    ROUTES_URL,
    data=data,
    headers={
      "Content-Type": "application/json",
      "X-Goog-Api-Key": api_key,
      "X-Goog-FieldMask": FIELD_MASK,
    },
    method="POST",
  )
  cloudlog.info("navrecorder: requesting Google route origin_type=%s destination_type=%s avoid_tolls=%s avoid_highways=%s avoid_ferries=%s",
                "gps" if "location" in origin else "address",
                "gps" if "location" in destination else "address",
                bool(avoid.get("tolls", False)),
                bool(avoid.get("highways", False)),
                bool(avoid.get("ferries", False)))
  status, response = _http_json(request, timeout=30)
  if status >= 400:
    cloudlog.warning("navrecorder: Google route request failed http_status=%d", status)
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "routes api request failed", "httpStatus": status, "response": response})
  routes = response.get("routes") or []
  route = routes[0] if routes else {}
  cloudlog.info("navrecorder: Google route request succeeded route_count=%d distance_m=%s duration=%s",
                len(routes), route.get("distanceMeters"), route.get("duration"))
  return response


def _status() -> dict[str, Any]:
  params = Params()
  pending = _typed_json_param(params, "PendingNavigationRecording")
  last = _typed_json_param(params, "LastNavigationRecording")
  current_route = params.get("CurrentRoute")
  pending_route = (pending or {}).get("routeName")
  current_gps = _string_json_param(params, "LastGPSPositionLLK") or _string_json_param(params, "LastGPSPosition")
  return {
    "hasApiKey": bool(params.get("GoogleMapsApiKey")),
    "currentRoute": current_route,
    "hasPendingNavigation": pending is not None,
    "pendingRoute": pending_route,
    "pendingDestination": (pending or {}).get("destinationText"),
    "pendingDestinationWaypoint": (pending or {}).get("destination"),
    "pendingOrigin": (pending or {}).get("originText") or (pending or {}).get("origin"),
    "pendingEncodedPolyline": ((pending or {}).get("summary") or {}).get("encodedPolyline"),
    "pendingStepCount": len(((pending or {}).get("summary") or {}).get("steps") or []),
    "lastRecordedRoute": (last or {}).get("routeName"),
    "lastDestination": (last or {}).get("destinationText"),
    "currentGps": current_gps,
  }


def _save_api_key(body: dict[str, Any]) -> dict[str, Any]:
  api_key = body.get("apiKey", "")
  if not isinstance(api_key, str) or not api_key.strip():
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "apiKey is required"})
  Params().put("GoogleMapsApiKey", api_key.strip())
  cloudlog.info("navrecorder: Google Maps API key saved")
  return {"ok": True}


def _maps_api_key() -> dict[str, Any]:
  api_key = Params().get("GoogleMapsApiKey")
  if not isinstance(api_key, str) or not api_key.strip():
    raise RequestError(HTTPStatus.NOT_FOUND, {"error": "Google Maps API key is not saved"})
  return {"apiKey": api_key.strip()}


def _preview_route(body: dict[str, Any]) -> dict[str, Any]:
  params = Params()
  api_key = body.get("apiKey") or params.get("GoogleMapsApiKey")
  if not isinstance(api_key, str) or not api_key.strip():
    cloudlog.warning("navrecorder: route preview rejected, missing Google Maps API key")
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "Google Maps API key is required"})
  api_key = api_key.strip()

  manual_origin = _waypoint(body.get("origin"))
  origin = manual_origin or _current_origin(params)
  destination = _waypoint(body.get("destination"))
  if origin is None:
    cloudlog.warning("navrecorder: route preview rejected, current GPS unavailable and no origin override provided")
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "origin is required because current GPS is unavailable"})
  if destination is None:
    cloudlog.warning("navrecorder: route preview rejected, missing destination")
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "destination is required"})

  avoid = body.get("avoid") if isinstance(body.get("avoid"), dict) else {}
  cloudlog.info("navrecorder: route preview started origin_source=%s destination_type=%s",
                "manual" if manual_origin is not None else "gps",
                "address" if "address" in destination else "gps")
  routes_response = _compute_route(api_key, origin, destination, avoid)
  routes = routes_response.get("routes") or []
  if not routes:
    cloudlog.warning("navrecorder: Google returned no preview routes")
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "Google returned no routes", "response": routes_response})

  route = routes[0]
  encoded_polyline = _route_polyline(route)
  cloudlog.info("navrecorder: route preview succeeded distance_m=%s duration=%s has_polyline=%s",
                route.get("distanceMeters"), route.get("duration"), bool(encoded_polyline))
  return {
    "ok": True,
    "distanceMeters": route.get("distanceMeters"),
    "duration": route.get("duration"),
    "encodedPolyline": encoded_polyline,
    "googleMapsUrl": _google_maps_route_url(routes_response, origin, destination, _summarize_steps(route)),
  }


def _plan_route(body: dict[str, Any]) -> dict[str, Any]:
  params = Params()
  api_key = body.get("apiKey") or params.get("GoogleMapsApiKey")
  if not isinstance(api_key, str) or not api_key.strip():
    cloudlog.warning("navrecorder: route planning rejected, missing Google Maps API key")
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "Google Maps API key is required"})
  api_key = api_key.strip()
  params.put("GoogleMapsApiKey", api_key)

  manual_origin = _waypoint(body.get("origin"))
  origin = manual_origin or _current_origin(params)
  destination = _waypoint(body.get("destination"))
  if origin is None:
    cloudlog.warning("navrecorder: route planning rejected, current GPS unavailable and no origin override provided")
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "origin is required because current GPS is unavailable"})
  if destination is None:
    cloudlog.warning("navrecorder: route planning rejected, missing destination")
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "destination is required"})

  avoid = body.get("avoid") if isinstance(body.get("avoid"), dict) else {}
  current_route = params.get("CurrentRoute") or None
  cloudlog.info("navrecorder: route planning started current_route=%s origin_source=%s destination_type=%s",
                current_route or "none",
                "manual" if manual_origin is not None else "gps",
                "address" if "address" in destination else "gps")
  routes_response = _compute_route(api_key, origin, destination, avoid)
  routes = routes_response.get("routes") or []
  if not routes:
    cloudlog.warning("navrecorder: Google returned no routes")
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "Google returned no routes", "response": routes_response})

  route = routes[0]
  steps = _summarize_steps(route)
  cloudlog.info("navrecorder: route summary distance_m=%s duration=%s step_count=%d",
                route.get("distanceMeters"), route.get("duration"), len(steps))
  speed_limits = _fetch_speed_limits(api_key, _route_polyline(route))
  summary = {
    "distanceMeters": route.get("distanceMeters"),
    "duration": route.get("duration"),
    "encodedPolyline": _route_polyline(route),
    "steps": steps,
  }
  if speed_limits is not None:
    summary["speedLimits"] = speed_limits

  snapshot = {
    "schemaVersion": 1,
    "createdAtUnixSeconds": math.floor(time.time()),
    "source": "google_maps",
    "routeName": current_route,
    "origin": origin,
    "destination": destination,
    "originText": body.get("origin") if isinstance(body.get("origin"), str) else None,
    "destinationText": body.get("destination") if isinstance(body.get("destination"), str) else None,
    "routes": routes_response,
    "summary": summary,
  }
  params.put("PendingNavigationRecording", snapshot)
  cloudlog.info("navrecorder: pending navigation snapshot saved route=%s destination_set=%s speed_limits_included=%s",
                current_route or "next-route", bool(snapshot["destinationText"] or snapshot["destination"]),
                speed_limits is not None)

  return {
    "ok": True,
    "distanceMeters": snapshot["summary"]["distanceMeters"],
    "duration": snapshot["summary"]["duration"],
    "stepCount": len(snapshot["summary"]["steps"]),
    "speedLimitsIncluded": speed_limits is not None,
    "currentRoute": snapshot["routeName"],
    "origin": snapshot["origin"],
    "destination": snapshot["destination"],
    "encodedPolyline": snapshot["summary"]["encodedPolyline"],
    "googleMapsUrl": _google_maps_route_url(routes_response, origin, destination, snapshot["summary"]["steps"]),
    "message": "Navigation data will be written once to navigation.json for the current or next recording route.",
  }


def _clear_pending_navigation() -> dict[str, Any]:
  params = Params()
  params.remove("PendingNavigationRecording")
  cloudlog.info("navrecorder: cleared pending navigation recording")
  return {"ok": True, "message": "Pending navigation cleared."}


def _client_log(body: dict[str, Any]) -> dict[str, Any]:
  level = body.get("level")
  message = body.get("message")
  context = body.get("context")
  if not isinstance(level, str):
    level = "info"
  if not isinstance(message, str) or not message.strip():
    raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "message is required"})

  context_text = ""
  if context is not None:
    try:
      context_text = " " + json.dumps(context, sort_keys=True)
    except TypeError:
      context_text = f" {context!r}"

  log_message = "navrecorder client: " + message.strip() + context_text
  if level == "error":
    cloudlog.error(log_message)
  elif level == "warning":
    cloudlog.warning(log_message)
  else:
    cloudlog.info(log_message)
  return {"ok": True}


class NavRecorderHandler(SimpleHTTPRequestHandler):
  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, directory=STATIC_DIR, **kwargs)

  def log_message(self, fmt: str, *args: Any) -> None:
    cloudlog.info("navrecorder: " + fmt, *args)

  def _read_json(self) -> dict[str, Any]:
    length = int(self.headers.get("Content-Length", "0"))
    raw = self.rfile.read(length).decode("utf-8") if length else "{}"
    try:
      body = json.loads(raw)
    except json.JSONDecodeError as e:
      raise RequestError(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {e}"})
    if not isinstance(body, dict):
      raise RequestError(HTTPStatus.BAD_REQUEST, {"error": "JSON body must be an object"})
    return body

  def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def do_GET(self) -> None:
    try:
      if self.path == "/status":
        self._send_json(_status())
        return
      if self.path == "/maps_api_key":
        self._send_json(_maps_api_key())
        return
      if self.path == "/":
        self.path = "/index.html"
      super().do_GET()
    except RequestError as e:
      cloudlog.warning("navrecorder: request failed path=%s status=%d error=%s", self.path, e.status, e.payload.get("error"))
      self._send_json(e.payload, e.status)
    except Exception as e:
      cloudlog.exception("navrecorder request failed")
      self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

  def do_POST(self) -> None:
    try:
      body = self._read_json()
      if self.path == "/api_key":
        self._send_json(_save_api_key(body))
      elif self.path == "/preview_route":
        self._send_json(_preview_route(body))
      elif self.path == "/route":
        self._send_json(_plan_route(body))
      elif self.path == "/clear_nav":
        self._send_json(_clear_pending_navigation())
      elif self.path == "/client_log":
        self._send_json(_client_log(body))
      else:
        raise RequestError(HTTPStatus.NOT_FOUND, {"error": "not found"})
    except RequestError as e:
      cloudlog.warning("navrecorder: request failed path=%s status=%d error=%s", self.path, e.status, e.payload.get("error"))
      self._send_json(e.payload, e.status)
    except Exception as e:
      cloudlog.exception("navrecorder request failed")
      self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
  server = ThreadingHTTPServer(("0.0.0.0", 5050), NavRecorderHandler)
  cloudlog.info("navrecorder listening on 0.0.0.0:5050")
  server.serve_forever()


if __name__ == "__main__":
  main()
