# Navigation Mode PRD

## Summary

Navigation Mode is a proof of concept for destination-driven driving assistance in sunnypilot. The first version should let a user open a local web page from a phone on the same network as the device, enter a destination address, and start a route that is visible to the backend and, where practical, to the on-device UI.

The core product bet is that route context can become a small queryable memory during a drive. For the initial POC, that memory only needs to cover the next few minutes of road geometry, turns, speed limits, road names, and route progress. Later versions can expand this into a larger hybrid map memory that supports offline and online sources.

## Goals

* Provide a simple phone-accessible interface for destination entry.
* Resolve the destination into route data and persist the active route locally.
* Preprocess route geometry and metadata into a queryable short-horizon memory.
* Update route progress using ego vehicle GPS from `liveLocationKalman` or the best available location service.
* Expose navigation state to inference and UI surfaces without blocking existing driving flows.
* Support offline route/map memory as the MVP path for training and inference.

## Non-goals

* Full consumer navigation parity with Google Maps, Apple Maps, or Mapbox.
* Cloud account flows, saved places, or user profile sync.
* Replacing existing sunnypilot `mapd` speed-limit behavior in the first POC.
* Depending on online map APIs at inference time for the MVP.
* Automated lane changes, turns, or control-policy changes without separate safety review.

## Users

* Driver: enters a destination before a drive and wants the route context available to sunnypilot.
* Developer/researcher: records drives with aligned route memory for training and replay.
* Inference pipeline: queries upcoming route context as ego position changes.

## User Experience

The phone web page should be reachable from the same local network as the sunnypilot device. The page should be intentionally minimal:

* Destination address search/input.
* Current route status: inactive, resolving, active, rerouting, complete, or error.
* Route preview with destination, ETA/distance if available, and next maneuver.
* A live navigation view when available. For the POC this can be a simple map/route polyline or structured route progress panel.
* Controls to start, cancel, and refresh/reroute.

The on-device UI can remain minimal for the first version. If the existing nav surfaces are available, show active route state using `navInstruction`, `navRoute`, `navThumbnail`, or `mapRenderState`. If not, backend availability for logging and inference is sufficient for the first milestone.

## Backend Concept

Navigation Mode has four logical pieces:

* Local web server: serves the phone UI and accepts destination/route commands.
* Route resolver: geocodes the destination, computes candidate routes, and stores the selected route.
* Route memory builder: preprocesses geometry, maneuvers, speed limits, road names, intersections, and segment metadata into a compact queryable structure.
* Progress tracker: matches ego GPS to the active route and publishes current/upcoming context.

The progress tracker should use existing route and map conventions where possible. Relevant local integration points include `sunnypilot/navd/helpers.py`, `sunnypilot/mapd/live_map_data`, `liveMapDataSP`, and the existing cereal navigation messages.

## Route Memory

For the POC, route memory should cover approximately the next 5 minutes of travel. It should be cheap to update and easy for inference to query.

Minimum useful fields:

* Route ID and version.
* Destination label and coordinate.
* Full route polyline or list of coordinates.
* Segment list with start/end distance along route.
* Maneuvers with distance along route, type, modifier, and display text.
* Road names per segment where available.
* Posted speed limits per segment where available.
* Route progress: nearest segment, distance along route, distance to next maneuver, route confidence, and off-route flag.

Useful query shapes:

* `context_at(distance_m)`: route metadata at a distance along the route.
* `context_near(lat, lon)`: nearest route segment and upcoming metadata.
* `window_from_ego(horizon_s|horizon_m)`: compact route context ahead of the vehicle.
* `next_maneuver()`: next user-visible instruction.

## API Sketch

The exact transport can be HTTP plus local messaging for the POC.

### Phone UI HTTP

* `GET /navigation` - serves the local web interface.
* `GET /api/navigation/status` - returns active route state and progress.
* `POST /api/navigation/route` - body contains `destination`, optional `origin`, and route preferences.
* `POST /api/navigation/cancel` - clears the active route.
* `POST /api/navigation/reroute` - recomputes from current ego location.

Example status response:

```json
{
  "state": "active",
  "destination": "1 Market St, San Francisco, CA",
  "distanceRemainingM": 12450,
  "etaSeconds": 1120,
  "distanceAlongRouteM": 820,
  "offRoute": false,
  "nextManeuver": {
    "distanceM": 350,
    "type": "turn",
    "modifier": "right",
    "text": "Turn right"
  },
  "speedLimitMps": 13.4,
  "roadName": "Market St"
}
```

### Internal State

* `NavigationModeEnabled`: feature flag.
* `NavigationActiveRoute`: persisted active route metadata.
* `NavigationDestination`: selected destination.
* `NavigationRouteMemory`: compact serialized route memory.

Candidate published messages for implementation planning:

* Existing: `navInstruction`, `navRoute`, `navThumbnail`, `mapRenderState`.
* Existing sunnypilot map data: `liveMapDataSP` for speed limits and road names.
* New only if needed: `navigationMemorySP` for inference-specific route context.

## Map and Route Data Sources

Offline map support is the MVP path for training and inference. The system should prefer locally cached or preprocessed map/route data while driving.

Online services can be used during collection or route preparation when available. The pasted Google Roads API context suggests:

* Speed limits can be requested by path or road-segment place IDs.
* A path request snaps points to roads before returning speed limits.
* Requests should be batched, sparse calls should be avoided, and costs scale with returned speed-limit entries.
* Speed-limit data can be missing, estimated, stale, or unavailable in some regions.

The route resolver should treat online speed limits as optional metadata, not as required control input. Missing speed limits should degrade to existing mapd behavior or no speed-limit annotation.

## Data Collection

Training logs should preserve current modalities and add route-memory alignment. Each logged sample should be reproducible without online map access.

Minimum additions:

* Active route ID/version.
* Ego GPS and route-matched position.
* Distance along route.
* Queryable route-memory window used at that timestamp.
* Next maneuver and road/speed metadata.
* Route confidence and off-route status.

The dataset should continue to work when no route is active. Route memory should be nullable and explicitly marked unavailable instead of changing existing modality semantics.

## Inference Use

Inference should consume a compact route context, not raw external API responses. The initial shape can be a fixed horizon around ego position:

* Current road segment metadata.
* Next N maneuvers.
* Polyline points ahead, normalized relative to ego.
* Speed-limit and road-class annotations when available.
* Validity/confidence flags.

The POC can start with a 5-minute or distance-bounded horizon, whichever is smaller. The key requirement is deterministic replay: the same log should produce the same memory window without an online dependency.

## Safety and Privacy

* Destination entry must be unavailable or clearly constrained while driving unless handled by a passenger-safe interaction model.
* Route context must not directly authorize new vehicle maneuvers without a separate control and safety design.
* Do not expose the local web server beyond the local network by default.
* Avoid storing destination history longer than needed for the active route unless the user explicitly opts in.
* API keys for online route providers must not be shipped in logs or exposed to the phone client.

## Milestones

### M0: PRD and Shape

* Define this PRD.
* Identify existing nav/map messages and code paths.
* Decide whether the first UI is hosted by an existing local service or a small new service.

### M1: Local Destination UI

* Serve `/navigation` on the device LAN.
* Accept destination input and return structured status.
* Store active/inactive route state.

### M2: Route Memory POC

* Resolve one route and build compact route memory.
* Match ego GPS to route progress.
* Expose status and next maneuver from route memory.
* Log route progress alongside existing modalities.

### M3: Live View and Inference Hook

* Show route preview/progress on the phone page.
* Publish a compact route-context window for inference.
* Validate deterministic replay from logged route memory.

### M4: Offline MVP

* Build or import offline route/map data for a test area.
* Run destination-to-route and route-memory queries without online APIs.
* Compare online-enriched and offline-only route memory quality.

## Open Questions

* Which process should own the local web server?
* Should route memory be stored in Params, a local database, or route log artifacts?
* What is the minimum route-context schema needed by the first inference experiment?
* Which online provider is acceptable for route preparation, if any?
* How should rerouting work when offline map data cannot compute a new route?
* What UI surface should be considered required for on-device display versus phone-only display?

## Success Criteria

* A phone on the same LAN can enter a destination and see route status.
* The backend can produce a route-memory window from current ego GPS.
* Route memory is logged and replayable without online access.
* Existing no-route driving behavior is unchanged.
* Missing map metadata degrades cleanly without breaking navigation state.
