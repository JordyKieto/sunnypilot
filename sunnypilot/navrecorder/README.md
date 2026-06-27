# Navigation Recorder

## Local Dev

Run the self-contained mock server to test the UI without a comma device:

```bash
python3 -m sunnypilot.navrecorder.mock_server
```

Open:

```text
http://localhost:5050
```

This mode keeps the nav UI and route-planning flows working locally.
If you enter a real Google Maps API key, the UI will save it through the local server
and use it for map loading and route requests.
