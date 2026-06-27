async function getJson(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    throw new Error(`${path} failed: ${resp.status}`);
  }
  return await resp.json();
}

function fmt(v) {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") return Number.isInteger(v) ? `${v}` : v.toFixed(4);
  return String(v);
}

function render(status) {
  $("#status").text(status.message || (status.running ? "running" : "idle"));
  $("#sample-index").text(fmt(status.sample_index));
  $("#source").text(fmt(status.source));
  $("#status").text(`${fmt(status.manual_infer_mode ? "manual" : status.message)}`);
  const pred = status.prediction || {};
  const mapped = pred.chevy_bolt_controls || {};
  const policy = pred.policy_outputs || {};
  $("#accelerator").text(fmt(mapped.accelerator_pedal));
  $("#brake").text(fmt(mapped.brake_pedal));
  $("#steering").text(fmt(mapped.steering_wheel));
  $("#raw-output").text(JSON.stringify(policy, null, 2));
  $("#infer-log").text(status.log || "");
}

async function refresh() {
  try {
    render(await getJson("/status"));
  } catch (e) {
    $("#status").text("offline");
    $("#raw-output").text(String(e));
  }
}

$("#start-btn").on("click", async () => {
  render(await getJson("/start", { method: "POST" }));
});

$("#stop-btn").on("click", async () => {
  render(await getJson("/stop", { method: "POST" }));
});

$("#step-btn").on("click", async () => {
  render(await getJson("/step", { method: "POST" }));
});

refresh();
setInterval(refresh, 1000);
