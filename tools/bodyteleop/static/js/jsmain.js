async function getJson(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    throw new Error(`${path} failed: ${resp.status}`);
  }
  return await resp.json();
}

async function uploadModel(file) {
  const form = new FormData();
  form.append("model", file);
  return await getJson("/model", { method: "POST", body: form });
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
  $("#status").text(status.manual_infer_mode ? "manual" : (status.message || "idle"));
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

$("#model-input").on("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  $("#infer-log").text(`Uploading ${file.name}...`);
  try {
    const result = await uploadModel(file);
    $("#infer-log").text(`Uploaded ${result.path} (${result.bytes} bytes)`);
    await refresh();
  } catch (err) {
    $("#infer-log").text(String(err));
  }
});

refresh();
setInterval(refresh, 1000);
