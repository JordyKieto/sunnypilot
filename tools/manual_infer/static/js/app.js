async function j(path, opts = {}) {
  const r = await fetch(path, opts);
  return await r.json();
}
async function refresh() {
  const s = await j("/status");
  const payload = s.log ? JSON.parse(s.log) : {};
  const action = payload.payload?.action || {};
  document.getElementById("log").textContent = s.log || (s.running ? "running" : "idle");
  document.getElementById("curvature").textContent = action.desiredCurvature ?? "-";
  document.getElementById("accel").textContent = action.desiredAcceleration ?? "-";
  document.getElementById("should-stop").textContent = action.shouldStop ?? "-";
}
document.getElementById("start").onclick = async () => { await j("/start", {method: "POST"}); refresh(); };
document.getElementById("stop").onclick = async () => { await j("/stop", {method: "POST"}); refresh(); };
setInterval(refresh, 1000);
refresh();
