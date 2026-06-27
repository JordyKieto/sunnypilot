async function j(path, opts = {}) {
  const r = await fetch(path, opts);
  return await r.json();
}
async function refresh() {
  const s = await j("/status");
  document.getElementById("log").textContent = s.log || (s.running ? "running" : "idle");
}
document.getElementById("start").onclick = async () => { await j("/start", {method: "POST"}); refresh(); };
document.getElementById("stop").onclick = async () => { await j("/stop", {method: "POST"}); refresh(); };
setInterval(refresh, 1000);
refresh();
