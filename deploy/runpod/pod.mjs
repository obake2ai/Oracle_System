// RunPod pod helper for the Organs stream (IAMAS 2026). REST API v2 (v1 was retired on 2026-11-15).
// The key comes from the environment and is never printed.
// Run: node --env-file=<file with RUNPOD_API_KEY> deploy/runpod/pod.mjs <list|create|status|wait|stop|start|delete> [podId]
//   GPU="NVIDIA GeForce RTX 4090"  one GPU type per pod in v2 (no fallback list); retry with another type if sold out
const KEY = process.env.RUNPOD_API_KEY;
if (!KEY) { console.log("no RUNPOD_API_KEY"); process.exit(1); }
const H = { "content-type": "application/json", authorization: `Bearer ${KEY}` };
const API = "https://api.runpod.io/v2";
const [cmd, podId] = process.argv.slice(2);

const POD_SPEC = {
  name: "organs-iamas-stream",
  image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
  gpu: { id: process.env.GPU || "NVIDIA GeForce RTX 4090", count: 1 },
  cloud: "SECURE",
  disk: 30,
  mounts: { persistent: { size: 40, path: "/workspace" } },
  ports: ["22/tcp", "8000/http"], // v2 has no default ports
  startSsh: true,
  env: {},
};

async function call(method, path, body) {
  const r = await fetch(API + path, { method, headers: H, body: body ? JSON.stringify(body) : undefined });
  const t = await r.text();
  let j; try { j = JSON.parse(t); } catch { j = t; }
  return { status: r.status, body: j };
}
const brief = p => ({
  id: p.id, name: p.name, status: p.status, gpu: p.gpu?.id, cost: p.cost,
  ssh: p.ssh?.direct ? `${p.ssh.direct.host}:${p.ssh.direct.port}` : null,
  ports: p.runtime?.ports ?? null, createdAt: p.createdAt,
});
const show = (label, r, ok = 200) =>
  console.log(label + ":", r.status, JSON.stringify(r.status === ok ? brief(r.body) : r.body).slice(0, 800));

if (cmd === "list") {
  const r = await call("GET", "/pods");
  console.log("pods:", r.status, JSON.stringify(r.body?.pods ? r.body.pods.map(brief) : r.body).slice(0, 2000));
} else if (cmd === "create") {
  show("create", await call("POST", "/pods", POD_SPEC), 201);
} else if (cmd === "status") {
  show("status", await call("GET", `/pods/${podId}`));
} else if (cmd === "wait") {
  // Poll until RUNNING with a direct SSH mapping (max ~8 min). Prints "SSH <host> <port>".
  for (let i = 0; i < 48; i++) {
    const r = await call("GET", `/pods/${podId}`);
    const p = r.body ?? {};
    console.log(new Date().toISOString().slice(11, 19), r.status, p.status ?? "-", p.ssh?.direct?.host ?? "-", p.ssh?.direct?.port ?? "-");
    if (p.status === "RUNNING" && p.ssh?.direct) { console.log("SSH", p.ssh.direct.host, p.ssh.direct.port); process.exit(0); }
    if (p.status === "ERROR" || p.status === "TERMINATED") process.exit(3);
    await new Promise(res => setTimeout(res, 10000));
  }
  console.log("TIMEOUT"); process.exit(2);
} else if (cmd === "stop" || cmd === "start") {
  show(cmd, await call("POST", `/pods/${podId}/action`, { action: cmd }));
} else if (cmd === "delete") {
  const r = await call("DELETE", `/pods/${podId}`);
  console.log("delete:", r.status, r.status === 204 ? "terminated" : JSON.stringify(r.body).slice(0, 300));
} else {
  console.log("usage: list | create | status <id> | wait <id> | stop <id> | start <id> | delete <id>");
}
