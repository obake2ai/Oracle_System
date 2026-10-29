// Organs venue player (IAMAS 2026)
// Plays the live HLS stream; when it stalls, crossfades to a local fallback loop and keeps
// retrying the live stream in the background, switching back once it is stable again.
(function () {
  "use strict";
  const cfg = Object.assign({}, window.ORGANS_PLAYER || {});
  const params = new URLSearchParams(location.search);
  if (params.get("src")) cfg.streamUrl = params.get("src");
  if (params.get("fallback")) cfg.fallbackUrl = params.get("fallback");
  const debug = params.get("debug") === "1";

  const live = document.getElementById("live");
  const fb = document.getElementById("fallback");
  const dbg = document.getElementById("debug");
  if (debug) dbg.style.display = "block";

  let hls = null;
  let showing = "none";          // "live" | "fallback" | "none"
  let lastLiveTime = -1;
  let lastProgressAt = 0;        // when live.currentTime last advanced
  let attemptStartedAt = 0;
  let stableSince = 0;           // when the current attempt started advancing continuously
  let lastError = "";
  let retryTimer = null;

  const now = () => performance.now() / 1000;

  function log(msg) {
    lastError = new Date().toLocaleTimeString() + " " + msg;
    if (debug) console.log("[organs]", msg);
  }

  function show(which) {
    if (showing === which) return;
    showing = which;
    live.classList.toggle("on", which === "live");
    fb.classList.toggle("on", which === "fallback");
    if (which === "fallback") {
      fb.play().catch(() => {});
    } else if (which === "live") {
      // keep the fallback decoded but paused to save power
      setTimeout(() => { if (showing === "live") fb.pause(); }, 2000);
    }
    log("showing " + which);
  }

  function stopLive() {
    if (hls) { try { hls.destroy(); } catch (e) {} hls = null; }
    live.removeAttribute("src");
    try { live.load(); } catch (e) {}
    lastLiveTime = -1;
    stableSince = 0;
  }

  function startLive() {
    stopLive();
    attemptStartedAt = now();
    lastProgressAt = now();
    const url = cfg.streamUrl + (cfg.streamUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
    if (window.Hls && window.Hls.isSupported()) {
      hls = new Hls({ liveSyncDurationCount: 3, liveMaxLatencyDurationCount: 8,
                      manifestLoadingMaxRetry: 2, levelLoadingMaxRetry: 4, fragLoadingMaxRetry: 4,
                      enableWorker: true, lowLatencyMode: false });
      hls.on(Hls.Events.ERROR, (_, data) => {
        if (data && data.fatal) {
          log("hls fatal: " + data.type + " / " + data.details);
          fail();
        }
      });
      hls.loadSource(url);
      hls.attachMedia(live);
      hls.on(Hls.Events.MANIFEST_PARSED, () => live.play().catch((e) => log("play(): " + e)));
    } else if (live.canPlayType("application/vnd.apple.mpegurl")) {
      live.src = url;                       // Safari native HLS
      live.play().catch((e) => log("play(): " + e));
    } else {
      log("HLS not supported in this browser");
      fail();
    }
  }

  function fail() {
    stopLive();
    show(fb.getAttribute("src") ? "fallback" : "none");
    clearTimeout(retryTimer);
    retryTimer = setTimeout(startLive, cfg.retrySeconds * 1000);
  }

  live.addEventListener("timeupdate", () => {
    const t = live.currentTime;
    if (t > lastLiveTime + 0.01) {
      if (stableSince === 0) stableSince = now();
      lastProgressAt = now();
      lastLiveTime = t;
    }
  });
  live.addEventListener("error", () => {
    if (!hls && !live.getAttribute("src")) return;   // ignore errors caused by our own teardown
    log("video error");
    fail();
  });

  // watchdog
  setInterval(() => {
    const n = now();
    const advancing = n - lastProgressAt < 2;
    if (!advancing) stableSince = 0;

    if (showing === "live") {
      if (n - lastProgressAt > cfg.stallSeconds) { log("live stalled"); fail(); }
    } else if (hls || live.getAttribute("src")) {
      // an attempt is running behind the fallback
      if (stableSince && n - stableSince >= cfg.healthySeconds) {
        clearTimeout(retryTimer);
        show("live");
      } else if (n - attemptStartedAt > cfg.stallSeconds + cfg.healthySeconds) {
        log("live attempt did not stabilise");
        fail();
      }
    }

    if (debug) {
      dbg.textContent =
        "showing   " + showing + "\n" +
        "stream    " + cfg.streamUrl + "\n" +
        "live t    " + live.currentTime.toFixed(1) + " s, progress " + (n - lastProgressAt).toFixed(1) + " s ago\n" +
        "stable    " + (stableSince ? (n - stableSince).toFixed(1) + " s" : "-") + "\n" +
        "last      " + lastError;
    }
  }, 1000);

  // daily reload as a safety net for multi-day runs
  if (cfg.dailyReloadAt) {
    let lastCheck = "";
    setInterval(() => {
      const hm = new Date().toTimeString().slice(0, 5);
      if (hm === cfg.dailyReloadAt && lastCheck !== hm) location.reload();
      lastCheck = hm;
    }, 20 * 1000);
  }

  // start: fallback first (instant picture), live crossfades in once stable
  if (cfg.fallbackUrl) {
    fb.src = cfg.fallbackUrl;
    fb.addEventListener("error", () => { log("fallback missing: " + cfg.fallbackUrl); fb.removeAttribute("src"); }, { once: true });
    show("fallback");
  }
  startLive();
})();
