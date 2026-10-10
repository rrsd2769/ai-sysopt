// App shell: hash router, status bar, and the shared live connection.

import { LiveSocket, getJSON, postJSON } from "./api.js";
import { readinessBannerText } from "./format.js";
import * as live from "./views/live.js";

const READINESS_POLL_MS = 3000;

const CONN_LABELS = {
  connecting: "Connecting",
  open: "Connected",
  reconnecting: "Reconnecting…",
  down: "Server unreachable",
};

function placeholder(title) {
  return {
    mount(root) {
      root.innerHTML = `<div class="card"><h2>${title}</h2><p class="placeholder">This view is not built yet.</p></div>`;
      return () => {};
    },
  };
}

const VIEWS = {
  live,
  accuracy: placeholder("Accuracy"),
  reports: placeholder("Reports"),
  model: placeholder("Model"),
};

// A tiny event bus the views subscribe to: "state" (each live snapshot)
// and "open" (the socket reconnected after a drop).
const app = {
  lastState: null,
  listeners: new Map(),
  on(event, fn) {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event).add(fn);
    return () => this.listeners.get(event).delete(fn);
  },
  emit(event, payload) {
    for (const fn of this.listeners.get(event) ?? []) {
      try {
        fn(payload);
      } catch (err) {
        console.error(`${event} handler failed`, err);
      }
    }
  },
};

// --- Router -------------------------------------------------------------------

const viewEl = document.getElementById("view");
let unmountView = null;

function route() {
  const name = location.hash.replace(/^#\/?/, "") || "live";
  const key = Object.hasOwn(VIEWS, name) ? name : "live";
  unmountView?.();
  viewEl.innerHTML = "";
  unmountView = VIEWS[key].mount(viewEl, app);
  for (const link of document.querySelectorAll(".sidebar a")) {
    link.classList.toggle("active", link.dataset.route === key);
  }
}

// --- Status bar ---------------------------------------------------------------

const connEl = document.getElementById("conn");
const readinessEl = document.getElementById("readiness");

let wasOpen = false;

function setConnection(state) {
  connEl.dataset.state = state;
  connEl.querySelector(".conn-label").textContent = CONN_LABELS[state];
  if (state === "open") {
    // Views already loaded history on mount; only a reconnect needs a backfill.
    if (wasOpen) app.emit("open");
    wasOpen = true;
    refreshStatus();
  }
}

function renderReadiness(r) {
  readinessEl.dataset.state = r.state;
  readinessEl.textContent =
    r.state === "replay"
      ? "REPLAY: recorded session"
      : readinessBannerText(r.state, r.detail, r.history_seconds, r.history_seconds_required);
  readinessEl.title = r.detail;
}

const toggles = [
  { el: document.getElementById("toggle-monitoring"), base: "/monitoring" },
  { el: document.getElementById("toggle-live"), base: "/live" },
];

async function refreshStatus() {
  try {
    renderReadiness(await getJSON("/readiness"));
  } catch {
    readinessEl.dataset.state = "error";
    readinessEl.textContent = "Status unavailable";
    readinessEl.title = "";
  }
  await Promise.all(
    toggles.map(async (t) => {
      if (t.busy) return;
      try {
        const { running } = await getJSON(`${t.base}/status`);
        t.el.setAttribute("aria-pressed", String(running));
        t.el.disabled = false;
      } catch {
        t.el.disabled = true;
      }
    }),
  );
}

for (const t of toggles) {
  t.el.addEventListener("click", async () => {
    const running = t.el.getAttribute("aria-pressed") === "true";
    t.busy = true;
    t.el.disabled = true;
    try {
      const res = await postJSON(`${t.base}/${running ? "stop" : "start"}`);
      t.el.setAttribute("aria-pressed", String(res.running));
    } catch (err) {
      console.error(err);
    } finally {
      t.busy = false;
      t.el.disabled = false;
      refreshStatus();
    }
  });
}

// --- Start ----------------------------------------------------------------------

const socket = new LiveSocket({
  onMessage(state) {
    app.lastState = state;
    app.emit("state", state);
  },
  onStatus: setConnection,
});

window.addEventListener("hashchange", route);
route();
socket.start();
refreshStatus();
setInterval(refreshStatus, READINESS_POLL_MS);
