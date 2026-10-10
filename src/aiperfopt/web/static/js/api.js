// HTTP and WebSocket access to the PRISM API (same origin).

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(method, path, { params, body } = {}) {
  const url = new URL(path, location.origin);
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value != null) url.searchParams.set(key, value);
  }
  const init = { method, headers: {} };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(url, init);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data?.detail) message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch { /* body wasn't JSON */ }
    throw new ApiError(res.status, message);
  }
  return res.json();
}

export const getJSON = (path, params) => request("GET", path, { params });
export const postJSON = (path, body) => request("POST", path, { body });

// The /ws/live stream. Reconnects with capped exponential backoff and
// reports its state: "connecting" | "open" | "reconnecting" | "down".
// "down" means several attempts in a row failed; it keeps retrying anyway.
export class LiveSocket {
  constructor({ onMessage, onStatus }) {
    this.onMessage = onMessage;
    this.onStatus = onStatus;
    this.failures = 0;
    this.ws = null;
    this.timer = null;
    this.closed = false;
  }

  start() {
    this.closed = false;
    this.#connect();
  }

  stop() {
    this.closed = true;
    clearTimeout(this.timer);
    this.ws?.close();
  }

  #connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    this.onStatus(this.failures === 0 ? "connecting" : this.failures >= 5 ? "down" : "reconnecting");
    const ws = new WebSocket(`${scheme}://${location.host}/ws/live`);
    this.ws = ws;
    ws.onopen = () => {
      this.failures = 0;
      this.onStatus("open");
    };
    ws.onmessage = (event) => {
      try {
        this.onMessage(JSON.parse(event.data));
      } catch (err) {
        console.error("Bad live message", err);
      }
    };
    ws.onclose = () => {
      if (this.closed) return;
      this.failures += 1;
      this.onStatus(this.failures >= 5 ? "down" : "reconnecting");
      const delay = Math.min(10_000, 500 * 2 ** (this.failures - 1));
      this.timer = setTimeout(() => this.#connect(), delay);
    };
  }
}
