// Pure display helpers. No DOM access, so `node --test` can run them.
// Ported from the Qt UI's formatting.py; strings match it exactly.

// One plain-language label per diagnosis category (insight/taxonomy.py).
export const CATEGORY_LABELS = {
  cpu_bound_single_process: "One process is using most of your CPU",
  cpu_bound_multi_process_contention: "Several processes together are straining your CPU",
  memory_pressure_gradual: "Memory usage is climbing steadily",
  memory_pressure_sudden: "Memory usage just spiked sharply",
  disk_io_bottleneck: "Disk activity is becoming a bottleneck",
  network_saturation: "Network usage is approaching capacity",
  cross_resource_contention: "Multiple resources are under pressure at once",
  unusual_pattern_anomaly_flagged: "Unusual system behavior detected",
};

// Unknown categories degrade to a readable sentence, not a raw enum string.
export function humanizeCategory(category) {
  if (Object.hasOwn(CATEGORY_LABELS, category)) return CATEGORY_LABELS[category];
  const text = String(category).replaceAll("_", " ").trim().toLowerCase();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function humanizeBytes(numBytes) {
  if (numBytes == null) return "--";
  let value = Number(numBytes);
  for (const unit of ["B", "KB", "MB", "GB", "TB"]) {
    if (Math.abs(value) < 1024) return `${value.toFixed(1)} ${unit}`;
    value /= 1024;
  }
  return `${value.toFixed(1)} PB`;
}

export function formatPercent(value, decimals = 1) {
  if (value == null) return "--";
  return `${Number(value).toFixed(decimals)}%`;
}

// Confidence is stored as a [0, 1] fraction.
export function formatConfidence(value) {
  if (value == null) return "--";
  return `${(value * 100).toFixed(0)}%`;
}

export function formatProbability(value) {
  if (value == null) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatTimestamp(epochSeconds) {
  if (epochSeconds == null) return "--";
  const d = new Date(epochSeconds * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

// 135 -> "2m15s", 45 -> "45s".
export function formatDurationShort(seconds) {
  const total = Math.max(0, Math.trunc(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return minutes ? `${minutes}m${String(secs).padStart(2, "0")}s` : `${secs}s`;
}

const READINESS_LABELS = {
  no_data: "No data yet",
  needs_training: "No model trained",
  ready: "Live",
  replay: "REPLAY",
};

// One-line status text for a GET /readiness state; `detail` is the fallback
// for any state this table doesn't know.
export function readinessBannerText(state, detail, historySeconds, historySecondsRequired) {
  if (state === "collecting") {
    return `Collecting - ${formatDurationShort(historySeconds)} of ${formatDurationShort(historySecondsRequired)}`;
  }
  return READINESS_LABELS[state] ?? detail;
}

// Conservative on purpose: nothing is painted "high" unless it really is.
export function confidenceTier(value) {
  if (value == null) return "unknown";
  if (value >= 0.75) return "high";
  if (value >= 0.5) return "medium";
  return "low";
}

// 30 -> "in 30 s", 120 -> "in 2 min", 90 -> "in 1.5 min".
export function formatHorizon(seconds) {
  if (seconds == null) return "--";
  if (seconds < 90) return `in ${Math.round(seconds)} s`;
  const minutes = seconds / 60;
  return `in ${Number.isInteger(minutes) ? minutes : minutes.toFixed(1)} min`;
}

const METRIC_LABELS = {
  cpu_percent: "CPU",
  mem_percent: "Memory",
  swap_percent: "Swap",
  load_avg_1: "Load (1 min)",
};

export function metricLabel(name) {
  return METRIC_LABELS[name] ?? humanizeCategory(name);
}

// [30, 60] -> "in 30–60 s", [120, 300] -> "in 2–5 min".
export function formatHorizonRange(range) {
  if (!Array.isArray(range) || range.length !== 2) return "--";
  const [lo, hi] = range;
  if (hi < 90) return `in ${Math.round(lo)}–${Math.round(hi)} s`;
  const min = (s) => {
    const m = s / 60;
    return Number.isInteger(m) ? String(m) : m.toFixed(1);
  };
  return `in ${min(lo)}–${min(hi)} min`;
}

export function isPercentMetric(name) {
  return String(name).endsWith("_percent");
}

// A metric value in its own unit: percentages get "%", byte counters
// read as sizes, everything else is a plain number.
export function formatMetricValue(name, value, decimals = 1) {
  if (value == null) return "--";
  if (isPercentMetric(name)) return formatPercent(value, decimals);
  if (String(name).endsWith("_bytes")) return humanizeBytes(value);
  return formatNumber(value, decimals);
}

export function formatNumber(value, decimals = 2) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(decimals);
}

// "2026-10-10 14:05", local time.
export function formatDateTime(epochSeconds) {
  if (epochSeconds == null) return "--";
  const d = new Date(epochSeconds * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Contention contributor names: "current_mem_percent" -> "Memory now",
// "forecast_cpu_percent" -> "CPU forecast", "anomaly_score" -> "Anomaly score".
export function contributorLabel(name) {
  const m = /^(current|forecast)_(.+)$/.exec(String(name));
  if (!m) return metricLabel(name);
  return `${metricLabel(m[2])} ${m[1] === "current" ? "now" : "forecast"}`;
}
