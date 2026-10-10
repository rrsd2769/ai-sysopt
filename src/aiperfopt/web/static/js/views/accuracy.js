// Accuracy view: what the model predicted against what actually happened,
// for one metric and horizon over a lookback window.

import { getJSON } from "../api.js";
import { Chart, alpha, axes } from "../charts.js";
import {
  escapeHtml,
  formatHorizonRange,
  formatMetricValue,
  formatNumber,
  formatTimestamp,
  isPercentMetric,
  metricLabel,
} from "../format.js";
import { accuracyStats } from "../stats.js";
import { segmented } from "../ui.js";

const LOOKBACKS = [
  { value: 15 * 60, label: "15 min" },
  { value: 3600, label: "1 h" },
  { value: 6 * 3600, label: "6 h" },
  { value: 24 * 3600, label: "24 h" },
];
const HORIZONS = ["short", "medium"];
const AUTO_REFRESH_MS = 30_000;
// Forecasts arrive every few seconds; a longer silence is a gap
// (monitoring was off) and the lines break there instead of bridging it.
const GAP_SECONDS = 60;

// Kept across visits to the view within one page load.
const selection = { metric: "cpu_percent", horizon: "short", lookback: 15 * 60 };

const TEMPLATE = `
  <section class="card controls">
    <label class="field"><span>Metric</span><select data-ref="metric"></select></label>
    <div class="field"><span>Horizon</span><div data-ref="horizon"></div></div>
    <div class="field"><span>Lookback</span><div data-ref="lookback"></div></div>
    <span class="spacer"></span>
    <span class="muted small" data-ref="status"></span>
    <button type="button" class="btn" data-ref="refresh">Refresh</button>
  </section>
  <section class="card chart-card">
    <div class="chart-head">
      <span class="chart-title" data-ref="title"></span>
      <span class="readout" data-ref="readout"></span>
      <span class="chart-legend"><span><i class="swatch"></i>Actual</span><span><i class="swatch dashed"></i>Predicted</span></span>
    </div>
    <div class="chart" data-ref="chart"></div>
    <div class="chart-empty" data-ref="empty" hidden></div>
  </section>
  <section class="stats" data-ref="stats"></section>
`;

function withGaps(rows) {
  const xs = [];
  const actual = [];
  const predicted = [];
  let prev = null;
  for (const r of rows) {
    if (prev != null && r.target_timestamp - prev > GAP_SECONDS) {
      xs.push(prev + 1);
      actual.push(null);
      predicted.push(null);
    }
    xs.push(r.target_timestamp);
    actual.push(r.actual_value);
    predicted.push(r.predicted_value);
    prev = r.target_timestamp;
  }
  return [xs, actual, predicted];
}

function statCard(label, value, caption = "") {
  return `
    <div class="card stat">
      <span class="tile-label">${escapeHtml(label)}</span>
      <span class="tile-value">${escapeHtml(value)}</span>
      <span class="tile-caption">${escapeHtml(caption)}</span>
    </div>`;
}

export function mount(root) {
  root.innerHTML = TEMPLATE;
  const $ = (ref) => root.querySelector(`[data-ref="${ref}"]`);

  let rows = [];
  let loading = false;
  let horizonRanges = {};
  let seq = 0;

  const metricSel = $("metric");
  metricSel.innerHTML = `<option value="${selection.metric}">${escapeHtml(metricLabel(selection.metric))}</option>`;

  const horizonCtl = segmented($("horizon"), HORIZONS.map((h) => ({ value: h, label: h[0].toUpperCase() + h.slice(1) })), selection.horizon, (v) => {
    selection.horizon = v;
    load();
  });
  const lookbackCtl = segmented($("lookback"), LOOKBACKS, selection.lookback, (v) => {
    selection.lookback = v;
    load();
  });
  metricSel.addEventListener("change", () => {
    selection.metric = metricSel.value;
    load();
  });
  $("refresh").addEventListener("click", () => load());

  // --- Chart -----------------------------------------------------------------

  const chart = new Chart($("chart"), (t) => {
    const color = t[selection.metric === "mem_percent" ? "mem" : "cpu"];
    const percent = isPercentMetric(selection.metric);
    return {
      scales: {
        x: { time: true },
        y: {
          range: (u, min, max) => {
            if (min == null) return [0, 1];
            const pad = Math.max((max - min) * 0.1, percent ? 1 : Math.abs(max) * 0.05 || 1);
            const lo = min - pad;
            const hi = max + pad;
            return percent ? [Math.max(0, lo), Math.min(100, hi)] : [lo, hi];
          },
        },
      },
      axes: axes(t, { ySplits: null, yFormat: (v) => formatMetricValue(selection.metric, v, 0) }),
      legend: { show: false },
      cursor: { points: { show: false }, y: false, drag: { x: true, y: false } },
      series: [
        {},
        { stroke: color, width: 2, points: { show: false } },
        { stroke: alpha(color, 0.9), width: 1.5, dash: [6, 4], points: { show: false } },
      ],
      hooks: { setCursor: [(u) => renderReadout(u.cursor.idx, u.data)] },
    };
  }, 260);

  function renderReadout(idx, data) {
    const el = $("readout");
    if (idx == null || data?.[0]?.[idx] == null || data[2][idx] == null) {
      el.textContent = "";
      return;
    }
    const m = selection.metric;
    const actual = data[1][idx];
    const predicted = data[2][idx];
    const parts = [formatTimestamp(data[0][idx]), `predicted ${formatMetricValue(m, predicted)}`];
    parts.push(actual == null ? "actual not observed" : `actual ${formatMetricValue(m, actual)}`);
    el.textContent = parts.join(" · ");
  }

  // --- Render ----------------------------------------------------------------

  function horizonText(h) {
    const range = horizonRanges[h];
    return range ? formatHorizonRange(range) : h;
  }

  function emptyText(error, predicting) {
    if (error) return `Couldn't load predictions: ${error}`;
    if (predicting) return "Predictions are running, but none were persisted for this metric and horizon in this window.";
    return "No forecasts persisted in this window, start Predictions.";
  }

  function render(error, predicting = false) {
    const m = selection.metric;
    $("title").textContent = `${metricLabel(m)} · forecast ${horizonText(selection.horizon)}`;
    const empty = $("empty");
    const chartEl = $("chart");
    const s = accuracyStats(rows);
    if (error || !rows.length) {
      empty.textContent = emptyText(error, predicting);
      empty.hidden = false;
      chartEl.hidden = true;
    } else {
      empty.hidden = true;
      chartEl.hidden = false;
      chart.setData(withGaps(rows));
    }
    const unit = (v) => (v == null ? "--" : formatMetricValue(m, v, 2));
    const bias = s.bias == null ? "" : s.bias > 0 ? "predictions run high" : s.bias < 0 ? "predictions run low" : "";
    $("stats").innerHTML = [
      statCard("Mean absolute error", unit(s.mae), "average miss, either direction"),
      statCard("RMSE", unit(s.rmse), "penalises large misses more"),
      statCard("Bias", s.bias == null ? "--" : `${s.bias > 0 ? "+" : ""}${unit(s.bias)}`, bias),
      statCard(
        "Points compared",
        `${s.compared.toLocaleString()}`,
        s.total ? `of ${s.total.toLocaleString()} forecasts · ${s.pending.toLocaleString()} not yet observed` : "",
      ),
    ].join("");
  }

  async function load() {
    const mine = ++seq;
    loading = true;
    $("refresh").disabled = true;
    $("status").textContent = "Loading…";
    const end = Date.now() / 1000;
    let error = null;
    try {
      const data = await getJSON("/reporting/predicted-vs-actual", {
        metric_name: selection.metric,
        horizon_label: selection.horizon,
        start: end - selection.lookback,
        end,
      });
      if (mine !== seq) return;
      rows = data.sort((a, b) => a.target_timestamp - b.target_timestamp);
    } catch (err) {
      if (mine !== seq) return;
      rows = [];
      error = err.message;
    }
    loading = false;
    $("refresh").disabled = false;
    $("status").textContent = error ? "" : `Updated ${formatTimestamp(Date.now() / 1000)}`;
    let predicting = false;
    if (!error && !rows.length) {
      predicting = await getJSON("/live/status").then((r) => r.running, () => false);
      if (mine !== seq) return;
    }
    chart.reset();
    render(error, predicting);
  }

  async function loadOptions() {
    const [metrics, config] = await Promise.allSettled([getJSON("/reporting/metrics"), getJSON("/config")]);
    if (metrics.status === "fulfilled") {
      // Percent metrics first: those are the ones the model forecasts by default.
      const names = [...metrics.value].sort((a, b) => isPercentMetric(b) - isPercentMetric(a) || a.localeCompare(b));
      metricSel.innerHTML = names
        .map((n) => `<option value="${escapeHtml(n)}"${n === selection.metric ? " selected" : ""}>${escapeHtml(metricLabel(n))}</option>`)
        .join("");
    }
    if (config.status === "fulfilled") {
      const f = config.value.forecasting ?? {};
      horizonRanges = { short: f.short_horizon_seconds, medium: f.medium_horizon_seconds };
      const ctl = $("horizon");
      for (const b of ctl.querySelectorAll("button")) {
        const h = b.dataset.value;
        b.textContent = `${h[0].toUpperCase() + h.slice(1)} · ${horizonText(h)}`;
      }
      $("title").textContent = `${metricLabel(selection.metric)} · forecast ${horizonText(selection.horizon)}`;
    }
  }

  loadOptions();
  load();
  const timer = setInterval(() => {
    if (!loading && selection.lookback <= 3600 && document.visibilityState === "visible") load();
  }, AUTO_REFRESH_MS);

  return () => {
    seq += 1;
    clearInterval(timer);
    chart.destroy();
    horizonCtl.disable(true);
    lookbackCtl.disable(true);
  };
}
