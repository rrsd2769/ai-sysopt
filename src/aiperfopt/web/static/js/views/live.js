// Live view: tiles, CPU and Memory charts with forecasts, the contention
// strip, and what the system thinks is happening right now.

import { getJSON } from "../api.js";
import { Chart, alpha, axes, nowMarker, sparkline, theme } from "../charts.js";
import {
  confidenceTier,
  escapeHtml,
  formatConfidence,
  formatHorizon,
  formatPercent,
  formatProbability,
  formatTimestamp,
  humanizeCategory,
} from "../format.js";

const WINDOW_SECONDS = 15 * 60;
const MIN_SPAN_SECONDS = 5 * 60;
const SPARK_POINTS = 60;
// `confidence` is an uncalibrated 0-1 score, not an interval. The band
// widens as confidence drops, up to this many points either side; it is a
// display hint, not a statistical bound.
const MAX_BAND_HALF_WIDTH = 15;
const SYNC_KEY = "prism-live";

const METRICS = [
  { key: "cpu_percent", label: "CPU", color: "cpu" },
  { key: "mem_percent", label: "Memory", color: "mem" },
];

const TEMPLATE = `
  <div class="banner danger" data-ref="error" hidden>
    <p></p><button type="button" class="close" aria-label="Dismiss">×</button>
  </div>
  <div class="banner warn" data-ref="anomaly" hidden>
    <p>Unusual system behavior detected. Prediction confidence has been reduced because the
    current workload differs from what's normally seen.</p>
  </div>
  <section class="tiles">
    ${tile("cpu", "CPU")}
    ${tile("mem", "Memory")}
    ${tile("swap", "Swap")}
    ${tile("contention", "Contention risk")}
  </section>
  ${METRICS.map((m) => chartCard(m.key, m.label, true)).join("")}
  ${chartCard("contention", "Contention risk", false)}
  <section class="insight">
    <div class="card">
      <h2>What's happening</h2>
      <div data-ref="diagnosis"></div>
      <div class="actions" data-ref="actions"></div>
    </div>
    <div class="card">
      <h2>Top contributing processes</h2>
      <div data-ref="processes"><p class="muted">No contention reading yet.</p></div>
    </div>
  </section>
`;

function tile(key, label) {
  return `
    <div class="card tile" data-tile="${key}">
      <span class="tile-label">${label}</span>
      <span class="tile-value">--</span>
      <span class="tile-caption"></span>
      <svg class="spark" aria-hidden="true"></svg>
    </div>`;
}

function chartCard(key, label, forecast) {
  const legend = forecast
    ? `<span><i class="swatch"></i>Actual</span><span><i class="swatch dashed"></i>Forecast</span><span><i class="swatch band"></i>Confidence</span>`
    : `<span><i class="swatch"></i>Probability</span><span><i class="swatch dashed"></i>Threshold</span>`;
  return `
    <section class="card chart-card" data-chart="${key}">
      <div class="chart-head">
        <span class="chart-title">${label}</span>
        <span class="readout"></span>
        <span class="chart-legend" data-legend="${key}">${legend}</span>
      </div>
      <div class="chart"></div>
      <div class="chart-empty" hidden>Waiting for data…</div>
    </section>`;
}

// Concatenate, sort by `t`, drop duplicates, keep the trailing window.
function merge(existing, incoming) {
  const all = existing.concat(incoming).sort((a, b) => a.t - b.t);
  const out = [];
  for (const item of all) {
    if (out.length && out[out.length - 1].t === item.t) out[out.length - 1] = item;
    else out.push(item);
  }
  const newest = out.length ? out[out.length - 1].t : 0;
  return out.filter((item) => item.t >= newest - WINDOW_SECONDS);
}

const sampleOf = (m) => ({ t: m.timestamp, cpu_percent: m.cpu_percent, mem_percent: m.mem_percent, swap_percent: m.swap_percent });
const contentionOf = (c) => ({ t: c.generated_at, p: c.probability, threshold: c.threshold_used, triggered: c.triggered });

// Only the most recent forecast cycle is drawn.
function latestCycle(forecasts) {
  if (!forecasts?.length) return [];
  const newest = Math.max(...forecasts.map((f) => f.generated_at));
  return forecasts.filter((f) => f.generated_at === newest);
}

export function mount(root, app) {
  root.innerHTML = TEMPLATE;
  const $ = (sel) => root.querySelector(sel);

  const buf = { samples: [], contention: [], forecasts: [], range: null };
  let dismissedError = null;
  let lastError = null;

  // --- Charts --------------------------------------------------------------

  const xRange = (u, min, max) => buf.range ?? [min, max];
  const now = () => (buf.samples.length ? buf.samples[buf.samples.length - 1].t : null);
  const cursor = { sync: { key: SYNC_KEY }, points: { show: false }, y: false, drag: { x: false, y: false, setScale: false } };

  const metricCharts = METRICS.map((metric) => {
    const card = $(`[data-chart="${metric.key}"]`);
    const readout = card.querySelector(".readout");
    const chart = new Chart(card.querySelector(".chart"), (t) => {
      const color = t[metric.color];
      return {
        scales: { x: { time: true, range: xRange }, y: { range: [0, 100] } },
        axes: axes(t, { xLabels: false }),
        legend: { show: false },
        cursor,
        series: [
          {},
          { stroke: color, width: 2, points: { show: false } },
          { stroke: color, width: 2, dash: [6, 4], points: { show: false } },
          { stroke: "transparent", points: { show: false } },
          { stroke: "transparent", points: { show: false } },
        ],
        bands: [{ series: [3, 4], fill: alpha(color, 0.18) }],
        hooks: {
          draw: [nowMarker(now)],
          setCursor: [(u) => renderReadout(metric, readout, u.cursor.idx, u.data)],
        },
      };
    }, 170);
    card.querySelector(".chart-legend").style.color = `var(--series-${metric.color})`;
    return { metric, card, readout, chart };
  });

  const contentionCard = $('[data-chart="contention"]');
  const contentionReadout = contentionCard.querySelector(".readout");
  const shadeTriggered = (u) => {
    const points = buf.contention;
    if (!points.length) return;
    const ctx = u.ctx;
    const { top, height } = u.bbox;
    ctx.save();
    ctx.fillStyle = alpha(theme().danger, 0.16);
    points.forEach((p, i) => {
      if (!p.triggered) return;
      const next = points[i + 1]?.t ?? p.t + 5;
      const x0 = u.valToPos(p.t, "x", true);
      const x1 = u.valToPos(next, "x", true);
      ctx.fillRect(x0, top, Math.max(1, x1 - x0), height);
    });
    ctx.restore();
  };
  const contentionChart = new Chart(contentionCard.querySelector(".chart"), (t) => ({
    scales: { x: { time: true, range: xRange }, y: { range: [0, 100] } },
    axes: axes(t, { ySplits: [0, 50, 100] }),
    legend: { show: false },
    cursor,
    series: [
      {},
      { stroke: t.contention, width: 2, points: { show: false } },
      { stroke: t.danger, width: 1, dash: [4, 4], points: { show: false } },
    ],
    hooks: {
      drawAxes: [shadeTriggered],
      draw: [nowMarker(now)],
      setCursor: [(u) => renderContentionReadout(u.cursor.idx, u.data)],
    },
  }), 130);
  contentionCard.querySelector(".chart-legend").style.color = "var(--series-contention)";

  // --- Data -> charts ------------------------------------------------------

  function forecastsFor(metricKey) {
    return buf.forecasts
      .filter((f) => f.metric_name === metricKey)
      .sort((a, b) => a.target_timestamp - b.target_timestamp);
  }

  function metricData(metricKey) {
    const xs = buf.samples.map((s) => s.t);
    const actual = buf.samples.map((s) => s[metricKey] ?? null);
    const n = xs.length;
    const forecast = new Array(n).fill(null);
    const upper = new Array(n).fill(null);
    const lower = new Array(n).fill(null);
    const anchor = actual[n - 1];
    const ahead = forecastsFor(metricKey).filter((f) => n && f.target_timestamp > xs[n - 1]);
    if (n && anchor != null && ahead.length) {
      forecast[n - 1] = upper[n - 1] = lower[n - 1] = anchor;
      for (const f of ahead) {
        const half = (1 - (f.confidence ?? 0.5)) * MAX_BAND_HALF_WIDTH;
        xs.push(f.target_timestamp);
        actual.push(null);
        forecast.push(f.predicted_value);
        upper.push(Math.min(100, f.predicted_value + half));
        lower.push(Math.max(0, f.predicted_value - half));
      }
    }
    return [xs, actual, forecast, upper, lower];
  }

  function sharedRange() {
    if (!buf.samples.length) return null;
    const first = buf.samples[0].t;
    let max = buf.samples[buf.samples.length - 1].t;
    for (const f of buf.forecasts) max = Math.max(max, f.target_timestamp);
    return [Math.min(first, max - MIN_SPAN_SECONDS), max];
  }

  function renderReadout(metric, el, idx, data) {
    if (idx != null && data?.[0]?.[idx] != null) {
      const actual = data[1][idx];
      const forecast = data[2][idx];
      const value = actual ?? forecast;
      const kind = actual != null ? "" : " forecast";
      el.textContent = `${formatTimestamp(data[0][idx])} · ${formatPercent(value)}${kind}`;
      return;
    }
    const latest = buf.samples[buf.samples.length - 1]?.[metric.key];
    const parts = [`now ${formatPercent(latest)}`];
    for (const f of forecastsFor(metric.key)) {
      parts.push(`${formatHorizon(f.horizon_seconds)} → ${formatPercent(f.predicted_value)}`);
    }
    el.textContent = latest == null ? "" : parts.join(" · ");
  }

  function renderContentionReadout(idx, data) {
    if (idx != null && data?.[1]?.[idx] != null) {
      contentionReadout.textContent = `${formatTimestamp(data[0][idx])} · ${formatPercent(data[1][idx])}`;
      return;
    }
    const last = buf.contention[buf.contention.length - 1];
    contentionReadout.textContent = last
      ? `${formatProbability(last.p)} · threshold ${formatProbability(last.threshold)}${last.triggered ? " · triggered" : ""}`
      : "";
  }

  function renderCharts() {
    buf.range = sharedRange();
    for (const { metric, card, readout, chart } of metricCharts) {
      const empty = !buf.samples.length;
      card.querySelector(".chart-empty").hidden = !empty;
      card.querySelector(".chart").hidden = empty;
      if (!empty) chart.setData(metricData(metric.key));
      renderReadout(metric, readout, null, null);
    }
    const c = buf.contention.filter((p) => !buf.range || p.t >= buf.range[0]);
    const emptyC = !c.length;
    contentionCard.querySelector(".chart-empty").hidden = !emptyC;
    contentionCard.querySelector(".chart").hidden = emptyC;
    contentionCard.querySelector(".chart-empty").textContent = "No contention predictions in the last 15 minutes.";
    if (!emptyC) {
      contentionChart.setData([c.map((p) => p.t), c.map((p) => p.p * 100), c.map((p) => p.threshold * 100)]);
    }
    renderContentionReadout(null, null);
  }

  // --- Tiles ---------------------------------------------------------------

  function renderTiles() {
    const t = theme();
    const recent = buf.samples.slice(-SPARK_POINTS);
    const last = recent[recent.length - 1];
    for (const [key, field, color] of [["cpu", "cpu_percent", t.cpu], ["mem", "mem_percent", t.mem], ["swap", "swap_percent", t.swap]]) {
      const el = $(`[data-tile="${key}"]`);
      el.querySelector(".tile-value").textContent = formatPercent(last?.[field]);
      sparkline(el.querySelector(".spark"), recent.map((s) => s[field]), color);
    }
    const el = $('[data-tile="contention"]');
    const recentC = buf.contention.slice(-SPARK_POINTS);
    const lastC = recentC[recentC.length - 1];
    el.classList.toggle("triggered", Boolean(lastC?.triggered));
    el.querySelector(".tile-value").textContent = formatProbability(lastC?.p);
    el.querySelector(".tile-caption").textContent = lastC
      ? `threshold ${formatProbability(lastC.threshold)}${lastC.triggered ? " · triggered" : ""}`
      : "";
    sparkline(el.querySelector(".spark"), recentC.map((p) => p.p * 100), lastC?.triggered ? t.danger : t.contention);
  }

  // --- Insight -------------------------------------------------------------

  function renderInsight(state) {
    const diagnosis = state.diagnosis;
    const diagEl = $('[data-ref="diagnosis"]');
    if (diagnosis) {
      const tier = confidenceTier(diagnosis.confidence);
      diagEl.innerHTML = `
        <p class="diag-title">${escapeHtml(humanizeCategory(diagnosis.category))}</p>
        <p class="diag-body">${escapeHtml(diagnosis.plain_language_explanation)}</p>
        <span class="chip" data-tier="${tier}">Confidence ${formatConfidence(diagnosis.confidence)}</span>`;
    } else {
      diagEl.innerHTML = `<p class="all-clear">All clear. Nothing needs your attention right now.</p>`;
    }

    const actions = (state.recommendation?.actions ?? []).slice(0, 3);
    $('[data-ref="actions"]').innerHTML = actions
      .map((a) => `
        <div class="action">
          <div class="action-title">${escapeHtml(a.title)}</div>
          <div class="action-desc">${escapeHtml(a.description)}</div>
        </div>`)
      .join("");

    const contention = state.contention;
    const procEl = $('[data-ref="processes"]');
    const processes = contention?.contributing_processes ?? [];
    if (!contention) {
      procEl.innerHTML = `<p class="muted">No contention reading yet.</p>`;
    } else if (!processes.length) {
      procEl.innerHTML = `<p class="muted">No single process stands out.</p>`;
    } else {
      const dominant = (contention.contributing_metrics ?? []).find((m) => m.metric !== "anomaly_score")?.metric ?? "";
      const by = /mem|swap/.test(dominant) ? "memory" : "CPU";
      procEl.innerHTML = `
        <ul class="proc-list">
          ${processes.map((p) => `
            <li>
              <span class="proc-name">${escapeHtml(p.name)}<span class="proc-pid">pid ${p.pid}</span></span>
              <span class="num">${formatPercent(p.value)}</span>
            </li>`).join("")}
        </ul>
        <p class="muted" style="margin:8px 0 0;font-size:12px">Ranked by ${by} use.</p>`;
    }

    $('[data-ref="anomaly"]').hidden = !state.anomaly_notification;
    lastError = state.error ?? null;
    renderError();
  }

  function renderError() {
    const el = $('[data-ref="error"]');
    const show = lastError && lastError !== dismissedError;
    el.hidden = !show;
    if (show) el.querySelector("p").textContent = `Something went wrong: ${lastError}`;
  }

  $('[data-ref="error"] .close').addEventListener("click", () => {
    dismissedError = lastError;
    renderError();
  });

  // --- Feed ----------------------------------------------------------------

  function onState(state) {
    if (state.latest_metrics) buf.samples = merge(buf.samples, [sampleOf(state.latest_metrics)]);
    if (state.contention) buf.contention = merge(buf.contention, [contentionOf(state.contention)]);
    if (state.forecasts) buf.forecasts = state.forecasts;
    renderTiles();
    renderCharts();
    renderInsight(state);
  }

  async function prefill() {
    const since = Date.now() / 1000 - WINDOW_SECONDS;
    try {
      const [metrics, contention, forecasts] = await Promise.all([
        getJSON("/metrics/system", { start: since }),
        getJSON("/contention", { start: since }),
        getJSON("/forecasts", { start: Date.now() / 1000 - 120 }),
      ]);
      buf.samples = merge(buf.samples, metrics.map(sampleOf));
      buf.contention = merge(buf.contention, contention.map(contentionOf));
      if (!buf.forecasts.length) buf.forecasts = latestCycle(forecasts);
      renderTiles();
      renderCharts();
    } catch (err) {
      console.warn("Live prefill failed", err);
    }
  }

  renderTiles();
  renderCharts();
  renderInsight({});
  prefill();
  if (app.lastState) onState(app.lastState);

  const offState = app.on("state", onState);
  const offOpen = app.on("open", prefill); // backfill the gap after a reconnect

  return () => {
    offState();
    offOpen();
    for (const { chart } of metricCharts) chart.destroy();
    contentionChart.destroy();
  };
}
