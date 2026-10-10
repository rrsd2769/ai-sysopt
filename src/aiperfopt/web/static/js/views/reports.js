// Reports view: the daily or weekly summary from GET /reporting/summary.

import { getJSON } from "../api.js";
import {
  escapeHtml,
  formatConfidence,
  formatDateTime,
  formatMetricValue,
  formatProbability,
  humanizeCategory,
  metricLabel,
} from "../format.js";
import { barList, segmented } from "../ui.js";

const PERIODS = [
  { value: "daily", label: "Last 24 h" },
  { value: "weekly", label: "Last 7 days" },
];

// The summary also carries byte counters and load figures; their
// avg/min/max over a day isn't meaningful on its own, so the context
// table sticks to the three percentages the rest of the app is built on.
const CONTEXT_METRICS = ["cpu_percent", "mem_percent", "swap_percent"];

let period = "daily";

const TEMPLATE = `
  <section class="card controls">
    <div class="field"><span>Period</span><div data-ref="period"></div></div>
    <span class="spacer"></span>
    <span class="muted small" data-ref="status"></span>
    <a class="btn" data-ref="download" download>Download JSON</a>
    <button type="button" class="btn" data-ref="refresh">Refresh</button>
  </section>
  <div class="banner danger" data-ref="error" hidden><p></p></div>
  <div class="report-grid" data-ref="body"></div>
`;

function stat(label, value) {
  return `<div class="kv"><span class="kv-value num">${escapeHtml(value)}</span><span class="kv-label">${escapeHtml(label)}</span></div>`;
}

function table(headers, rows, emptyText) {
  const head = headers.map((h, i) => `<th${i ? ' class="n"' : ""}>${escapeHtml(h)}</th>`).join("");
  const body = rows.length
    ? rows.map((r) => `<tr>${r.map((c, i) => (i ? `<td class="n">${c}</td>` : `<td>${c}</td>`)).join("")}</tr>`).join("")
    : `<tr><td colspan="${headers.length}" class="muted">${escapeHtml(emptyText)}</td></tr>`;
  return `<div class="table-scroll"><table class="table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function countBars(counts, emptyText) {
  const entries = Object.entries(counts ?? {})
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([category, n]) => ({ label: humanizeCategory(category), value: n, text: n.toLocaleString() }));
  return entries.length ? barList(entries) : `<p class="muted">${escapeHtml(emptyText)}</p>`;
}

function renderReport(r) {
  const context = CONTEXT_METRICS.map((name) => r.system_metrics.find((m) => m.metric_name === name)).filter(Boolean);
  const contextRows = context.map((m) => [
    escapeHtml(metricLabel(m.metric_name)),
    `<strong>${formatMetricValue(m.metric_name, m.avg)}</strong>`,
    formatMetricValue(m.metric_name, m.minimum),
    formatMetricValue(m.metric_name, m.maximum),
  ]);
  const accuracyRows = r.prediction_accuracy.map((p) => [
    escapeHtml(metricLabel(p.metric_name)),
    escapeHtml(p.horizon_label[0].toUpperCase() + p.horizon_label.slice(1)),
    formatMetricValue(p.metric_name, p.mae, 2),
    formatMetricValue(p.metric_name, p.rmse, 2),
    formatProbability(p.directional_accuracy),
    `${p.matched_sample_count.toLocaleString()} / ${p.prediction_count.toLocaleString()}`,
  ]);
  const c = r.contention;
  const d = r.diagnosis;
  const rec = r.recommendation;
  return `
    <section class="card">
      <h2>System context</h2>
      ${table(["Metric", "Average", "Min", "Max"], contextRows, "No readings in this window.")}
    </section>
    <section class="card">
      <h2>Prediction accuracy</h2>
      ${table(["Metric", "Horizon", "MAE", "RMSE", "Direction right", "Matched / total"], accuracyRows, "No predictions in this window yet.")}
    </section>
    <section class="card">
      <h2>Contention events</h2>
      <div class="kvs">
        ${stat("predictions", c.prediction_count.toLocaleString())}
        ${stat("triggered", c.triggered_count.toLocaleString())}
        ${stat("average probability", formatProbability(c.avg_probability))}
      </div>
    </section>
    <section class="card">
      <h2>Recommendations issued</h2>
      <div class="kvs">
        ${stat("recommendations", rec.total_count.toLocaleString())}
        ${stat("actions suggested", rec.total_actions.toLocaleString())}
      </div>
      ${countBars(rec.counts_by_diagnosis_category, "None issued in this window.")}
    </section>
    <section class="card span-2">
      <h2>Diagnoses by category</h2>
      <div class="kvs">
        ${stat("diagnoses", d.total_count.toLocaleString())}
        ${stat("actionable", d.actionable_count.toLocaleString())}
        ${stat("average confidence", formatConfidence(d.avg_confidence))}
      </div>
      ${countBars(d.counts_by_category, "No diagnoses in this window.")}
    </section>`;
}

export function mount(root) {
  root.innerHTML = TEMPLATE;
  const $ = (ref) => root.querySelector(`[data-ref="${ref}"]`);
  let seq = 0;

  const periodCtl = segmented($("period"), PERIODS, period, (v) => {
    period = v;
    load();
  });
  $("refresh").addEventListener("click", () => load());

  async function load() {
    const mine = ++seq;
    $("download").href = `/reporting/summary?period=${period}&download=true`;
    $("refresh").disabled = true;
    periodCtl.disable(true);
    $("status").textContent = "Building report…";
    $("error").hidden = true;
    try {
      const report = await getJSON("/reporting/summary", { period });
      if (mine !== seq) return;
      const readings = Math.max(0, ...report.system_metrics.filter((m) => CONTEXT_METRICS.includes(m.metric_name)).map((m) => m.sample_count));
      $("status").textContent = `${formatDateTime(report.start)} – ${formatDateTime(report.end)} · ${readings.toLocaleString()} readings`;
      $("body").innerHTML = renderReport(report);
    } catch (err) {
      if (mine !== seq) return;
      $("status").textContent = "";
      $("error").querySelector("p").textContent = `Couldn't build the report: ${err.message}`;
      $("error").hidden = false;
    } finally {
      if (mine === seq) {
        $("refresh").disabled = false;
        periodCtl.disable(false);
      }
    }
  }

  load();
  return () => {
    seq += 1;
  };
}
