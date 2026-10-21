// Compare view: graphs of the saved LightGBM / Random Forest / Ridge
// comparison (GET /comparison/models), drawn on demand. The comparison
// itself is run with `aiperfopt compare`; this view only reads its results.

import { getJSON } from "../api.js";
import { cssVar } from "../charts.js";
import { escapeHtml, formatDateTime, metricLabel } from "../format.js";
import { comparisonSummary } from "../stats.js";
import { segmented } from "../ui.js";

const REFERENCE = "Persistence (reference)";

const MODELS = {
  LightGBM: { label: "LightGBM", color: "--model-lightgbm" },
  "Random Forest": { label: "Random Forest", color: "--model-rf" },
  "Ridge Regression": { label: "Ridge", color: "--model-ridge" },
  [REFERENCE]: { label: "No change (persistence)", color: "--model-persistence" },
  Persistence: { label: "No change (persistence)", color: "--model-persistence" },
};
const modelLabel = (m) => MODELS[m]?.label ?? m;
const modelColor = (m) => cssVar(MODELS[m]?.color ?? "--text-muted");

const MEASURES = [
  {
    value: "accuracy", label: "Accuracy",
    yLabel: "Accuracy (%) = 100 − average error", note: "Higher is better. Axis starts near the lowest value so the differences are visible.",
    get: (r) => 100 - r.mae, fmt: (v) => `${v.toFixed(1)}%`, zoom: true,
  },
  {
    value: "mae", label: "Error (MAE)",
    yLabel: "Average error (percentage points)", note: "Lower is better.",
    get: (r) => r.mae, fmt: (v) => v.toFixed(2),
  },
  {
    value: "skill", label: "vs. no change",
    yLabel: "Error vs. \"no change\" (%)", note: "Below 0 = less error than simply repeating the current value. Lower is better.",
    get: (r, base) => (base ? (r.mae / base - 1) * 100 : null), fmt: (v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}%`,
  },
  {
    value: "direction", label: "Direction right",
    yLabel: "Direction predicted correctly (%)", note: "Higher is better: did it get up / down / steady right?",
    get: (r) => (r.directional_accuracy == null ? null : r.directional_accuracy * 100), fmt: (v) => `${v.toFixed(0)}%`,
  },
];

// Kept across visits to the view within one page load.
const selection = { measure: "accuracy", target: null };

const TEMPLATE = `
  <section class="card controls">
    <div class="field grow">
      <span class="compare-title">Model comparison</span>
      <span class="muted">LightGBM vs Random Forest vs Ridge, each scored against a "no change" forecast on the newest 20 % of recorded usage, which no model saw in training.</span>
    </div>
    <span class="muted small" data-ref="status"></span>
    <button type="button" class="btn primary" data-ref="generate">Generate comparison graph</button>
  </section>
  <div class="banner danger" data-ref="error" hidden><p></p></div>
  <div data-ref="body" class="compare-body" hidden>
    <section class="compare-tiles" data-ref="tiles"></section>
    <section class="card chart-card">
      <div class="chart-head">
        <span class="chart-title">Holdout results by forecast</span>
        <div data-ref="measure"></div>
        <span class="chart-legend" data-ref="legend"></span>
      </div>
      <div class="barchart" data-ref="main"></div>
      <p class="muted small" data-ref="main-note"></p>
    </section>
    <div class="report-grid">
      <section class="card chart-card" data-ref="regime-card">
        <div class="chart-head">
          <span class="chart-title">Load steady vs. changing</span>
          <label class="field"><select data-ref="regime-target"></select></label>
          <span class="chart-legend" data-ref="regime-legend"></span>
        </div>
        <div class="barchart" data-ref="regime"></div>
        <p class="muted small">Average error (percentage points), lower is better. "Changing" = the largest 25 % of moves in the test data.</p>
      </section>
      <section class="card chart-card" data-ref="cv-card">
        <div class="chart-head">
          <span class="chart-title">Walk-forward cross-validation</span>
          <label class="field"><select data-ref="cv-target"></select></label>
          <span class="chart-legend" data-ref="cv-legend"></span>
        </div>
        <div class="barchart" data-ref="cv"></div>
        <p class="muted small">Average error per fold (percentage points), lower is better. Each fold trains on all earlier data and tests on the next block.</p>
      </section>
    </div>
    <section class="card">
      <h2>Running cost</h2>
      <div data-ref="cost"></div>
    </section>
  </div>
`;

// ---- grouped bar chart (SVG) ------------------------------------------------

function niceStep(span, count = 5) {
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

// groups: x labels. series: [{ name, color, values }], values aligned to
// groups (null = no bar). Redraws at the container's width.
function drawBars(el, { groups, series, yLabel, fmt, zoom = false, height = 300 }) {
  const width = el.clientWidth || 800;
  const m = { top: 22, right: 12, bottom: 34, left: 62 };
  const pw = width - m.left - m.right;
  const ph = height - m.top - m.bottom;
  const vals = series.flatMap((s) => s.values).filter((v) => v != null);
  if (!vals.length) {
    el.innerHTML = `<div class="chart-empty">Nothing to show.</div>`;
    return;
  }
  let lo = Math.min(0, ...vals);
  let hi = Math.max(0, ...vals);
  if (zoom) {
    lo = Math.max(0, Math.floor(Math.min(...vals) - 1));
    hi = Math.min(100, Math.ceil(Math.max(...vals) + 0.5));
  }
  const step = niceStep(hi - lo || 1);
  lo = Math.floor(lo / step) * step;
  hi = Math.ceil((hi + (zoom ? 0 : (hi - lo) * 0.08)) / step) * step;
  const y = (v) => m.top + ph - ((v - lo) / (hi - lo)) * ph;
  const t = { text: cssVar("--text"), muted: cssVar("--text-muted"), grid: cssVar("--grid"), border: cssVar("--border") };

  const parts = [];
  for (let v = lo; v <= hi + step / 2; v += step) {
    const yy = y(v).toFixed(1);
    parts.push(`<line x1="${m.left}" x2="${width - m.right}" y1="${yy}" y2="${yy}" stroke="${v === 0 && lo < 0 ? t.muted : t.grid}"/>`);
    parts.push(`<text x="${m.left - 6}" y="${yy}" dy="0.32em" text-anchor="end" fill="${t.muted}">${escapeHtml(fmt(+v.toFixed(6)))}</text>`);
  }
  const gw = pw / groups.length;
  const bw = Math.min(46, (gw * 0.78) / series.length);
  const base = y(Math.max(lo, 0));
  groups.forEach((g, gi) => {
    const x0 = m.left + gi * gw + (gw - bw * series.length) / 2;
    series.forEach((s, si) => {
      const v = s.values[gi];
      if (v == null) return;
      const top = y(v);
      const x = x0 + si * bw;
      const yTop = Math.min(top, base);
      const h = Math.max(1, Math.abs(base - top));
      parts.push(`<rect x="${(x + 1).toFixed(1)}" y="${yTop.toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${s.color}"><title>${escapeHtml(`${s.name} · ${g}: ${fmt(v)}`)}</title></rect>`);
      if (bw >= 30) {
        const ly = v >= 0 || zoom ? yTop - 5 : yTop + h + 12;
        parts.push(`<text x="${(x + bw / 2).toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" fill="${t.text}" class="bar-num">${escapeHtml(fmt(v))}</text>`);
      }
    });
    parts.push(`<text x="${(m.left + gi * gw + gw / 2).toFixed(1)}" y="${height - m.bottom + 20}" text-anchor="middle" fill="${t.text}">${escapeHtml(g)}</text>`);
  });
  parts.push(`<text transform="translate(14 ${m.top + ph / 2}) rotate(-90)" text-anchor="middle" fill="${t.muted}">${escapeHtml(yLabel)}</text>`);
  el.innerHTML = `<svg width="${width}" height="${height}" role="img" aria-label="${escapeHtml(yLabel)}">${parts.join("")}</svg>`;
}

function legend(models) {
  return models.map((m) => `<span><i class="swatch block" style="color:${modelColor(m)}"></i>${escapeHtml(modelLabel(m))}</span>`).join("");
}

// ---- view -------------------------------------------------------------------

export function mount(root) {
  root.innerHTML = TEMPLATE;
  const $ = (ref) => root.querySelector(`[data-ref="${ref}"]`);
  let data = null;
  let alive = true;
  const draws = [];

  const targetLabel = (target) => {
    const [, metric, horizon] = /^(.*)_(short|medium)$/.exec(target) ?? [null, target, null];
    const secs = data?.horizon_seconds?.[horizon];
    return secs ? `${metricLabel(metric)} · ${secs} s ahead` : metricLabel(metric);
  };
  const uniq = (xs) => [...new Set(xs)];

  function drawMain() {
    const measure = MEASURES.find((x) => x.value === selection.measure);
    const rows = data.holdout;
    const targets = uniq(rows.map((r) => r.target));
    let models = uniq(rows.map((r) => r.model));
    if (measure.value === "skill") models = models.filter((m) => m !== REFERENCE);
    const base = new Map(rows.filter((r) => r.model === REFERENCE).map((r) => [r.target, r.mae]));
    drawBars($("main"), {
      groups: targets.map(targetLabel),
      series: models.map((m) => ({
        name: modelLabel(m),
        color: modelColor(m),
        values: targets.map((tg) => {
          const r = rows.find((x) => x.target === tg && x.model === m);
          return r ? measure.get(r, base.get(tg)) : null;
        }),
      })),
      yLabel: measure.yLabel, fmt: measure.fmt, zoom: measure.zoom,
    });
    $("legend").innerHTML = legend(models);
    $("main-note").textContent = measure.note;
  }

  function drawByTarget(ref, rows, groupKey, groupLabel) {
    const target = $(`${ref}-target`).value;
    const sel = rows.filter((r) => r.target === target);
    const groups = uniq(sel.map((r) => r[groupKey]));
    const models = uniq(sel.map((r) => r.model));
    drawBars($(ref), {
      groups: groups.map(groupLabel),
      series: models.map((m) => ({
        name: modelLabel(m), color: modelColor(m),
        values: groups.map((g) => sel.find((r) => r[groupKey] === g && r.model === m)?.mae ?? null),
      })),
      yLabel: "Average error (points)", fmt: (v) => v.toFixed(1), height: 260,
    });
    $(`${ref}-legend`).innerHTML = legend(models);
  }

  function fillTargets(select, rows) {
    const targets = uniq(rows.map((r) => r.target));
    select.innerHTML = targets.map((tg) => `<option value="${escapeHtml(tg)}">${escapeHtml(targetLabel(tg))}</option>`).join("");
    if (targets.includes(selection.target)) select.value = selection.target;
  }

  function renderTiles() {
    const { models, winner } = comparisonSummary(data.holdout, REFERENCE);
    $("tiles").innerHTML = models.map((m) => {
      const extra = m.skill == null || m.isReference ? "the reference every model must beat" : `${m.skill >= 0 ? "" : "+"}${(-m.skill * 100).toFixed(0)}% error vs. no change on average`;
      return `
        <div class="card stat${m.model === winner ? " winner" : ""}">
          <span class="tile-label"><i class="swatch block" style="color:${modelColor(m.model)}"></i>${escapeHtml(modelLabel(m.model))}${m.model === winner ? ' <span class="tag" data-tone="ok">Chosen</span>' : ""}</span>
          <span class="tile-value">${m.accuracy.toFixed(1)}%</span>
          <span class="tile-caption">accuracy · ${escapeHtml(extra)}</span>
        </div>`;
    }).join("");
  }

  function renderCost() {
    const rows = data.holdout.filter((r) => r.latency_ms != null);
    const models = uniq(rows.map((r) => r.model));
    const avg = (m, k) => {
      const xs = rows.filter((r) => r.model === m).map((r) => r[k]);
      return xs.reduce((a, b) => a + b, 0) / xs.length;
    };
    const size = (kb) => (kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(0)} kB`);
    $("cost").innerHTML = `<div class="table-scroll"><table class="table">
      <thead><tr><th>Model</th><th class="n">Time per prediction</th><th class="n">Training time per target</th><th class="n">Model size</th></tr></thead>
      <tbody>${models.map((m) => `<tr><td>${escapeHtml(modelLabel(m))}</td><td class="n">${avg(m, "latency_ms").toFixed(1)} ms</td><td class="n">${avg(m, "fit_seconds").toFixed(0)} s</td><td class="n">${size(avg(m, "size_kb"))}</td></tr>`).join("")}</tbody>
    </table></div>`;
  }

  function renderAll() {
    renderTiles();
    drawMain();
    $("regime-card").hidden = !data.regimes.length;
    if (data.regimes.length) {
      fillTargets($("regime-target"), data.regimes);
      drawByTarget("regime", data.regimes, "regime", (g) => (g === "stable" ? "Load steady" : g === "moving" ? "Load changing" : g));
    }
    $("cv-card").hidden = !data.cv.length;
    if (data.cv.length) {
      fillTargets($("cv-target"), data.cv);
      drawByTarget("cv", data.cv, "fold", (f) => `Fold ${f + 1}`);
    }
    renderCost();
  }

  segmented($("measure"), MEASURES.map(({ value, label }) => ({ value, label })), selection.measure, (v) => {
    selection.measure = v;
    drawMain();
  });
  $("regime-target").addEventListener("change", (e) => {
    selection.target = e.target.value;
    drawByTarget("regime", data.regimes, "regime", (g) => (g === "stable" ? "Load steady" : g === "moving" ? "Load changing" : g));
  });
  $("cv-target").addEventListener("change", (e) => {
    selection.target = e.target.value;
    drawByTarget("cv", data.cv, "fold", (f) => `Fold ${f + 1}`);
  });

  $("generate").addEventListener("click", async () => {
    $("generate").disabled = true;
    $("status").textContent = "Loading results…";
    $("error").hidden = true;
    try {
      const res = await getJSON("/comparison/models");
      if (!alive) return;
      data = res;
      $("body").hidden = false;
      renderAll();
      $("status").textContent = `Results from ${formatDateTime(data.generated_at)}`;
      $("generate").textContent = "Regenerate graph";
    } catch (err) {
      if (!alive) return;
      $("status").textContent = "";
      $("error").querySelector("p").textContent = `Couldn't load the comparison: ${err.message}`;
      $("error").hidden = false;
    } finally {
      if (alive) $("generate").disabled = false;
    }
  });

  // Redraw at the new width, and with the new palette when the theme flips.
  let frame = 0;
  const redraw = () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => data && !$("body").hidden && renderAll());
  };
  const resizer = new ResizeObserver(redraw);
  resizer.observe($("main"));
  const lightQuery = matchMedia("(prefers-color-scheme: light)");
  lightQuery.addEventListener("change", redraw);
  draws.push(() => resizer.disconnect(), () => lightQuery.removeEventListener("change", redraw));

  return () => {
    alive = false;
    cancelAnimationFrame(frame);
    for (const off of draws) off();
  };
}
