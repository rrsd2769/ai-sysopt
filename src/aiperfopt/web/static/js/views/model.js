// Model view: the active model and its deployments, what's behind the
// latest diagnosis and contention reading, the raw feature vector, and
// retraining (run, compare, confirm/reject, roll back).

import { getJSON, postJSON } from "../api.js";
import {
  confidenceTier,
  contributorLabel,
  escapeHtml,
  formatConfidence,
  formatDateTime,
  formatMetricValue,
  formatNumber,
  formatProbability,
  formatTimestamp,
  humanizeCategory,
  metricLabel,
} from "../format.js";
import { filterFeatures } from "../stats.js";
import { barList, twoClick } from "../ui.js";

const DIAGNOSIS_LOOKBACK_SECONDS = 6 * 3600;
const DEPLOYMENT_ROWS = 8;

// Retraining outlives the view: a run takes minutes, and leaving the tab
// shouldn't lose the request or the candidate it returns.
const retrain = { inflight: null, startedAt: null, candidate: null, message: "", error: false };

const TEMPLATE = `
  <section class="model-top">
    <div class="card">
      <h2>Active model</h2>
      <p class="big-value" data-ref="active">--</p>
      <p class="muted small" data-ref="active-caption"></p>
    </div>
    <div class="card">
      <h2>Deployments</h2>
      <div data-ref="deployments"><p class="muted">Loading…</p></div>
    </div>
  </section>

  <section class="card">
    <h2>Why the latest diagnosis</h2>
    <div data-ref="diagnosis"><p class="muted">Loading…</p></div>
  </section>

  <section class="card">
    <h2>Contention internals</h2>
    <div data-ref="contention"><p class="muted">No contention reading yet.</p></div>
  </section>

  <details class="card features">
    <summary><h2>Feature vector</h2><span class="muted small" data-ref="feature-caption"></span></summary>
    <input type="search" class="input" data-ref="feature-search" placeholder="Filter features" aria-label="Filter features">
    <div class="table-scroll"><table class="table"><thead><tr><th>Feature</th><th class="n">Value</th></tr></thead><tbody data-ref="feature-rows"></tbody></table></div>
  </details>

  <section class="card">
    <h2>Retraining</h2>
    <p class="muted">Trains a candidate on your recorded usage, compares it with the active model and runs the
    quality gate. Nothing changes until you confirm.</p>
    <form class="controls bare" data-ref="form">
      <label class="field"><span>New version name</span><input class="input" data-ref="version" required></label>
      <label class="field grow"><span>Metrics (optional)</span><input class="input" data-ref="metrics" placeholder="cpu_percent, mem_percent. Blank trains the defaults."></label>
      <button type="submit" class="btn primary" data-ref="run">Run retraining</button>
    </form>
    <p class="retrain-status" data-ref="status" hidden></p>
    <div data-ref="result"></div>
    <div class="decision-row">
      <button type="button" class="btn primary" data-ref="confirm" disabled>Confirm deployment</button>
      <button type="button" class="btn" data-ref="reject" disabled>Reject candidate</button>
      <span class="spacer"></span>
      <button type="button" class="btn danger" data-ref="rollback">Rollback to last known good</button>
    </div>
  </section>
`;

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const words = (s) => cap(String(s ?? "").replaceAll("_", " "));
const yesNo = (b) => (b ? "yes" : "no");

function defaultVersionName() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `real-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
}

function meta(pairs) {
  return `<dl class="meta">${pairs
    .map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${v}</dd></div>`)
    .join("")}</dl>`;
}

function decisionChip(decision) {
  const tone = decision === "deployed" || decision === "rolled_back" ? "ok" : decision === "rejected" ? "warn" : "muted";
  return `<span class="tag" data-tone="${tone}">${escapeHtml(words(decision))}</span>`;
}

// --- Static renderers -------------------------------------------------------------

function renderDeployments(rows) {
  if (!rows.length) return `<p class="muted">No deployments recorded yet.</p>`;
  const body = rows
    .slice(0, DEPLOYMENT_ROWS)
    .map((d) => `
      <tr>
        <td>${formatDateTime(d.decided_at)}</td>
        <td><code>${escapeHtml(d.candidate_version)}</code></td>
        <td>${decisionChip(d.decision)}</td>
        <td class="muted">${d.previous_version ? `<code>${escapeHtml(d.previous_version)}</code>` : "--"}</td>
        <td class="muted">${escapeHtml(d.confirmed_by ?? "--")}</td>
      </tr>`)
    .join("");
  return `<div class="table-scroll"><table class="table">
    <thead><tr><th>When</th><th>Version</th><th>Decision</th><th>Replaced</th><th>By</th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

function renderExplanation(e) {
  const features = e.top_features ?? [];
  const head = `
    <div class="explain-head">
      <strong>${escapeHtml(metricLabel(e.metric_name))}</strong>
      <span class="muted small">${escapeHtml(e.forecaster_kind ?? "")}${e.tier ? ` · ${escapeHtml(words(e.tier))}` : ""}</span>
    </div>`;
  const bars = features.length
    ? barList(features.map((f) => ({
        label: f.feature_name,
        value: f.value,
        text: `${f.value > 0 ? "+" : ""}${formatNumber(f.value, 3)}`,
      })))
    : "";
  const note = e.note ? `<p class="muted small">${escapeHtml(e.note)}</p>` : "";
  return `<div class="explain">${head}${bars}${note}</div>`;
}

function renderDiagnosis(d, live) {
  if (!d) return `<p class="muted">No diagnosis in the last ${DIAGNOSIS_LOOKBACK_SECONDS / 3600} hours.</p>`;
  const when = live ? "this cycle" : `at ${formatTimestamp(d.generated_at)}`;
  const explanations = d.forecast_explanations ?? [];
  return `
    <p class="diag-title">${escapeHtml(humanizeCategory(d.category))} <span class="muted small">${when}</span></p>
    <p class="diag-body">${escapeHtml(d.technical_explanation)}</p>
    ${meta([
      ["Category", `<code>${escapeHtml(d.category)}</code>`],
      ["Actionable", yesNo(d.is_actionable)],
      ["Confidence", `<span class="chip" data-tier="${confidenceTier(d.confidence)}">${formatConfidence(d.confidence)}</span>`],
      ["Explanation tier", escapeHtml(words(d.explanation_tier))],
      ["Trigger", escapeHtml(words(d.trigger_reason))],
      ["Model", `<code>${escapeHtml(d.model_version)}</code>`],
    ])}
    <h3>Feature attribution</h3>
    ${explanations.length
      ? `<div class="explain-grid">${explanations.map(renderExplanation).join("")}</div>`
      : `<p class="muted">No feature attribution was recorded for this diagnosis.</p>`}`;
}

function renderContention(c) {
  if (!c) return `<p class="muted">No contention reading yet.</p>`;
  const metrics = c.contributing_metrics ?? [];
  const processes = c.contributing_processes ?? [];
  return `
    ${meta([
      ["Probability", formatProbability(c.probability)],
      ["Threshold", formatProbability(c.threshold_used)],
      ["Triggered", c.triggered ? `<span class="danger-text">yes</span>` : "no"],
      ["Anomaly score", formatNumber(c.anomaly_score, 3)],
      ["Model", `<code>${escapeHtml(c.model_version)}</code>`],
      ["At", formatTimestamp(c.generated_at)],
    ])}
    <div class="two-col">
      <div>
        <h3>Contributing metrics</h3>
        ${metrics.length
          ? barList(metrics.map((m) => ({
              label: contributorLabel(m.metric),
              value: m.contribution,
              text: `${m.contribution > 0 ? "+" : ""}${formatNumber(m.contribution, 3)}`,
            })))
          : `<p class="muted">None reported.</p>`}
      </div>
      <div>
        <h3>Contributing processes</h3>
        ${processes.length
          ? `<table class="table"><thead><tr><th>Process</th><th class="n">PID</th><th class="n">Value</th></tr></thead><tbody>
              ${processes.map((p) => `<tr><td>${escapeHtml(p.name)}</td><td class="n">${p.pid}</td><td class="n">${formatNumber(p.value, 2)}</td></tr>`).join("")}
            </tbody></table>`
          : `<p class="muted">No single process stands out.</p>`}
      </div>
    </div>`;
}

function renderCandidate(candidate) {
  const cmp = candidate.comparison;
  const gate = candidate.gate;
  const rec = cmp.recommendation;
  const reasons = (list) => (list?.length ? `<ul class="reasons">${list.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>` : "");
  const mae = (m, v) => formatMetricValue(m, v, 2);
  // Rounds first so a tiny change reads "0.00", not "-0.00".
  const signed = (m, v) => {
    const r = Math.round(v * 100) / 100 || 0;
    return `${r > 0 ? "+" : ""}${mae(m, r)}`;
  };
  const cmpRows = cmp.metric_comparisons
    .map((c) => {
      const delta = c.candidate_mae != null && c.baseline_mae != null ? c.candidate_mae - c.baseline_mae : null;
      return `
        <tr class="${c.regressed ? "bad" : ""}">
          <td>${escapeHtml(metricLabel(c.metric_name))}</td>
          <td>${escapeHtml(cap(c.horizon_label))}</td>
          <td class="n">${mae(c.metric_name, c.candidate_mae)}</td>
          <td class="n">${mae(c.metric_name, c.baseline_mae)}</td>
          <td class="n">${delta == null ? "--" : signed(c.metric_name, delta)}</td>
          <td class="n">${formatProbability(c.candidate_directional_accuracy)}</td>
          <td class="n">${formatProbability(c.baseline_directional_accuracy)}</td>
          <td>${c.regressed ? "Regressed" : "OK"}${c.note ? ` <span class="muted small">${escapeHtml(c.note)}</span>` : ""}</td>
        </tr>`;
    })
    .join("");
  const cc = cmp.contention_comparison;
  const ccRow = cc
    ? `<tr class="${cc.regressed ? "bad" : ""}"><td>Contention</td><td>F1</td>
        <td class="n">${formatNumber(cc.candidate_f1, 3)}</td><td class="n">${formatNumber(cc.baseline_f1, 3)}</td>
        <td class="n">--</td><td class="n"></td><td class="n"></td>
        <td>${cc.regressed ? "Regressed" : "OK"}${cc.note ? ` <span class="muted small">${escapeHtml(cc.note)}</span>` : ""}</td></tr>`
    : "";

  const gateRows = gate.forecast_criteria
    .map((c) => `
      <tr class="${c.passed ? "" : "bad"}">
        <td>${escapeHtml(metricLabel(c.metric_name))}</td>
        <td>${escapeHtml(cap(c.horizon_label))}</td>
        <td class="n">${mae(c.metric_name, c.candidate_mae)}</td>
        <td class="n">${mae(c.metric_name, c.persistence_mae)}</td>
        <td class="n">${formatProbability(c.candidate_directional_accuracy)}</td>
        <td>${c.passed ? "Pass" : "Blocked"}${c.note ? ` <span class="muted small">${escapeHtml(c.note)}</span>` : ""}</td>
      </tr>`)
    .join("");
  const gc = gate.contention_criterion;
  const gcRow = gc
    ? `<tr class="${gc.passed ? "" : "bad"}"><td>Contention</td><td>F1</td>
        <td class="n">${formatNumber(gc.candidate_f1, 3)}</td><td class="n">${formatNumber(gc.baseline_f1, 3)}</td><td class="n"></td>
        <td>${gc.passed ? "Pass" : "Blocked"}${gc.note ? ` <span class="muted small">${escapeHtml(gc.note)}</span>` : ""}</td></tr>`
    : "";

  return `
    <div class="candidate-head">
      <span>Candidate <code>${escapeHtml(cmp.candidate_version)}</code> vs. active
        ${cmp.active_version ? `<code>${escapeHtml(cmp.active_version)}</code>` : "<em>none</em>"}</span>
      <span class="tag" data-tone="${rec === "deploy" ? "ok" : "warn"}">Recommendation: ${escapeHtml(words(rec))}</span>
      <span class="muted small">Data: ${escapeHtml(words(candidate.data_source))}</span>
    </div>
    ${reasons(cmp.reasons)}
    <h3>Candidate vs. active</h3>
    <div class="table-scroll"><table class="table">
      <thead><tr><th>Metric</th><th>Horizon</th><th class="n">MAE new</th><th class="n">MAE active</th><th class="n">Change</th>
      <th class="n">Direction new</th><th class="n">Direction active</th><th>Result</th></tr></thead>
      <tbody>${cmpRows}${ccRow}</tbody></table></div>
    <div class="banner ${gate.passed ? "ok" : "danger"}">
      <p>${gate.passed ? "Quality gate passed. The candidate beats the naive baselines." : "Quality gate blocks this deployment. Confirm stays disabled."}</p>
    </div>
    ${reasons(gate.reasons)}
    <h3>Quality gate</h3>
    <div class="table-scroll"><table class="table">
      <thead><tr><th>Metric</th><th>Horizon</th><th class="n">MAE new</th><th class="n">Baseline</th>
      <th class="n">Direction new</th><th>Result</th></tr></thead>
      <tbody>${gateRows}${gcRow}</tbody></table></div>`;
}

// --- Mount --------------------------------------------------------------------------

export function mount(root, app) {
  root.innerHTML = TEMPLATE;
  const $ = (ref) => root.querySelector(`[data-ref="${ref}"]`);
  let alive = true;
  let activeVersion = null;
  let persistedDiagnosis = null;
  let featureVector = null;
  let featureTime = null;
  let tick = null;

  // --- Model + deployments --------------------------------------------------

  async function loadModel() {
    const [active, deployments] = await Promise.allSettled([
      getJSON("/retraining/active-version"),
      getJSON("/deployments"),
    ]);
    if (!alive) return;
    if (active.status === "fulfilled") {
      activeVersion = active.value.active_model_version;
      $("active").textContent = activeVersion ?? "None";
    }
    if (deployments.status === "fulfilled") {
      const rows = [...deployments.value].sort((a, b) => b.decided_at - a.decided_at);
      $("deployments").innerHTML = renderDeployments(rows);
      const current = rows.find((d) => d.candidate_version === activeVersion && d.decision !== "rejected");
      $("active-caption").textContent = current
        ? `Deployed ${formatDateTime(current.decided_at)}${current.previous_version ? `, replacing ${current.previous_version}` : ""}`
        : "";
    } else {
      $("deployments").innerHTML = `<p class="muted">Couldn't load deployments: ${escapeHtml(deployments.reason.message)}</p>`;
    }
  }

  // --- Live-driven sections ----------------------------------------------------

  function renderFeatures() {
    const rows = filterFeatures(featureVector, $("feature-search").value);
    const total = Object.keys(featureVector ?? {}).length;
    $("feature-caption").textContent = total
      ? `${total} features${featureTime ? ` at ${formatTimestamp(featureTime)}` : ""}`
      : "waiting for a live snapshot";
    $("feature-rows").innerHTML = rows.length
      ? rows.map(([n, v]) => `<tr><td><code>${escapeHtml(n)}</code></td><td class="n">${v == null ? "--" : formatNumber(v, 4)}</td></tr>`).join("")
      : `<tr><td colspan="2" class="muted">${total ? "No feature matches." : "No feature vector yet. Start Monitoring and Predictions."}</td></tr>`;
  }
  $("feature-search").addEventListener("input", renderFeatures);

  function onState(state) {
    if (state.active_model_version && !activeVersion) $("active").textContent = state.active_model_version;
    $("diagnosis").innerHTML = renderDiagnosis(state.diagnosis ?? persistedDiagnosis, Boolean(state.diagnosis));
    if (state.contention) $("contention").innerHTML = renderContention(state.contention);
    if (state.feature_vector) {
      featureVector = state.feature_vector;
      featureTime = state.timestamp;
      renderFeatures();
    }
  }

  async function loadPersisted() {
    const since = Date.now() / 1000 - DIAGNOSIS_LOOKBACK_SECONDS;
    const [diagnoses, contention] = await Promise.allSettled([
      getJSON("/diagnoses", { start: since }),
      app.lastState?.contention ? Promise.resolve([]) : getJSON("/contention", { start: Date.now() / 1000 - 120 }),
    ]);
    if (!alive) return;
    if (diagnoses.status === "fulfilled") persistedDiagnosis = diagnoses.value.at(-1) ?? null;
    if (!app.lastState?.diagnosis) $("diagnosis").innerHTML = renderDiagnosis(persistedDiagnosis, false);
    if (contention.status === "fulfilled" && contention.value.length && !app.lastState?.contention) {
      $("contention").innerHTML = renderContention(contention.value.at(-1));
    }
  }

  // --- Retraining ------------------------------------------------------------

  const controls = ["version", "metrics", "run", "confirm", "reject", "rollback"].map($);
  $("version").value = defaultVersionName();

  function setStatus(message, error = false) {
    retrain.message = message;
    retrain.error = error;
    const el = $("status");
    el.hidden = !message;
    el.textContent = message;
    el.classList.toggle("danger-text", error);
  }

  function syncButtons() {
    const busy = Boolean(retrain.inflight);
    const c = retrain.candidate;
    for (const el of controls) el.disabled = busy;
    $("confirm").disabled = busy || !c || !c.gate.passed;
    $("confirm").title = c && !c.gate.passed ? "The quality gate blocks this candidate." : "";
    $("reject").disabled = busy || !c;
    $("result").innerHTML = c ? renderCandidate(c) : "";
  }

  // Runs one retraining request with the panel locked; the promise lives in
  // module state so a remounted view picks it up.
  function act(label, request, onDone) {
    retrain.startedAt = Date.now();
    retrain.inflight = request()
      .then((res) => onDone(res))
      .catch((err) => setStatus(`${label} failed: ${err.message}`, true))
      .finally(() => {
        retrain.inflight = null;
        if (alive) {
          clearInterval(tick);
          syncButtons();
          setStatus(retrain.message, retrain.error);
          loadModel();
        }
      });
    watchInflight(label);
  }

  function watchInflight(label) {
    syncButtons();
    const update = () => {
      const s = Math.round((Date.now() - retrain.startedAt) / 1000);
      setStatus(`${label}… ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")} elapsed`);
    };
    update();
    clearInterval(tick);
    tick = setInterval(update, 1000);
    retrain.inflight.finally(() => clearInterval(tick));
  }

  $("form").addEventListener("submit", (event) => {
    event.preventDefault();
    const version = $("version").value.trim();
    if (!version) return;
    const metrics = $("metrics").value.split(",").map((m) => m.trim()).filter(Boolean);
    retrain.candidate = null;
    act(
      "Training the candidate",
      () => postJSON("/retraining/run", { new_model_version: version, target_metrics: metrics.length ? metrics : null }),
      (candidate) => {
        retrain.candidate = candidate;
        setStatus(`Candidate ${candidate.model_version} is ready for review.`);
      },
    );
  });

  function decide(confirmed) {
    const c = retrain.candidate;
    act(
      confirmed ? "Deploying" : "Rejecting",
      () => postJSON("/retraining/confirm", {
        candidate_version: c.model_version,
        comparison: c.comparison,
        gate: c.gate,
        confirmed,
      }),
      (record) => {
        retrain.candidate = null;
        setStatus(`Decision recorded: ${words(record.decision).toLowerCase()} ${record.candidate_version}.`);
      },
    );
  }

  const confirmBtn = twoClick($("confirm"), () => `Confirm: this replaces ${activeVersion ?? "nothing"}`, () => decide(true));
  $("reject").addEventListener("click", () => decide(false));
  const rollbackBtn = twoClick($("rollback"), () => `Roll back: this replaces ${activeVersion ?? "nothing"}`, () =>
    act(
      "Rolling back",
      () => postJSON("/retraining/rollback", { reason: "Manual rollback from the dashboard." }),
      (record) => setStatus(`Rolled back to ${record.candidate_version}.`),
    ),
  );

  // --- Start -------------------------------------------------------------------

  renderFeatures();
  if (retrain.inflight) {
    watchInflight("Still working");
    retrain.inflight.finally(() => {
      if (alive) {
        syncButtons();
        setStatus(retrain.message, retrain.error);
        loadModel();
      }
    });
  } else {
    syncButtons();
    setStatus(retrain.message, retrain.error);
  }
  loadModel();
  loadPersisted();
  if (app.lastState) onState(app.lastState);
  const offState = app.on("state", onState);

  return () => {
    alive = false;
    clearInterval(tick);
    confirmBtn.disarm();
    rollbackBtn.disarm();
    offState();
  };
}
