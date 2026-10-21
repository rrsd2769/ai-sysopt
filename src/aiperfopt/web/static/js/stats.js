// Pure numeric helpers for the Accuracy and Model views. No DOM access,
// so `node --test` can run them.

// Summarises /reporting/predicted-vs-actual rows. Rows whose actual value
// hasn't been observed yet count as pending, not as errors.
export function accuracyStats(rows) {
  const matched = (rows ?? []).filter((r) => r.actual_value != null && r.predicted_value != null);
  const total = rows?.length ?? 0;
  if (!matched.length) return { total, compared: 0, pending: total, mae: null, rmse: null, bias: null };
  let abs = 0;
  let sq = 0;
  let signed = 0;
  for (const r of matched) {
    const e = r.predicted_value - r.actual_value;
    abs += Math.abs(e);
    sq += e * e;
    signed += e;
  }
  const n = matched.length;
  return { total, compared: n, pending: total - n, mae: abs / n, rmse: Math.sqrt(sq / n), bias: signed / n };
}

// Feature-vector entries sorted by name, filtered by a case-insensitive
// substring of the name.
export function filterFeatures(featureVector, query = "") {
  const q = query.trim().toLowerCase();
  return Object.entries(featureVector ?? {})
    .filter(([name]) => !q || name.toLowerCase().includes(q))
    .sort(([a], [b]) => a.localeCompare(b));
}

// Bar lengths as a percentage of the largest magnitude, so signed values
// (SHAP, logistic contributions) share one scale.
export function barWidths(values) {
  const max = Math.max(0, ...values.map((v) => Math.abs(v ?? 0)));
  return values.map((v) => (max ? (Math.abs(v ?? 0) / max) * 100 : 0));
}

// Summarises /comparison/models holdout rows, one entry per model in
// first-seen order. accuracy = 100 - MAE (the targets are 0-100 %);
// skill = 1 - MAE / persistence MAE on the same target, so persistence
// itself scores 0. `winner` follows the comparison's own rule: highest
// mean skill among the candidates (persistence is the reference, not one).
export function comparisonSummary(rows, reference = "Persistence (reference)") {
  const baseline = new Map((rows ?? []).filter((r) => r.model === reference).map((r) => [r.target, r.mae]));
  const byModel = new Map();
  for (const r of rows ?? []) {
    if (!byModel.has(r.model)) byModel.set(r.model, []);
    byModel.get(r.model).push(r);
  }
  const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const models = [...byModel].map(([model, rs]) => {
    const skills = rs.filter((r) => baseline.get(r.target)).map((r) => 1 - r.mae / baseline.get(r.target));
    return {
      model,
      isReference: model === reference,
      accuracy: mean(rs.map((r) => 100 - r.mae)),
      skill: mean(skills),
      worstSkill: skills.length ? Math.min(...skills) : null,
      directional: mean(rs.map((r) => r.directional_accuracy).filter((v) => v != null)),
    };
  });
  const candidates = models.filter((m) => !m.isReference && m.skill != null);
  const winner = candidates.length ? candidates.reduce((a, b) => (b.skill > a.skill ? b : a)).model : null;
  return { models, winner };
}
