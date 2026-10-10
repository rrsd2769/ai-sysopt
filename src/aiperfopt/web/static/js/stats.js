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
