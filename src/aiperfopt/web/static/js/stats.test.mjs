// Run: node --test src/aiperfopt/web/static/js/

import assert from "node:assert/strict";
import { test } from "node:test";

import { accuracyStats, barWidths, comparisonSummary, filterFeatures } from "./stats.js";

test("accuracyStats on no rows", () => {
  assert.deepEqual(accuracyStats([]), { total: 0, compared: 0, pending: 0, mae: null, rmse: null, bias: null });
  assert.equal(accuracyStats(null).total, 0);
});

test("accuracyStats skips rows without an observed actual", () => {
  const s = accuracyStats([
    { predicted_value: 10, actual_value: 12 },
    { predicted_value: 20, actual_value: 16 },
    { predicted_value: 30, actual_value: null },
  ]);
  assert.equal(s.total, 3);
  assert.equal(s.compared, 2);
  assert.equal(s.pending, 1);
  assert.equal(s.mae, 3);
  assert.equal(s.rmse, Math.sqrt(10));
  assert.equal(s.bias, 1);
});

test("accuracyStats when nothing is observed yet", () => {
  const s = accuracyStats([{ predicted_value: 5, actual_value: null }]);
  assert.equal(s.compared, 0);
  assert.equal(s.pending, 1);
  assert.equal(s.mae, null);
});

test("filterFeatures sorts and filters case-insensitively", () => {
  const fv = { mem_lag_1: 2, cpu_lag_1: 1, CPU_roll_mean: 3 };
  assert.deepEqual(filterFeatures(fv).map(([n]) => n), ["cpu_lag_1", "CPU_roll_mean", "mem_lag_1"]);
  assert.deepEqual(filterFeatures(fv, " cpu ").map(([n]) => n), ["cpu_lag_1", "CPU_roll_mean"]);
  assert.deepEqual(filterFeatures(null), []);
});

test("barWidths scales by the largest magnitude", () => {
  assert.deepEqual(barWidths([2, -4, 1]), [50, 100, 25]);
  assert.deepEqual(barWidths([0, 0]), [0, 0]);
  assert.deepEqual(barWidths([null, 3]), [0, 100]);
});

test("comparisonSummary scores models against persistence and picks the highest mean skill", () => {
  const rows = [
    { target: "a", model: "LightGBM", mae: 2, directional_accuracy: 0.6 },
    { target: "b", model: "LightGBM", mae: 3, directional_accuracy: 0.8 },
    { target: "a", model: "Ridge", mae: 1, directional_accuracy: 0.5 },
    { target: "b", model: "Ridge", mae: 6, directional_accuracy: 0.5 },
    { target: "a", model: "Persistence (reference)", mae: 2, directional_accuracy: 0.1 },
    { target: "b", model: "Persistence (reference)", mae: 4, directional_accuracy: 0.1 },
  ];
  const { models, winner } = comparisonSummary(rows);
  const lgbm = models.find((m) => m.model === "LightGBM");
  assert.equal(lgbm.accuracy, 97.5);
  assert.equal(lgbm.skill, 0.125);
  assert.equal(lgbm.worstSkill, 0);
  assert.equal(lgbm.directional, 0.7);
  assert.equal(models.find((m) => m.model === "Ridge").skill, 0);
  assert.equal(models.find((m) => m.isReference).skill, 0);
  assert.equal(winner, "LightGBM");
});

test("comparisonSummary on no rows", () => {
  assert.deepEqual(comparisonSummary([]), { models: [], winner: null });
});
