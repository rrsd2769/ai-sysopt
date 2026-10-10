// Mirrors the Qt UI's tests/frontend/test_formatting.py cases.
// Run: node --test src/aiperfopt/web/static/js/

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  CATEGORY_LABELS,
  confidenceTier,
  contributorLabel,
  formatConfidence,
  formatDurationShort,
  escapeHtml,
  formatDateTime,
  formatHorizon,
  formatHorizonRange,
  formatMetricValue,
  formatPercent,
  formatProbability,
  formatTimestamp,
  humanizeBytes,
  humanizeCategory,
  metricLabel,
  readinessBannerText,
} from "./format.js";

test("humanizeCategory has a real label for every known category", () => {
  for (const category of Object.keys(CATEGORY_LABELS)) {
    const label = humanizeCategory(category);
    assert.notEqual(label, category);
    assert.match(label[0], /[A-Z]/);
  }
});

test("humanizeCategory falls back gracefully for an unknown value", () => {
  assert.equal(humanizeCategory("some_future_category"), "Some future category");
});

test("humanizeBytes", () => {
  assert.equal(humanizeBytes(null), "--");
  assert.equal(humanizeBytes(0), "0.0 B");
  assert.equal(humanizeBytes(1536), "1.5 KB");
  assert.equal(humanizeBytes(1024 * 1024 * 3), "3.0 MB");
});

test("formatPercent", () => {
  assert.equal(formatPercent(null), "--");
  assert.equal(formatPercent(undefined), "--");
  assert.equal(formatPercent(53.256), "53.3%");
  assert.equal(formatPercent(53.256, 0), "53%");
});

test("formatConfidence and formatProbability turn fractions into percents", () => {
  assert.equal(formatConfidence(null), "--");
  assert.equal(formatConfidence(0.876), "88%");
  assert.equal(formatProbability(0.4321), "43.2%");
});

test("formatTimestamp", () => {
  assert.equal(formatTimestamp(null), "--");
  assert.match(formatTimestamp(1700000000), /^\d\d:\d\d:\d\d$/);
});

test("confidenceTier thresholds", () => {
  assert.equal(confidenceTier(null), "unknown");
  assert.equal(confidenceTier(0.9), "high");
  assert.equal(confidenceTier(0.6), "medium");
  assert.equal(confidenceTier(0.1), "low");
  assert.equal(confidenceTier(0.75), "high");
  assert.equal(confidenceTier(0.5), "medium");
});

test("formatDurationShort", () => {
  assert.equal(formatDurationShort(45), "45s");
  assert.equal(formatDurationShort(135), "2m15s");
  assert.equal(formatDurationShort(0), "0s");
});

test("readinessBannerText for each state", () => {
  assert.equal(readinessBannerText("no_data", "detail", 0, 300), "No data yet");
  assert.equal(readinessBannerText("collecting", "detail", 30, 300), "Collecting - 30s of 5m00s");
  assert.equal(readinessBannerText("needs_training", "detail", 300, 300), "No model trained");
  assert.equal(readinessBannerText("ready", "detail", 300, 300), "Live");
  assert.equal(readinessBannerText("replay", "detail", 300, 300), "REPLAY");
});

test("readinessBannerText falls back to detail for an unknown state", () => {
  assert.equal(readinessBannerText("some_future_state", "a detail string", 0, 0), "a detail string");
});

test("formatHorizon", () => {
  assert.equal(formatHorizon(null), "--");
  assert.equal(formatHorizon(30), "in 30 s");
  assert.equal(formatHorizon(60), "in 60 s");
  assert.equal(formatHorizon(120), "in 2 min");
  assert.equal(formatHorizon(90), "in 1.5 min");
});

test("metricLabel", () => {
  assert.equal(metricLabel("cpu_percent"), "CPU");
  assert.equal(metricLabel("mem_percent"), "Memory");
  assert.equal(metricLabel("disk_read_bytes_rate"), "Disk read bytes rate");
});

test("formatHorizonRange", () => {
  assert.equal(formatHorizonRange([30, 60]), "in 30–60 s");
  assert.equal(formatHorizonRange([120, 300]), "in 2–5 min");
  assert.equal(formatHorizonRange([90, 150]), "in 1.5–2.5 min");
  assert.equal(formatHorizonRange(null), "--");
});

test("formatMetricValue picks a unit from the metric name", () => {
  assert.equal(formatMetricValue("cpu_percent", 42.26), "42.3%");
  assert.equal(formatMetricValue("disk_read_bytes", 2048), "2.0 KB");
  assert.equal(formatMetricValue("load_avg_1", 3.456), "3.5");
  assert.equal(formatMetricValue("cpu_percent", null), "--");
});

test("formatDateTime is local and zero-padded", () => {
  const t = new Date(2026, 0, 5, 7, 3).getTime() / 1000;
  assert.equal(formatDateTime(t), "2026-01-05 07:03");
  assert.equal(formatDateTime(null), "--");
});

test("escapeHtml", () => {
  assert.equal(escapeHtml(`<a href="x">'&'</a>`), "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;");
  assert.equal(escapeHtml(null), "");
});

test("contributorLabel", () => {
  assert.equal(contributorLabel("current_mem_percent"), "Memory now");
  assert.equal(contributorLabel("forecast_cpu_percent"), "CPU forecast");
  assert.equal(contributorLabel("anomaly_score"), "Anomaly score");
});
