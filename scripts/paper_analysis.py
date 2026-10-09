"""Deeper analysis of the three-model comparison, for the IEEE paper.

Refits the three candidates from ``compare_models.py`` on the same chronological
80/20 split (no CV -- that is already in ``results/model_comparison_cv.csv``)
and keeps every per-sample holdout prediction, which the comparison script
throws away. From those it computes:

* Diebold-Mariano tests on absolute error (Newey-West variance, lag = horizon
  in samples) -- is LightGBM's edge over each rival statistically real?
* Moving-block bootstrap 95% CIs for each model's holdout skill.
* Error by regime: "stable" samples (smallest 75% of actual |change|) versus
  "moving" samples (largest 25%) -- where does each model win or lose?
* LightGBM gain importance, per feature and per feature group.
* Data characterisation: utilisation distribution and the full time series.

Uses the same preprocessing as ``compare_models.py`` (``--raw`` to skip it).
Run against a pinned snapshot so the numbers match the published tables:

    AIPERFOPT_HOME=paper_snapshot python scripts/paper_analysis.py

Writes ``results/paper/*`` (``results/paper_raw/*`` with ``--raw``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("compare_models", REPO_ROOT / "scripts" / "compare_models.py")
cm = importlib.util.module_from_spec(_spec)
sys.modules["compare_models"] = cm
_spec.loader.exec_module(cm)

from aiperfopt.bootstrap import bootstrap  # noqa: E402
from aiperfopt.features import FeatureEngineeringConfig, compute_feature_matrix  # noqa: E402
from aiperfopt.features.storage_bridge import aligned_process_summary  # noqa: E402
from aiperfopt.forecasting.dataset import build_supervised_dataset  # noqa: E402
from aiperfopt.forecasting.evaluation import feature_columns_of  # noqa: E402
from aiperfopt.lifecycle.training_data import (  # noqa: E402
    REAL_USAGE_SCENARIO_NAME,
    build_real_usage_training_data,
)

OUT_DIR = REPO_ROOT / "results" / "paper"  # main() switches to paper_raw for --raw
POLL_SECONDS = 1.5
BOOT_REPS = 1000
BLOCK = 400  # samples (~10 min) -- longer than either horizon, keeps autocorrelation inside blocks
MOVING_QUANTILE = 0.75
PERSIST = "Persistence"


def diebold_mariano(abs_err_a: np.ndarray, abs_err_b: np.ndarray, lag: int) -> tuple[float, float]:
    """DM statistic and two-sided p on d = |e_a| - |e_b|; negative stat means A is better."""
    d = abs_err_a - abs_err_b
    n = len(d)
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        lrv += 2 * weight * np.dot(dc[k:], dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def block_bootstrap_skill(abs_err: np.ndarray, abs_err_persist: np.ndarray, rng) -> tuple[float, float]:
    n = len(abs_err)
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK, size=(BOOT_REPS, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(BOOT_REPS, -1)[:, :n]
    skill = 1 - abs_err[idx].mean(axis=1) / abs_err_persist[idx].mean(axis=1)
    lo, hi = np.percentile(skill, [2.5, 97.5])
    return float(lo), float(hi)


def feature_group(name: str) -> str:
    if name.startswith("proc_"):
        return "Process activity"
    for key, group in (("roll", "Rolling-window stats"), ("mean", "Rolling-window stats"),
                       ("std", "Rolling-window stats"), ("min", "Rolling-window stats"),
                       ("max", "Rolling-window stats"), ("lag", "Lagged values"),
                       ("rate", "Rate of change"), ("roc", "Rate of change"), ("delta", "Rate of change"),
                       ("diff", "Rate of change"), ("slope", "Rate of change")):
        if key in name:
            return group
    return "Current values / other"


def characterise_data(db_path: Path, split_ts: float) -> dict:
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql("select timestamp, cpu_percent, mem_percent, swap_percent from raw_system_metrics "
                         "order by timestamp", con)
    info = {
        "samples": len(df),
        "cpu_median": float(df.cpu_percent.median()),
        "cpu_p90": float(df.cpu_percent.quantile(0.9)),
        "cpu_p99": float(df.cpu_percent.quantile(0.99)),
        "cpu_share_below_10": float((df.cpu_percent < 10).mean()),
        "cpu_share_below_20": float((df.cpu_percent < 20).mean()),
        "cpu_share_above_70": float((df.cpu_percent >= 70).mean()),
        "mem_median": float(df.mem_percent.median()),
        "mem_p10": float(df.mem_percent.quantile(0.1)),
        "mem_p90": float(df.mem_percent.quantile(0.9)),
        "mem_share_above_85": float((df.mem_percent >= 85).mean()),
        "swap_share_above_50": float((df.swap_percent >= 50).mean()),
    }
    gaps = np.diff(df.timestamp.to_numpy())
    info["recorded_hours"] = float(gaps[gaps < 60].sum() / 3600)
    info["gaps_over_1h"] = [round(g / 3600, 1) for g in gaps if g > 3600]

    t = pd.to_datetime(df.timestamp, unit="s")
    s = df.set_index(t)[["cpu_percent", "mem_percent"]].resample("5min").mean()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(s.index, s.cpu_percent, lw=0.8, color="#2a6fdb", label="CPU %")
    ax.plot(s.index, s.mem_percent, lw=0.8, color="#8e44ad", label="Memory %")
    ax.axvline(pd.to_datetime(split_ts, unit="s"), color="k", ls="--", lw=0.8)
    ax.text(pd.to_datetime(split_ts, unit="s"), 102, " holdout →", fontsize=7, va="bottom")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Utilisation (%)")
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "timeseries.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
    axes[0].hist(df.cpu_percent, bins=50, color="#2a6fdb")
    axes[0].set_xlabel("CPU %")
    axes[1].hist(df.mem_percent, bins=50, color="#8e44ad")
    axes[1].set_xlabel("Memory %")
    for ax in axes:
        ax.set_ylabel("Samples")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distributions.pdf")
    plt.close(fig)
    return info


def main() -> int:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", action="store_true", help="skip preprocessing")
    preprocess = not parser.parse_args().raw
    if not preprocess:
        OUT_DIR = REPO_ROOT / "results" / "paper_raw"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = bootstrap(None)
    settings = ctx.settings.forecasting
    feature_config = FeatureEngineeringConfig()
    rng = np.random.default_rng(0)

    print("Loading real usage data ...", flush=True)
    scenarios, holdout_samples = build_real_usage_training_data(ctx.storage_engine, holdout_fraction=0.2)
    train_samples = scenarios[REAL_USAGE_SCENARIO_NAME]
    print(f"  train={len(train_samples)} holdout={len(holdout_samples)}", flush=True)
    tol = feature_config.tolerance_seconds
    train_fm = compute_feature_matrix(
        train_samples, feature_config, aligned_process_summary(ctx.storage_engine, train_samples, tol))
    holdout_fm = compute_feature_matrix(
        holdout_samples, feature_config, aligned_process_summary(ctx.storage_engine, holdout_samples, tol))

    db_path = Path(str(ctx.storage_engine.url.database))
    data_info = characterise_data(db_path, holdout_samples[0].timestamp)
    data_info["train_samples"] = len(train_samples)
    data_info["holdout_samples"] = len(holdout_samples)

    horizons = (("short", settings.short_horizon_target_seconds),
                ("medium", settings.medium_horizon_target_seconds))
    dm_rows, ci_rows, regime_rows, importance_rows = [], [], [], []
    for metric in cm.METRICS:
        for horizon_label, horizon_seconds in horizons:
            target = f"{metric}_{horizon_label}"
            print(f"{target} ...", flush=True)
            train = build_supervised_dataset(train_samples, train_fm, metric, horizon_seconds, tol)
            holdout = build_supervised_dataset(holdout_samples, holdout_fm, metric, horizon_seconds, tol)
            if preprocess:
                train, _ = cm.clean_rows(train, feature_config)
                holdout, _ = cm.clean_rows(holdout, feature_config)
            cols = feature_columns_of(train)
            truth = holdout["target"].to_numpy()
            current = holdout["current_value"].to_numpy()

            abs_err = {PERSIST: np.abs(truth - current)}
            for cand in cm.make_candidates(horizon_label, horizon_seconds, preprocess):
                cand.fit(train[cols], train["target"] - train["current_value"])
                abs_err[cand.name] = np.abs(truth - (cand.predict_delta(holdout[cols]) + current))
                if cand.name == "LightGBM":
                    booster = getattr(cand, "inner", cand)._model.booster
                    gain = booster.feature_importance(importance_type="gain")
                    total = gain.sum() or 1.0
                    for fname, g in zip(booster.feature_name(), gain):
                        importance_rows.append({"target": target, "feature": fname,
                                                "group": feature_group(fname), "gain_share": g / total})
                print(f"  {cand.name}: MAE={abs_err[cand.name].mean():.3f}", flush=True)

            lag = int(round(horizon_seconds / POLL_SECONDS))
            for rival in ("Random Forest", "Ridge Regression", PERSIST):
                stat, p = diebold_mariano(abs_err["LightGBM"], abs_err[rival], lag)
                dm_rows.append({"target": target, "comparison": f"LightGBM vs {rival}",
                                "mean_abs_err_diff": float((abs_err["LightGBM"] - abs_err[rival]).mean()),
                                "dm_stat": stat, "p_value": p})
            for name in cm.CANDIDATE_NAMES:
                lo, hi = block_bootstrap_skill(abs_err[name], abs_err[PERSIST], rng)
                ci_rows.append({"target": target, "model": name,
                                "skill": float(1 - abs_err[name].mean() / abs_err[PERSIST].mean()),
                                "ci_low": lo, "ci_high": hi})

            change = abs_err[PERSIST]  # |actual change| == persistence error
            cut = np.quantile(change, MOVING_QUANTILE)
            for regime, mask in (("stable", change <= cut), ("moving", change > cut)):
                for name, err in abs_err.items():
                    regime_rows.append({"target": target, "regime": regime, "model": name,
                                        "mae": float(err[mask].mean()), "share": float(mask.mean()),
                                        "cut_pp": float(cut)})

    pd.DataFrame(dm_rows).to_csv(OUT_DIR / "diebold_mariano.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(OUT_DIR / "skill_ci.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(OUT_DIR / "regimes.csv", index=False)
    imp = pd.DataFrame(importance_rows)
    imp.to_csv(OUT_DIR / "lightgbm_importance.csv", index=False)
    (OUT_DIR / "data_summary.json").write_text(json.dumps(data_info, indent=2))

    print(json.dumps(data_info, indent=2))
    print(pd.DataFrame(dm_rows).round(4).to_string())
    print(pd.DataFrame(ci_rows).round(3).to_string())
    print(pd.DataFrame(regime_rows).pivot_table(index=["target", "regime"], columns="model",
                                                  values="mae").round(3).to_string())
    print(imp.groupby(["target", "group"]).gain_share.sum().unstack().round(3).to_string())
    for target, g in imp.groupby("target"):
        print(target, g.nlargest(8, "gain_share")[["feature", "gain_share"]].round(3).to_dict("records"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
