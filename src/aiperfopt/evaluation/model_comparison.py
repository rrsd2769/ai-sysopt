"""Compare three ML forecasters head-to-head on real usage data and pick one.

Candidates (all regress the same delta target -- ``target - current_value``
-- on the same feature matrix, exactly as the deployed forecaster does; see
:class:`aiperfopt.forecasting.model.MetricForecaster`'s ``fit`` docstring):

* **LightGBM** -- gradient-boosted trees, the project's production model
  (ADL-13), with its production hyperparameters unchanged.
* **Random Forest** -- bagged trees; the classic non-boosted ensemble.
* **Ridge regression** -- a regularised linear model. Needs imputation +
  scaling (trees handle NaN lag features natively, a linear model can't).

Protocol (identical for every candidate):

1. Real accumulated usage history, split chronologically: oldest 80% train,
   newest 20% held out (``build_real_usage_training_data``) -- the same split
   ``aiperfopt evaluate`` uses.
2. System + process features (the full, deployed feature set), then
   preprocessing (:mod:`aiperfopt.features.preprocessing`): invalid and
   warm-up rows are dropped, byte-rate features log-transformed, constant and
   redundant features removed. The column decisions are fitted on each
   training fold only and reused unchanged on its validation/holdout rows.
   ``--raw`` skips this step (the original, unprocessed comparison).
3. Walk-forward CV (``TimeSeriesSplit``, ``forecasting.walk_forward_splits``
   folds) on the training portion, then one refit on all training rows and
   a score on the untouched holdout.
4. Naive persistence ("no change") is scored alongside as the reference
   every model must beat -- it is not a candidate.

Selection rule (fixed *before* looking at results, so the choice can't be
fitted to the numbers):

    skill = 1 - model_MAE / persistence_MAE, per target, on the holdout.
    Winner = highest mean skill across the four targets. If two candidates
    are within 0.02 mean skill, the one with lower per-prediction latency
    wins (the forecaster runs every tick, so cost matters when accuracy
    doesn't separate them).

    aiperfopt compare          # with preprocessing
    aiperfopt compare --raw    # without

Writes ``results/model_comparison{,_raw}.csv``, ``..._cv.csv``, ``.txt`` and
``.png``.
"""

from __future__ import annotations

import csv
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aiperfopt.bootstrap import AppContext
from aiperfopt.features import FeatureEngineeringConfig, compute_feature_matrix
from aiperfopt.features.preprocessing import FeaturePreprocessor, clean_supervised_rows
from aiperfopt.features.storage_bridge import aligned_process_summary
from aiperfopt.forecasting.dataset import build_supervised_dataset
from aiperfopt.forecasting.evaluation import (
    compute_directional_accuracy,
    compute_mae_rmse,
    feature_columns_of,
)
from aiperfopt.forecasting.model import MetricForecaster
from aiperfopt.lifecycle.training_data import (
    REAL_USAGE_SCENARIO_NAME,
    build_real_usage_training_data,
)

METRICS = ("cpu_percent", "mem_percent")
TIE_MARGIN = 0.02
LATENCY_SAMPLE_ROWS = 200


class LightGBMCandidate:
    name = "LightGBM"

    def __init__(self, horizon_label: str, horizon_seconds: int) -> None:
        self._model = MetricForecaster(
            metric_name="candidate", horizon_label=horizon_label,
            horizon_seconds=horizon_seconds, model_version="compare",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict_delta(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.booster.predict(X.astype(float))

    def size_bytes(self) -> int:
        return len(self._model.booster.model_to_string().encode())


class SklearnCandidate:
    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self._estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._estimator.fit(X.astype(float), y)

    def predict_delta(self, X: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict(X.astype(float))

    def size_bytes(self) -> int:
        return len(pickle.dumps(self._estimator))


class Preprocessed:
    """Fits a :class:`FeaturePreprocessor` on whatever rows the candidate is fitted on.

    Wrapping at this level (rather than preprocessing the matrix once up
    front) is what keeps walk-forward CV honest: each fold's column
    decisions see only that fold's training rows.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name
        self.preprocessor: FeaturePreprocessor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.preprocessor = FeaturePreprocessor().fit(X)
        self.inner.fit(self.preprocessor.transform(X), y)

    def predict_delta(self, X: pd.DataFrame) -> np.ndarray:
        return self.inner.predict_delta(self.preprocessor.transform(X))

    def size_bytes(self) -> int:
        return self.inner.size_bytes()


def make_candidates(horizon_label: str, horizon_seconds: int, preprocess: bool = True) -> list:
    candidates = [
        LightGBMCandidate(horizon_label, horizon_seconds),
        SklearnCandidate(
            "Random Forest",
            RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=0,
            ),
        ),
        SklearnCandidate(
            "Ridge Regression",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        ),
    ]
    return [Preprocessed(c) for c in candidates] if preprocess else candidates


CANDIDATE_NAMES = ("LightGBM", "Random Forest", "Ridge Regression")


@dataclass
class Score:
    target: str
    model: str
    mae: float
    rmse: float
    directional_accuracy: float
    fit_seconds: float | None = None
    latency_ms: float | None = None
    size_kb: float | None = None


def _score(current, truth, predicted, threshold) -> tuple[float, float, float]:
    mae, rmse = compute_mae_rmse(truth, predicted)
    return mae, rmse, compute_directional_accuracy(current, truth, predicted, threshold)


def _single_row_latency_ms(candidate, X: pd.DataFrame) -> float:
    """Median wall time for one-row prediction -- what the live loop pays each tick."""
    rows = X.iloc[: min(LATENCY_SAMPLE_ROWS, len(X))]
    timings = []
    for i in range(len(rows)):
        start = time.perf_counter()
        candidate.predict_delta(rows.iloc[[i]])
        timings.append(time.perf_counter() - start)
    return float(np.median(timings) * 1000)


def evaluate_target(train: pd.DataFrame, holdout: pd.DataFrame, target: str, horizon_label: str,
                    horizon_seconds: int, n_splits: int, threshold: float, preprocess: bool):
    feature_cols = feature_columns_of(train)
    cv_rows: list[dict] = []

    for fold, (tr_idx, te_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(train)):
        tr, te = train.iloc[tr_idx], train.iloc[te_idx]
        for candidate in make_candidates(horizon_label, horizon_seconds, preprocess):
            candidate.fit(tr[feature_cols], tr["target"] - tr["current_value"])
            predicted = candidate.predict_delta(te[feature_cols]) + te["current_value"].to_numpy()
            mae, rmse, da = _score(te["current_value"], te["target"], predicted, threshold)
            cv_rows.append({"target": target, "fold": fold, "model": candidate.name,
                            "mae": mae, "rmse": rmse, "directional_accuracy": da})
        mae, rmse, da = _score(te["current_value"], te["target"], te["current_value"], threshold)
        cv_rows.append({"target": target, "fold": fold, "model": "Persistence (reference)",
                        "mae": mae, "rmse": rmse, "directional_accuracy": da})
        print(f"    fold {fold} done")

    holdout_scores: list[Score] = []
    for candidate in make_candidates(horizon_label, horizon_seconds, preprocess):
        start = time.perf_counter()
        candidate.fit(train[feature_cols], train["target"] - train["current_value"])
        fit_seconds = time.perf_counter() - start
        predicted = candidate.predict_delta(holdout[feature_cols]) + holdout["current_value"].to_numpy()
        mae, rmse, da = _score(holdout["current_value"], holdout["target"], predicted, threshold)
        holdout_scores.append(Score(
            target, candidate.name, mae, rmse, da, fit_seconds,
            _single_row_latency_ms(candidate, holdout[feature_cols]), candidate.size_bytes() / 1024,
        ))
    mae, rmse, da = _score(holdout["current_value"], holdout["target"], holdout["current_value"], threshold)
    holdout_scores.append(Score(target, "Persistence (reference)", mae, rmse, da))
    return holdout_scores, cv_rows


def choose_winner(scores: list[Score]) -> tuple[str, dict[str, float], dict[str, float], str]:
    persistence = {s.target: s.mae for s in scores if s.model == "Persistence (reference)"}
    skill: dict[str, float] = {}
    latency: dict[str, float] = {}
    for name in CANDIDATE_NAMES:
        mine = [s for s in scores if s.model == name]
        skill[name] = float(np.mean([1 - s.mae / persistence[s.target] for s in mine]))
        latency[name] = float(np.mean([s.latency_ms for s in mine]))

    ranked = sorted(CANDIDATE_NAMES, key=lambda n: skill[n], reverse=True)
    best, runner_up = ranked[0], ranked[1]
    if skill[best] - skill[runner_up] < TIE_MARGIN and latency[runner_up] < latency[best]:
        reason = (f"{runner_up} is within {TIE_MARGIN} mean skill of {best} and has lower latency, "
                  f"so the tie-break selects it.")
        return runner_up, skill, latency, reason
    reason = (f"{best} has the highest mean skill over persistence "
              f"({skill[best]:+.3f} vs {runner_up} {skill[runner_up]:+.3f}).")
    return best, skill, latency, reason


def _write_outputs(
    scores: list[Score], cv_rows: list[dict], dataset_info: dict, stem: str, out_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(Score.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(s.__dict__ for s in scores)
    with (out_dir / f"{stem}_cv.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cv_rows[0]))
        writer.writeheader()
        writer.writerows(cv_rows)

    winner, skill, latency, reason = choose_winner(scores)
    cv = pd.DataFrame(cv_rows)
    title = "Three-model forecaster comparison (real usage data)"
    title += " -- with preprocessing" if stem == "model_comparison" else " -- raw features, no preprocessing"
    lines = [title, "=" * len(title)]
    lines += [f"{k:<22}: {v}" for k, v in dataset_info.items()]
    lines += ["", "Holdout MAE (lower is better)"]
    for target in dict.fromkeys(s.target for s in scores):
        cells = ", ".join(f"{s.model}={s.mae:.3f}" for s in scores if s.target == target)
        lines.append(f"  {target:<20} {cells}")
    lines += ["", "Walk-forward CV MAE, mean +/- std across folds"]
    for (target, model), grp in cv.groupby(["target", "model"], sort=False):
        lines.append(f"  {target:<20} {model:<24} {grp['mae'].mean():.3f} +/- {grp['mae'].std():.3f}")
    lines += ["", "Mean skill vs persistence (1 - MAE/persistence MAE; higher is better)"]
    lines += [f"  {n:<18} {skill[n]:+.3f}   latency {latency[n]:.2f} ms/prediction" for n in CANDIDATE_NAMES]
    lines += ["", f"SELECTED: {winner}", f"reason  : {reason}"]
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    targets = list(dict.fromkeys(s.target for s in scores))
    models = list(CANDIDATE_NAMES) + ["Persistence (reference)"]
    colors = ["#2a6fdb", "#e0892b", "#3aa17e", "#9aa0a6"]
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [next(s.mae for s in scores if s.target == t and s.model == model) for t in targets]
        ax.bar(np.arange(len(targets)) + (i - 1.5) * width, values, width, label=model, color=color)
    ax.set_xticks(np.arange(len(targets)), targets)
    ax.set_ylabel("Holdout MAE (percentage points)")
    ax.set_title("Forecaster comparison on held-out real usage (lower is better)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    plt.close(fig)
    return [out_dir / f"{stem}{suffix}" for suffix in (".csv", "_cv.csv", ".txt", ".png")]


def clean_rows(dataset: pd.DataFrame, feature_config: FeatureEngineeringConfig):
    """Row-level preprocessing: warm-up is the longest lookback window."""
    return clean_supervised_rows(
        dataset, gap_seconds=feature_config.max_gap_seconds,
        warmup_seconds=max(feature_config.lookback_windows_seconds),
    )


@dataclass
class ModelComparisonResult:
    winner: str
    reason: str
    skill: dict[str, float]
    scores: list[Score] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)


def run_model_comparison(
    ctx: AppContext,
    out_dir: Path,
    *,
    preprocess: bool = True,
    feature_config: FeatureEngineeringConfig | None = None,
) -> ModelComparisonResult:
    """Run the comparison on the recorded usage in ``ctx`` and write the result files to ``out_dir``."""
    stem = "model_comparison" if preprocess else "model_comparison_raw"
    settings = ctx.settings.forecasting
    feature_config = feature_config or FeatureEngineeringConfig()

    print("Loading real usage data ...")
    scenarios, holdout_samples = build_real_usage_training_data(ctx.storage_engine, holdout_fraction=0.2)
    train_samples = scenarios[REAL_USAGE_SCENARIO_NAME]
    tol = feature_config.tolerance_seconds
    print("Computing features (system + process) ...")
    train_fm = compute_feature_matrix(
        train_samples, feature_config, aligned_process_summary(ctx.storage_engine, train_samples, tol))
    holdout_fm = compute_feature_matrix(
        holdout_samples, feature_config, aligned_process_summary(ctx.storage_engine, holdout_samples, tol))

    horizons = (("short", settings.short_horizon_target_seconds),
                ("medium", settings.medium_horizon_target_seconds))
    scores: list[Score] = []
    cv_rows: list[dict] = []
    cleaning: dict[str, str] = {}
    for metric in METRICS:
        for horizon_label, horizon_seconds in horizons:
            target = f"{metric}_{horizon_label}"
            print(f"Evaluating {target} ...")
            train = build_supervised_dataset(train_samples, train_fm, metric, horizon_seconds, tol)
            holdout = build_supervised_dataset(holdout_samples, holdout_fm, metric, horizon_seconds, tol)
            if preprocess:
                train, train_report = clean_rows(train, feature_config)
                holdout, holdout_report = clean_rows(holdout, feature_config)
                kept = FeaturePreprocessor().fit(train[feature_columns_of(train)])
                cleaning[target] = (
                    f"train {train_report.input_rows}->{train_report.output_rows} "
                    f"(invalid {train_report.invalid_rows}, warm-up {train_report.warmup_rows}, "
                    f"{train_report.segments} segments); holdout {holdout_report.input_rows}->"
                    f"{holdout_report.output_rows} (invalid {holdout_report.invalid_rows}, warm-up "
                    f"{holdout_report.warmup_rows}); features {len(kept.input_columns)}->{len(kept.kept_columns)} "
                    f"(log {len(kept.log_columns)}, constant {len(kept.constant_columns)}, "
                    f"redundant {len(kept.redundant_columns)})"
                )
                print(f"  {cleaning[target]}")
            s, c = evaluate_target(train, holdout, target, horizon_label, horizon_seconds,
                                   settings.walk_forward_splits, settings.directional_stability_threshold,
                                   preprocess)
            scores += s
            cv_rows += c

    dataset_info = {
        "training_samples": len(train_samples),
        "holdout_samples": len(holdout_samples),
        # Wall clock, so it includes any stretch the collector was stopped.
        "window_hours_wall_clock": round((holdout_samples[-1].timestamp - train_samples[0].timestamp) / 3600, 1),
        "features": len(feature_columns_of(train)),
        "cv_folds": settings.walk_forward_splits,
    }
    for target, summary in cleaning.items():
        dataset_info[f"preprocessing {target}"] = summary
    files = _write_outputs(scores, cv_rows, dataset_info, stem, out_dir)
    winner, skill, _latency, reason = choose_winner(scores)
    return ModelComparisonResult(winner=winner, reason=reason, skill=skill, scores=scores, files=files)
