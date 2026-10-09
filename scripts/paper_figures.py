"""Draw every results figure used in the paper and the deck.

Reads the CSVs written by ``compare_models.py`` (with and without ``--raw``)
and ``paper_analysis.py``; writes PDFs to ``paper/figures/`` for LaTeX and
PNGs to ``paper/figures/png/`` for slides.

    python scripts/paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "paper" / "figures"
PNG = OUT / "png"

MODELS = ["LightGBM", "Random Forest", "Ridge Regression"]
PERSIST = "Persistence (reference)"
COLORS = {"LightGBM": "#2a6fdb", "Random Forest": "#e08a2c", "Ridge Regression": "#3aa07e",
          PERSIST: "#9aa0a6", "Persistence": "#9aa0a6"}
TARGETS = ["cpu_percent_short", "cpu_percent_medium", "mem_percent_short", "mem_percent_medium"]
LABELS = ["CPU 45 s", "CPU 210 s", "Memory 45 s", "Memory 210 s"]


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(PNG / f"{name}.png", dpi=200)
    plt.close(fig)


def _clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _mae(df: pd.DataFrame, model: str, target: str) -> float:
    return df[(df.model == model) & (df.target == target)].mae.item()


def holdout_mae(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.1))
    x, w = np.arange(4), 0.2
    for i, m in enumerate(MODELS + [PERSIST]):
        ax.bar(x + (i - 1.5) * w, [_mae(d, m, t) for t in TARGETS], w,
               label=m.replace(" (reference)", ""), color=COLORS[m])
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Holdout MAE (pp)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _clean(ax)
    _save(fig, "holdout_mae")


def relative_error(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.3))
    x, w = np.arange(4), 0.26
    top = 0.0
    for i, m in enumerate(MODELS):
        r = [_mae(d, m, t) / _mae(d, PERSIST, t) for t in TARGETS]
        top = max(top, *r)
        ax.bar(x + (i - 1) * w, r, w, label=m, color=COLORS[m])
        for xi, ri in zip(x + (i - 1) * w, r):
            ax.text(xi, ri + 0.04, f"{ri:.2f}", ha="center", fontsize=6.5)
    ax.axhline(1, color="k", ls="--", lw=0.9, label="Persistence (= 1.0)")
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("MAE ÷ persistence MAE\n(lower is better)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _clean(ax)
    _save(fig, "relative_error")


def preprocessing_effect(raw: pd.DataFrame, pre: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    x, w = np.arange(4), 0.36
    for ax, m in zip(axes, MODELS):
        before = [_mae(raw, m, t) for t in TARGETS]
        after = [_mae(pre, m, t) for t in TARGETS]
        ax.bar(x - w / 2, before, w, color="#c9ced6", label="Raw features")
        ax.bar(x + w / 2, after, w, color=COLORS[m], label="Preprocessed")
        ax.set_title(m, fontsize=9)
        ax.set_xticks(x, ["CPU\n45 s", "CPU\n210 s", "Mem\n45 s", "Mem\n210 s"], fontsize=7)
        ax.tick_params(labelsize=7)
        _clean(ax)
        ax.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel("Holdout MAE (pp)", fontsize=8)
    _save(fig, "preprocessing_effect")


def cv_folds(cv: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for ax, t, lab in zip(axes, TARGETS, LABELS):
        for m in MODELS + [PERSIST]:
            s = cv[(cv.target == t) & (cv.model == m)].sort_values("fold")
            ax.plot(s.fold + 1, s.mae, marker="o", ms=3, lw=1, color=COLORS[m],
                    ls="--" if m == PERSIST else "-", label=m.replace(" (reference)", ""))
        ax.set_title(lab, fontsize=8)
        ax.set_xticks([1, 2, 3, 4])
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
        _clean(ax)
    axes[0].set_ylabel("MAE (pp)", fontsize=8)
    fig.supxlabel("Walk-forward fold", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "cv_folds.pdf")
    fig.savefig(PNG / "cv_folds.png", dpi=200)
    plt.close(fig)


def cost(d: pd.DataFrame) -> None:
    agg = d[d.model != PERSIST].groupby("model")[["latency_ms", "fit_seconds", "size_kb"]].mean().loc[MODELS]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
    for ax, (col, title) in zip(axes, [("latency_ms", "Latency / prediction (ms)"),
                                       ("fit_seconds", "Fit time (s)"), ("size_kb", "Model size (kB)")]):
        v = agg[col]
        ax.bar(range(3), v, color=[COLORS[m] for m in MODELS])
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(3), ["LGBM", "RF", "Ridge"], fontsize=7)
        ax.tick_params(labelsize=7)
        for i, val in enumerate(v):
            ax.text(i, val * 1.15, f"{val:.2f}" if val < 10 else f"{val:,.0f}", ha="center", fontsize=6.5)
        ax.set_ylim(v.min() / 3, v.max() * 4)
        _clean(ax)
    _save(fig, "cost")


def regimes(r: pd.DataFrame) -> None:
    models = MODELS + ["Persistence"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x, w = np.arange(4), 0.2
    for ax, reg, title in zip(axes, ["stable", "moving"],
                              ["Stable: smallest 75% of |change|", "Moving: largest 25% of |change|"]):
        for i, m in enumerate(models):
            v = [r[(r.target == t) & (r.regime == reg) & (r.model == m)].mae.item() for t in TARGETS]
            ax.bar(x + (i - 1.5) * w, v, w, color=COLORS[m], label=m)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(x, LABELS, fontsize=7)
        ax.tick_params(labelsize=7)
        _clean(ax)
    axes[0].set_ylabel("Holdout MAE (pp)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "regimes.pdf")
    fig.savefig(PNG / "regimes.png", dpi=200)
    plt.close(fig)


def importance(imp: pd.DataFrame) -> None:
    g = imp.groupby(["target", "group"]).gain_share.sum().unstack().reindex(TARGETS).fillna(0)
    order = ["Rate of change", "Current values / other", "Rolling-window stats", "Lagged values", "Process activity"]
    colors = ["#c0392b", "#34495e", "#16a085", "#f39c12", "#8e44ad"]
    fig, ax = plt.subplots(figsize=(7, 2.4))
    left = np.zeros(4)
    for grp, c in zip(order, colors):
        v = g[grp].to_numpy() if grp in g else np.zeros(4)
        ax.barh(LABELS, v, left=left, color=c, label=grp.replace("Current values / other", "Current value"))
        for yi, (lv, vv) in enumerate(zip(left, v)):
            if vv > 0.06:
                ax.text(lv + vv / 2, yi, f"{vv:.0%}", ha="center", va="center", color="white", fontsize=7)
        left += v
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of LightGBM total split gain", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.45, 1.22))
    _clean(ax)
    _save(fig, "importance")


def main() -> int:
    PNG.mkdir(parents=True, exist_ok=True)
    pre = pd.read_csv(RESULTS / "model_comparison.csv")
    raw = pd.read_csv(RESULTS / "model_comparison_raw.csv")
    holdout_mae(pre)
    relative_error(pre)
    preprocessing_effect(raw, pre)
    cv_folds(pd.read_csv(RESULTS / "model_comparison_cv.csv"))
    cost(pre)
    regimes(pd.read_csv(RESULTS / "paper" / "regimes.csv"))
    importance(pd.read_csv(RESULTS / "paper" / "lightgbm_importance.csv"))
    for name in ("timeseries", "distributions"):
        src = RESULTS / "paper" / f"{name}.pdf"
        if src.exists():
            (OUT / f"{name}.pdf").write_bytes(src.read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
