"""
Summarise the 10-seed comparison between DI-GNN and the baseline GNN (CIN10_COUT3).

Reads results/eval_results_10seeds.json and writes:
    results/summary_10seeds.csv    mean and standard deviation per model and metric
    figures/seed_comparison.png    paired per-seed scores

Usage
-----
    python scripts/summarize_seeds.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(42, 52)
METRICS = {
    "balanced_accuracy": "Balanced accuracy",
    "f1": "F1 score",
    "recall": "Recall",
    "precision": "Precision",
    "auc_roc": "AUC-ROC",
    "avg_precision": "Average precision",
}


def main():
    res = json.loads((ROOT / "results" / "eval_results_10seeds.json").read_text())
    scores = {m: {k: np.array([res[f"{m}_cin10_cout3_seed{s}"][k] for s in SEEDS])
                  for k in METRICS} for m in ("dignn", "baseline")}

    rows = []
    for k, label in METRICS.items():
        a, b = scores["dignn"][k], scores["baseline"][k]
        rows.append({
            "metric": k,
            "dignn_mean": a.mean(), "dignn_std": a.std(ddof=1),
            "baseline_mean": b.mean(), "baseline_std": b.std(ddof=1),
            "dignn_wins": int((a > b).sum()),
            "wilcoxon_p": wilcoxon(a, b).pvalue,
        })
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "summary_10seeds.csv", index=False, float_format="%.4f")
    print(table.round(3).to_string(index=False))

    shown = ["balanced_accuracy", "f1", "auc_roc", "avg_precision"]
    fig, axes = plt.subplots(1, len(shown), figsize=(12, 3.6))
    for ax, k in zip(axes, shown):
        a, b = scores["baseline"][k], scores["dignn"][k]
        for x0, x1 in zip(a, b):
            ax.plot([0, 1], [x0, x1], color="#B0B7BF", lw=1, zorder=1)
        ax.scatter(np.zeros(len(a)), a, color="#5D6D7E", s=28, zorder=2)
        ax.scatter(np.ones(len(b)), b, color="#1A5276", s=28, zorder=2)
        ax.set_xticks([0, 1], ["Baseline GNN", "DI-GNN"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(METRICS[k], fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Paired scores over 10 seeds, CIN10_COUT3, period 2023-2024", fontsize=12)
    fig.tight_layout()
    out = ROOT / "figures" / "seed_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
