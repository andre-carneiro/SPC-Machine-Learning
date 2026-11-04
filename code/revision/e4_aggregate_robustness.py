"""
Aggregates the robustness experiments (E4 full procedure, E4b fixed configuration, E5 distributions, E6 splits,
E7 lasso) into summary tables and the stability heatmap.
  - selected instants per source and period; number of reviews; losses vs the conventional schedule (test set)
  - window stability: share of runs that select an instant within +-10 days of each reference instant
Outputs: results/E4_mc_replicates/robustness_summary.csv, robustness_windows.csv,
         figures + manuscript/Paper/Images: stability_heatmap.pdf
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, V2_DIR  # noqa: E402

REF = {"first": [40, 80, 100], "second": [160, 200], "third": [300]}
PERIODS = ["first", "second", "third"]
FIGS = [os.path.join(V2_DIR, "figures"), os.path.join(V2_DIR, "manuscript", "Paper", "Images")]


def days(s):
    return [int(x) for x in str(s).split(",") if str(x).strip() not in ("", "nan")]


def collect():
    rows = []
    ref = pd.read_csv(os.path.join(RESULTS, "E3_inference", "e3d_selection.csv"))
    for _, r in ref.iterrows():
        rows.append({"source": "Manuscript dataset", "group": "full", "run": "reference", "period": r.period,
                     "review_periods": r.review_periods, "diff_test": r.diff_test, "non_inferior": r.non_inferior_test})
    for sub, label in [*[(f"E4_mc_replicates/tri_rep{i}", f"Replication {i}") for i in range(1, 11)],
                       ("E5_distributions/gamma", "Gamma"), ("E5_distributions/cmpoisson", "CMP"), ("E5_distributions/pert", "Beta-PERT")]:
        f = os.path.join(RESULTS, sub, "selection_final.csv")
        if os.path.exists(f):
            for _, r in pd.read_csv(f).iterrows():
                rows.append({"source": label, "group": "full" if "Replication" in label else "distribution", "run": sub.split("/")[-1],
                             "period": r.period, "review_periods": r.review_periods, "diff_test": r.diff_test, "non_inferior": r.non_inferior_test})
    for f, grp, key in [(os.path.join(RESULTS, "E4_mc_replicates", "fixed_config.csv"), "fixed_replication", "dataset"),
                        (os.path.join(RESULTS, "E6_splits", "splits.csv"), "split", "split_seed")]:
        if os.path.exists(f):
            for _, r in pd.read_csv(f).iterrows():
                rows.append({"source": f"{grp} {r[key]}", "group": grp, "run": str(r[key]), "period": r.period,
                             "review_periods": r.review_periods, "diff_test": np.nan, "non_inferior": np.nan,
                             "auc_test": r.auc_test_100k})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = collect()
    d["n"] = d.review_periods.apply(lambda s: len(days(s)))
    d.to_csv(os.path.join(RESULTS, "E4_mc_replicates", "robustness_summary.csv"), index=False)

    # window stability
    w = []
    for grp, g in d.groupby("group"):
        for period in PERIODS:
            gp = g[g.period == period]
            if not len(gp):
                continue
            for ref_day in REF[period]:
                hit_exact = gp.review_periods.apply(lambda s: ref_day in days(s)).mean()
                hit_win = gp.review_periods.apply(lambda s: any(abs(x - ref_day) <= 10 for x in days(s))).mean()
                w.append({"group": grp, "period": period, "reference_day": ref_day, "runs": len(gp),
                          "share_exact": round(hit_exact, 3), "share_within_10_days": round(hit_win, 3),
                          "n_reviews_median": gp.n.median(), "n_reviews_min": gp.n.min(), "n_reviews_max": gp.n.max(),
                          "non_inferior_share": gp.non_inferior.mean() if gp.non_inferior.notna().any() else np.nan})
    pd.DataFrame(w).to_csv(os.path.join(RESULTS, "E4_mc_replicates", "robustness_windows.csv"), index=False)
    print(pd.DataFrame(w).to_string(index=False))

    # selection-frequency figure: share of runs selecting each day, on the real time axis
    groups = [("full", "Complete procedure: manuscript dataset + 10 replications (n = 11)"),
              ("distribution", "Complete procedure: Gamma, CMP, and Beta-PERT distributions (n = 3)"),
              ("fixed_replication", "Fixed configuration: 20 replications (n = 20)"),
              ("split", "Fixed configuration: 20 train-test partitions (n = 20)")]
    windows = [(40, 55), (80, 85), (95, 100), (140, 160), (195, 200), (220, 245), (295, 300)]
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 6.2), sharex=True, sharey=True)
    for ax, (g, title) in zip(axes, groups):
        gd = d[d.group == g]
        nruns = gd.run.nunique()
        counts = {}
        for s_ in gd.review_periods:
            for x in days(s_):
                counts[x] = counts.get(x, 0) + 1
        for a, b in windows:
            ax.axvspan(a - 2, b + 2, color="#EEEEEE", zorder=0)
        ax.bar(list(counts), [v / nruns for v in counts.values()], width=3.2, color="black", zorder=2)
        for b in [100, 200]:
            ax.axvline(b + 2.5, color="#777777", linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1])
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlim(0, 305); axes[-1].set_xticks(range(0, 301, 20))
    axes[-1].set_xlabel("Review period (day)")
    fig.text(0.005, 0.5, "Share of runs selecting the day", va="center", rotation=90)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    for p_ in FIGS:
        fig.savefig(os.path.join(p_, "stability_heatmap.pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS[0], "stability_heatmap.png"), dpi=200, bbox_inches="tight")
    print("ok")
