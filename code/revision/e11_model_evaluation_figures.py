"""
Regenerates the Model Evaluation figures with the FINAL models of Table 4 on the independent test set
(100,000 projects): predicted-probability histograms by outcome and threshold analysis (TP/FP/TN/FN shares).
Outputs: figures + manuscript/Paper/Images: probability_distribution_final.pdf, confusion_matrix_threshold_final.pdf;
         results/E3_inference/final_models_threshold_table.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403
from sklearn.metrics import roc_auc_score  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

FIGS = [os.path.join(V2_DIR, "figures"), os.path.join(V2_DIR, "manuscript", "Paper", "Images")]
TITLES = {"first": "First period", "second": "Second period", "third": "Third period"}


def savefig(fig, name):
    for d in FIGS:
        fig.savefig(os.path.join(d, name), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS[0], name.replace(".pdf", ".png")), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    e2b = pd.read_csv(os.path.join(RESULTS, "E2_cv_selection", "e2b_boruta_all_candidates.csv"))
    sel = pd.read_csv(os.path.join(RESULTS, "E3_inference", "e3d_selection.csv"))
    df, test = load_dataset("reference"), load_dataset("test_triangular")
    y = test.late_project.astype(int).values
    probs = {}
    for _, s in sel.iterrows():
        c = e2b[(e2b.period == s.period) & (e2b.model == s.model) & (e2b.time_interval == s.k)].iloc[0]
        feats = ["period_" + f for f in str(c.features_final).split(",")]
        bp = eval(c.best_params_final, {"np": np}); bp["classifier_name"] = c.model
        _, _, probs[s.period] = fit_predict(df, feats, bp, 42, test)
        print(s.period, "AUROC", round(roc_auc_score(y, probs[s.period]), 4), flush=True)

    # probability histograms
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (period, p) in zip(axes, probs.items()):
        for cls, col, lab, hatch in [(0, "#BDBDBD", "On-time", None), (1, "#000000", "Late", None)]:
            w = np.full((y == cls).sum(), 100 / (y == cls).sum())
            ax.hist(p[y == cls], bins=bins, weights=w, histtype="step" if cls else "stepfilled",
                    color=col, linewidth=1.5, label=lab, alpha=1 if cls else 0.8)
        ax.set_title(TITLES[period], fontsize=10); ax.set_xlabel("Predicted probability of delay")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Projects of each outcome (%)"); axes[0].legend(frameon=False)
    fig.tight_layout(); savefig(fig, "probability_distribution_final.pdf")

    # threshold analysis
    rows = []
    for period, p in probs.items():
        for thr in np.round(np.arange(0.05, 1.0, 0.05), 2):
            pred = p >= thr
            rows.append({"period": period, "threshold": thr, "TP": np.mean(pred & (y == 1)), "FP": np.mean(pred & (y == 0)),
                         "TN": np.mean(~pred & (y == 0)), "FN": np.mean(~pred & (y == 1)),
                         "recall": (pred & (y == 1)).sum() / (y == 1).sum(), "precision": (pred & (y == 1)).sum() / max(pred.sum(), 1)})
    tab = pd.DataFrame(rows); tab.to_csv(os.path.join(RESULTS, "E3_inference", "final_models_threshold_table.csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    styles = {"TP": ("black", "-", "o"), "FP": ("#4D4D4D", "--", "s"), "TN": ("#8C8C8C", ":", "D"), "FN": ("#BDBDBD", "-.", "^")}
    for ax, period in zip(axes, probs):
        t = tab[tab.period == period]
        for m, (col, ls, mk) in styles.items():
            ax.plot(t.threshold, t[m], color=col, linestyle=ls, marker=mk, markersize=4, linewidth=1.4, label=m)
        ax.set_title(TITLES[period], fontsize=10); ax.set_xlabel("Decision threshold")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Share of projects"); axes[-1].legend(frameon=False, loc="center right")
    fig.tight_layout(); savefig(fig, "confusion_matrix_threshold_final.pdf")
    print(tab[tab.threshold.isin([0.1, 0.2, 0.3, 0.5])].round(3).to_string(index=False))
