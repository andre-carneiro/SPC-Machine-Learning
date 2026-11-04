"""
E10 - Interpretation of the selected review periods (R1.10).
  (1) Baseline schedule (CPM forward/backward pass) -> planned start/end of each activity and critical path.
  (2) Univariate predictive value of each day's DPI: AUROC of -DPI_t for the delay outcome on the validation set.
  (3) Figure: univariate AUROC over time with the evaluation periods, the selected review periods and the
      baseline completion events.
  (4) SHAP summary of the final models (Table 4) on 5,000 projects of the test set.
Outputs: results/E10_interpretation/{baseline_schedule,univariate_auc,events_near_selected}.csv,
         figures + manuscript/Paper/Images: interpretation_univariate_auc.pdf, shap_final_models.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403
import mc_simulator as mc  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(RESULTS, "E10_interpretation")
FIGS = [os.path.join(V2_DIR, "figures"), os.path.join(V2_DIR, "manuscript", "Paper", "Images")]
SELECTED = {"first": [40, 80, 100], "second": [160, 200], "third": [300]}


def savefig(fig, name):
    for d in FIGS:
        fig.savefig(os.path.join(d, name), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS[0], name.replace(".pdf", ".png")), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    # (1) baseline schedule and critical path
    net = mc.load_network()
    preds = mc._pred_index(net)
    ps, pe = mc._forward_pass(net.expectation.values[None, :], preds)
    ps, pe = ps[0], pe[0]
    n, T = len(net), pe.max()
    succ = [[j for j in range(n) if i in preds[j]] for i in range(n)]
    ls = np.zeros(n)
    for i in reversed(range(n)):
        lf = min([ls[j] for j in succ[i]], default=T)
        ls[i] = lf - net.expectation.values[i]
    sched = pd.DataFrame({"activity": net.activity.astype(int), "planned_start": ps, "planned_end": pe,
                          "planned_duration": net.expectation.values, "total_float": ls - ps})
    sched["critical"] = sched.total_float.abs() < 1e-9
    sched.to_csv(os.path.join(OUT, "baseline_schedule.csv"), index=False)

    # (2) univariate AUROC of each day's DPI (validation set)
    val = load_dataset("val_triangular")
    y = val.late_project.astype(int).values
    uni = pd.DataFrame({"t": range(1, 301), "auc_univariate": [roc_auc_score(y, -val[f"period_{t}"].values) for t in range(1, 301)]})
    uni.to_csv(os.path.join(OUT, "univariate_auc.csv"), index=False)

    # events within 10 days of each selected review period
    ev = []
    for period, days in SELECTED.items():
        for d in days:
            ends = sched[(sched.planned_end >= d - 10) & (sched.planned_end <= d + 10)]
            starts = sched[(sched.planned_start >= d - 10) & (sched.planned_start <= d + 10) & (sched.planned_start > 0)]
            ev.append({"period": period, "day": d, "auc_univariate": float(uni.auc_univariate[d - 1]),
                       "activities_ending_within_10d": ";".join(f"{a}@{e:.0f}" for a, e in zip(ends.activity, ends.planned_end)),
                       "activities_starting_within_10d": ";".join(f"{a}@{s:.0f}" for a, s in zip(starts.activity, starts.planned_start))})
    pd.DataFrame(ev).to_csv(os.path.join(OUT, "events_near_selected.csv"), index=False)

    # (3) figure
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(uni.t, uni.auc_univariate, color="black", linewidth=1.4, label="AUROC of $DPI_t$ alone (validation set)")
    for b in [100, 200]:
        ax.axvline(b, color="#999999", linewidth=0.8, linestyle=":")
    ends = sched.planned_end.round().astype(int).value_counts().sort_index()
    for d, c in ends.items():
        if d <= 300:
            ax.plot([d, d], [0.5, 0.5 + 0.012 * c], color="#8C8C8C", linewidth=2)
    allsel = [d for v in SELECTED.values() for d in v]
    ax.scatter(allsel, uni.auc_univariate.values[np.array(allsel) - 1], s=60, color="black", zorder=4, label="Selected review periods")
    for d in allsel:
        ax.annotate(str(d), (d, uni.auc_univariate[d - 1]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlim(0, 305); ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Review period $t$ (days)"); ax.set_ylabel("AUROC")
    for x, lab in [(50, "First evaluation period"), (150, "Second evaluation period"), (250, "Third evaluation period")]:
        ax.text(x, 1.005, lab, ha="center", va="bottom", fontsize=9, color="#4D4D4D")
    ax.plot([], [], color="#8C8C8C", linewidth=2, label="Planned activity completions (bar height = number of activities)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.94), frameon=False, fontsize=8.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    savefig(fig, "interpretation_univariate_auc.pdf")

    # (4) SHAP of the final models
    import shap
    e2b = pd.read_csv(os.path.join(RESULTS, "E2_cv_selection", "e2b_boruta_all_candidates.csv"))
    sel = pd.read_csv(os.path.join(RESULTS, "E3_inference", "e3d_selection.csv"))
    df = load_dataset("reference")
    test = load_dataset("test_triangular").sample(5000, random_state=42)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (_, s) in zip(axes, sel.iterrows()):
        c = e2b[(e2b.period == s.period) & (e2b.model == s.model) & (e2b.time_interval == s.k)].iloc[0]
        feats = ["period_" + f for f in str(c.features_final).split(",")]
        bp = eval(c.best_params_final, {"np": np}); bp["classifier_name"] = c.model
        ytr = df.late_project.astype(int).values
        tr, _ = train_test_split(np.arange(len(df)), test_size=0.25, random_state=42, stratify=ytr)
        sc = StandardScaler().fit(df[feats].values[tr])
        m = instantiate_model(dict(bp)); m.fit(sc.transform(df[feats].values[tr]), ytr[tr])
        Xs = pd.DataFrame(sc.transform(test[feats].values), columns=[f"$F_{{{f.split('_')[1]}}}$" for f in feats])
        sv = shap.TreeExplainer(m).shap_values(Xs)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
        plt.sca(ax)
        shap.summary_plot(sv, Xs, show=False, plot_size=None, color_bar=False, cmap="Greys")
        ax.set_title({"first": "First period", "second": "Second period", "third": "Third period"}[s.period], fontsize=10)
        ax.set_xlabel("SHAP value (log-odds of delay)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.2, 0.012, 0.6])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="Greys"), cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("DPI value", fontsize=9)
    savefig(fig, "shap_final_models.pdf")
    print("ok")
