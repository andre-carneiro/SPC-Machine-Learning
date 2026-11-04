"""Figure: loss in AUROC vs number of review instants (validation set), 16 candidates per period (E3d)."""
import os
import pandas as pd
import matplotlib.pyplot as plt

V2 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
c = pd.read_csv(os.path.join(V2, "results", "E3_inference", "e3d_candidates.csv"))
sel = pd.read_csv(os.path.join(V2, "results", "E3_inference", "e3d_selection.csv"))
style = {1: ("o", "#BDBDBD", "Daily (k = 1)"), 5: ("s", "#8C8C8C", "Weekly (k = 5)"),
         10: ("^", "#4D4D4D", "Biweekly (k = 10)"), 20: ("D", "#000000", "Monthly (k = 20)")}
titles = {"first": "First period (0, 100]", "second": "Second period (100, 200]", "third": "Third period (200, 300]"}
plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, period in zip(axes, titles):
    g = c[c.period == period]
    for k, (m, col, lab) in style.items():
        h = g[g.k == k]
        ax.scatter(h.n, h.diff_val, marker=m, s=34, facecolor="white", edgecolor=col, linewidth=1.3, label=lab, zorder=3)
    s = sel[sel.period == period].iloc[0]
    ax.scatter([s.n], [s.diff_val], marker=style[int(s.k)][0], s=70, color="black", zorder=4)
    ax.annotate("selected: {" + str(s.review_periods).replace(",", ", ") + "}", (s.n, s.diff_val), xytext=(10, 12),
                textcoords="offset points", fontsize=8.5, ha="left", arrowprops=dict(arrowstyle="-", color="black", lw=0.7))
    ax.axhspan(-0.01, 0.012, color="#F2F2F2", zorder=0)
    ax.axhline(-0.01, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.set_title(titles[period], fontsize=10, pad=10)
    ax.set_xlabel("Number of review instants |M|")
    ax.set_xticks(range(1, int(c.n.max()) + 1))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
axes[0].set_ylim(c.diff_val.min() - 0.004, 0.012)
axes[0].set_ylabel("ΔAUROC = AUROC(M) − AUROC(B$^e$)\n(validation set)")
for ax in axes:
    ax.text(ax.get_xlim()[1], -0.0112, "−δ", ha="right", va="top", fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, title="Candidate frequency (filled marker = selected)", title_fontsize=9)
fig.tight_layout(rect=(0, 0.14, 1, 1))
for out in [os.path.join(V2, "figures", "tradeoff_reviews_auroc.pdf"), os.path.join(V2, "manuscript", "Paper", "Images", "tradeoff_reviews_auroc.pdf"),
            os.path.join(V2, "figures", "tradeoff_reviews_auroc.png")]:
    fig.savefig(out, dpi=200, bbox_inches="tight")
print("ok")
