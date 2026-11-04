"""Vector versions of Figures 3-5 (R2-m7): ECDF of the total duration, DPI distribution per day, DPI median
by outcome, lag correlations of the DPI, and boxplot of the configuration experiment.

Same data and statistics as the original plotly figures of code/original_pipeline/1_EDA.ipynb (cells 19-36)
and 2_model_analyses.ipynb (cell 5); only the rendering changes (matplotlib, PDF, larger fonts).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

V2 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
OUT = [os.path.join(V2, "figures"), os.path.join(V2, "manuscript", "Paper", "Images")]
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 11})


def save(fig, name):
    fig.tight_layout()
    for d in OUT:
        fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT[0], name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def clean(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)


df = pd.read_parquet(os.path.join(V2, "data", "processed", "reference.parquet"))
periods = [c for c in df.columns if c.startswith("period_")]
days = np.array([int(c.split("_")[1]) for c in periods])

# (a) ECDF of the total duration
d = np.sort(df.total_duration.values)
ecdf = np.arange(1, len(d) + 1) / len(d)
p300, p320 = (d <= 300).mean(), (d <= 320).mean()
print("P(D<=300) =", round(p300, 3), "P(D<=320) =", round(p320, 3))
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.step(d, ecdf, where="post", color="black", linewidth=1.4)
for x, p in [(300, p300), (320, p320)]:
    ax.plot([x], [p], marker="+", color="black", markersize=11, markeredgewidth=1.6)
    ax.annotate(f"({x}, {p:.3f})", (x, p), xytext=(-8, 6), textcoords="offset points", ha="right", fontsize=10)
ax.set_xlabel("Total duration $D$ (days)")
ax.set_ylabel("Cumulative probability")
clean(ax)
save(fig, "ecdf_total_duration")

# (b) DPI distribution per day: median with min-max range
X = df[periods]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
med, lo, hi = X.median().values, X.min().values, X.max().values
ax.vlines(days, lo, hi, color="#8C8C8C", linewidth=0.8)
ax.plot(days, med, ".", color="black", markersize=3)
ax.set_xlabel("Day $t$")
ax.set_ylabel("$DPI_t$ (median and range)")
clean(ax)
save(fig, "dpi_distribution")

# (c) DPI median by outcome
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(days, X[~df.late_project].median().values, color="black", linewidth=1.6, label="On-time")
ax.plot(days, X[df.late_project].median().values, color="black", linewidth=1.6, linestyle="--", label="Late")
ax.set_xlabel("Day $t$")
ax.set_ylabel("Median of $DPI_t$")
ax.legend(frameon=False, title="Outcome")
clean(ax)
save(fig, "dpi_median_outcome")

# (d) correlation between DPI_t and DPI_{t-lag}, t <= 300
fig, ax = plt.subplots(figsize=(5.2, 3.8))
styles = {1: ("black", "-"), 5: ("black", "--"), 10: ("#7F7F7F", "-"), 20: ("#7F7F7F", "--")}
names = {1: "1 (daily)", 5: "5 (weekly)", 10: "10 (biweekly)", 20: "20 (monthly)"}
for lag, (col, ls) in styles.items():
    t = np.arange(lag + 1, 301)
    r = [df[f"period_{i}"].corr(df[f"period_{i - lag}"]) for i in t]
    ax.plot(t, r, color=col, linestyle=ls, linewidth=1.4, label=names[lag])
ax.set_xlabel("Day $t$")
ax.set_ylabel("corr($DPI_t$, $DPI_{t-\\mathrm{lag}}$)")
ax.legend(frameon=False, title="Lag (days)", fontsize=9, title_fontsize=9, loc="lower right")
clean(ax)
save(fig, "correlation_lag")

# boxplot of the configuration experiment (test AUROC of the 576 configurations per period)
exp = pd.read_csv(os.path.join(V2, "data", "raw", "df_exp_simulation.csv"))
order = ["dt", "rf", "xgb", "lgbm"]
fills = {"dt": "black", "rf": "#696969", "xgb": "#A0A0A0", "lgbm": "#DCDCDC"}
titles = {"(0-100]": "First period, $t\\in(0,100]$", "(100-200]": "Second period, $t\\in(100,200]$",
          "(200-300]": "Third period, $t\\in(200,300]$"}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
for ax, (p, title) in zip(axes, titles.items()):
    g = exp[exp.Period == p]
    bp = ax.boxplot([g[g.model == m].score_test for m in order], patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=1.4),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgecolor="#555555"))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(fills[m]); patch.set_edgecolor("black")
    bp["medians"][3].set_color("black")
    ax.set_xticks(range(1, 5), [m.upper() for m in order])
    ax.set_title(title)
    ax.set_xlabel("Model ($H_{12}$)")
    clean(ax)
axes[0].set_ylabel("Test AUROC")
save(fig, "boxplot_models_performance")
print("ok")
