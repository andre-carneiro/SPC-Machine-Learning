"""Figure: baseline schedule (Gantt chart) of the EPC project, with total float from a CPM pass.

The network and planned durations are those of Table S2 (Supporting Information); the baseline
start/finish dates are checked against data/raw/gantt_chart.csv (the source of the original figure).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

V2 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

# activity: (predecessors, planned duration PD_i); dummies 1, 9, 19 (start of each discipline) and 36 (end) omitted
NET = {
    2: ([], 15), 3: ([2], 30), 4: ([2], 30), 5: ([3, 4], 55), 6: ([3], 70), 7: ([3], 90), 8: ([2], 70),
    10: ([6], 25), 11: ([6], 85), 12: ([7], 85), 13: ([8], 80), 14: ([8], 110), 15: ([8], 80),
    16: ([13, 15], 30), 17: ([16], 15), 18: ([2], 190),
    20: ([3], 55), 21: ([20], 60), 22: ([4], 55), 23: ([5, 22], 95), 24: ([21], 25), 25: ([24], 20),
    26: ([21], 40), 27: ([24], 40), 28: ([10], 45), 29: ([24, 28], 85), 30: ([11, 25, 26], 80),
    31: ([12, 27], 60), 32: ([14, 23], 15), 33: ([32, 17, 18], 60), 34: ([29, 33], 15), 35: ([34, 30, 31], 15),
}
DISCIPLINE = {**{a: "Engineering" for a in range(2, 9)}, **{a: "Procurement" for a in range(10, 19)},
              **{a: "Construction" for a in range(20, 36)}}

# forward pass
es, ef = {}, {}
for a in sorted(NET):
    preds, d = NET[a]
    es[a] = max([ef[p] for p in preds], default=0)
    ef[a] = es[a] + d
bpd = max(ef.values())
# backward pass
succ = {a: [b for b in NET if a in NET[b][0]] for a in NET}
lf, ls = {}, {}
for a in sorted(NET, reverse=True):
    lf[a] = min([ls[s] for s in succ[a]], default=bpd)
    ls[a] = lf[a] - NET[a][1]
tf = {a: ls[a] - es[a] for a in NET}

# consistency with the source of the original figure
g = pd.read_csv(os.path.join(V2, "data", "raw", "gantt_chart.csv"), sep=";", encoding="utf-8-sig").dropna(subset=["baseline_start"])
g = g[g.baseline_start != "NaT"]
for _, r in g.iterrows():
    a = int(r.activity)
    assert (es[a], ef[a]) == (int(r.baseline_start.split()[0]), int(r.baseline_end.split()[0])), a
assert bpd == 300 and sum(d for _, d in NET.values()) == 1825
print("BPD =", bpd, "| activities with positive total float:", {a: f for a, f in tf.items() if f > 0})

plt.rcParams.update({"font.family": "serif", "font.size": 9})
acts = sorted(NET)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for y, a in enumerate(acts):
    ax.barh(y, NET[a][1], left=es[a], height=0.62, color="black" if tf[a] == 0 else "#7F7F7F", zorder=3)
    if tf[a] > 0:
        ax.barh(y, tf[a], left=ef[a], height=0.62, color="#D0D0D0", zorder=3)
for c in (100, 200, 300):
    ax.axvline(c, color="black", linestyle="--", linewidth=0.8, zorder=2)
ax.set_yticks(range(len(acts)))
ax.set_yticklabels(acts, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlim(0, 310)
ax.set_xticks(range(0, 301, 20))
ax.set_xlabel("Day of the baseline schedule")
ax.set_ylabel("Activity")
ax.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
# discipline labels
for name, (lo, hi) in {"Engineering": (2, 8), "Procurement": (10, 18), "Construction": (20, 35)}.items():
    y0, y1 = acts.index(lo), acts.index(hi)
    ax.text(311, (y0 + y1) / 2, name, rotation=270, va="center", ha="left", fontsize=8)
    if lo != 2:
        ax.axhline(y0 - 0.5, color="#BDBDBD", linewidth=0.6, zorder=1)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="black", label="Zero total float"), Patch(color="#7F7F7F", label="Positive total float"),
                   Patch(color="#D0D0D0", label="Total float")], loc="lower left", frameon=False, fontsize=8)
fig.tight_layout()
for out in [os.path.join(V2, "figures", "gantt_chart.pdf"), os.path.join(V2, "figures", "gantt_chart.png"),
            os.path.join(V2, "manuscript", "Paper", "Images", "gantt_chart.pdf")]:
    fig.savefig(out, dpi=300, bbox_inches="tight")
print("ok")
