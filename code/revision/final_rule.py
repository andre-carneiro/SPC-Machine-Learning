"""
Final selection rule (decisions D13, D17) for any training dataset, reusing its candidates.csv
(16 candidates per period: tuning + Boruta already done by pipeline_v2.run_dataset).

For each candidate subset M (frequency k), the benchmark is the conventional schedule of the same frequency:
every k days of the period, no correlation filter, with the BEST of the four model families by training CV
(tuned here with the same hyperopt protocol; cached per (period, k)). Daily monitoring (k = 1) is also the
upper-bound reference reported in the appendix.
Rule V': the fewest review periods with one-sided 95% DeLong lower limit of AUROC_val(M) - AUROC_val(B) > -delta
(ties: largest difference); if none, largest difference. The test set is used once for the selected subset.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403
from scipy import stats  # noqa: E402

Z95 = stats.norm.ppf(0.95)


def candidate_days(period, k):
    start, end = PERIODS[period]
    return [f"period_{t}" for t in range(start + k, end + 1, k)]


def reference_families():
    d = load_experiments()
    g = d[(d.remove_correlation == False) & (d.apply_fa == False) & d.list_lag_features.isna() & d.apply_rebalance_strategy.isna()]
    best = g.loc[g.groupby(["Period", "time_interval"]).score_train.idxmax()]
    return {(PERIOD_LABELS[r.Period], int(r.time_interval)): r.model for _, r in best.iterrows()}


REFERENCE_FAMILY = reference_families()


class Benchmarks:
    """Best-of-four conventional schedule per (period, k), tuned on the training set, predictions on val+test."""

    def __init__(self, df, both, split_seed=42, cache_path=None):
        self.df, self.both, self.seed, self.cache_path = df, both, split_seed, cache_path
        self.info = pd.read_csv(cache_path).to_dict("records") if cache_path and os.path.exists(cache_path) else []
        self.pred = {}

    def get(self, period, k):
        key = (period, k)
        if key not in self.pred:
            row = next((r for r in self.info if r["period"] == period and int(r["k"]) == k), None)
            if row is None:
                # family = best-CV family on the manuscript dataset for this (period, k) (stable across replications;
                # checked on replication 1: 9/10 identical), tuned on this dataset with the same protocol
                families = [REFERENCE_FAMILY[(period, k)]] if REFERENCE_FAMILY else MODELS
                tuned = {m: tune(self.df, base_params(period, k, self.seed, corr=False), m) for m in families}
                best = max(tuned, key=lambda m: tuned[m]["cv"])
                row = {"period": period, "k": k, "model": best, "cv": tuned[best]["cv"],
                       "best_params": repr(tuned[best]["best_params"])}
                self.info.append(row)
                if self.cache_path:
                    pd.DataFrame(self.info).to_csv(self.cache_path, index=False)
            bp = eval(row["best_params"], {"np": np})
            _, _, p = fit_predict(self.df, candidate_days(period, k), bp, self.seed, self.both)
            self.pred[key] = (row["model"], p)
        return self.pred[key]


def select_final(df, cands, df_val, df_test, out_dir, split_seed=42, delta=DELTA):
    both = pd.concat([df_val, df_test], ignore_index=True)
    nv = len(df_val)
    y_val, y_test = df_val.late_project.astype(int).values, df_test.late_project.astype(int).values
    bench = Benchmarks(df, both, split_seed, os.path.join(out_dir, "benchmarks.csv"))
    rows = []
    for period in PERIODS:
        g = cands[cands.period == period].copy()
        evaluated, chosen = [], None
        for n in sorted(g.n_final.unique()):
            level = []
            for _, c in g[g.n_final == n].iterrows():
                feats = ["period_" + f for f in str(c.features_final).split(",")]
                bp = eval(c.best_params_final, {"np": np}) if isinstance(c.best_params_final, str) else c.best_params_final
                bp = dict(bp); bp["classifier_name"] = c.model
                _, _, p = fit_predict(df, feats, bp, split_seed, both)
                bmodel, pb = bench.get(period, int(c.k))
                v = delong_paired(y_val, p[:nv], pb[:nv], delta=delta)
                level.append({"period": period, "model": c.model, "k": int(c.k), "review_periods": c.features_final,
                              "n": int(n), "benchmark_model": bmodel, "diff_val": v["diff"],
                              "lower95_val": v["diff"] - Z95 * v["se"], "_p": p, "_pb": pb})
            evaluated += level
            ok = [r for r in level if r["lower95_val"] > -delta]
            if ok:
                chosen = max(ok, key=lambda r: r["diff_val"]); met = True
                break
        if chosen is None:
            chosen = max(evaluated, key=lambda r: r["diff_val"]); met = False
        t = delong_paired(y_test, chosen["_p"][nv:], chosen["_pb"][nv:], delta=delta)
        _, pd_daily = bench.get(period, 1)
        td = delong_paired(y_test, chosen["_p"][nv:], pd_daily[nv:], delta=delta)
        rows.append({k: v for k, v in chosen.items() if not k.startswith("_")} | {
            "constraint_met_on_validation": met, "auc_test": t["auc_1"], "auc_test_bench": t["auc_2"],
            "diff_test": t["diff"], "ci_test_low": t["ci_low"], "ci_test_high": t["ci_high"],
            "non_inferior_test": t["non_inferior"], "auc_test_daily": td["auc_2"], "diff_test_daily": td["diff"],
            "ci_test_daily_low": td["ci_low"], "ci_test_daily_high": td["ci_high"]})
        print(out_dir.split("/")[-1], period, chosen["review_periods"], round(chosen["diff_val"], 4), round(t["diff"], 4), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "selection_final.csv"), index=False)
    return out


DATASETS = {**{f"E4_mc_replicates/tri_rep{r}": (f"tri_rep{r}", "val_triangular", "test_triangular") for r in range(1, 11)},
            "E5_distributions/gamma": ("gamma_reference", "val_gamma", "test_gamma"),
            "E5_distributions/cmpoisson": ("cmp_reference", "val_cmpoisson", "test_cmpoisson"),
            "E5_distributions/pert": ("pert_train", "val_pert", "test_pert")}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(DATASETS)
    for sub in targets:
        out_dir = os.path.join(RESULTS, sub)
        cpath = os.path.join(out_dir, "candidates.csv")
        if os.path.exists(os.path.join(out_dir, "selection_final.csv")):
            continue
        if not os.path.exists(cpath) or len(pd.read_csv(cpath)) < 48:
            print("skip (candidates incomplete)", sub, flush=True)
            continue
        train, val, test = DATASETS[sub]
        select_final(load_dataset(train), pd.read_csv(cpath), load_dataset(val), load_dataset(test), out_dir)
