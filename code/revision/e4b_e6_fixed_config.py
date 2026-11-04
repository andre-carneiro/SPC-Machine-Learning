"""
E4b - 20 Monte Carlo replications (tri_rep1..20) with the selected configuration FIXED per period
      (first: LGBM, biweekly candidates; second: LGBM, monthly; third: LGBM, monthly), re-running only the
      data-dependent steps (correlation filter, tuning, Boruta) -> stability of the selected review instants.
E6  - 20 stratified train-test partitions (split seeds 1..20) of the manuscript dataset, same fixed configuration.
Both report the selected instants, CV AUROC and the AUROC on the independent test set (100,000 projects).
Usage: .venv/bin/python code/revision/e4b_e6_fixed_config.py e4b|e6
Output: results/E4_mc_replicates/fixed_config.csv, results/E6_splits/splits.csv (resumable)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403

FIXED = {"first": ("lgbm", 10), "second": ("lgbm", 20), "third": ("lgbm", 20)}

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "e4b":
        jobs = [(f"tri_rep{r}", 42) for r in range(1, 21)]
        out = os.path.join(RESULTS, "E4_mc_replicates", "fixed_config.csv")
    else:
        jobs = [("reference", s) for s in range(1, 21)]
        out = os.path.join(RESULTS, "E6_splits", "splits.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    df_test = load_dataset("test_triangular")
    y_test = df_test.late_project.astype(int).values
    from sklearn.metrics import roc_auc_score
    for name, seed in jobs:
        df = None
        for period, (model, k) in FIXED.items():
            if len(done) and ((done.dataset == name) & (done.split_seed == seed) & (done.period == period)).any():
                continue
            df = load_dataset(name) if df is None else df
            r = run_candidate(df, period, model, k, split_seed=seed)
            feats = ["period_" + f for f in r["features_final"].split(",")]
            _, _, p = fit_predict(df, feats, eval(r["best_params_final"], {"np": np}), seed, df_test)
            row = {"dataset": name, "split_seed": seed, "period": period, "model": model, "k": k,
                   "features_intermediate": r["features_intermediate"], "review_periods": r["features_final"],
                   "n": r["n_final"], "cv_final": r["cv_final"], "auc_test_100k": roc_auc_score(y_test, p),
                   "seconds": r["seconds"]}
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(out, index=False)
            print(mode, name, seed, period, row["review_periods"], round(row["auc_test_100k"], 4), flush=True)
