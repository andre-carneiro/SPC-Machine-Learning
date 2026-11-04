"""
E3d - Final selection rule (decisions D13 + 31/10 confirmation):
  benchmark of each candidate = full candidate set B^e of ITS OWN frequency k (every k days in the period),
  no correlation filter / FA / lags / rebalancing, with the BEST model family by training CV at that frequency
  (grid rows of the factorial design, hyperparameters already tuned by 5-fold CV on the training set).
  Using the same family instead would let weak families pass against equally weak benchmarks (observed with DT).
  Rule V': among the 16 candidates, fewest review periods such that the one-sided 95% DeLong lower bound
  of AUROC_val(M) - AUROC_val(B^e) > -delta (delta = 0.01), ties -> smallest loss; if none, smallest loss.
  Validation: val_triangular (100k), test: test_triangular (100k, used once). Daily monitoring is reported
  for the selected sets as the upper bound of information (appendix).
Output: results/E3_inference/e3d_candidates.csv, e3d_selection.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403
from scipy import stats  # noqa: E402

OUT = os.path.join(RESULTS, "E3_inference")


def grid_benchmark_row(df_exp, period, k):
    """Best-CV model family among DT/RF/XGB/LGBM monitoring every k days, no preprocessing."""
    g = df_exp[(df_exp.Period.map(PERIOD_LABELS) == period) & (df_exp.time_interval == k) &
               (df_exp.remove_correlation == False) & (df_exp.apply_fa == False) &
               (df_exp.list_lag_features.isna()) & (df_exp.apply_rebalance_strategy.isna())]
    assert len(g) == 4, (period, k, len(g))
    return g.score_train.idxmax()


def candidate_days(period, k):
    start, end = PERIODS[period]
    return [f"period_{t}" for t in range(start + k, end + 1, k)]


if __name__ == "__main__":
    df_exp = load_experiments()
    e2b = pd.read_csv(os.path.join(RESULTS, "E2_cv_selection", "e2b_boruta_all_candidates.csv")).rename(columns={"time_interval": "k"})
    daily = pd.read_csv(os.path.join(RESULTS, "E3_inference", "e3b_independent_test.csv")).set_index("period")
    df, df_val, df_test = load_dataset("reference"), load_dataset("val_triangular"), load_dataset("test_triangular")
    both = pd.concat([df_val, df_test], ignore_index=True)
    nv = len(df_val)
    y_val, y_test = df_val.late_project.astype(int).values, df_test.late_project.astype(int).values
    z = stats.norm.ppf(0.95)

    bench_cache, rows, sel = {}, [], []
    for period in PERIODS:
        start, end = PERIODS[period]
        _, _, p_daily = fit_predict(df, [f"period_{t}" for t in range(start + 1, end + 1)],
                                    eval(daily.loc[period, "daily_best_params"], {"np": np}), 42, both)
        for _, c in e2b[e2b.period == period].iterrows():
            key = (period, int(c.k))
            if key not in bench_cache:
                bidx = grid_benchmark_row(df_exp, *key)
                bench_cache[key + ("model",)] = df_exp.loc[bidx, "model"]
                bp = obtain_experiment_parameters(df_exp, bidx)["best_params"]
                _, _, bench_cache[key] = fit_predict(df, candidate_days(period, int(c.k)), bp, 42, both)
                bench_cache[key + ("cv",)] = float(df_exp.loc[bidx, "score_train"])
            p_b = bench_cache[key]
            feats = ["period_" + f for f in str(c.features_final).split(",")]
            bp = eval(c.best_params_final, {"np": np}); bp["classifier_name"] = c.model
            _, _, p = fit_predict(df, feats, bp, 42, both)
            v = delong_paired(y_val, p[:nv], p_b[:nv], delta=DELTA)
            t = delong_paired(y_test, p[nv:], p_b[nv:], delta=DELTA)
            td = delong_paired(y_test, p[nv:], p_daily[nv:], delta=DELTA)
            rows.append({"period": period, "model": c.model, "k": int(c.k), "n_benchmark": len(candidate_days(period, int(c.k))),
                         "benchmark_model": bench_cache[key + ("model",)],
                         "review_periods": c.features_final, "n": int(c.n_final), "cv_final": c.cv_final,
                         "cv_benchmark": bench_cache[key + ("cv",)],
                         "auc_val": v["auc_1"], "auc_val_bench": v["auc_2"], "diff_val": v["diff"], "lower95_val": v["diff"] - z * v["se"],
                         "auc_test": t["auc_1"], "auc_test_bench": t["auc_2"], "diff_test": t["diff"],
                         "ci_test_low": t["ci_low"], "ci_test_high": t["ci_high"], "p_test": t["p_two_sided"],
                         "non_inferior_test": t["non_inferior"],
                         "auc_test_daily": td["auc_2"], "diff_test_daily": td["diff"],
                         "ci_test_daily_low": td["ci_low"], "ci_test_daily_high": td["ci_high"]})
        cands = pd.DataFrame([r for r in rows if r["period"] == period])
        ok = cands[cands.lower95_val > -DELTA]
        pick = ok.sort_values(["n", "diff_val"], ascending=[True, False]).iloc[0] if len(ok) \
            else cands.sort_values("diff_val", ascending=False).iloc[0]
        sel.append({**pick.to_dict(), "constraint_met_on_validation": bool(len(ok))})
        print(period, pick.model, pick.k, pick.review_periods, round(pick.diff_val, 4), round(pick.diff_test, 4), flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e3d_candidates.csv"), index=False)
    pd.DataFrame(sel).to_csv(os.path.join(OUT, "e3d_selection.csv"), index=False)
