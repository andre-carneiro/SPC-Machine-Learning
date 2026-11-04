"""
E8 - Sensitivity to the correlation threshold c (R2.7) on the manuscript dataset, with the selected
configuration of each period fixed (first: LGBM biweekly; second and third: LGBM monthly) and
c in {0.80, 0.85, 0.90, 0.95}. For each case: instants after the filter, instants confirmed by Boruta,
and the loss against the conventional schedule of the same frequency (best-CV grid model, no filter)
on the validation (100k) and test (100k) sets, with DeLong 95% CIs.
(The full 16-candidate run with c = 0.95 for the second period is E8a.)
Output: results/E8_thresholds/correlation_threshold.csv (resumable)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403
from final_rule import candidate_days, Z95  # noqa: E402
from e3d_candidate_benchmark_rule import grid_benchmark_row  # noqa: E402

FIXED = {"first": ("lgbm", 10), "second": ("lgbm", 20), "third": ("lgbm", 20)}
OUT = os.path.join(RESULTS, "E8_thresholds", "correlation_threshold.csv")

if __name__ == "__main__":
    df, df_val, df_test, df_exp = load_dataset("reference"), load_dataset("val_triangular"), load_dataset("test_triangular"), load_experiments()
    both = pd.concat([df_val, df_test], ignore_index=True); nv = len(df_val)
    y_val, y_test = df_val.late_project.astype(int).values, df_test.late_project.astype(int).values
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    for c_thr in [0.80, 0.85, 0.90, 0.95]:
        for period, (model, k) in FIXED.items():
            if len(done) and ((done.threshold == c_thr) & (done.period == period)).any():
                continue
            r = run_candidate(df, period, model, k, corr_threshold=c_thr)
            feats = ["period_" + f for f in r["features_final"].split(",")]
            _, _, p = fit_predict(df, feats, eval(r["best_params_final"], {"np": np}), 42, both)
            bidx = grid_benchmark_row(df_exp, period, k)
            _, _, pb = fit_predict(df, candidate_days(period, k), obtain_experiment_parameters(df_exp, bidx)["best_params"], 42, both)
            v = delong_paired(y_val, p[:nv], pb[:nv], delta=DELTA)
            t = delong_paired(y_test, p[nv:], pb[nv:], delta=DELTA)
            row = {"threshold": c_thr, "period": period, "model": model, "k": k,
                   "after_filter": r["features_intermediate"], "review_periods": r["features_final"], "n": r["n_final"],
                   "cv_final": r["cv_final"], "diff_val": v["diff"], "lower95_val": v["diff"] - Z95 * v["se"],
                   "auc_test": t["auc_1"], "diff_test": t["diff"], "ci_test_low": t["ci_low"], "ci_test_high": t["ci_high"],
                   "non_inferior_test": t["non_inferior"]}
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(OUT, index=False)
            print("e8", c_thr, period, row["review_periods"], round(row["diff_test"], 4), flush=True)
