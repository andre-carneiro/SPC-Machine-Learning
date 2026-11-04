"""
E8a - Correlation threshold 0.95 (instead of 0.9) on the manuscript dataset: 16 candidates per period
(tuning + Boruta on the training set), then rule V (delta = 0.01) with the 100k validation set and a
single evaluation on the 100k independent test set, against the same daily-monitoring benchmark as E3c.
Usage: .venv/bin/python code/revision/e8a_corr095.py second [first third]
Output: results/E8_thresholds/corr095_candidates.csv, corr095_val_test.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403
from e3c_validation_rule import load_validation  # noqa: E402
from scipy import stats  # noqa: E402

OUT = os.path.join(RESULTS, "E8_thresholds")

if __name__ == "__main__":
    periods = sys.argv[1:] or list(PERIODS)
    df = load_dataset("reference")
    cpath = os.path.join(OUT, "corr095_candidates.csv")
    cands = pd.read_csv(cpath) if os.path.exists(cpath) else pd.DataFrame()
    for period in periods:
        for model in MODELS:
            for k in INTERVALS:
                if len(cands) and ((cands.period == period) & (cands.model == model) & (cands.k == k)).any():
                    continue
                r = run_candidate(df, period, model, k, corr_threshold=0.95)
                cands = pd.concat([cands, pd.DataFrame([r])], ignore_index=True)
                cands.to_csv(cpath, index=False)
                print(period, model, k, r["features_final"], round(r["cv_final"], 4), flush=True)

    daily = pd.read_csv(os.path.join(RESULTS, "E3_inference", "e3b_independent_test.csv")).set_index("period")
    df_val, df_test = load_validation(), load_dataset("test_triangular")
    both = pd.concat([df_val, df_test], ignore_index=True)
    y_val, y_test = df_val.late_project.astype(int).values, df_test.late_project.astype(int).values
    z, rows = stats.norm.ppf(0.95), []
    for period in periods:
        start, end = PERIODS[period]
        dbp = eval(daily.loc[period, "daily_best_params"], {"np": np})
        _, _, p_daily = fit_predict(df, [f"period_{t}" for t in range(start + 1, end + 1)], dbp, 42, both)
        for _, c in cands[cands.period == period].iterrows():
            feats = ["period_" + f for f in str(c.features_final).split(",")]
            _, _, p = fit_predict(df, feats, eval(c.best_params_final, {"np": np}), 42, both)
            v = delong_paired(y_val, p[:len(df_val)], p_daily[:len(df_val)], delta=DELTA)
            t = delong_paired(y_test, p[len(df_val):], p_daily[len(df_val):], delta=DELTA)
            rows.append({"period": period, "model": c.model, "k": c.k, "review_periods": c.features_final, "n": c.n_final,
                         "cv_final": c.cv_final, "diff_val": v["diff"], "lower95_val": v["diff"] - z * v["se"],
                         "auc_test": t["auc_1"], "auc_test_daily": t["auc_2"], "diff_test": t["diff"],
                         "ci_test_low": t["ci_low"], "ci_test_high": t["ci_high"], "non_inferior_test": t["non_inferior"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "corr095_val_test.csv"), index=False)
    print("done", flush=True)
