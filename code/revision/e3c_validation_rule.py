"""
E3c - Rule V: selection on a large simulated VALIDATION set (three-way protocol suggested by Reviewer 2).

Motivation (E3b): with only 7,500 training projects, 5-fold CV cannot enforce a margin as fine as
delta = 0.005; in the second period the CV-selected set {160, 200} lost 0.013 AUROC on 100,000
independent projects.

Protocol (reference dataset, candidates and hyperparameters from E2b, no re-tuning):
  train      : manuscript training split (7,500)
  validation : 100,000 new projects (seed 900011) -> used ONLY to choose the review periods
  test       : 100,000 new projects (seed 900001) -> used ONCE to report the selected model
Rule V: fewest review periods such that the one-sided 95% DeLong lower bound of
        AUROC_val(candidate) - AUROC_val(daily monitoring) > -delta.
Output: results/E3_inference/e3c_validation_rule_{candidates,selection}.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403
import mc_simulator as mc  # noqa: E402
from scipy import stats  # noqa: E402

VAL_SEED = 900011


def load_validation(dist="triangular"):
    return load_dataset(f"val_{dist}")   # seed 900011 for triangular (data/processed/manifest.json)


if __name__ == "__main__":
    e2b = pd.read_csv(os.path.join(RESULTS, "E2_cv_selection", "e2b_boruta_all_candidates.csv")).rename(columns={"time_interval": "k"})
    daily = pd.read_csv(os.path.join(RESULTS, "E3_inference", "e3b_independent_test.csv")).set_index("period")
    df, df_val, df_test = load_dataset("reference"), load_validation(), load_dataset("test_triangular")
    y_val, y_test = df_val.late_project.astype(int).values, df_test.late_project.astype(int).values
    z = stats.norm.ppf(0.95)

    rows, sel_rows = [], []
    for period, (start, end) in PERIODS.items():
        days = [f"period_{t}" for t in range(start + 1, end + 1)]
        dbp = eval(daily.loc[period, "daily_best_params"], {"np": np})
        both = pd.concat([df_val, df_test], ignore_index=True)
        _, _, p_daily = fit_predict(df, days, dbp, 42, both)
        pd_val, pd_test = p_daily[:len(df_val)], p_daily[len(df_val):]
        for _, c in e2b[e2b.period == period].iterrows():
            feats = ["period_" + f for f in str(c.features_final).split(",")]
            bp = eval(c.best_params_final, {"np": np}); bp["classifier_name"] = c.model
            _, _, p = fit_predict(df, feats, bp, 42, both)
            v = delong_paired(y_val, p[:len(df_val)], pd_val, delta=DELTA)
            t = delong_paired(y_test, p[len(df_val):], pd_test, delta=DELTA)
            rows.append({"period": period, "model": c.model, "k": c.k, "review_periods": c.features_final,
                         "n": c.n_final, "cv_final": c.cv_final, "auc_val": v["auc_1"], "auc_val_daily": v["auc_2"],
                         "diff_val": v["diff"], "lower95_val": v["diff"] - z * v["se"],
                         "auc_test": t["auc_1"], "auc_test_daily": t["auc_2"], "diff_test": t["diff"],
                         "ci_test_low": t["ci_low"], "ci_test_high": t["ci_high"],
                         "non_inferior_test": t["non_inferior"]})
        cands = pd.DataFrame([r for r in rows if r["period"] == period])
        ok = cands[cands.lower95_val > -DELTA]
        # constraint met: fewest review periods (ties -> smallest loss); otherwise: smallest loss
        pick = ok.sort_values(["n", "diff_val"], ascending=[True, False]).iloc[0] if len(ok) \
            else cands.sort_values("diff_val", ascending=False).iloc[0]
        sel_rows.append({**pick.to_dict(), "constraint_met_on_validation": bool(len(ok))})
        print(period, pick.review_periods, round(pick.diff_val, 4), round(pick.diff_test, 4), flush=True)

    out = os.path.join(RESULTS, "E3_inference")
    tag = str(DELTA).replace(".", "")
    pd.DataFrame(rows).to_csv(os.path.join(out, "e3c_validation_rule_candidates.csv"), index=False)
    pd.DataFrame(sel_rows).to_csv(os.path.join(out, f"e3c_validation_rule_selection_delta{tag}.csv"), index=False)
