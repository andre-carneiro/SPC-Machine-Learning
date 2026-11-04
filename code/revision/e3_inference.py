"""
E3 - Inference on AUROC differences for the models selected with rule B (decision D2, delta = 0.005).

For each evaluation period, on the SAME test set (seed 42, stratified 75/25):
  final        = rule-B configuration, Boruta features, re-tuned hyperparameters (from E2b)
  intermediate = same configuration before Boruta
  daily (B1)   = monitoring at every day of the period, no correlation removal / FA / lags / rebalancing,
                 (a) same model family as the final model, (b) highest-CV model among the four families
Tests: DeLong paired test + one-sided non-inferiority (delta = 0.005 and 0.01) + stratified bootstrap CI.

Output: results/E3_inference/e3_inference.csv, e3_predictions.parquet-like csv
Run: .venv/bin/python code/revision/e3_inference.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa: E402,F403
from inference import delong_paired, bootstrap_auc_diff  # noqa: E402

OUT = os.path.join(RESULTS, "E3_inference")
RULE_B_ROWS = {"first": 134, "second": 705, "third": 1213}
DELTAS = [0.005, 0.01]


def parse_params(text):
    return eval(text, {"np": np})  # stored by hyperopt/space_eval with numpy scalars


def predict(df_dpi, dict_parameters, best_params, columns=None):
    """Rebuild train/test with the given features, fit with fixed hyperparameters, return y_test, p_test."""
    params = {k: dict_parameters[k] for k in PREPROCESSING_KEYS}
    params["list_columns_filter"] = columns
    X_train, X_test, y_train, y_test, _ = preprocess_dataframe(df_dpi, **params)
    model = instantiate_model(dict(best_params))
    model.fit(X_train, y_train.values.ravel())
    return y_test.values.ravel(), model.predict_proba(X_test)[:, 1], list(X_train.columns)


def daily_rows(df_exp, period_label):
    g = df_exp[(df_exp.Period.map(PERIOD_LABELS) == period_label) & (df_exp.time_interval == 1) &
               (df_exp.remove_correlation == False) & (df_exp.apply_fa == False) &
               (df_exp.list_lag_features.isna()) & (df_exp.apply_rebalance_strategy.isna())]
    return g


if __name__ == "__main__":
    df_dpi = load_dpi()
    df_exp = load_experiments()
    e2b = pd.read_csv(os.path.join(RESULTS, "E2_cv_selection", "e2b_boruta_all_candidates.csv")).set_index("index_df")

    rows, preds = [], {}
    for period, idx in RULE_B_ROWS.items():
        dp = obtain_experiment_parameters(df_exp, idx)
        model_name = dp["model"]

        # final (Boruta features + re-tuned hyperparameters)
        feats = ["period_" + f for f in str(e2b.loc[idx, "features_final"]).split(",")]
        bp = parse_params(e2b.loc[idx, "best_params_final"]); bp["classifier_name"] = model_name
        y, p_final, _ = predict(df_dpi, dp, bp, feats)

        # intermediate (before Boruta)
        _, p_inter, feats_inter = predict(df_dpi, dp, dp["best_params"])

        # daily benchmarks
        g = daily_rows(df_exp, period)
        bench = {"daily_same_model": g[g.model == model_name].index[0], "daily_best_cv": g.score_train.idxmax()}
        p_bench = {}
        for name, bidx in bench.items():
            bdp = obtain_experiment_parameters(df_exp, bidx)
            yb, p_bench[name], _ = predict(df_dpi, bdp, bdp["best_params"])
            assert np.array_equal(yb, y), "test sets differ"

        preds[period] = {"y": y, "final": p_final, "intermediate": p_inter, **p_bench}
        comparisons = [("intermediate", p_inter, idx), ("daily_same_model", p_bench["daily_same_model"], bench["daily_same_model"]),
                       ("daily_best_cv", p_bench["daily_best_cv"], bench["daily_best_cv"])]
        for ref_name, p_ref, ref_idx in comparisons:
            boot = bootstrap_auc_diff(y, p_final, p_ref, n_boot=2000)
            for delta in DELTAS:
                r = delong_paired(y, p_final, p_ref, delta=delta)
                rows.append({"period": period, "final_model": model_name, "final_review_periods": ",".join(f.replace("period_", "") for f in feats),
                             "reference": ref_name, "reference_grid_row": int(ref_idx),
                             "reference_model": df_exp.loc[ref_idx, "model"],
                             "auc_final": r["auc_1"], "auc_reference": r["auc_2"], "diff": r["diff"], "se_delong": r["se"],
                             "ci95_delong_low": r["ci_low"], "ci95_delong_high": r["ci_high"],
                             "ci95_boot_low": boot["diff_ci"][0], "ci95_boot_high": boot["diff_ci"][1],
                             "auc_final_ci95_boot": boot["auc_1_ci"], "p_two_sided": r["p_two_sided"],
                             "delta": delta, "p_noninferiority": r["p_noninferiority"], "non_inferior": r["non_inferior"]})
        print(period, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e3_inference.csv"), index=False)
    for period, d in preds.items():
        pd.DataFrame(d).to_csv(os.path.join(OUT, f"e3_test_predictions_{period}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df[["period", "reference", "reference_model", "auc_final", "auc_reference", "diff", "ci95_delong_low", "ci95_delong_high",
              "p_two_sided", "delta", "p_noninferiority", "non_inferior"]].round(4).to_string(index=False))
