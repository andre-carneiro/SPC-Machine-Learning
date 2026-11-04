"""
E3b - Models selected with rule B on the manuscript dataset (from E2b, no re-tuning of candidates),
evaluated on 100,000 independent simulated projects (triangular, seed 900001) against daily monitoring.
Output: results/E3_inference/e3b_independent_test.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403

if __name__ == "__main__":
    e2b = pd.read_csv(os.path.join(RESULTS, "E2_cv_selection", "e2b_boruta_all_candidates.csv"))
    cands = e2b.rename(columns={"time_interval": "k"})
    df, df_ext = load_dataset("reference"), load_dataset("test_triangular")
    out = []
    for period in PERIODS:
        sel = rule_b(cands[cands.period == period]).copy()
        bp = eval(sel["best_params_final"], {"np": np}); bp["classifier_name"] = sel["model"]
        sel["best_params_final"] = bp
        out.append(evaluate_selection(df, df_ext, period, sel))
        print(out[-1], flush=True)
    pd.DataFrame(out).assign(train="reference", test="test_triangular").to_csv(
        os.path.join(RESULTS, "E3_inference", "e3b_independent_test.csv"), index=False)
