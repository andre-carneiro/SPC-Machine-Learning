"""
E2b - Boruta stage for ALL 16 candidate configurations of the simplicity subset, per period (48 runs).

Purpose: apply the problem statement (Eq. X of the revised manuscript) on the cross-validation scale,
i.e. choose the configuration with the FEWEST final review periods among those whose CV AUROC
(after Boruta) is within delta of the best CV AUROC. This formalizes the "simplest pipeline" criterion
used qualitatively in the submission, using only training data. The test set is read here only to be
stored; selection rules in the analysis must use cv_* columns exclusively.

Output: results/E2_cv_selection/e2b_boruta_all_candidates.csv
Run: .venv/bin/python code/revision/e2b_boruta_all_candidates.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa: E402,F403

OUT = os.path.join(RESULTS, "E2_cv_selection", "e2b_boruta_all_candidates.csv")

if __name__ == "__main__":
    df_dpi = load_dpi()
    df_exp = load_experiments()
    subset = simplicity_subset(df_exp)

    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    rows = done.to_dict("records")
    done_idx = set(done.index_df) if len(done) else set()

    for index_df in subset.index:
        if index_df in done_idx:
            continue
        r = boruta_stage(df_dpi, df_exp, index_df)
        r["n_intermediate"] = len(r["features_intermediate"])
        r["n_final"] = len(r["features_final"])
        r["features_intermediate"] = ",".join(f.replace("period_", "") for f in r["features_intermediate"])
        r["features_final"] = ",".join(f.replace("period_", "") for f in r["features_final"])
        rows.append(r)
        pd.DataFrame(rows).to_csv(OUT, index=False)  # checkpoint after each run
        print(r["period"], r["model"], r["time_interval"], r["features_final"], round(r["cv_final"], 4), flush=True)
