"""
E1 - Reproduction of the intermediate and final (post-Boruta) models reported in Table 4.
E2 - Selection of intermediate models using ONLY the 5-fold CV score on the training set
     (answer to Reviewer 2, comment 1: test set touched once).

Outputs (results/E1_reproduction and results/E2_cv_selection):
  d2_simplicity_subset_ranking.csv : the 16 candidate configurations per period, with CV, test,
                                     ranks and number of review periods after correlation removal
  e1_reproduction.json             : manuscript models (grid rows 60, 637, 1250) re-run through Boruta
  e2_cv_selected.json              : CV-selected models re-run through Boruta
Run: .venv/bin/python code/revision/e1_e2_reproduction_cv_selection.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa: E402,F403

OUT_E1 = os.path.join(RESULTS, "E1_reproduction")
OUT_E2 = os.path.join(RESULTS, "E2_cv_selection")

#rows of df_exp_simulation.csv used in the manuscript (3_model_new_iteration.ipynb)
MANUSCRIPT_ROWS = {"first": 60, "second": 637, "third": 1250}


def number_review_periods(df_dpi, df_exp, index_df):
    """Review periods that survive correlation removal (computed on the training set only)."""
    params = obtain_experiment_parameters(df_exp, index_df)
    X_train, *_ = preprocess_dataframe(df_dpi, **{k: params[k] for k in PREPROCESSING_KEYS})
    return list(X_train.columns)


if __name__ == "__main__":
    start = time.time()
    df_dpi = load_dpi()
    df_exp = load_experiments()
    subset = simplicity_subset(df_exp)

    #--- D2 table: ranking of the candidate configurations by CV and by test
    rows = []
    for period, group in subset.groupby("Period"):
        group = group.copy()
        group["rank_cv"] = group.score_train.rank(ascending=False, method="min").astype(int)
        group["rank_test"] = group.score_test.rank(ascending=False, method="min").astype(int)
        group["rank_test_full_grid"] = [int((df_exp[df_exp.Period == period].score_test > s).sum()) + 1
                                        for s in group.score_test]
        for index_df, row in group.iterrows():
            features = number_review_periods(df_dpi, df_exp, index_df)
            rows.append({"period": PERIOD_LABELS[period], "grid_row": index_df, "model": row.model,
                         "H4_interval_days": int(row.time_interval), "n_review_periods": len(features),
                         "review_periods": ",".join(f.replace("period_", "") for f in features),
                         "cv_auroc_train": round(row.score_train, 4), "test_auroc": round(row.score_test, 4),
                         "rank_cv": row.rank_cv, "rank_test": row.rank_test,
                         "rank_test_full_grid": row.rank_test_full_grid,
                         "used_in_manuscript": index_df == MANUSCRIPT_ROWS[PERIOD_LABELS[period]]})
    df_rank = pd.DataFrame(rows).sort_values(["period", "rank_cv"])
    df_rank.to_csv(os.path.join(OUT_E2, "d2_simplicity_subset_ranking.csv"), index=False)
    print(df_rank.to_string(index=False), flush=True)

    #--- E1: reproduce manuscript models
    e1 = [boruta_stage(df_dpi, df_exp, idx) for idx in MANUSCRIPT_ROWS.values()]
    save_json(e1, os.path.join(OUT_E1, "e1_reproduction.json"))
    print("\nE1", pd.DataFrame(e1)[["period", "model", "features_final", "test_intermediate", "test_final"]], flush=True)

    #--- E2: CV-only selection (highest CV AUROC within the simplicity subset)
    cv_rows = df_rank.loc[df_rank.rank_cv == 1].drop_duplicates("period")
    e2 = []
    for _, r in cv_rows.iterrows():
        same = [x for x in e1 if x["index_df"] == r.grid_row]
        e2.append(same[0] if same else boruta_stage(df_dpi, df_exp, int(r.grid_row)))
    save_json(e2, os.path.join(OUT_E2, "e2_cv_selected.json"))
    print("\nE2", pd.DataFrame(e2)[["period", "model", "features_final", "test_intermediate", "test_final"]], flush=True)
    print(f"\nTotal time: {time.time() - start:.0f} s")
