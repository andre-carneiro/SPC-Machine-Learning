"""
E7 - Alternative selection procedures (R1.11, R2.5, R2.6) on the manuscript dataset and, for the lasso,
also on the 20 replications (stability).
  (a) Boruta with each tree-based family at the selected frequency -> tabulated from E2b (no computation).
  (b) L1-penalized logistic regression (lasso) on ALL daily instants of the period (no prior filtering):
      penalty chosen by 5-fold CV (AUROC) on the training split; selected instants = non-zero coefficients.
  (c) Permutation importance (model-agnostic): best-CV tree model monitoring every k days (k of the selected
      configuration), importance on the 25% descriptive split (30 repeats, AUROC); instants kept if
      mean - 2 sd > 0; refit with the same hyperparameters.
All evaluated on the independent test set (100,000 projects).
Output: results/E7_selection_methods/{boruta_families,lasso,permutation}.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v2 import *  # noqa: E402,F403
from final_rule import candidate_days  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = os.path.join(RESULTS, "E7_selection_methods")
SELECTED_K = {"first": 10, "second": 20, "third": 20}


def split(df, cols, seed=42):
    y = df.late_project.astype(int).values
    tr, te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=seed, stratify=y)
    X = df[cols].values
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X[tr]), y[tr], sc.transform(X[te]), y[te], sc


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df_test = load_dataset("test_triangular")
    y_test = df_test.late_project.astype(int).values

    # (a) Boruta with each family (from E2b)
    e2b = pd.read_csv(os.path.join(RESULTS, "E2_cv_selection", "e2b_boruta_all_candidates.csv"))
    rows = [{"period": p, "model": r.model, "k": k, "review_periods": r.features_final, "n": r.n_final, "cv_final": r.cv_final}
            for p, k in SELECTED_K.items() for _, r in e2b[(e2b.period == p) & (e2b.time_interval == k)].iterrows()]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "boruta_families.csv"), index=False)

    # (b) lasso on daily instants: reference + 20 replications
    lasso_rows = []
    for name in ["reference"] + [f"tri_rep{r}" for r in range(1, 21)]:
        df = load_dataset(name)
        for period in PERIODS:
            cols = candidate_days(period, 1)
            Xtr, ytr, _, _, sc = split(df, cols)
            lr = LogisticRegressionCV(Cs=np.logspace(-3, 1, 12), penalty="l1", solver="liblinear", cv=5, scoring="roc_auc",
                                      tol=1e-3, max_iter=2000, n_jobs=5, random_state=42).fit(Xtr, ytr)
            sel = [c.replace("period_", "") for c, w in zip(cols, lr.coef_[0]) if abs(w) > 1e-8]
            p = lr.predict_proba(sc.transform(df_test[cols].values))[:, 1]
            lasso_rows.append({"dataset": name, "period": period, "C": float(lr.C_[0]), "review_periods": ",".join(sel),
                               "n": len(sel), "auc_test_100k": roc_auc_score(y_test, p)})
            print("lasso", name, period, len(sel), ",".join(sel)[:80], flush=True)
        pd.DataFrame(lasso_rows).to_csv(os.path.join(OUT, "lasso.csv"), index=False)

    # (c) permutation importance (reference)
    df_exp, df = load_experiments(), load_dataset("reference")
    perm_rows = []
    for period, k in SELECTED_K.items():
        g = df_exp[(df_exp.Period.map(PERIOD_LABELS) == period) & (df_exp.time_interval == k) &
                   (df_exp.remove_correlation == False) & (df_exp.apply_fa == False) &
                   (df_exp.list_lag_features.isna()) & (df_exp.apply_rebalance_strategy.isna())]
        bp = obtain_experiment_parameters(df_exp, g.score_train.idxmax())["best_params"]
        cols = candidate_days(period, k)
        Xtr, ytr, Xte, yte, sc = split(df, cols)
        m = instantiate_model(dict(bp)); m.fit(Xtr, ytr)
        pi = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=30, random_state=42)
        keep = [c for c, mu, sd in zip(cols, pi.importances_mean, pi.importances_std) if mu - 2 * sd > 0]
        _, _, p = fit_predict(df, keep, bp, 42, df_test)
        perm_rows.append({"period": period, "model": bp["classifier_name"] if "classifier_name" in bp else g.loc[g.score_train.idxmax(), "model"],
                          "k": k, "review_periods": ",".join(c.replace("period_", "") for c in keep), "n": len(keep),
                          "auc_test_100k": roc_auc_score(y_test, p),
                          "importances": ";".join(f"{c.replace('period_', '')}:{mu:.4f}±{sd:.4f}" for c, mu, sd in zip(cols, pi.importances_mean, pi.importances_std))})
        print("perm", period, perm_rows[-1]["review_periods"], round(perm_rows[-1]["auc_test_100k"], 4), flush=True)
    pd.DataFrame(perm_rows).to_csv(os.path.join(OUT, "permutation.csv"), index=False)
