"""
Revision pipeline (Algorithm 1 of the revised manuscript), reused by E3b, E4, E5 and E6.

For one training dataset (10,000 simulated projects) and each evaluation period:
  1. 16 candidate configurations of the simplicity subset: model in {DT, RF, XGB, LGBM} x
     candidate interval k in {1, 5, 10, 20}; correlation filter (|rho| > 0.9, training set only);
     no FA, no feature engineering, no rebalancing. Hyperparameters: hyperopt TPE, 60 evaluations,
     5-fold CV AUROC on the training set (identical search spaces to the manuscript).
  2. Boruta on the training set of each candidate, then re-tuning on the confirmed review periods.
  3. Rule B (decision D2): fewest final review periods among candidates with CV AUROC >= best - delta.
  4. Benchmark "daily monitoring": same model family, every day of the period, no filtering, tuned the same way.
  5. Evaluation: AUROC on the dataset's own 25% test split and on a large INDEPENDENT test set
     (100,000 new simulated projects), with DeLong non-inferiority of the final model vs the benchmark.

Datasets are regenerated deterministically from mc_simulator (see data/processed/MANIFEST.md).
"""
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa: E402,F403
from inference import delong_paired  # noqa: E402
import mc_simulator as mc  # noqa: E402

PERIODS = {"first": (0, 100), "second": (100, 200), "third": (200, 300)}
MODELS = ["dt", "rf", "xgb", "lgbm"]
INTERVALS = [1, 5, 10, 20]
DELTA = 0.01   # decision D3 (revised 31/10/2026 after E3b/E3c: 0.005 not attainable in the 2nd period)
MAX_EVALS, TIMEOUT, CV, SCORING = 60, 3600, 5, "roc_auc"

SP_DIR = os.environ.get("SPC_LEGACY_DIR", "") + "/"  # legacy pickles of the original loop-based simulator (only to re-create the legacy datasets)


# ------------------------------------------------------------------ datasets
def to_frame(dpi, result):
    """Same layout as data/raw/df_dpi.csv: period_1..period_330 + late_project."""
    df = dpi.copy()
    df["late_project"] = result.late_project.values.astype(bool)
    return df


def load_dataset(name):
    """
    Named datasets (all with deadline 320, horizon 330):
      reference            -> data/raw/df_dpi.csv (manuscript; = legacy triangular seeds 1..100)
      gamma_reference      -> legacy gamma run (= reference_dpi_gamma.pkl, verified bit-exact)
      cmp_reference        -> reference_dpi_cmpoisson.pkl (+ forward fill), original pymc3 run
      tri_rep{r}           -> triangular, Generator seed 1000 + r (independent replications)
      pert_train           -> Beta-PERT, seed 2001
      test_{dist}          -> 100,000 independent projects: seeds 900001 (tri), 900002 (gamma),
                              900003 (cmp), 900004 (pert)
    """
    stored = os.path.join(V2_DIR, "data", "processed", f"{name}.parquet")
    if os.path.exists(stored):
        # published copy (see data/processed/manifest.json for seed and SHA-256); identical to regeneration
        return pd.read_parquet(stored).drop(columns="total_duration")
    if name == "reference":
        return load_dpi()
    if name == "gamma_reference":
        return to_frame(*mc.simulate(n_sim=10000, rng_mode="legacy", dist="gamma"))
    if name == "cmp_reference":
        dpi = pd.read_pickle(SP_DIR + "reference_dpi_cmpoisson.pkl").astype(float).T.reset_index(drop=True)
        dpi.columns = [f"period_{t}" for t in range(1, 331)]
        res = pd.read_pickle(SP_DIR + "reference_result_cmpoisson.pkl").reset_index(drop=True)
        return to_frame(dpi.ffill(axis=1), res.assign(late_project=res.late_project.astype(bool)))
    val_seeds = {"val_triangular": (900011, "triangular"), "val_gamma": (900012, "gamma"),
                 "val_cmpoisson": (900013, "cmpoisson"), "val_pert": (900014, "pert")}
    if name in val_seeds:
        seed, dist = val_seeds[name]
        return to_frame(*mc.simulate(n_sim=100000, seed=seed, dist=dist))
    if name.startswith("tri_rep"):
        return to_frame(*mc.simulate(n_sim=10000, seed=1000 + int(name[7:]), dist="triangular"))
    if name == "pert_train":
        return to_frame(*mc.simulate(n_sim=10000, seed=2001, dist="pert"))
    test_seeds = {"test_triangular": (900001, "triangular"), "test_gamma": (900002, "gamma"),
                  "test_cmpoisson": (900003, "cmpoisson"), "test_pert": (900004, "pert")}
    if name in test_seeds:
        seed, dist = test_seeds[name]
        return to_frame(*mc.simulate(n_sim=100000, seed=seed, dist=dist))
    raise ValueError(name)


# ------------------------------------------------------------------ modeling steps
def base_params(period, k, split_seed=42, corr=True, columns=None, corr_threshold=0.9):
    start, end = PERIODS[period]
    return {"initial_date": start, "final_date": end, "time_interval": k,
            "create_lag_features_abs": False, "create_lag_features_rel": False, "list_lag_features": None,
            "list_columns_filter": columns, "test_size": 0.25,
            "remove_correlation": corr,
            "remove_correlation_params": {"method": "pearson", "correlation": corr_threshold} if corr else None,
            "apply_fa_bool": False, "fa_params": None,
            "apply_rebalance_strategy": None, "rebalance_params": None, "seed": split_seed}


def tune(df, params, model):
    """Hyperopt + 5-fold CV on train, single test evaluation (original run_full_pipeline)."""
    r = run_full_pipeline(df, **params, model_params_search=model_search_space(model),
                          max_evals=MAX_EVALS, timeout=TIMEOUT, cv=CV, scoring=SCORING).iloc[0]
    bp = dict(r.best_params); bp["classifier_name"] = model
    return {"cv": float(r.score_train), "test": float(r.score_test), "best_params": bp}


def run_candidate(df, period, model, k, split_seed=42, corr_threshold=0.9):
    t0 = time.time()
    params = base_params(period, k, split_seed, corr_threshold=corr_threshold)
    inter = tune(df, params, model)
    X_train, X_test, y_train, y_test, _ = preprocess_dataframe(df, **params)
    clf = instantiate_model(dict(inter["best_params"]))
    clf.fit(X_train, y_train.values.ravel())
    feats = run_boruta(clf, X_train, y_train)
    if len(feats) == 0:                       # Boruta confirmed nothing: keep the intermediate set
        feats = list(X_train.columns)
    final = tune(df, {**params, "list_columns_filter": feats}, model)
    return {"period": period, "model": model, "k": k,
            "features_intermediate": ",".join(c.replace("period_", "") for c in X_train.columns),
            "cv_intermediate": inter["cv"], "test_intermediate": inter["test"],
            "features_final": ",".join(c.replace("period_", "") for c in feats), "n_final": len(feats),
            "cv_final": final["cv"], "test_final": final["test"], "best_params_final": repr(final["best_params"]),
            "seconds": round(time.time() - t0, 1)}


def rule_b(cands, delta=DELTA):
    best = cands.cv_final.max()
    ok = cands[cands.cv_final >= best - delta]
    return ok.sort_values(["n_final", "cv_final"], ascending=[True, False]).iloc[0]


def fit_predict(df_train, features, best_params, split_seed, df_ext):
    """Refit on the training split with fixed hyperparameters; predict own test split and external test."""
    y = df_train.late_project.astype(int).values
    idx = np.arange(len(df_train))
    tr, te = train_test_split(idx, test_size=0.25, random_state=split_seed, stratify=y)
    X = df_train[features].values
    scaler = StandardScaler().fit(X[tr])
    model = instantiate_model(dict(best_params))
    model.fit(scaler.transform(X[tr]), y[tr])
    p_own = model.predict_proba(scaler.transform(X[te]))[:, 1]
    p_ext = model.predict_proba(scaler.transform(df_ext[features].values))[:, 1]
    return y[te], p_own, p_ext


def evaluate_selection(df, df_ext, period, sel, split_seed=42):
    """Final model vs daily-monitoring benchmark (same family) on own test and on the external test."""
    feats = ["period_" + f for f in str(sel["features_final"]).split(",")]
    bp = eval(sel["best_params_final"], {"np": np}) if isinstance(sel["best_params_final"], str) else sel["best_params_final"]
    y_own, p_own, p_ext = fit_predict(df, feats, bp, split_seed, df_ext)
    bench = tune(df, base_params(period, 1, split_seed, corr=False), sel["model"])
    start, end = PERIODS[period]
    daily = [f"period_{t}" for t in range(start + 1, end + 1)]
    _, b_own, b_ext = fit_predict(df, daily, bench["best_params"], split_seed, df_ext)
    y_ext = df_ext.late_project.astype(int).values
    own = delong_paired(y_own, p_own, b_own, delta=DELTA)
    ext = delong_paired(y_ext, p_ext, b_ext, delta=DELTA)
    ext10 = delong_paired(y_ext, p_ext, b_ext, delta=0.01)
    return {"period": period, "model": sel["model"], "k": int(sel["k"]), "review_periods": sel["features_final"],
            "n_review_periods": int(sel["n_final"]), "cv_final": sel["cv_final"],
            "auc_own_test": own["auc_1"], "auc_own_test_daily": own["auc_2"],
            "auc_ext": ext["auc_1"], "auc_ext_daily": ext["auc_2"], "diff_ext": ext["diff"],
            "ci_ext_low": ext["ci_low"], "ci_ext_high": ext["ci_high"], "p_ext_two_sided": ext["p_two_sided"],
            "non_inferior_ext_0005": ext["non_inferior"], "non_inferior_ext_001": ext10["non_inferior"],
            "daily_cv": bench["cv"], "daily_best_params": repr(bench["best_params"])}


def run_dataset(train_name, test_name, out_dir, split_seed=42, corr_threshold=0.9, models=MODELS, intervals=INTERVALS):
    """Full pipeline for one dataset with checkpointing (resumable)."""
    os.makedirs(out_dir, exist_ok=True)
    cand_path = os.path.join(out_dir, "candidates.csv")
    sel_path = os.path.join(out_dir, "selection_evaluation.csv")
    df = load_dataset(train_name)
    cands = pd.read_csv(cand_path) if os.path.exists(cand_path) else pd.DataFrame()
    for period in PERIODS:
        for model in models:
            for k in intervals:
                if len(cands) and ((cands.period == period) & (cands.model == model) & (cands.k == k)).any():
                    continue
                r = run_candidate(df, period, model, k, split_seed, corr_threshold)
                cands = pd.concat([cands, pd.DataFrame([r])], ignore_index=True)
                cands.to_csv(cand_path, index=False)
                print(train_name, period, model, k, r["features_final"], round(r["cv_final"], 4), flush=True)
    if os.path.exists(sel_path):
        return pd.read_csv(sel_path)
    df_ext = load_dataset(test_name)
    evals = []
    for period in PERIODS:
        sel = rule_b(cands[cands.period == period])
        evals.append(evaluate_selection(df, df_ext, period, sel, split_seed))
        print(train_name, "selected", period, sel["features_final"], flush=True)
    out = pd.DataFrame(evals).assign(train=train_name, test=test_name, split_seed=split_seed, corr_threshold=corr_threshold)
    out.to_csv(sel_path, index=False)
    return out
