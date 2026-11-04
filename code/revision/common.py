"""
Shared utilities for the QREI revision experiments (E1-E10).

The original pipeline (preprocessing_functions, modeling_functions, experiment_pipeline)
is imported unchanged from ../original_pipeline so that every revision result is produced
with exactly the code used in the submitted manuscript. The Boruta stage below is a direct
port of the functions defined inside 3_model_new_iteration.ipynb (obtain_experiment_parameters,
reproduce_results, run_boruta, create_new_model, run_boruta_pipeline).
"""
#path configuration
import os
import sys
import ast
import time
import json

#data manipulation
import numpy as np
import pandas as pd

V2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(V2_DIR, "code", "original_pipeline"))

from preprocessing_functions import *  # noqa: E402,F401
from modeling_functions import *  # noqa: E402,F401
from experiment_pipeline import *  # noqa: E402,F401
from hyperopt import hp  # noqa: E402
from boruta import BorutaPy  # noqa: E402

#reference seed
seed = 42

DATA_RAW = os.path.join(V2_DIR, "data", "raw")
RESULTS = os.path.join(V2_DIR, "results")

PERIOD_LABELS = {"(0-100]": "first", "(100-200]": "second", "(200-300]": "third"}


def load_dpi():
    """Load the simulated DPI dataset used in the manuscript (10,000 x 330 + late_project)."""
    return pd.read_csv(os.path.join(DATA_RAW, "df_dpi.csv"))


def load_experiments():
    """Load the 1,728 experiment results of the full factorial grid."""
    return pd.read_csv(os.path.join(DATA_RAW, "df_exp_simulation.csv")).drop(columns="Unnamed: 0")


def simplicity_subset(df_exp):
    """
    Subset used in the manuscript to pick intermediate models: correlation removal applied (H2=2),
    no factor analysis (H3=1), no feature engineering (H5=1), no rebalancing (H7=1).
    """
    return df_exp[(df_exp.apply_fa == False) &
                  (df_exp.list_lag_features.isna()) &
                  (df_exp.apply_rebalance_strategy.isna()) &
                  (df_exp.remove_correlation == True)]


def _literal(value):
  """Parse a dict/list stored as string in the experiment csv (returns None if empty)."""
  if isinstance(value, (dict, list)):
    return value
  try:
    return ast.literal_eval(value)
  except Exception:
    return None


def obtain_experiment_parameters(df_exp, index_df):
  """Obtain all preprocessing and modeling parameters of a specific experiment row."""
  row = df_exp.loc[index_df].to_dict()
  best_params = _literal(row["best_params"])
  best_params["classifier_name"] = row["model"]
  return {"seed": row["seed"],
          "initial_date": row["initial_date"],
          "final_date": row["final_date"],
          "time_interval": row["time_interval"],
          "create_lag_features_abs": row["create_lag_features_abs"],
          "create_lag_features_rel": row["create_lag_features_rel"],
          "list_lag_features": _literal(row["list_lag_features"]),
          "list_columns_filter": _literal(row["list_columns_filter"]),
          "test_size": row["test_size"],
          "remove_correlation": row["remove_correlation"],
          "remove_correlation_params": _literal(row["remove_correlation_params"]),
          "apply_fa_bool": row["apply_fa"],
          "fa_params": _literal(row["fa_params"]),
          "apply_rebalance_strategy": row["apply_rebalance_strategy"] if isinstance(row["apply_rebalance_strategy"], str) else None,
          "rebalance_params": _literal(row["rebalance_params"]),
          "model": row["model"],
          "best_params": best_params,
          "scoring": row["scoring"],
          "cv": row["cv"],
          "max_evals": row["max_evals"],
          "timeout": row["timeout"],
          "score_train": row["score_train"],
          "score_test": row["score_test"]}


PREPROCESSING_KEYS = ["initial_date", "final_date", "time_interval", "create_lag_features_abs",
                      "create_lag_features_rel", "list_lag_features", "list_columns_filter", "test_size",
                      "remove_correlation", "remove_correlation_params", "apply_fa_bool", "fa_params",
                      "apply_rebalance_strategy", "rebalance_params", "seed"]


def reproduce_results(df_dpi, dict_parameters):
    """Rebuild train/test data and refit the model with the stored best hyperparameters."""
    dict_preprocessing = {key: dict_parameters[key] for key in PREPROCESSING_KEYS}
    X_train, X_test, y_train, y_test, _ = preprocess_dataframe(df_dpi, **dict_preprocessing)
    model_classifier = instantiate_model(dict(dict_parameters["best_params"]))
    model_classifier.fit(X_train, y_train.values.ravel())
    return X_train, y_train, X_test, y_test, model_classifier


def run_boruta(model_classifier, X_train, y_train, random_state=42):
    """Boruta (BorutaPy) on the training set only; returns the confirmed features."""
    if not hasattr(model_classifier, "n_estimators"):
        # single decision tree: BorutaPy's n_estimators="auto" would call set_params(n_estimators=...),
        # which a DecisionTreeClassifier does not accept; the tree is used as is
        model_classifier.set_params = lambda **kwargs: model_classifier
    boruta_classifier = BorutaPy(estimator=model_classifier,
                                 n_estimators="auto",
                                 alpha=0.01,
                                 max_iter=300,
                                 random_state=random_state,
                                 verbose=0)
    boruta_classifier.fit(X_train.values, y_train.values.ravel())
    return list(X_train.columns[boruta_classifier.support_])


def model_search_space(classifier_name):
  """Hyperopt search spaces, identical to the ones used in the manuscript."""
  spaces = {
      "dt": {"classifier_name": "dt", "random_state": seed,
             "criterion": hp.choice('dt_criterion', ["gini", "entropy"]),
             "max_depth": hp.randint('dt_max_depth', 3, 7),
             "min_samples_split": hp.choice('dt_min_samples_split', np.linspace(0.02, 0.05, 4)),
             "min_samples_leaf": hp.choice('dt_min_samples_leaf', np.linspace(0.005, 0.015, 3))},
      "rf": {"classifier_name": "rf", "random_state": seed,
             'n_estimators': hp.choice('rf_n_estimators', [int(i) for i in np.linspace(100, 500, 5)]),
             'max_depth': hp.randint('rf_max_depth', 3, 7),
             "min_samples_split": hp.choice("rf_min_samples_split", np.linspace(0.02, 0.05, 4)),
             "min_samples_leaf": hp.choice("rf_min_samples_leaf", np.linspace(0.005, 0.015, 3))},
      "xgb": {"classifier_name": "xgb", "seed": seed, "verbosity": 0,
              'n_estimators': hp.choice('xgb_n_estimators', [int(i) for i in np.linspace(200, 1000, 5)]),
              'max_depth': hp.randint('xgb_max_depth', 3, 7),
              'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.6, 1.0),
              'subsample': hp.uniform('xgb_subsample', 0.6, 1.0),
              'eta': hp.choice('xgb_eta', np.geomspace(0.01, 0.3, 5)),
              'gamma': hp.choice('xgb_gamma', np.linspace(0, 8, 5)),
              'lambda': hp.choice('xgb_lambda', np.linspace(0, 8, 5)),
              'alpha': hp.choice('xgb_alpha', np.linspace(0, 8, 5))},
      "lgbm": {"classifier_name": "lgbm", "random_state": seed, "verbosity": -1,
               "learning_rate": hp.choice("lgbm_learning_rate", np.geomspace(0.01, 0.3, 5)),
               'max_depth': hp.randint('lgbm_max_depth', 3, 7),
               'n_estimators': hp.choice('lgbm_n_estimators', [int(i) for i in np.linspace(100, 500, 5)]),
               'num_leaves': hp.choice('lgbm_num_leaves', [int(i) for i in np.linspace(18, 52, 8)]),
               'subsample': hp.uniform('lgbm_subsample', 0.6, 1.0),
               'colsample_bytree': hp.uniform('lgbm_colsample_bytree', 0.6, 1.0),
               'reg_alpha': hp.choice('lgbm_alpha', np.linspace(0, 8, 5)),
               'reg_lambda': hp.choice('lgbm_lambda', np.linspace(0, 8, 5))},
  }
  return spaces[classifier_name]


def retrain_with_features(df_dpi, dict_parameters, list_columns_filter):
  """Re-tune (hyperopt + 5-fold CV on train) and evaluate once on test with the Boruta features."""
  params = {key: dict_parameters[key] for key in PREPROCESSING_KEYS}
  params["list_columns_filter"] = list_columns_filter
  return run_full_pipeline(df_dpi, **params,
                           model_params_search=model_search_space(dict_parameters["model"]),
                           max_evals=dict_parameters["max_evals"],
                           timeout=dict_parameters["timeout"],
                           cv=dict_parameters["cv"],
                           scoring=dict_parameters["scoring"])


def boruta_stage(df_dpi, df_exp, index_df):
    """
    Intermediate model (row index_df of the grid) -> Boruta on train -> re-tune -> single test evaluation.
    One Boruta iteration is used: in the manuscript, iterations 2 and 3 never changed the feature set.
    Returns a dict with intermediate/final features and scores, plus timings.
    """
    t0 = time.time()
    dict_parameters = obtain_experiment_parameters(df_exp, index_df)
    X_train, y_train, X_test, y_test, model_classifier = reproduce_results(df_dpi, dict_parameters)
    t_reproduce = time.time() - t0

    t1 = time.time()
    features_final = run_boruta(model_classifier, X_train, y_train)
    t_boruta = time.time() - t1

    t2 = time.time()
    df_final = retrain_with_features(df_dpi, dict_parameters, features_final)
    t_retrain = time.time() - t2

    return {"index_df": int(index_df),
            "period": PERIOD_LABELS.get(df_exp.loc[index_df, "Period"], df_exp.loc[index_df, "Period"]),
            "model": dict_parameters["model"],
            "time_interval": int(dict_parameters["time_interval"]),
            "features_intermediate": list(X_train.columns),
            "cv_intermediate": float(dict_parameters["score_train"]),
            "test_intermediate": float(dict_parameters["score_test"]),
            "features_final": features_final,
            "cv_final": float(df_final.score_train.iloc[0]),
            "test_final": float(df_final.score_test.iloc[0]),
            "best_params_final": str(df_final.best_params.iloc[0]),
            "seconds_reproduce": round(t_reproduce, 1),
            "seconds_boruta": round(t_boruta, 1),
            "seconds_retrain": round(t_retrain, 1)}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
