#path and notebook configuration
import os
import sys
import ipynb

#data manipulation
import pandas as pd
import numpy as np

#model creation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

#hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval

#model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

#import warning
import warnings
warnings.filterwarnings("ignore")

#reference seed
seed = 42
np.random.seed(seed)

def instantiate_model(params):
  """
  Function that instantiate the model with the corresponding hyperparameters

  Params:
    params: dictionary with model information (name and hyperparameters)

  Return:
    model_classifier: model with hyperparameters
  """

  #obtain classifer name
  classifier_name = params["classifier_name"]
  del params["classifier_name"]

  #instantiate model
  if classifier_name == "dt":
    model_classifier = DecisionTreeClassifier(**params)

  # elif classifier_name == "catb":
  #   model_classifier = CatBoostClassifier(**params)

  elif classifier_name == "rf":
    model_classifier = RandomForestClassifier(**params)

  elif classifier_name == "xgb":
    model_classifier = XGBClassifier(**params)

  elif classifier_name == "lgbm":
    model_classifier = LGBMClassifier(**params)

  return model_classifier

def function_optimize(params_search):
  """
  Function used for the hyperopt optimization

  Params:
    params_search: dictionary with search space of the model

  Return:
    {loss: ..., status: ...}: loss function with train metric 
  """

  #instantiate the model
  model_classifier = instantiate_model(params_search)

  #use cross validation
  list_metric_score_train = cross_val_score(model_classifier,
                                            X = X_train_global,
                                            y = y_train_global.values.ravel(),
                                            cv = cv_global,
                                            scoring = scoring_global,
                                            n_jobs = -1)
  

  #metric average
  if scoring_global == "roc_auc":
    metric_score_train = np.mean(list_metric_score_train)

  return {'loss': - metric_score_train,
          'status': STATUS_OK}

def run_hyperopt(params_search,
                 X_train,
                 y_train,
                 max_evals,
                 timeout,
                 cv,
                 scoring):
  """
  Apply the hyperopt function over the training dataset

  Params:
    params_search: dictionary with search space of the model
    X_train: features training dataframe
    y_train: response training dataframe
    cv: number of cross validation
    scoring: score to optimize
    max_evals: amout of evaluation
    timeout: seconds for timeout

  Return:
    best_params: best hyperparameters that maximize the optimization metric
  """

  #create global variables of training dataset for hyperopt
  global X_train_global
  global y_train_global
  global cv_global
  global scoring_global

  X_train_global = X_train.copy()
  y_train_global = y_train.copy()
  cv_global = cv
  scoring_global = scoring

  #obtain best params
  best_result = fmin(fn = function_optimize,
                     space = params_search,
                     algo = tpe.suggest,
                     max_evals = max_evals,
                     timeout = timeout,
                     rstate = np.random.default_rng(seed))

  #obtain best_params dictionary
  best_params = space_eval(params_search, best_result)

  return best_params

def evaluate_model(best_params,
                   X_train,
                   y_train,
                   X_test,
                   y_test,
                   cv,
                   scoring):
  """
  Use the best hyperparameters to recreate the model and obtain the training and testing performance

  Params:
    best_params: dictionary with search space of the model
    X_train: features training dataframe
    y_train: features testing dataframe
    X_test: features testing dataframe
    y_test: response testing dataframe
    cv: number of cross validation
    scoring: score to optimize

  Return:
    metric_score_train: optimization metric for the training set 
    metric_score_test: optimization metric for the testing set
  """

  #instantiate the model with best hyperparameters
  model_classifier = instantiate_model(best_params)

  #fit the model over training dataset
  model_classifier.fit(X_train, y_train.values.ravel())

  #predict probabilities in test dataset
  y_prob = model_classifier.predict_proba(X_test)[:, 1]

  #evaluate the model according to scoring
  if scoring == "roc_auc":

    #train dataset (using this due to hyperopt limitation)
    list_metric_score_train = cross_val_score(model_classifier,
                                              X = X_train,
                                              y = y_train.values.ravel(),
                                              cv = cv,
                                              scoring = scoring,
                                              n_jobs = -1)

    metric_score_train = np.mean(list_metric_score_train)

    #test dataset
    metric_score_test = roc_auc_score(y_test, y_prob)

  return metric_score_train, metric_score_test

def incorporate_results_experiment(df_experiment,
                                   params_search,
                                   best_params,
                                   scoring,
                                   metric_score_train,
                                   metric_score_test,
                                   max_evals,
                                   timeout,
                                   cv):
  """
  Incorporate the results of the simulation 

  Params:
    df_experiment: dataframe that contains all experiment parameters
    params_search: dictionary with search space of the model
    best_params: best hyperparameters that maximize the optimization metric
    scoring: score to optimize
    metric_score_train: optimization metric for the training set 
    metric_score_test: optimization metric for the testing set
    max_evals: amout of evaluation
    timeout: seconds for timeout
    cv: number of cross validation

  Return:
    df_experiment: dataframe that contains all experiment parameters, added model parameters
  """

  #obtain classifer name
  classifier_name = params_search["classifier_name"]

  #create the tmp_experiment with model information
  tmp_experiment = pd.DataFrame(data = {"model": [classifier_name],
                                        "best_params": [best_params],
                                        "scoring": [scoring],
                                        "max_evals": [max_evals],
                                        "timeout": [timeout],
                                        "cv": [cv],
                                        "score_train": [metric_score_train],
                                        "score_test": [metric_score_test],
                                        }
                                )

  #incorporate the results
  df_experiment = pd.concat([df_experiment, tmp_experiment], axis = 1)

  return df_experiment

def create_evaluate_model(df_experiment,
                          model_params_search,
                          X_train,
                          y_train,
                          X_test,
                          y_test,
                          max_evals,
                          timeout,
                          cv,
                          scoring):
  """
  Function that instantiate the model with the corresponding hyperparameters

  Params:
    df_experiment: dataframe that contains all experiment parameters
    params_search: dictionary with search space of the model
    X_train: features training dataframe
    y_train: features testing dataframe
    X_test: features testing dataframe
    y_test: response testing dataframe
    max_evals: amout of evaluation
    timeout: seconds for timeout
    cv: number of cross validation
    scoring: score to optimize

  Return:
    df_experiment: dataframe that contains all experiment parameters, added model parameters
  """

  #obtain dict of best hyperparameters
  best_params = run_hyperopt(model_params_search,
                            X_train,
                            y_train,
                            max_evals,
                            timeout,
                            cv,
                            scoring)

  #evaluate the model
  metric_score_train, metric_score_test = evaluate_model(best_params,
                                                        X_train,
                                                        y_train,
                                                        X_test,
                                                        y_test,
                                                        cv,
                                                        scoring)

  #incorporate the results over experiment dataframe
  df_experiment = incorporate_results_experiment(df_experiment,
                                                 model_params_search,
                                                 best_params,
                                                 scoring,
                                                 metric_score_train,
                                                 metric_score_test,
                                                 max_evals,
                                                 timeout,
                                                 cv)

  return df_experiment