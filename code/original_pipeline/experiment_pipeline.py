#path configuration
import os
import sys
import ipynb

#data manipulation
import pandas as pd
import numpy as np
from datetime import datetime

#import warning
import warnings
warnings.filterwarnings("ignore")

#reference seed
seed = 42
np.random.seed(seed)

from preprocessing_functions import *
from modeling_functions import *

def run_full_pipeline(df,
                      initial_date,
                      final_date,
                      time_interval,
                      create_lag_features_abs,
                      create_lag_features_rel,
                      list_lag_features,
                      list_columns_filter,
                      test_size,
                      remove_correlation,
                      remove_correlation_params,
                      apply_fa_bool,
                      fa_params,
                      apply_rebalance_strategy,
                      rebalance_params,
                      seed,
                      model_params_search,
                      max_evals,
                      timeout,
                      cv,
                      scoring):
  
  """
  Run the full pipeline, for a specific set of parameters

  Params:
    df: dataframe with DPI indicator, per time, and the response variable
    initial_date: initial date to start (0, 100, 200)
    final_date: final date to finish (100, 200, 300)
    time_interval: time interval to obtain the features (1, 5, 10, 20)
    create_lag_features_abs: bool variable to create absolute difference (True, False)
    create_lag_features_rel: bool variable to create relative difference (True, False)
    list_lag_features: list of lag features to create ([1], [3], [1, 3])
    list_columns_filter: list of columns to filter the dataframe
    test_size: percentage of the test size
    remove_correlation: boolean variable that inform if correlated features will be removed (True, False)
    remove_correlation_params: dictionary with information of correlation removal
    apply_fa_bool: bool variable if FA should be applied (True, False)
    fa_params: Dictionary with FA parameters
    apply_rebalance_strategy: object that defines if the rebalancement should be "under_sampling", "over_sampling" or None
    rebalance_params: rebalancement dictionary with function parameters
    seed: seed used for all random_state and seed functions
    model_params_search: 
    max_evals: amout of evaluation
    timeout: seconds for timeout
    cv: number of cross validation
    scoring: score to optimize
    
  Return:
    df_experiment: dataframe that contains all experiment parameters
  """

  #preprocess and feature engineering
  X_train, X_test, y_train, y_test, df_experiment = preprocess_dataframe(df,
                                                                        initial_date,
                                                                        final_date,
                                                                        time_interval,
                                                                        create_lag_features_abs,
                                                                        create_lag_features_rel,
                                                                        list_lag_features,
                                                                        list_columns_filter,
                                                                        test_size,
                                                                        remove_correlation,
                                                                        remove_correlation_params,
                                                                        apply_fa_bool,
                                                                        fa_params,
                                                                        apply_rebalance_strategy,
                                                                        rebalance_params,
                                                                        seed)

  # create and evaluate model
  df_experiment = create_evaluate_model(df_experiment,
                                        model_params_search,
                                        X_train,
                                        y_train,
                                        X_test,
                                        y_test,
                                        max_evals,
                                        timeout,
                                        cv,
                                        scoring)

  return df_experiment

def adjust_df_experiment(list_tmp_experiment,
                         save_results,
                         name_results):
  """
  Run the full pipeline, for a specific set of parameters

  Params:
    list_tmp_experiment: list of temporary experiments dataframe
    save_results: bool variable to save the result
    name_results: name of the path
  
  Return
    df_experiment: dataframe that contains all experiment parameters
  """

  #add the tmp_experiment over the df_experiment
  df_experiment = pd.concat(list_tmp_experiment,
                            axis = 0)

  #create rank column for best scores
  df_experiment["score_train_rank"] = df_experiment.score_train.rank(method = "first", ascending = False)
  df_experiment["score_test_rank"] = df_experiment.score_test.rank(method = "first", ascending = False)

  #sort values according to test rank and reset index
  df_experiment.sort_values("score_test_rank", inplace = True)
  df_experiment.reset_index(inplace = True, drop = True)

  #save df_experiment
  if save_results == True:
    if name_results != None:
      df_experiment.to_csv(f'df_experiment_{name_results}.csv',
                           index=False)
    else:
      str_actual_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
      df_experiment.to_csv(f'df_experiment_{str_actual_time}.csv',
                           index=False)

  return df_experiment

def run_full_experiment(df,
                        seed,
                        list_initial_date,
                        list_final_date,
                        list_time_interval,
                        list_create_lag_features_abs,
                        list_create_lag_features_rel,
                        list_list_lag_features,
                        list_columns_filter,
                        test_size,
                        list_remove_correlation,
                        list_remove_correlation_params,
                        list_apply_fa_bool,
                        list_fa_params,
                        list_apply_rebalance_strategy,
                        list_rebalance_params,
                        list_model_params_search,
                        max_evals,
                        timeout,
                        scoring,
                        cv,
                        save_results,
                        name_results):
  
  """
  Run the full pipeline, for a set of different parameters

  Params:
    df: dataframe with DPI indicator, per time, and the response variable
    seed: seed used for all random_state and seed functions
    list_initial_date: initial date to start (0, 100, 200)
    list_final_date: final date to finish (100, 200, 300)
    list_time_interval: time interval to obtain the features (1, 5, 10, 20)
    list_create_lag_features_abs: bool variable to create absolute difference (True, False)
    list_create_lag_features_rel: bool variable to create relative difference (True, False)
    list_list_lag_features: list of lag features to create ([1], [3], [1, 3])
    list_columns_filter: list of columns to filter the dataframe
    test_size: percentage of the test size
    list_remove_correlation: boolean variable that inform if correlated features will be removed (True, False)
    list_remove_correlation_params: dictionary with information of correlation removal
    list_apply_fa_bool: bool variable if FA should be applied (True, False)
    list_fa_params: Dictionary with FA parameters
    list_apply_rebalance_strategy: object that defines if the rebalancement should be "under_sampling", "over_sampling" or None
    list_rebalance_params: rebalancement dictionary with function parameters
    list_model_params_search: 
    max_evals: amout of evaluation
    timeout: seconds for timeout
    scoring: score to optimize
    cv: number of cross validation
    save_results: bool variable to save the result
    name_results: name of the path
    
  Return:
    df_experiment: dataframe that contains all experiment parameters
  """

  #create initial empty df_experiment
  list_tmp_experiment = []

  #iterate over all possibilities
  for initial_date, final_date in zip(list_initial_date, list_final_date):
    for time_interval in list_time_interval:
      for create_lag_features_abs, create_lag_features_rel in zip(list_create_lag_features_abs, list_create_lag_features_rel):
        for list_lag_features in list_list_lag_features:

          #for the cases where abs and rel are false, we should run the simulation only once. otherwise, will create duplicated data
          if (
              (create_lag_features_abs == False) &
              (create_lag_features_rel == False) &
              (list_lag_features != list_list_lag_features[0])
          ):
            continue

          for remove_correlation in list_remove_correlation:
            for remove_correlation_params in list_remove_correlation_params:
              for apply_fa_bool in list_apply_fa_bool:
                for fa_params in list_fa_params:
                  for apply_rebalance_strategy in list_apply_rebalance_strategy:
                    for rebalance_params in list_rebalance_params:
                      for params_search in list_model_params_search:

                        #create tmp_experiment
                        tmp_experiment = run_full_pipeline(df,
                                                          initial_date,
                                                          final_date,
                                                          time_interval,
                                                          create_lag_features_abs,
                                                          create_lag_features_rel,
                                                          list_lag_features,
                                                          list_columns_filter,
                                                          test_size,
                                                          remove_correlation,
                                                          remove_correlation_params,
                                                          apply_fa_bool,
                                                          fa_params,
                                                          apply_rebalance_strategy,
                                                          rebalance_params,
                                                          seed,
                                                          params_search,
                                                          max_evals,
                                                          timeout,
                                                          cv,
                                                          scoring)

                        #append the tmp_experiment
                        list_tmp_experiment.append(tmp_experiment)

  #make final adjustments
  df_experiment = adjust_df_experiment(list_tmp_experiment,
                                       save_results,
                                       name_results)

  return df_experiment