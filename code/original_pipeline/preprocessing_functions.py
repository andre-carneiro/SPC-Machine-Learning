#path configuration
import os
import sys

#data manipulation
import pandas as pd
import numpy as np

#train test split
from sklearn.model_selection import train_test_split

#over and under sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

#import warning
import warnings
warnings.filterwarnings("ignore")

#reference seed
seed = 42
np.random.seed(seed)

def separate_and_filter_X_y_dataframes(df,
                                       initial_date,
                                       final_date,
                                       time_interval):
    """
    Separate the dataframe in features (X) and response variable (y), and filter it, if necessary

    Params:
        df: dataframe with DPI indicator, per time, and the response variable
        initial_date: initial date to start (0, 100, 200)
        final_date: final date to finish (100, 200, 300)
        time_interval: time interval to obtain the features (1, 5, 10, 20)

    Return:
        X: features dataframe
        y: response dataframe
    """

    #separate the original dataframe into X and y
    X = df.drop(columns = ["late_project"])
    y = df[["late_project"]].astype(int)

    #use time_interval to filter X dataframe (e.g.: if time_interval is 10 and final_date is 200, it goes as 10, 20 ... 200)
    X = X.iloc[:, initial_date + time_interval - 1 : final_date : time_interval]

    return X, y

def create_lag_features(X,
                        create_lag_features_abs,
                        create_lag_features_rel,
                        list_lag_features):
  
  """
  Create the lag features (difference from one review period to another). Can be both in absolute or relative terms, and can calculate differente
  lags at once (such as 1 and 3)

  Params:
      X: features dataframe
      create_lag_features_abs: bool variable to create absolute difference (True, False)
      create_lag_features_rel: bool variable to create relative difference (True, False)
      list_lag_features: list of lag features to create ([1], [3], [1, 3])

  Return:
      X: features dataframe, with lag features
  """

  #create the abs increase, according to a pre_defined lag list

  if (create_lag_features_abs == True) | (create_lag_features_rel == True):

    #number of columns
    number_columns = len(X.columns)

    # iterate over lag list
    for lag in list_lag_features:

      #iterate over all columns
      for col_index in range(lag, number_columns):

        #try to creat the column (will fail lag times)
        try:

          #obtain the name of the column (time) to iterate
          col_name_iterate = X.columns[col_index]

          #obtain the name of the column (time) to subtract
          col_name_subtract = X.columns[col_index-lag]

          #create the absolute new feature
          if create_lag_features_abs == True:

            #create the new column name
            col_name_new = f"{col_name_iterate}_minus_{col_name_subtract}_abs"

            #create the lag column
            X.loc[:,col_name_new] = X.iloc[:,col_index] - X.iloc[:,col_index-lag]

          #create the relative new feature
          if create_lag_features_rel == True:

            #create the new column name
            col_name_new = f"{col_name_iterate}_minus_{col_name_subtract}_rel"

            #create the lag column
            X.loc[:,col_name_new] = (X.iloc[:,col_index] - X.iloc[:,col_index-lag])/X.iloc[:,col_index]

        except:
          pass

  return X

def apply_train_test_split(X,
                           y,
                           test_size,
                           seed):
  """
  Split the data into train and test dataframes

  Params:
    X: features dataframe
    y: response dataframe
    test_size: percentage of the test size
    seed: seed parameter for stratification

  Return:
    X_train: features training dataframe
    X_test: features testing dataframe
    y_train: response traning dataframe
    y_test: response testing dataframe
  """

  #use sklearn train_test split function, and stratify according to y proportion
  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      test_size = test_size,
                                                      random_state = seed,
                                                      stratify = y)

  return X_train, X_test, y_train, y_test

def remove_correlated_features(X_train,
                               X_test,
                               remove_correlation,
                               remove_correlation_params):
    """
    Remove correlated features, based on the remove_correlation_params dictionary

    Params:
        X_train: features training dataframe
        X_test: features testing dataframe
        remove_correlation: boolean variable that inform if correlated features will be removed (True, False)
        remove_correlation_params: dictionary with information of correlation removal

    Return:
        X_train: features training dataframe
        X_test: features testing dataframe
        X_columns_names: list of new X dataframe columns (used after applying preprocessing)
        X_indexes_names:  list of new X dataframe indexes (used after applying preprocessing)
    """

    #verify if correlated features should be removed
    if remove_correlation == True:

        #obtain parameters of the dictionary
        method = remove_correlation_params["method"]
        correlation = remove_correlation_params["correlation"]

        #iterate at most all columns
        for _ in range (len(X_train.columns)):

            #obtain module of correlation matrix
            corr_matrix = X_train.corr(method).abs()

            #obtain the index with more correlated periods
            max_correlation_period = (corr_matrix > correlation).sum().idxmax()

            #obtain list of columns to drop
            list_periods_drop = corr_matrix.loc[corr_matrix[max_correlation_period] > correlation].index.to_list()

            #do not drop the period with the most correlations
            list_periods_drop.remove(max_correlation_period)

            #if the list is empty, break the loop
            if len(list_periods_drop) == 0:
                break

            #drop columns of original dataframe
            X_train.drop(list_periods_drop, axis=1, inplace=True)
            X_test.drop(list_periods_drop, axis=1, inplace=True)

    else:
        pass

    #obtain columns name
    X_columns_names = X_train.columns
    X_indexes_names_train = X_train.index
    X_indexes_names_test = X_test.index

    return X_train, X_test, X_columns_names, X_indexes_names_train, X_indexes_names_test

def apply_standard_scaler(X_train,
                          X_test):
    """
    Standardize dataframe

    Params:
        X_train: features training dataframe
        X_test: features testing dataframe

    Return:
        X_train: features training dataframe, standardized
        X_test: features testing dataframe, standardized
    """

    #instantiate standard scaler
    standard_scaler = StandardScaler()

    #Fit X_train
    standard_scaler.fit(X_train)

    #Transform X_train and X_test
    X_train = standard_scaler.transform(X_train)
    X_test = standard_scaler.transform(X_test)

    return X_train, X_test

def apply_fa(X_train,
              X_test,
              apply_fa_bool,
              fa_params,
              seed):
  """
  Apply FA, if needed. The FA params (such as explained variance) are inside the fa_params dictionary.

  Params:
      X_train: features training dataframe
      X_test: features testing dataframe
      apply_fa_bool: bool variable if FA should be applied (True, False)
      fa_params: Dictionary with FA parameters
      seed: seed parameter for FA

  Return:
      X_train: features training dataframe, with components
      X_test: features testing dataframe, with components
  """

  #verify if FA should be applied
  if apply_fa_bool == True:

    #obtain parameter
    explained_variance = fa_params["explained_variance"]

    #instantiate PCA to obtain the amount of components
    pca = PCA(n_components = explained_variance,
              random_state = seed)

    #fit x_train
    pca.fit(X_train)

    #obtain amount of components
    n_components_pca = len(pca.components_)

    #apply FA with varimax rotation
    fa = FactorAnalysis(n_components = n_components_pca,
                        rotation = "varimax",
                        max_iter = 100,
                        random_state = seed)
    
    #fit X_train
    fa.fit(X_train)

    #Transform X_train and X_test
    X_train = fa.transform(X_train)
    X_test = fa.transform(X_test)

  else:
    pass

  return X_train, X_test

def apply_rebalancement(X_train,
                        y_train,
                        apply_rebalance_strategy,
                        rebalance_params):
  """
  Apply rebalancement, if needed. It is possible to have a under_sampling

  Params:
    X_train: features training dataframe
    y_train: response training dataframe
    apply_rebalance_strategy: object that defines if the rebalancement should be "under_sampling", "over_sampling" or None
    rebalance_params: rebalancement dictionary with function parameters

  Return:
    X_train: features training dataframe, with (or not) rebalancement
    y_train: response training dataframe, with (or not) rebalancement
  """

  if apply_rebalance_strategy == "under_sampling":
    X_train, y_train = apply_under_sampling(X_train,
                                            y_train,
                                            rebalance_params)

  elif apply_rebalance_strategy == "over_sampling":
    X_train, y_train = apply_over_sampling(X_train,
                                           y_train,
                                           rebalance_params)
  else:
    pass

  return X_train, y_train

def apply_under_sampling(X_train,
                         y_train,
                         rebalance_params):
  """
  Apply undersampling technique

  Params:
      X_train: features training dataframe
      y_train: response training dataframe
      rebalance_params: rebalancement dictionary with function parameters

  Return:
      X_train: features training dataframe, with undersampling
      y_train: response training dataframe, with undersampling
  """
  
  #instantiate under sampler
  rus = RandomUnderSampler(**rebalance_params)

  #apply undersampler
  X_train, y_train = rus.fit_resample(X_train, y_train)

  return X_train, y_train

def apply_over_sampling(X_train,
                        y_train,
                        rebalance_params):
  """
  Apply oversampling technique

  Params:
      X_train: features training dataframe
      y_train: response training dataframe
      rebalance_params: rebalancement dictionary with function parameters

  Return:
      X_train: features training dataframe, with oversampling
      y_train: response training dataframe, with oversampling
  """

  #instantiate SMOTE
  sm = SMOTE(**rebalance_params)

  #apply SMOTE
  X_train, y_train = sm.fit_resample(X_train, y_train)

  return X_train, y_train

def adjust_columns(X_train,
                   y_train,
                   X_test,
                   apply_fa_bool,
                   apply_rebalance_strategy,
                   X_columns_names,
                   X_indexes_names_train, 
                   X_indexes_names_test):
  
  """
  Adjust training and testing dataframe columns and indexes

  Params:
      X_train: features training dataframe
      y_train: response training dataframe
      X_test: features testing dataframe
      apply_fa_bool: boolean variable that determines if FA was applied to the dataframe. If so, the name of the features are components, 
    instead of the original period time
      apply_rebalance_strategy: 
      X_columns_names: list of columns names
      X_indexes_names: list of indexes names

  Return:
      X_train: features training dataframe, with columns and indexes adjusted
      X_test: features testing dataframe, with columns and indexes adjusted
      y_train: response training dataframe
  """

  #using FA and not using rebalance strategy
  if (apply_fa_bool == True) & (apply_rebalance_strategy != None):
    X_train = pd.DataFrame(X_train, columns = [f"component_{i}" for i in range(1, len(X_train[0]) + 1)])
    y_train = pd.DataFrame(y_train, columns = ["late_project"])
    X_test = pd.DataFrame(X_test, columns = [f"component_{i}" for i in range(1, len(X_test[0]) + 1)], index = X_indexes_names_test)
  
  #not using FA and not using rebalance strategy
  elif (apply_fa_bool == False) & (apply_rebalance_strategy != None):
    X_train = pd.DataFrame(X_train, columns = X_columns_names)
    y_train = pd.DataFrame(y_train, columns = ["late_project"])
    X_test = pd.DataFrame(X_test, columns = X_columns_names, index = X_indexes_names_test)

  #using FA and not using rebalance strategy
  elif (apply_fa_bool == True) & (apply_rebalance_strategy == None):
    X_train = pd.DataFrame(X_train, columns = [f"component_{i}" for i in range(1, len(X_train[0]) + 1)], index = X_indexes_names_train)
    X_test = pd.DataFrame(X_test, columns = [f"component_{i}" for i in range(1, len(X_test[0]) + 1)], index = X_indexes_names_test)

  #not using FA and not using rebalance strategy
  elif (apply_fa_bool == False) & (apply_rebalance_strategy == None):
    X_train = pd.DataFrame(X_train, columns = X_columns_names, index = X_indexes_names_train)
    X_test = pd.DataFrame(X_test, columns = X_columns_names, index = X_indexes_names_test)

  return X_train, X_test, y_train

def filter_X_columns(X_train,
                     X_test,
                     list_columns_filter):
  """
  Filter X Columns, used after feature selection technique

  Params:
    X_train: features training dataframe
    X_test: features testing dataframe
    list_columns_filter: list of columns to filter the dataframe

  Return:
    X_train: features training dataframe, filtered
    X_test: features testing dataframe, filtered
  """

  #select only wanted columns for X. Mostly used after feature selection
  if list_columns_filter == None:
    pass

  else:
    X_train = X_train.loc[:,list_columns_filter]
    X_test = X_test.loc[:,list_columns_filter]

  return X_train, X_test

def create_experiment_df(initial_date,
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
                         seed):
  
  """
  Create the experiment dataframe, which contains all parameters used for running the simulation

  Params:
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
    
  Return:
    df_experiment: dataframe that contains all experiment parameters
  """

  #obtain params, if none was used
  if apply_rebalance_strategy == None:
    rebalance_params = np.nan

  if apply_fa_bool == False:
    fa_params = np.nan

  if (create_lag_features_abs == False) and (create_lag_features_rel == False):
    list_lag_features = None

  if remove_correlation == False:
    remove_correlation_params = None

  #create the df_experiment
  df_experiment = pd.DataFrame(data = {"initial_date": [initial_date],
                                      "final_date": [final_date],
                                       "time_interval": [time_interval],
                                       "create_lag_features_abs": [create_lag_features_abs],
                                       "create_lag_features_rel": [create_lag_features_rel],
                                       "list_lag_features": [list_lag_features],
                                       "list_columns_filter": [list_columns_filter],
                                       "seed": [seed],
                                       "test_size": [test_size],
                                       "remove_correlation": [remove_correlation],
                                       "remove_correlation_params": [remove_correlation_params],
                                       "apply_fa": [apply_fa_bool],
                                       "fa_params": [fa_params],
                                       "apply_rebalance_strategy": [apply_rebalance_strategy],
                                       "rebalance_params": [rebalance_params]})

  return df_experiment

def preprocess_dataframe(df,
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
                         seed):

  """
  Create the experiment dataframe, which contains all parameters used for running the simulation

  Params:
    df: 
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
    
  Return:
    X_train: features training dataframe
    X_test: features testing dataframe
    y_train: response training dataframe
    y_test: response testing dataframe
    df_experiment: dataframe that contains all experiment parameters
  """

  #separate and filter the original dataframe
  X, y = separate_and_filter_X_y_dataframes(df,
                                            initial_date,
                                            final_date,
                                            time_interval)

  #create new features, if necessary
  X = create_lag_features(X,
                          create_lag_features_abs,
                          create_lag_features_rel,
                          list_lag_features)

  #train test split
  X_train, X_test, y_train, y_test = apply_train_test_split(X,
                                                            y,
                                                            test_size,
                                                            seed)

  #remove high correlated features
  X_train, X_test, X_columns_names, X_indexes_names_train, X_indexes_names_test = remove_correlated_features(X_train,
                                                                             X_test,
                                                                             remove_correlation,
                                                                             remove_correlation_params)

  #standardize data
  X_train, X_test = apply_standard_scaler(X_train,
                                          X_test)

  #apply, if necessary, the FA
  X_train, X_test = apply_fa(X_train,
                              X_test,
                              apply_fa_bool,
                              fa_params,
                              seed)

  #rebalance training dataframe
  X_train, y_train = apply_rebalancement(X_train,
                                         y_train,
                                         apply_rebalance_strategy,
                                         rebalance_params)

  # adjust the columns of treated data
  X_train, X_test, y_train = adjust_columns(X_train,
                                            y_train,
                                            X_test,
                                            apply_fa_bool,
                                            apply_rebalance_strategy,
                                            X_columns_names,
                                            X_indexes_names_train,
                                            X_indexes_names_test)

  #select wanted columns
  X_train, X_test = filter_X_columns(X_train,
                                     X_test,
                                     list_columns_filter)

  #crete experiment dataframe
  df_experiment = create_experiment_df(initial_date,
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

  return X_train, X_test, y_train, y_test, df_experiment