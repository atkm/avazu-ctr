import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tools.cv_tools import train_test_split, score_params, fit_and_score, best_param

import time



class CountEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, col, count_col_name):
        """
        col: String - the name of the column to encode
        col_name: String - the name of a column to output the resulting count feature.
        """
        self.col = col
        self.count_col_name = count_col_name
        self.counts = None # a pd.Series
    
    def fit(self, X, y=None):
        self.counts = X.groupby(self.col).size().to_frame(self.count_col_name)
        return self
    
    def transform(self, X):
        X = pd.merge(X, self.counts, how='left', on=self.col)
        X = X.fillna({self.count_col_name: 0})
        # return only the count column
        return X[[self.count_col_name]]

class HourlyCountEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, col, count_col_name):
        """
        col: String - the name of the column to encode
        col_name: String - the name of a column to output the resulting count feature.
        """
        self.col = col
        self.count_col_name = count_col_name
        self.counts = None # a pd.Series
    
    def fit(self, X, y=None):
        self.counts = X.groupby([X.hour, self.col]).size().to_frame(self.count_col_name)
        return self
    
    def transform(self, X):
        X = pd.merge(X, self.counts, how='left', on=['hour', self.col])
        X = X.fillna({self.count_col_name: 0})
        return X[[self.count_col_name]]



class ClickRateEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols, col_name):
        self.cols = cols
        self.col_name = col_name
        self.click_rate = None
        
    def fit(self, X, y=None):
        self.click_rate = X.groupby(self.cols)\
                .agg({'click': 'mean'}).rename({'click': self.col_name}, axis=1)
        return self
        
    def transform(self, X):
        X = pd.merge(X, self.click_rate, how='left', on=self.cols)
        X = X.fillna({self.col_name: 0})
        # test sets don't have a click column
        if 'click' in X.columns:
            X = X.drop('click', axis=1)
            
        return X.drop(self.cols, axis=1)

def create_user(df):
    """
    Adds a user column.
    """
    device_id_null = 'a99f214a'
    df['device_ip_model'] = df.device_ip.str.cat(df.device_model, sep='_')
    df['user'] = df.device_id.where(df.device_id != device_id_null,
                                    df.device_ip_model)
    return df

def tune_logistic_regression_pipeline(df, pipeline, params):
    """
    Tunes 'logistic_regression__C'.
    """
    if df.hour.dtype != np.dtype('datetime64'):
        df.hour = pd.to_datetime(df.hour, format="%y%m%d%H") 

    X_train, y_train, X_test, y_test = train_test_split(df, df.click, 30)
    test_day_ls = [25,26,27,28,29]

    C_kwd = 'logistic_regression__C'
    params_dict_ls = [{C_kwd: p} for p in params]

    train_begin = time.time()
    scores = score_params(X_train, y_train, pipeline, params_dict_ls, test_day_ls)
    train_time = time.time() - train_begin
    print("Tuning time: ", train_time)
    best_C = best_param(scores, params_dict_ls)

    # Use the best parameter to evaluate the model on the test set.
    test_score = fit_and_score(X_train, y_train, X_test, y_test, pipeline, best_C)

    return best_C, params_dict_ls, scores, test_score
