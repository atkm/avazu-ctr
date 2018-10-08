import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tools.cv_tools import train_test_split, score_params, fit_and_score, best_param

import time

class ClickRateEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols, click_rate_col_name):
        self.cols = cols
        self.col_name = click_rate_col_name
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
