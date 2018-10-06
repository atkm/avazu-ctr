import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from tools.cv_tools import (
    fit_and_score, train_test_split, score_one_param, score_one_test_day,
    score_one_param, score_params, best_param
)

def eval_model_one(df, params):
    """
    df: pd.DataFrame. An output of pd.read_csv('train.csv') with its hour column formatted.
    """
    model_one_cols = ['C1',
                 'banner_pos',
                 'site_category',
                 'app_category',
                 'device_type',
                 'device_conn_type',
                 'C15',
                 'C16',
                 'C18',
                 'C19',
                 'C21']

    clicks = df.click
    # need the hour column for splitting
    df = df[model_one_cols + ['hour']]
    # Day 30 is for testing
    X_train, y_train, X_test, y_test = train_test_split(df, clicks, 30)
    test_day_ls = [25,26,27,28,29]


    oh_encoder = OneHotEncoder(handle_unknown='ignore')
    lg = LogisticRegression(solver='lbfgs')
    pipeline = Pipeline([('one_hot_encoding', oh_encoder),
                     ('logistic_regression', lg)])

    C_kwd = 'logistic_regression__C'
    params_dict_ls = [{C_kwd: p} for p in params]

    train_begin = time.time()
    scores = score_params(X_train, y_train, pipeline, params_dict_ls, test_day_ls)
    train_time = time.time() - train_begin
    print("Train time: ", train_time)
    best_C = best_param(scores, params_dict_ls)
    print("Best C: ", best_C)

    # Use the best parameter to evaluate the model on the test set.
    test_begin = time.time()
    test_score = fit_and_score(X_train, y_train, X_test, y_test, pipeline, best_C)
    test_time = time.time() - test_begin
    print("Test time: ", test_time)

    return params_dict_ls, scores, test_score

if __name__=='__main__':
    train_tiny_csv = pkg_resources.resource_stream('data', 'train_tiny.csv')
    df = pd.read_csv(train_tiny_csv)
    df.hour = pd.to_datetime(df.hour, format="%y%m%d%H")
    params = np.logspace(-3, 0, num=10)
    params_dict_ls, scores, test_score = eval_model_one(df, params)
    best_C = best_param(scores, params_dict_ls)
    print(dict(zip(params, scores)))
    print('Test score: ', test_score)
