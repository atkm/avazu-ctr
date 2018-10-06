import pytest
import pkg_resources

import pandas as pd
import numpy as np
import models.model_one
import tools.cv_tools

def test_model_one():
    train_tiny_csv = pkg_resources.resource_stream('data', 'train_tiny.csv')
    df = pd.read_csv(train_tiny_csv)
    df.hour = pd.to_datetime(df.hour, format="%y%m%d%H")
    params = np.logspace(-3, 0, num=10)
    params_dict_ls, scores, test_score = models.model_one.eval_model_one(df, params)
    best_C = tools.cv_tools.best_param(scores, params_dict_ls)
