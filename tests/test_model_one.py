import pytest
import pkg_resources

import pandas as pd
import numpy as np
import models.model_one
import tools.cv_tools

from tests.fixtures import df_train_tiny

def test_model_one(df_train_tiny):
    df = df_train_tiny.copy()
    params = np.logspace(-3, 0, num=10)
    params_dict_ls, scores, test_score = models.model_one.eval_model_one(df, params)
    best_C = tools.cv_tools.best_param(scores, params_dict_ls)
