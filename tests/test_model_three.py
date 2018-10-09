import pytest
import pkg_resources

import pandas as pd
import numpy as np
import models.model_three
from tests.util import ColumnChecker, get_nuniques
from tests.fixtures import df_train_tiny

def test_model_three(df_train_tiny):
    df = df_train_tiny.copy()
    params = np.logspace(-5, -3, num=8)
    best_C, params_dict_ls, scores, test_score = models.model_three.tune_model_three(df, params)
    best_C, dict(zip(params, scores)), test_score

def test_model_three_cols(df_train_tiny):
    df = df_train_tiny.copy()
    pipeline = models.model_three.get_model_three_pipeline()
    pipeline.steps.pop() # remove logistic regression

    categorical_features = models.model_three.categorical_features
    nuniques = get_nuniques(df[categorical_features])
    n_categorical_cols = sum(nuniques.values())
    n_cols = n_categorical_cols + 2 # 2 click rate columns
    checker = ColumnChecker(n_cols)
    pipeline.steps.append(['inspect', checker])
    pipeline.fit_transform(df)
