import pytest
import pkg_resources

import pandas as pd
import numpy as np
import models.model_one_plus
from tests.util import ColumnChecker, get_nuniques

def test_model_one_plus_cols():
    train_tiny_csv = pkg_resources.resource_stream('data', 'train_tiny.csv')
    df = pd.read_csv(train_tiny_csv)
    high_cardinality_features = ['site_id','app_id','device_id']
    pipeline = models.model_one_plus.get_model_one_plus_pipeline(high_cardinality_features)
    pipeline.steps.pop() # remove logistic regression

    categorical_features = models.model_one_plus.categorical_features
    categorical_features = categorical_features + high_cardinality_features
    print(categorical_features)
    nuniques = get_nuniques(df[categorical_features])
    n_cols = sum(nuniques.values())
    checker = ColumnChecker(n_cols)
    pipeline.steps.append(['inspect', checker])
    pipeline.fit_transform(df)
