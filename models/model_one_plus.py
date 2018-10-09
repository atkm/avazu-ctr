import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from models.base import tune_logistic_regression_pipeline


categorical_features = ['C1',
        'banner_pos',
        'site_category',
        'app_category',
        'device_type',
        'device_conn_type',
        'C15',
        'C16',
        'C17',
        'C18',
        'C19',
        'C20',
        'C21']

high_cardinality_features = [
        'site_id',
        'app_id',
        'device_id',
        'device_model',
        'C14'
        ]

def get_model_one_plus_pipeline(high_cardinality_cols):
    features = categorical_features.copy()
    for c in high_cardinality_cols:
        assert c in high_cardinality_cols
        features.append(c)

    oh_encoder = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer([
        ('one_hot_encoding', oh_encoder, features),
        ])
    lg = LogisticRegression(solver='liblinear')
    pipeline = Pipeline([
                    ('preprocessing', preprocessor),
                     ('logistic_regression', lg)])
    return pipeline

def tune_model_one_plus(df, params, high_cardinality_cols):
    pipeline = get_model_one_plus_pipeline(interaction)
    return tune_logistic_regression_pipeline(df, pipeline, params)
