import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from models.base import tune_logistic_regression_pipeline, CountEncoder, HourlyCountEncoder


model_four_cols = ['id','click','hour','banner_pos',
                        'device_id','device_ip','device_model',
                        'device_conn_type','C14','C17','C20','C21']

categorical_features = ['banner_pos', 'site_id', 'site_category',
        'app_id', 'app_category', 'user',
                        'device_conn_type','C14','C17','C20','C21']

def get_model_four_pipeline():
    device_id_encoder = CountEncoder('device_id', 'device_id_count')
    device_ip_encoder = CountEncoder('device_ip', 'device_ip_count')
    user_encoder = CountEncoder('user', 'user_count')
    hourly_user_encoder = HourlyCountEncoder('user', 'hourly_user_count')
    oh_encoder = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer([
        ('one_hot_encoding', oh_encoder, categorical_features),
        ('count_encoding_user', user_encoder, ['user']),
        ('hourly_count_encoding_user', hourly_user_encoder, ['hour', 'user']),
        ('count_encoding_device_id', device_id_encoder, ['device_id']),
        ('count_encoding_device_ip', device_ip_encoder, ['device_ip'])
    ])
    
    lg = LogisticRegression(solver='liblinear')
    pipeline = Pipeline([
                    ('preprocessing', preprocessor),
                     ('logistic_regression', lg)])
    return pipeline

def tune_model_four(df, params):
    pipeline = get_model_four_pipeline()
    return tune_logistic_regression_pipeline(df, pipeline, params)
