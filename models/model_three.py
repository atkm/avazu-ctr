import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

model_three_cols = ['C1',
                  'click',
                 'banner_pos',
                  'app_id',
                  'site_id',
                  'device_id',
                 'device_type',
                 'device_conn_type',
                 'C15',
                 'C16',
                 'C18',
                 'C19',
                 'C21']

categorical_features = ['C1',
              'banner_pos',
              'app_id',
              'site_id',
             'device_type',
             'device_conn_type',
             'C15',
             'C16',
             'C18',
             'C19',
             'C21']

click_rate_cols = ['click', 'app_id',
              'site_id',
              'device_id']

class ClickRateBySiteEncoder(BaseEstimator, TransformerMixin):
    
    user_site_interaction_cols = ['site_id', 'device_id']
    user_app_interaction_cols = ['app_id', 'device_id']
    
    def __init__(self, interaction='both'):
        """
        interaction can be 'user-site', 'user-app', or 'both' (default).
        """
        assert interaction in ['user-app','user-site','both']
        self.interaction = interaction
        self.click_rates_by_site_id = None
        self.click_rates_by_app_id = None
    
    def fit(self, X, y=None):
        """
        X must have the following columns: 'click', 'site_id', 'app_id', and 'device_id'.
        Returns a transformed DataFrame with 'click_rate_site_id' and 'click_rate_app_id'.
        The 'click' column is dropped.
        """
        if self.interaction != 'app-site':
            self.click_rates_by_site_id = X.groupby(ClickRateBySiteEncoder.user_site_interaction_cols)\
                .agg({'click': 'mean'}).rename({'click': 'click_rate_site'}, axis=1)
        
        if self.interaction != 'user-site':
            self.click_rates_by_app_id = X.groupby(ClickRateBySiteEncoder.user_app_interaction_cols)\
                .agg({'click': 'mean'}).rename({'click': 'click_rate_app'}, axis=1)
    
        return self
    
    def transform(self, X):
        if self.interaction != 'app-site':
            X = pd.merge(X, self.click_rates_by_site_id, how='left',
                      on=ClickRateBySiteEncoder.user_site_interaction_cols)
            X = X.fillna({'click_rate_site': 0})
            
        if self.interaction != 'user-site':
            X = pd.merge(X, self.click_rates_by_app_id, how='left',
                      on=ClickRateBySiteEncoder.user_app_interaction_cols)
            X = X.fillna({'click_rate_app': 0})
        
        # test sets don't have a click column
        if 'click' in X.columns:
            X = X.drop('click', axis=1)
            
        return X.drop(['device_id','site_id','app_id'], axis=1)


def get_model_three_pipeline(interaction='both'):
    cr_encoder = ClickRateBySiteEncoder(interaction)
    oh_encoder = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer([
        ('one_hot_encoding', oh_encoder, categorical_features),
        ('click_rate_encoding', cr_encoder, click_rate_cols)
    ])

    lg = LogisticRegression(solver='liblinear')
    pipeline = Pipeline([
                    ('preprocessing', preprocessor),
                     ('logistic_regression', lg)])
    return pipeline
