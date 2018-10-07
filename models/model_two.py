import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

model_two_cols = ['C1',
                  'click',
                 'banner_pos',
                  'app_category',
                  'site_category',
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
                 #'app_category',
                  'site_category',
                 'device_type',
                 'device_conn_type',
                 'C15',
                 'C16',
                 'C18',
                 'C19',
                 'C21']

click_rate_cols = ['click', 'app_category',
                  'site_category',
                  'device_id']


class ClickRateByCategoryEncoder(BaseEstimator, TransformerMixin):
    
    user_site_interaction_cols = ['site_category', 'device_id']
    user_app_interaction_cols = ['app_category', 'device_id']
    
    def __init__(self, interaction='both'):
        """
        interaction can be 'user-site', 'user-app', or 'both' (default).
        """
        assert interaction in ['user-app','user-site','both']
        self.interaction = interaction
        self.click_rates_by_site_category = None
        self.click_rates_by_app_category = None
    
    def fit(self, X, y=None):
        """
        X must have the following columns: 'click', 'site_category', 'app_category', and 'device_id'.
        Returns a transformed DataFrame with 'click_rate_site' and 'click_rate_app'.
        The 'click' column is dropped.
        """
        if self.interaction != 'user-app':
            self.click_rates_by_site_category = X.groupby(ClickRateByCategoryEncoder.user_site_interaction_cols)\
                .agg({'click': 'mean'}).rename({'click': 'click_rate_site'}, axis=1)
        
        if self.interaction != 'user-site':
            self.click_rates_by_app_category = X.groupby(ClickRateByCategoryEncoder.user_app_interaction_cols)\
                .agg({'click': 'mean'}).rename({'click': 'click_rate_app'}, axis=1)
    
        return self
    
    def transform(self, X):
        # TODO: need to deal with nulls that appear on rows without matching (device_id, category) rows.
        if self.interaction != 'user-app':
            X = pd.merge(X, self.click_rates_by_site_category, how='left',
                      on=ClickRateByCategoryEncoder.user_site_interaction_cols)
            X = X.fillna({'click_rate_site': 0})
            
        if self.interaction != 'user-site':
            X = pd.merge(X, self.click_rates_by_app_category, how='left',
                      on=ClickRateByCategoryEncoder.user_app_interaction_cols)
            X = X.fillna({'click_rate_app': 0})
        
        # test sets don't have a click column
        if 'click' in X.columns:
            X = X.drop('click', axis=1)
        return X.drop(['device_id','app_category','site_category'], axis=1)


def get_model_two_pipeline(interaction='both'):
    """
    interaction in ['both', 'user-app', 'user-site']
    """
    cr_encoder = ClickRateByCategoryEncoder(interaction)
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
