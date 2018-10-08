import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
