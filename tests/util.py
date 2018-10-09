import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def get_nuniques(df):
    unique_vals = dict()
    for c in df:
        unique_vals[c] = df[c].nunique()
    return unique_vals


class ColumnChecker(BaseEstimator, TransformerMixin):

    def __init__(self, n_cols):
        self.n_cols = n_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n = X.shape[1]
        assert n == self.n_cols, f"Expected {self.n_cols} columns, but got {n}."
        
        return X
