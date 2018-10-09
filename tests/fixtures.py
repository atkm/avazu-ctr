import pytest
import pkg_resources

import pandas as pd

@pytest.fixture(scope='session')
def df_train_tiny():
    train_tiny_csv = pkg_resources.resource_stream('data', 'train_tiny.csv')
    df = pd.read_csv(train_tiny_csv)
    df.hour = pd.to_datetime(df.hour, format="%y%m%d%H")
    return df
