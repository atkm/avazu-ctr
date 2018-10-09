#!/usr/bin/env python
# coding: utf-8

# Done: ['site_id', 'app_id']
# 
# ['site_id', 'app_id', 'device_id']

# In[1]:


import pandas as pd
import numpy as np
from tools.kaggle_tools import predict_on_test
from models.model_one_plus import get_model_one_plus_pipeline, tune_model_one_plus


# In[2]:


high_cardinality_cols = ['device_id','site_id', 'app_id']
pipeline = get_model_one_plus_pipeline(high_cardinality_cols)
df_train = pd.read_csv('data/train_small.csv')

tune = False


# In[3]:



if tune:
    params = np.logspace(-2, 0.5, num=6)
    param, params_dict_ls, scores, test_score = tune_model_one_plus(df_train, params, high_cardinality_cols)
    print('Best C: ', best_C)
    print(dict(zip(params, scores)))
    print('Test score: ', test_score)
else:
    param = {'logistic_regression__C': 0.6309573444801934}


y_train = df_train.click
df_test = pd.read_csv('data/test.csv', dtype={'id': 'uint64'})
fname = 'submission.csv'
predict_on_test(df_train, y_train, pipeline, param, df_test, fname)

