#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tools.kaggle_tools import predict_on_test
from models.model_three import get_model_three_pipeline, tune_model_three


# In[2]:


pipeline = get_model_three_pipeline('both')

df_train = pd.read_csv('data/train_tiny.csv')
print("Loaded train.csv")
df_train = df_train.sample(frac=0.5, replace=False)
print("Took a subset of df_train")

param_tuning = True
if param_tuning:
    params = np.logspace(-5, -3, num=8)
    best_C, params_dict_ls, scores, test_score = tune_model_three(df_train, params)
    print('Best C: ', best_C)
    print(dict(zip(params, scores)))
    print('Test score: ', test_score)
    print('Tuning done.')
    param = best_C
else:
    param = {'logistic_regression__C': 0.00026826957952797245}


y_train = df_train.click
df_test = pd.read_csv('data/test_tiny.csv', dtype={'id': 'uint64'})
print("Loaded test.csv")


# In[3]:

fname = 'submission-test.csv'
predict_on_test(df_train, y_train, pipeline, param, df_test, fname)
