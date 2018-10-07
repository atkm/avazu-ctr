#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tools.kaggle_tools import predict_on_test
from models.model_three import get_model_three_pipeline


# In[2]:


pipeline = get_model_three_pipeline('both')
param = {'logistic_regression__C': 0.00026826957952797245}
df_train = pd.read_csv('data/train_small.csv')
print("Loaded train.csv")
df_train = df_train.sample(frac=0.5, replace=False)
print("Took a subset of df_train")
y_train = df_train.click
df_test = pd.read_csv('data/test.csv', dtype={'id': 'uint64'})
print("Loaded test.csv")


# In[3]:

fname = 'submission.csv'
predict_on_test(df_train, y_train, pipeline, param, df_test, fname)
