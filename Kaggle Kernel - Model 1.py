
# coding: utf-8

# 1204^th out of 1604 kernels.
# 
# - Tune parameters locally. So no cross-validation in this notebook.
# - Have an option to train on a subset of train.csv.
# - Need to run fit_encoder on the union of train and test.

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

import time


# In[2]:


# hepler functions

def fit_encoders(df):
    site_category_encoder = LabelEncoder()
    site_category_encoder.fit(df.site_category)
    app_category_encoder = LabelEncoder()
    app_category_encoder.fit(df.app_category)
    return site_category_encoder, app_category_encoder

def fit_transform_train(X_train, site_category_encoder, app_category_encoder):
    X_train.site_category = site_category_encoder.transform(X_train.site_category)
    X_train.app_category = app_category_encoder.transform(X_train.app_category)
    X_train = X_train.values
    # when transforming, an unknown categorical feature is mapped to a zero vector
    oh_encoder = OneHotEncoder(handle_unknown='ignore')
    X_train = oh_encoder.fit_transform(X_train)
    return X_train, oh_encoder

def transform_dev(X_dev, site_category_encoder, app_category_encoder, oh_encoder):
    X_dev.site_category = site_category_encoder.transform(X_dev.site_category)
    X_dev.app_category = app_category_encoder.transform(X_dev.app_category)
    X_dev = oh_encoder.transform(X_dev)
    return X_dev

def neg_log_loss_score(lg, X_dev, y_dev):
    return -log_loss(y_dev, lg.predict_proba(X_dev))


# In[3]:


df_train = pd.read_csv('data/train_small.csv')
# nRows = int(1e7) # 40,428,968 rows in train.csv
df_train = df_train.sample(frac=0.5, replace=False)
y_train = df_train.click

df_test = pd.read_csv('data/test.csv')


# In[4]:


# this step is memory intensive!
# Consider reading in only the columns necessary to fit encoders.
df_concat = pd.concat([df_train, df_test], axis=0, sort=False)
site_category_encoder, app_category_encoder = fit_encoders(df_concat)


# In[5]:


model_one_cols = ['C1',
                 'banner_pos',
                 'site_category',
                 'app_category',
                 'device_type',
                 'device_conn_type',
                 'C15',
                 'C16',
                 'C18',
                 'C19',
                 'C21']

X_train = df_train[model_one_cols]
X_test = df_test[model_one_cols]

X_train, oh_encoder = fit_transform_train(X_train, site_category_encoder, app_category_encoder)
param = 0.021544346900318832
lg = LogisticRegression(C=param)

train_begin = time.time()
lg.fit(X_train, y_train)
train_time = time.time() - train_begin
print("Train time: ", train_time)

X_test = transform_dev(X_test, site_category_encoder, app_category_encoder, oh_encoder)
y_pred = lg.predict_proba(X_test)


# In[6]:


not_click_proba = y_pred[:, 0]
click_proba = y_pred[:, 1]


# In[7]:


click_proba.shape


# In[19]:


df_test = pd.read_csv('data/test.csv', usecols=['id'], dtype={'id': 'uint64'})
row_ids = df_test.id.values


# In[21]:


with open('submission.csv','w') as f:
    f.write('id,click\n')
    for id, click_p in zip(row_ids, click_proba):
        f.write(f'{id},{click_p}\n')

