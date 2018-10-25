import pandas as pd
import numpy as np

from tools.cv_tools import train_test_split
from models.base import site_app_split

import os

def make_field_dict(df, fields):
    """
    fields: Array[String] - a list of column names
    A field dictionary is just an inverted column index.
    """
    return {col: i for i, col in enumerate(fields)}

def make_feature_dict(df, fields):
    # prepend a field name to each feature in order to distinguish
    # a feature name present in two or more fields.
    features = [f'{c}_' + df[c].astype('str') for c in fields]
    #for c in fields:
    #    df[c] = f'{c}_' + df[c].astype('str')
    # TODO: we could hash all features at this stage.
    # TODO: hash into a smaller space to make the dict smaller
    #df[fields] = df[fields].applymap(hash)
    # Index features from all fields.
    features_concat = pd.concat(features, ignore_index=True)
    #features_concat = pd.concat([df[c] for c in fields], ignore_index=True)
    uniques = features_concat.unique()
    return pd.Series(np.arange(len(uniques)),index=uniques)

def encode_features(df, feature_dict, fields):
    # df.replace(feature_dict) doesn't fit in memory.
    # optimize by splitting the feature_dict into dictionaries corresponding to fields.
    replace_dict = dict()
    for c in fields:
        replace_dict[c] = {k: v for k, v in feature_dict.items()
                           if k.startswith(c)}

    return df.replace(replace_dict)


# TODO: would be nice to be able to write column-by-column.
def ffm_row_generator(df, feature_dict, test=False):
    """
    Convert each row to the libffm format, accroding to a provided dict.
    All columns of df are used.
    If test=False, the last column of df must be 'click'; otherwise, 'click' is all one.
    """
    if test:
        assert df.columns[-1] == 'id'
    else:
        assert df.columns[-1] == 'click'

    for _, row in df.iterrows():
        ffm_row = []
        if test:
            ffm_row.append(str(1))
        else:
            ffm_row.append(str(row['click']))
        # if test, ignore 'id', else ignore 'click'.
        for i, v in enumerate(row[:-1]):
            col_name = df.columns[i]
            feature = '_'.join([col_name, str(v)])
            if feature in feature_dict:
                feature_idx = feature_dict[feature]
                ffm_row.append(f'{i}:{feature_idx}:1')
            else:
                ffm_row.append(f'{i}:{feature_dict["__unseen__"]}:1')

        yield ' '.join(ffm_row)
        
def df_to_ffm(df, categorical_features, fname, data_type='train', feature_dict_train=None):
    """
    df: pd.DataFrame
    categorical_features: List[String] - a list of columns to use.
    return: the feature dictionary created to transform columns.
    side effect: Writes to fname and f'{fname}.id'.
    """
    assert data_type in ['train', 'validate', 'test']

    if data_type=='test' or data_type=='validate':
        assert feature_dict_train is not None
        feature_dict = feature_dict_train
        # TODO: do I need unseen for each column?
        feature_dict['__unseen__'] = len(feature_dict)

        if data_type=='validate':
            df = df[categorical_features + ['click']]
        else:
            df = df[categorical_features + ['id']]
    else:
        df = df[categorical_features + ['click']]
        feature_dict = make_feature_dict(df, categorical_features)
    
    test = data_type == 'test'
    with open(fname, 'w') as f:
        for ffm_row in ffm_row_generator(df, feature_dict, test):
            f.write(ffm_row)
            f.write('\n')

    if test:
        with open(fname + '.id', 'w') as f:
            for x in df.id:
                f.write(str(x))
                f.write('\n')

    if data_type == 'train':
        return feature_dict

def submission_rows(prediction, prediction_id):
    with open(prediction, 'r') as probas, open(prediction_id, 'r') as ids:
        for i, val in zip(ids, probas):
            yield f'{i.strip()},{val.strip()}\n'
            
            
def write_submission(prediction_site, prediction_id_site, prediction_app, prediction_id_app, fout):
    with open(fout, 'w') as out:
        out.write('id,click\n')
        for row in submission_rows(prediction_site, prediction_id_site):
            out.write(row)
        for row in submission_rows(prediction_app, prediction_id_app):
            out.write(row)


def make_train_validate_data(df, categorical_features, name):
    test_day = 30
    df_train, df_validate = train_test_split(df, None, test_day)
    df_train_site, df_train_app = site_app_split(df_train)
    df_validate_site, df_validate_app = site_app_split(df_validate)

    ffm_data_path = './ffm-data/'
    train_site_out = os.path.join(ffm_data_path, f'train_site_{name}.ffm')
    validate_site_out = os.path.join(ffm_data_path, f'validate_site_{name}.ffm')
    train_app_out = os.path.join(ffm_data_path, f'train_app_{name}.ffm')
    validate_app_out = os.path.join(ffm_data_path, f'validate_app_{name}.ffm')

    print(train_site_out)
    feature_dict_site = df_to_ffm(df_train_site, categorical_features, train_site_out)
    print(validate_site_out)
    df_to_ffm(df_validate_site, categorical_features, validate_site_out, 'validate', feature_dict_site)
    print(train_app_out)
    feature_dict_app = df_to_ffm(df_train_app, categorical_features, train_app_out)
    print(validate_app_out)
    df_to_ffm(df_validate_app, categorical_features, validate_app_out, 'validate', feature_dict_app)

    return train_site_out, validate_site_out, feature_dict_site, train_app_out, validate_app_out, feature_dict_app
