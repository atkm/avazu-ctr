import pandas as pd
import numpy as np

def make_field_dict(df, fields):
    """
    fields: Array[String] - a list of column names
    A field dictionary is just an inverted column index.
    """
    return {col: i for i, col in enumerate(fields)}

def make_feature_dict(df, fields):
    # prepend a field name to each feature in order to distinguish
    # a feature name present in two or more fields.
    for c in fields:
        df[c] = f'{c}_' + df[c].astype('str')
    # TODO: we could hash all features at this stage.
    # TODO: hash into a smaller space to make the dict smaller
    #df[fields] = df[fields].applymap(hash)
    # Index features from all fields.
    features_concat = pd.concat([df[c] for c in fields], ignore_index=True)
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
            # TODO: what error to throw when v is not a key?
            ffm_row.append(f'{i}:{feature_dict[v]}:1')
        yield ' '.join(ffm_row)
        
def df_to_ffm(df, categorical_features, fname, test=False):
    """
    df: pd.DataFrame
    categorical_features: List[String] - a list of columns to use.
    return: None. Writes to fname and f'{fname}.id'.
    """
    if test:
        df = df[categorical_features + ['id']]
    else:
        df = df[categorical_features + ['click']]
    feature_dict = make_feature_dict(df, categorical_features)
    
    with open(fname, 'w') as f:
        for ffm_row in ffm_row_generator(df, feature_dict, test):
            f.write(ffm_row)
            f.write('\n')

    with open(fname + '.id', 'w') as f:
        for x in df.id:
            f.write(str(x))
            f.write('\n')
