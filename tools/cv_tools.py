from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

def fit_encoders(df, cols):
    """
    Fit LabelEncoder to given columns.
    """
    encoders_dict = dict()
    for c in cols:
        encoder = LabelEncoder()
        encoder.fit(df[c])
        encoders_dict[c] = encoder
    return encoders_dict


def neg_log_loss_score(predictor, X_test, y_test):
    return -log_loss(y_test, predictor.predict_proba(X_test))


def train_test_split(X, y, test_day):
    """
    X and y are pd.DataFrame. X must have an hour column of type pd.Timestamp.
    """
    test_day_mask = X.hour.dt.day == test_day
    train_day_mask = X.hour.dt.day < test_day
    X_test = X[test_day_mask]
    X_train = X[train_day_mask]

    if y is None:
        return X_train, X_test
    else:
        y_test = y[test_day_mask]
        y_train = y[train_day_mask]
        return X_train, y_train, X_test, y_test

def fit_and_score(X_train, y_train, X_dev, y_dev, pipeline, params):

    # Note: the pipeline model should select which columns to use via ColumnTransformer
    # the DataFrame needs the hour column for splitting. Drop the column right before training.
    #if 'hour' in X_train.columns:
    #    X_train = X_train.drop('hour', axis=1)
    #if 'hour' in X_dev.columns:
    #    X_dev = X_dev.drop('hour', axis=1)

    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    return neg_log_loss_score(pipeline, X_dev, y_dev)

def score_one_test_day(X_train, y_train, pipeline, params_dict, test_day):
    """
    params_dict: map from keywords (string) to values
    """
    X_train, y_train, X_dev, y_dev = train_test_split(X_train, y_train, test_day)
    return fit_and_score(X_train, y_train, X_dev, y_dev, pipeline, params_dict)

def score_one_param(X_train, y_train, pipeline, params_dict, test_day_ls):

    scores = []
    for test_day in test_day_ls:
        scores.append(score_one_test_day(X_train, y_train, pipeline, params_dict, test_day))

    assert len(scores) == len(test_day_ls)

    #print('Scores:', scores)
    mean_score = sum(scores)/len(scores)
    #print('Mean: ', mean_score)
    return mean_score

def score_params(X_train, y_train, pipeline, params_dict_ls, test_day_ls):
    """
    Returns a map: parameter -> mean score.
    """

    scores = []
    for params_dict in params_dict_ls:
        scores.append(score_one_param(X_train, y_train, pipeline, params_dict, test_day_ls))
    return scores

def best_param(scores, param_dict_ls):
    maxidx = scores.index(max(scores))
    return param_dict_ls[maxidx]
