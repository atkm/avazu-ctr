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


def neg_log_loss_score(lg, X_test, y_test):
    return -log_loss(y_test, lg.predict_proba(X_test))


def train_test_split(X, y, test_day):
    """
    X and y are pd.DataFrame. X must have an hour column of type pd.Timestamp.
    """
    test_day_mask = X.hour.dt.day == test_day
    train_day_mask = X.hour.dt.day < test_day
    X_test = X[test_day_mask]
    y_test = y[test_day_mask]
    X_train = X[train_day_mask]
    y_train = y[train_day_mask]

    return X_train, y_train, X_test, y_test
