import numpy as np
import time

def predict_on_test(df_train, y_train, pipeline, param, df_test, fname='submission.csv'):
    row_ids = df_test.id.values
    assert row_ids.dtype == np.dtype('uint64'), "Read click data with dtype={'id': 'uint64'} to ensure the id column is not corrupted."
    pipeline.set_params(**param)

    train_begin = time.time()
    pipeline.fit(df_train, y_train)
    train_time = time.time() - train_begin
    print("Train time: ", train_time)

    test_begin = time.time()
    y_pred = pipeline.predict_proba(df_test)
    test_time = time.time() - test_begin
    print("Prediction time: ", test_time)

    click_proba = y_pred[:, 1]

    if fname is None:
        print('Done. Results not written.')
        return

    with open(fname,'w') as f:
        f.write('id,click\n')
        for id, click_p in zip(row_ids, click_proba):
            f.write(f'{id},{click_p}\n')

    print('Done. Results written to ', fname)
