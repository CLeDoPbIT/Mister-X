import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import rr_metrics
from filter_rrs import MAX_RR_INTERVAL, MIN_RR_INTERVAL

WINDOW_SIZE = 10
MAX_RR_INTERVAL = 1.4
MIN_RR_INTERVAL = 0.4
COEF_B = 0.1
COEF_A = 0.9
MAX_RR_INTERVAL_DIFF = 0.25
MAX_RR_INTERVAL_DIFF_RATIO = 0.25


def calculate_params(row_rr, postfix=''):
    hrv_parameters = dict()
    hrv_parameters['dRR'+postfix]= np.max(row_rr) - np.min(row_rr)

    hrv_parameters['RMSSD'+postfix] = np.sqrt(np.sum(np.diff(row_rr) ** 2) / (len(row_rr) - 1))
    hrv_parameters['NN50'+postfix] = np.mean(abs(np.diff(row_rr)) > 0.05)
    hrv_parameters['SDNN'+postfix]= np.sqrt(np.sum((row_rr - np.average(row_rr)) ** 2) / (len(row_rr) - 1))

    hrv_parameters['CV'+postfix] = (hrv_parameters['SDNN'+postfix] / np.average(row_rr))
    hrv_parameters['IDM'+postfix] = 0.5 * hrv_parameters['RMSSD'+postfix] / np.average(row_rr)
    hrv_parameters['IMA'+postfix] = 1 - (0.5 * hrv_parameters['IDM'+postfix] / hrv_parameters['CV'+postfix]) - 0.3
    return hrv_parameters

def split_data_on_windows_with_features(data_df: pd.DataFrame, wnds_df, window_size, y_key, features_list):
    wnds_df = wnds_df[['id', 'ts_start', 'ts_end']+features_list]
    for feature in features_list:
        feature_vals = wnds_df[feature].values
        wnds_df[feature] = (feature_vals - feature_vals.min())/(feature_vals.max()-feature_vals.min())
    X = []
    y = []
    errors = []
    for i in wnds_df.id.unique():
        exp_metrics = wnds_df[wnds_df.id == i]
        exp_df = data_df[data_df.id == i]
        for j, row in exp_metrics.iterrows():
            start = np.round(row['ts_start'] * 1000)
            end = np.round(row['ts_end'] * 1000)
            x_time = exp_df.time.values
            wnd_start = np.argwhere(x_time == start)[0][0]
            chunk = exp_df.iloc[wnd_start:wnd_start + 16]
            x_chunk = chunk['x'].values
            y_chunk = chunk['cut_y'].values
            data = [x_chunk]
            for feature in features_list:
                data.append(np.array([row[feature]] * len(x_chunk)))
            errors_chunk = chunk['corrected'].values
            features_chunk = np.stack(data)
            if features_chunk.shape != (len(features_list)+1, 16):
                print(j, features_chunk.shape)
            X.append(features_chunk)
            y.append(y_chunk)
            errors.append(errors_chunk)
    X = np.array(X)
    y = np.array(y)
    errors = np.array(errors)
    return X, y, errors

def normalize_min_max(X: np.array, errors: np.array):
    X = (X - MIN_RR_INTERVAL) / (MAX_RR_INTERVAL - MIN_RR_INTERVAL)
    X[errors.astype(bool)] = -1
    return X

def normalize_min_max_expwise(df):
    xs = []
    for id in df.id.unique():
        df_exp = df[df.id == id]
        df_exp.loc[df.corrected == 1, 'x'] = np.mean(df_exp.x.values)
        min_x = df_exp.x.min()
        max_x = df_exp.x.max()
        new_x = (df_exp.x.values-np.mean(df_exp[df_exp.corrected == 0].x.values))/np.std(df_exp[df_exp.corrected == 0].x.values)
        xs.append(new_x)
    xs = np.concatenate(xs)
    df.x = xs
    df.loc[df.corrected==1, 'x'] = -1
    return df

def split_data_on_windows(df: pd.DataFrame, window_size=WINDOW_SIZE, y_key='y', return_mean_y=False, no_gt=False,
                          filter_train=False, generate_metrics=True):
    X = []
    y = []
    errors = []
    metrics = []
    ids = []
    for id in df['id'].unique():
        df_id = df[df['id'].values == id]
        for i in range(0, len(df_id) + 1 - window_size):
            x_chunk = df_id['x'].values[i: i + window_size]
            if not no_gt:
                y.append(df_id[y_key].values[i: i + window_size])
            errors.append(df_id['Error'].values[i: i + window_size])
            if generate_metrics:
                params = calculate_params(x_chunk)
                params = np.array(list(params.values()))
                metrics.append(params)
            X.append(x_chunk)
            X[-1][errors[-1] == 1] = 0
    X = np.array(X)
    y = np.array(y)
    errors = np.array(errors)
    metrics = np.array(metrics)
    if return_mean_y:
        y = np.mean(y, axis=1)
    return X, y, errors, metrics


def thresholder(rrs):
    corrected_mask = np.zeros(len(rrs), dtype=bool)
    rr_iir = np.median(rrs)
    for i in range(len(rrs)):
        rr_iir = COEF_A * rr_iir + COEF_B * rrs[i]
        if rrs[i] > MAX_RR_INTERVAL or rrs[i] < MIN_RR_INTERVAL:
            # rrs[i] = rr_iir  #(rrs[i-1] + rrs[i+1])/2
            corrected_mask[i] = True
    return corrected_mask


def thresholder_diff(rrs, y):
    corrected_mask = np.zeros(len(rrs), dtype=bool)
    rr_iir = np.median(rrs)
    for i in range(1, len(rrs)):
        rr_iir = COEF_A * rr_iir + COEF_B * rrs[i]
        if abs(rrs[i] - rrs[i-1]) > MAX_RR_INTERVAL_DIFF or abs(rrs[i] - rrs[i-1]) / rrs[i-1] > MAX_RR_INTERVAL_DIFF_RATIO:
            assert y[i] != 1 or y[i-1] != 1 or y[i+1] != 1, str(i) #  and y[i-1] != 1 fails
            rrs[i] = rr_iir
            corrected_mask[i] = True
    return corrected_mask


def normalize_min_max(X: np.array, errors=None):
    X = (X - MIN_RR_INTERVAL) / (MAX_RR_INTERVAL - MIN_RR_INTERVAL)
    if errors is None:
        errors = np.zeros(len(X), dtype=bool)
    X[errors.astype(bool)] = -1
    return X


def window_predicts_to_honest_predicts(test_df, windowed_predicts, errors, metrcis=None, window_size=WINDOW_SIZE, threshold=0,
                                       get_continious=False, predict_last_only=False):
    """
    :param test_df: dataframe with rr_intervals and labels
    :param windowed_predicts: 1d array with binary class assigned to window (Note:it shouldn't be shuffeled)
    :param window_size:
    :param threshold: if probability of rr-interval is higher than threshold, label it as covid
    :return: "honest" labeling for each rr-interval in test_df
    """
    def window_predicts_to_segmentation(preds, rrs_count, error, metrcis_exp, threshold=threshold):
        predicted_labels = np.zeros(rrs_count)
        counts = np.zeros(rrs_count)
        metrcis_out = np.zeros((rrs_count, 7))
        for i, pred in enumerate(preds):
            if predict_last_only:
                predicted_labels[i + window_size-1] = pred
            else:
                if len(preds.shape) == 1:
                    predicted_labels[i:i + window_size] += np.array([pred] * window_size)
                else:
                    predicted_labels[i:i + window_size] += pred
            if metrcis is not None:
                metrcis_out[i:i + window_size] += metrcis_exp[i]
            counts[i:i + window_size] += 1
        non_zero_mask = counts != 0
        if not predict_last_only:
            predicted_labels[non_zero_mask] = predicted_labels[non_zero_mask] / counts[non_zero_mask]
        return (predicted_labels > threshold).astype(int), predicted_labels, metrcis_out
    predicts_index = 0
    honest_predicts = []
    honest_predicts_continious = []
    metrcis_continious = []
    for exp_id, exp_df in test_df.groupby('id'):
        exp_len = len(exp_df)
        predicts_exp = windowed_predicts[predicts_index:predicts_index + exp_len - window_size + 1]
        errors_exp = errors[predicts_index:predicts_index + exp_len - window_size + 1]
        if metrcis is not None:
            metrcis_exp = metrcis[predicts_index:predicts_index + exp_len - window_size + 1]
        else:
            metrcis_exp = None
        segm_preds, segm_preds_continious, metrcis_exp = window_predicts_to_segmentation(predicts_exp, exp_len, errors_exp, metrcis_exp)
        predicts_index += len(predicts_exp)
        honest_predicts.append(segm_preds)
        honest_predicts_continious.append(segm_preds_continious)
        metrcis_continious.append(metrcis_exp)
    honest_predicts = np.concatenate(honest_predicts)
    honest_predicts_continious = np.concatenate(honest_predicts_continious)
    metrcis_continious = np.concatenate(metrcis_continious)
    if not get_continious:
        return honest_predicts
    else:
        return honest_predicts, honest_predicts_continious, metrcis_continious


if __name__ == '__main__':
    #example
    full_data_path = 'Z:\\users\\ashilov2\\colleguches\\data\\train_cut.csv'
    df = pd.read_csv(full_data_path)
    #1)  split on train|test
    train_mask = df['id'].values < 2 * 275 / 3
    test_mask = df['id'].values > 2 * 275 / 3
    df_train = df[train_mask]
    df_test = df[test_mask]
    # 2) split on windows
    X_train, y_train = split_data_on_windows(df_train)
    X_test, y_test = split_data_on_windows(df_test)
    print(X_train.shape, y_train.shape, X_test.shape)
    # 3) Learn classifier
    import sklearn
    import sklearn.neighbors
    from sklearn.metrics import f1_score
    import sklearn.neural_network
    import sklearn.linear_model
    import sklearn.ensemble
    clf = sklearn.neighbors.KNeighborsClassifier(n_jobs=11, n_neighbors=5)
    clf.fit(X_train, np.round(y_train))
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    # 4) Calculate f-score
    print('f-score train', f1_score(pred_train, np.round(y_train)))
    print('f-score test', f1_score(pred_test, np.round(y_test)))
    # 5) Calculate honest f-score
    print('honest f-score train', f1_score(window_predicts_to_honest_predicts(df_train, pred_train), df_train['y']))
    print('honest f-score test', f1_score(window_predicts_to_honest_predicts(df_test, pred_test)