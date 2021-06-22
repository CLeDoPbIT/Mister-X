import numpy as np
import pandas as pd
# import numba
MAX_NEIGHBOURS_RATIO = 1.3
MIN_NEIGHBOURS_RATIO = 0.8
MAX_RR_INTERVAL = 1100
MIN_RR_INTERVAL = 400


def consecutive(data):
    return np.split(data, np.where(np.diff(data) != 1)[0] + 1)

def label_individual_good_rrs_as_error(df, key, n_consecutive=5):
    consecutive_good_rrs = consecutive(df[df[key] == 0].index.values)
    for good_rrs_list in consecutive_good_rrs:
        if len(good_rrs_list) <= n_consecutive:
            df.loc[good_rrs_list, key] = 1
    return df

def filter_data(df):
    df['Error'] = ((df.x > MAX_RR_INTERVAL) | (df.x < MIN_RR_INTERVAL)).astype(int)
    ratio_errors = []
    for exp_id, exp_df in df.groupby('id'):
        exp_rrs = exp_df.x.values
        ratio = exp_rrs[:-1] / exp_rrs[1:]
        ratio = (ratio > MAX_NEIGHBOURS_RATIO) | (ratio < MIN_NEIGHBOURS_RATIO)
        ratio_error = np.zeros(len(ratio) + 1)
        ratio_error[1:] += ratio
        ratio_error[:-1] += ratio
        ratio_errors.append(ratio_error > 1)
    df['Error_ratio'] = np.concatenate(ratio_errors).astype(int)
    df['Error'] = ((df['Error'] == 1) | (df['Error_ratio'] == 1)).values.astype(int)
    df.drop(columns=['Error_ratio'], inplace=True)


    errors = []
    for exp_id, exp_df in df.groupby('id'):
        exp_df = label_individual_good_rrs_as_error(exp_df,key='Error')
        errors.append(exp_df.Error.values)
    errors = np.concatenate(errors)
    df['Error'] = errors

    # Corrected
    df_filtered = df.rename(columns={'Error':'corrected'})
    df_filtered['x_corr'] = df_filtered['x']
    df_filtered['time_corr'] = df_filtered['time']
    for exp_id, exp_df in df_filtered.groupby('id'):
        corrected_index = exp_df[exp_df.corrected==1].index
        if len(corrected_index) > 0:
            for inds in consecutive(corrected_index):
                start = inds[0]
                end = inds[-1]
                if start!=exp_df.index.values.min() and end!=exp_df.index.values.max():
                    mean_rr = (exp_df.loc[start-1].x + exp_df.loc[end+1].x)/2
                else:
                    mean_rr = (MAX_RR_INTERVAL+MIN_RR_INTERVAL)/2
                df_filtered.loc[inds, 'x_corr'] = int(mean_rr)
            exp_x_corr = df_filtered.loc[exp_df.index, 'x_corr'].values
            df_filtered.loc[exp_df.index, 'time_corr'] = np.cumsum(exp_x_corr)-exp_x_corr[0]
    return df_filtered




if __name__ == '__main__':
    full_data_path = 'Z:\\users\\ashilov2\\colleguches\\data\\train_cut_v2.csv'
    df = pd.read_csv(full_data_path)
    df.drop(columns=['corrected', 'x_corr', 'time_corr'], inplace=True)
    df = filter_data(df)
    df.to_csv('Z:\\users\\ashilov2\\colleguches\\data\\train_cut_v3.csv', index=False)
