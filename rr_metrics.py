import pandas as pd
import hrvanalysis as hrv
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from functools import partial

HEADERS_CLEAR = None
HEADERS_ALL = None

STEP = 1
WINDOWS = [4, 6, 8, 10, 12, 16]


def __calculate_amp_mode(np_rr):
    num_intervals = 12
    interval_size = (np.max(np_rr) - np.min(np_rr)) / num_intervals
    intervals_distribution = []
    values_distribution = OrderedDict()
    for i in range(num_intervals):
        start = np.min(np_rr) + i * interval_size
        finish = start + interval_size
        intervals_distribution.append({'start': start, 'finish': finish})
    for interval in intervals_distribution:
        values_distribution[(interval['start'], interval['finish'])] = len(
            np.where(((np_rr > interval['start']) & (np_rr <= interval['finish'])))[0])
    mode_interval = max(values_distribution.items(), key=operator.itemgetter(1))
    mode_interval_borders = mode_interval[0]
    index_mode_interval = list(values_distribution).index(mode_interval_borders)

    if index_mode_interval == len(values_distribution) - 1:
        mode_next = ((0, 0), 0)
    else:
        mode_next = list(values_distribution.items())[index_mode_interval + 1]
    if index_mode_interval == 0:
        mode_prev = ((0, 0), 0)
    else:
        mode_prev = list(values_distribution.items())[index_mode_interval - 1]
    if mode_interval[1] != 0:
        mode = mode_interval_borders[0] + interval_size * ((mode_interval[1] - mode_prev[1]) / (
                (mode_interval[1] - mode_prev[1]) + (mode_interval[1] - mode_next[1])))
    else:
        mode = 0

    return mode, mode_interval[1] / np_rr.shape[0]


def __nn50_no_corr(row_rr, corrected):
    diffs = []
    num_pairs = 0
    for i, j in zip(range(0, row_rr.shape[0] - 1), range(1, row_rr.shape[0])):
        if corrected[i] == 0 and corrected[j] == 0:
            diffs.append(abs(np.diff([row_rr[i], row_rr[j]])[0]))
            num_pairs += 1
    diffs = np.array(diffs)
    nn50 = np.sum(diffs > 0.05)
    pnn50 = nn50 / (num_pairs + 1)
    return nn50, pnn50


def __calculate_params(row_rr):
    hrv_parameters = dict()
    hrv_parameters['Mean'] = np.mean(row_rr)
    hrv_parameters['Median'] = np.median(row_rr)
    hrv_parameters['std'] = np.std(row_rr)
    hrv_parameters['min'] = np.min(row_rr)
    hrv_parameters['max'] = np.max(row_rr)

    hrv_parameters['Mean_Median_Load_Criteria'] = hrv_parameters['Mean'] - hrv_parameters['Median']
    hrv_parameters['Mean_Median_Load'] = 1 * (hrv_parameters['Mean_Median_Load_Criteria'] >= 0.025)
    hrv_parameters['dRR']= np.max(row_rr) - np.min(row_rr)

    hrv_parameters['RMSSD'] = np.sqrt(np.sum(np.diff(row_rr) ** 2) / (len(row_rr) - 1))
    hrv_parameters['NN50'] = np.sum(abs(np.diff(row_rr)) > 0.05)
    hrv_parameters['pNN50'] = hrv_parameters['NN50'] / len(row_rr)

    hrv_parameters['SDNN']= np.sqrt(np.sum((row_rr - np.average(row_rr)) ** 2) / (len(row_rr) - 1))

    hrv_parameters['CV'] = (hrv_parameters['SDNN'] / np.average(row_rr))
    hrv_parameters['mode'], hrv_parameters['amp_mode'] = __calculate_amp_mode(row_rr)

    hrv_parameters['IVR'] = hrv_parameters['amp_mode'] / hrv_parameters['dRR']


    if hrv_parameters['mode'] != 0:
        hrv_parameters['VPR'] = 1 / (hrv_parameters['mode'] * hrv_parameters['dRR'])
        hrv_parameters['PAPR'] = hrv_parameters['amp_mode'] / hrv_parameters['mode']
        hrv_parameters['IN'] = hrv_parameters['amp_mode'] / (
                2 * hrv_parameters['dRR'] * hrv_parameters['mode'])
    else:
        hrv_parameters['VPR'] = 0
        hrv_parameters['PAPR'] = 0
        hrv_parameters['IN'] = 0


    hrv_parameters['IDM'] = 0.5 * hrv_parameters['RMSSD'] / np.average(row_rr)
    hrv_parameters['CAT'] = hrv_parameters['amp_mode'] / hrv_parameters['IDM']
    hrv_parameters['IMA'] = 1 - (0.5 * hrv_parameters['IDM'] / hrv_parameters['CV']) - 0.3

    # Parameters is calculating by package hrnanalysis
    time_domain_features = hrv.get_time_domain_features(row_rr*1000)
    poincare_features = hrv.get_poincare_plot_features(row_rr*1000)
    csi_cvi_features = hrv.get_csi_cvi_features(row_rr*1000)
    freq_domain_features = hrv.get_frequency_domain_features(row_rr*1000)
    geom_features = hrv.get_geometrical_features(row_rr*1000)

    hrv_parameters['lf'] = freq_domain_features['lf']
    hrv_parameters['triangular_index'] = geom_features['triangular_index']
    hrv_parameters['hf'] = freq_domain_features['hf']
    hrv_parameters['lf_hf_ratio'] = freq_domain_features['lf_hf_ratio']
    hrv_parameters['lfnu'] = freq_domain_features['lfnu']
    hrv_parameters['hfnu'] = freq_domain_features['hfnu']
    hrv_parameters['total_power'] = freq_domain_features['total_power']
    hrv_parameters['vlf'] = freq_domain_features['vlf']

    hrv_parameters['mean_nni']= time_domain_features['mean_nni']
    hrv_parameters['SDSD'] = time_domain_features['sdsd']
    hrv_parameters['range_nni'] = time_domain_features['range_nni']
    hrv_parameters['CVSD'] = time_domain_features['cvsd']
    hrv_parameters['nni_20'] = time_domain_features['nni_20']
    hrv_parameters['pnni_20'] = time_domain_features['pnni_20']
    hrv_parameters['median_nni'] = time_domain_features['median_nni']
    hrv_parameters['range_nni'] = time_domain_features['range_nni']
    hrv_parameters['cvnni'] = time_domain_features['cvnni']
    hrv_parameters['mean_hr'] = time_domain_features['mean_hr']
    hrv_parameters['max_hr'] = time_domain_features['max_hr']
    hrv_parameters['min_hr'] = time_domain_features['min_hr']
    hrv_parameters['std_hr'] = time_domain_features['std_hr']

    hrv_parameters['CSI'] = csi_cvi_features['csi']
    hrv_parameters['CVI'] = csi_cvi_features['cvi']
    hrv_parameters['Mod_CSI'] =  csi_cvi_features['Modified_csi']
    hrv_parameters['SD1'] = poincare_features['sd1']
    hrv_parameters['SD2'] = poincare_features['sd2']
    hrv_parameters['SD1_SD2'] = poincare_features['ratio_sd2_sd1']

    return hrv_parameters


def __process_window(data_arrays, start, end, cur_id):
    ts_window = data_arrays['ts'][start:end]
    rr_window = data_arrays['rrs'][start:end]
    corr = data_arrays['corr'][start:end]
    ts_corr = data_arrays['ts_corr'][start:end]
    rr_window_corr = data_arrays['rrs_corr'][start:end]

    if len(rr_window) == 0:
        return

    rr_clear_ind = np.where(corr == 0)[0]

    dict_parameters_rr_all = __calculate_params(rr_window)
    values_all = [cur_id, ts_window[0], ts_window[-1], ts_corr[0], ts_corr[-1]] + list(dict_parameters_rr_all.values())

    global HEADERS_ALL, HEADERS_CLEAR
    if HEADERS_CLEAR is None:
        HEADERS_CLEAR = ['id', 'ts_start', 'ts_end', 'ts_corr_start', 'ts_corr_end'] + list(dict_parameters_rr_all.keys())

    if rr_window[rr_clear_ind].shape[0] > 1:
        dict_parameters_clear = __calculate_params(rr_window[rr_clear_ind])
        values_clear = [cur_id, ts_window[0], ts_window[-1], ts_corr[0], ts_corr[-1]] + list(dict_parameters_clear.values())
        
        if HEADERS_ALL is None:
            HEADERS_ALL =  ['id', 'ts_start', 'ts_end', 'ts_corr_start', 'ts_corr_end'] + list(dict_parameters_clear.keys())
    else:
        values_clear = None

    return values_all, values_clear

def __process_one_size(data_arrays, window):
    output_data_all, output_data_clear = [], []
    for cur_id in set(data_arrays['ids']):
        cur_exp_indexes = np.where(data_arrays['ids']==cur_id)[0]
        for i in cur_exp_indexes[:-window]:
            row_all, row_clear = __process_window(data_arrays, i, i+window, cur_id)
            if row_all is not None:
                output_data_all.append(row_all)
            if row_clear is not None:
                output_data_clear.append(row_clear)
    global HEADERS_ALL, HEADERS_CLEAR
    df_all = pd.DataFrame(output_data_all, columns=HEADERS_ALL)
    df_all.to_csv('data/clear_rr_metrics_'+str(window)+'.csv')
    df_clear = pd.DataFrame(output_data_clear, columns=HEADERS_CLEAR)
    df_clear.to_csv('data/all_rr_metrics_'+str(window)+'.csv')


def __process_experiments_rr(data_arrays):
    pool = Pool(cpu_count())
    f = partial(__process_one_size, data_arrays)
    pool.map(f, WINDOWS)    

def process(path_to_data):
    data = pd.read_csv(path_to_data)
    data_arrays = dict(rrs=np.array(data['x'])/1000,
                        rrs_corr=np.array(data['x_corr'])/1000,
                        corr=np.array(data['corrected']),
                        ts=np.array(data['time'])/1000,
                        ts_corr=np.array(data['time_corr'])/1000,
                        ids=np.array(data['id']))

    __process_experiments_rr(data_arrays)

if __name__ == "__main__":
    path_to_data = 'data/train_cut_v3.csv'

    process(path_to_data)
 