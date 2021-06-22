import pandas as pd
import numpy as np

PATH_TO_TRAIN_DATA = "train.csv"


def __get_abnormal_intervals_indices(arr):
    is_ones = np.concatenate(([0], np.equal(arr, 1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(is_ones))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def __split_targets_intervals(rr_arr):
    min_diff_threshold = 0.025
    delta_cut_one_to_several = 2
    min_len_covid = 3
    diff = np.abs(np.diff(rr_arr))
    indices_diff_more_threshold = np.where(diff > min_diff_threshold)[0]
    diff_indices = np.where(np.diff(indices_diff_more_threshold) >= delta_cut_one_to_several)[0]
    if diff_indices.shape[0] == 0:
        if indices_diff_more_threshold[0] != 0:
            begin = indices_diff_more_threshold[0] - 1
        else:
            begin = indices_diff_more_threshold[0]
        if indices_diff_more_threshold[-1] != diff.shape[0] - 1:
            end = indices_diff_more_threshold[-1] + 2
        else:
            end = indices_diff_more_threshold[-1] + 1
        indices_interval_abnormal = np.arange(begin, end)
        return [indices_interval_abnormal]
    else:
        indices_interval_abnormal = []
        if indices_diff_more_threshold[0] != 0:
            begin = indices_diff_more_threshold[0] - 1
        else:
            begin = indices_diff_more_threshold[0]
        for i in diff_indices:
            if indices_diff_more_threshold[i] != diff.shape[0] - 1:
                end = indices_diff_more_threshold[i] + 2
            else:
                end = indices_diff_more_threshold[i] + 1
            if end - begin >= min_len_covid:
                indices_interval_abnormal.append(np.arange(begin, end))
            begin = indices_diff_more_threshold[i + 1] - 1
        if indices_diff_more_threshold[-1] - begin >= min_len_covid:
            if indices_diff_more_threshold[-1] != diff.shape[0] - 1:
                end = indices_diff_more_threshold[-1] + 2
            else:
                end = indices_diff_more_threshold[-1] + 1
            indices_interval_abnormal.append(np.arange(begin, end))
        return indices_interval_abnormal


def __cut_abnormal_rr():
    init_data = pd.read_csv(PATH_TO_TRAIN_DATA)
    unique_ids = init_data["id"].unique()
    ids = init_data["id"].to_numpy()
    cutted_target = np.zeros(init_data.shape[0])
    for id in unique_ids:
        rr_data = init_data[init_data["id"] == id]
        rr_intervals = rr_data["x"] / 1000
        target = rr_data["y"]
        global_indices_exp = np.where(ids == id)[0]
        exp_abnormal_indices = __get_abnormal_intervals_indices(target)
        for item in exp_abnormal_indices:
            curr_rr = rr_intervals[item[0]:item[1] + 1]
            indices_for_mark = __split_targets_intervals(curr_rr)
            tmp_global = np.arange(item[0], item[1] + 1)
            for idx_arr in indices_for_mark:
                cutted_idx = tmp_global[idx_arr]
                cutted_target[global_indices_exp[cutted_idx]] = 1
    init_data["cut_y"] = cutted_target
    init_data.to_csv("train_cut.csv", index=False)


if __name__ == '__main__':
    __cut_abnormal_rr()
