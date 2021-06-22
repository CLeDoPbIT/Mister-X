import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime
import torch
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import f1_score, confusion_matrix
import dummy_net
import utils


def run_lstm(df, net):
    running_id = -1
    lstm_pred = np.zeros(len(df))
    net = net.cpu()
    for i in tqdm(range(len(df))):
        x = df['x'].values[i] / 1000
        err = df['Error'].values[i]
        pred = df['dummy_net_predict_continious'].values[i]
        id = df['id'].values[i]
        err_win =  df['Error'].values[max(i-4, 0):i+4]
        if err == 1 or np.mean(err_win) > 0.5:
            x = 0
            pred = 0
        if id != running_id:
            h = c = None
            running_id = id
        X = torch.tensor([[x, pred, running_id]]).unsqueeze(2).float()
        with torch.no_grad():
            p, h, c = net(X, h, c, return_hc = True)
        lstm_pred[i] = p.cpu().numpy()
    honest_predicts = lstm_pred > 0.40
    net = net.cuda()
    return  honest_predicts, lstm_pred


def split_data_on_windows(df: pd.DataFrame, window_size, y_key='y', return_mean_y=False, no_gt=False, get_metrics=True):
    X = []
    y = []
    errors = []
    metrics = []
    for id in df['id'].unique():
        df_id = df[df['id'].values == id]
        x = df_id['x'].values
        err = df_id['Error'].values
        pred = df_id['dummy_net_predict_continious'].values
        x[err == 1] = 0
        pred[err == 1] = 0
        for i in range(0, len(df_id) + 1 - window_size):
            x_chunk0 = x[i: i + window_size] / 1000
            x_chunk1 = pred[i: i + window_size]
            X.append([x_chunk0, x_chunk1, np.ones(window_size) * id])
            if get_metrics:
                params = utils.calculate_params(x_chunk0)
                params = np.array(list(params.values()))
                metrics.append(params)
            if not no_gt:
                y.append(df_id[y_key].values[i: i + window_size])
            errors.append(err[i: i + window_size])
    X = np.array(X)
    y = np.array(y)
    metrics = np.array(metrics)
    metrics[metrics != metrics] = 0
    if not no_gt:
        y = y[:, -1]
    errors = np.array(errors)
    if not no_gt and return_mean_y:
        y = np.mean(y, axis=1)
    return X, y, errors, metrics


class BoostingNet(nn.Module):
    def __init__(self, window_size=8):
        super(BoostingNet, self).__init__()
        self.num_layers = 4
        self.hidden_size = 32
        self.lstm = nn.LSTM(input_size=2, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_size, 1)
    
    def forward(self, x, h=None, c=None, return_hc=False, metrics=None):
        id = x[:, 2, :]
        x = x[:, :2, :]
        if h is None:
            h = c = torch.zeros((self.num_layers, x.shape[0], self.hidden_size), device=x.device)
        for i in range(x.shape[-1]):
            input = x[:, :, i]
            input = input.unsqueeze(1)
            output, (h, c) = self.lstm(input, (h, c))

        output = self.linear(output).squeeze()
        output = torch.sigmoid(output)
        if return_hc:
            return output, h, c
        else:
            return output


def main():
    use_test = True
    window_size = 13
    df_train, df_test, df_test_final = dummy_net.get_data(use_test, gb=True)
    X_train, y_train, errors_train, metrics_train = split_data_on_windows(df_train, window_size=window_size, get_metrics=False)
    X_test, y_test, errors_test, metrics_test = split_data_on_windows(df_test, window_size=window_size, get_metrics=False)
    X_true_test, _, errors_true_test, metrics_true_test = split_data_on_windows(df_test_final, window_size=window_size, no_gt=True,
                                                                                get_metrics=False)
    X = torch.from_numpy(X_train).cuda().float()
    X_test_tensor = torch.from_numpy(X_test).cuda().float()
    X_true_test_tensor = torch.from_numpy(X_true_test).cuda().float()
    metrcis_train = np.load("metrics_train_gb.npy")
    metrcis_test = np.load("metrics_test_gb.npy")
    metrcis_true_test = np.load("metrics_true_test_gb.npy")
    metrcis_train = dummy_net.metrics_factory(metrcis_train)
    metrcis_true_test = dummy_net.metrics_factory(metrcis_true_test)
    metrcis_test = dummy_net.metrics_factory(metrcis_test)
    y = torch.from_numpy(y_train).float().cuda().float()
    model_name = 'BoostingNet'
    net = getattr(sys.modules[__name__], model_name)(window_size).float()
    dummy_net.load_convenient(net, torch.load(f'{model_name}_weights.pth', map_location='cpu'))
    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
    f_scores = []
    loss_function = dummy_net.Loss()
    
    mask = torch.sum(X[:, 0, :], dim=1) != 0

    X_train = X[mask]
    y_train = y[mask]
    metrcis_train_train = metrcis_train[mask]
    tqdm_range = trange(2000, desc='f1 score: 0', leave=True)
    for i in tqdm_range:
        out = net(X_train, metrics=metrcis_train_train)
        optimizer.zero_grad()
        loss = loss_function(out, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        f_scores.append(dummy_net.test(net, X_test_tensor, y_test, metrcis_test))
        tqdm_range.set_description(f"f1 score: {round(np.mean(f_scores[-5:]), 3)}")
        tqdm_range.refresh()
    net.eval()    
    honest_predicts, honest_predicts_continous = run_lstm(df_test, net) 
    honest_predicts_train, honest_predicts_train_continious = run_lstm(df_train, net)
    honest_predicts_final, honest_predicts_continous_final = run_lstm(df_test_final, net)

    df_test['gb_net_predict'] = honest_predicts
    df_train['gb_net_predict'] = honest_predicts_train
    df_test['gb_net_predict_continious'] = honest_predicts_continous
    df_train['gb_net_predict_continious'] = honest_predicts_train_continious
    df_test_final['gb_net_predict_continious'] = honest_predicts_continous_final
    df_test_final['gb_net_predict'] = honest_predicts_final
    f1 = f1_score(honest_predicts, df_test['y'].values)
    print("F1 score not honest:", np.mean(f_scores[-5:]))
    print("F1 score:", f1)
    print(confusion_matrix(honest_predicts, df_test['y'].values))
    exp_id = str(datetime.now())[11:19].replace(':', '_')
    df_test_final.to_csv(f"gb_net_predict_test_final_{exp_id}.csv")
    df_train.to_csv(f"gb_net_predict_train_{exp_id}.csv", index=None)
    df_test.to_csv(f"gb_net_predict_test_{exp_id}.csv", index=None)
    torch.save(net.state_dict(), f'{model_name}_weights.pth')


if __name__ == '__main__':
    main()