import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
from tqdm import tqdm, trange
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
import filter_rrs
import utils
import sys
import os

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def sort(df_true_test):
    ids = df_true_test['id'].unique()
    df_new_full = []
    running_id = 0
    for id in ids:
        df_loc = df_true_test.loc[df_true_test['id'].values == id]
        df_new = df_loc.copy()
        df_new['original_id'] = df_loc['id'].values
        df_new['id'] = running_id
        running_id += 1
        df_new_full.append(df_new)
    return pd.concat(df_new_full)


def get_data(use_test, gb=False):
    shared_path = '/shared' if os.path.exists('/shared/users/') else '/media/shared'
    if not os.path.exists(shared_path):
        shared_path = '/home/ashilov2/shared'
    path = f'{shared_path}/users/ashilov2/colleguches/data'
    df_test = pd.read_csv(f"{path}/df_test_filtered.csv")  # just filter df_train using filter_rrs.py
    df_train = pd.read_csv(f"{path}/df_train_filtered.csv")
    if use_test:
        df_train = pd.concat([df_train, df_test])  # WARNING!!! FOR FINAL SUBMIT
    df_true_test = pd.read_csv(f'{path}/test_final/test_FINAL.csv')
    df_true_test = sort(df_true_test)
    df_true_test = filter_rrs.filter_data(df_true_test)
    df_true_test['Error'] = df_true_test['corrected'].values 
    if gb:
        pred_train = pd.read_csv(f"dummy_net_predict_train_21_11_02.csv")[:len(df_train)]
        pred_test = pd.read_csv(f"dummy_net_predict_test_21_11_02.csv")
        pred_true_test = pd.read_csv(f"dummy_net_predict_test_final_21_11_02.csv")
        df_train['dummy_net_predict_continious'] = pred_train['dummy_net_predict_continious'].values
        df_test['dummy_net_predict_continious'] = pred_test['dummy_net_predict_continious'].values
        df_true_test['dummy_net_predict_continious'] = pred_true_test['dummy_net_predict_continious'].values
    df_train = sort(df_train)
    metrics_cols = [f for f in pred_train.columns.values if 'metric_'in f]
    for col in metrics_cols:
        df_train[col] = pred_train[col]
        df_true_test[col] = pred_true_test[col]
        df_test[col] = pred_test[col]    
    return df_train, df_test, df_true_test


def test(net, X_test_tensor, y_test, metrics=None):
    net = net.eval()
    with torch.no_grad():
        out = net(X_test_tensor, metrics=metrics)
    pred_train = out.cpu().numpy()
    if len(y_test.shape) == 2:
        pred_train = np.mean(pred_train, axis=1)
        y_test = np.mean(y_test, axis=1)
    net = net.train()
    if pred_train.std() == 0:
        return 0
    return f1_score((pred_train > 0.3).astype(int), np.round(y_test))


def X_factory(x):
    X = torch.from_numpy(np.diff(x, axis=1, prepend=0))
    if len(X.shape) < 3:
        X = X.unsqueeze(1)
    return X.cuda().float()


def metrics_factory(metrics):
    metrics = torch.from_numpy(metrics).float().cuda()
    return metrics


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, x, y):
        loss = self.mse(x, y)
        return loss
            

class ConvLayer(nn.Module):
    def __init__(self, size, padding=1, pool_layer=nn.MaxPool1d(2, stride=2), stride=1,
                 bn=False, dropout=False, activation_fn=nn.ReLU()):
        super(ConvLayer, self).__init__()
        layers = []
        layers.append(nn.Conv1d(size[0], size[1], size[2], padding=padding, stride=stride))
        if pool_layer is not None:
            layers.append(pool_layer)
        if bn:
            layers.append(nn.BatchNorm1d(size[1]))
        if dropout:
            layers.append(nn.Dropout(dropout))
        if activation_fn is not None:
            layers.append(activation_fn)
            self.use_sin = False
        else:
            self.use_sin = True
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_sin:
            return torch.sin(self.model(x))
        else:
            return self.model(x)


class FullyConnected(nn.Module):
    def __init__(self, sizes, dropout=False, activation_fn=nn.Tanh):
        super(FullyConnected, self).__init__()
        
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if dropout:
                layers.append(nn.Dropout(dropout))		
            layers.append(activation_fn())
        else:
            layers.append(nn.Linear(sizes[-2], sizes[-1]))
        # layers.append(nn.Linear(sizes[0], sizes[1]))
        # if dropout:
        #     layers.append(nn.Dropout(dropout))
        # layers.append(activation_fn())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EncoderDecoder(nn.Module):
    def __init__(self, window_size=8):
        super(EncoderDecoder, self).__init__()
        self.window_size = window_size
        self._conv1 = ConvLayer([1, 8, 3], bn=False, padding=2, dropout=0)
        if window_size < 4:
            self._conv2 = ConvLayer([8, 16, 3], bn=False, dropout=0., pool_layer=None)
        else:
            self._conv2 = ConvLayer([8, 16, 3], bn=False, dropout=0.)
        if window_size < 6:
            self._conv3 = ConvLayer([16, 32, 3], bn=False, dropout=0., 
                                    pool_layer=None, activation_fn=nn.Sigmoid())
        else:
            self._conv3 = ConvLayer([16, 32, 3], bn=False, dropout=0., activation_fn=nn.Sigmoid())
        self.fc = FullyConnected([32 + 32, window_size], dropout=0.5, activation_fn=nn.Sigmoid)
        self.metrics_encoder = MetricsEncoder()

    def conv(self, x):
        x = self._conv1(x)
        x2 = self._conv2(x)
        x3 = self._conv3(x2)
        return x, x2, x3

    def forward(self, x, metrics, get_intermediate_features=False):
        x, x2, x3 = self.conv(x)
        metrics = metrics.permute(0,2,1)
        metrics = self.metrics_encoder(metrics)
        x4 = x3.view(x.shape[0], -1)
        x = torch.cat((x4, metrics), dim=1)
        x4 = self.fc(x)
        if get_intermediate_features:
            return x2, x3, x4
        return x4


class MetricsEncoder(nn.Module):
    def __init__(self):
        super(MetricsEncoder, self).__init__()
        self._conv1 = ConvLayer([1, 16, 3], bn=False, padding=1, dropout=0.2)
        self._conv2 = ConvLayer([16, 32, 3], bn=False, dropout=0.2)
        self._conv3 = ConvLayer([32, 64, 3], bn=False, dropout=0.2, pool_layer=None, activation_fn=nn.Sigmoid())
        self.fc1 = FullyConnected([64, 32], dropout=0.2)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self._conv1(x)
        x2 = self._conv2(x)
        x3 = self._conv3(x2)
        x3 = x3.view(x.shape[0], -1)
        x3 = self.fc1(x3)
        return x3


class Combined_Net(nn.Module):
    def __init__(self, window_size):
        super(Combined_Net, self).__init__()
        self.window_sizes = [4, 8, 10, 12]
        self.nets = nn.ModuleList([EncoderDecoder(win) for win in self.window_sizes])
        
    def forward(self, x, metrics):
        metrics = metrics.unsqueeze(1)
        out = torch.zeros((x.shape[0], x.shape[2]), device=x.device)
        for win, net in zip(self.window_sizes, self.nets):
            collection = torch.zeros(x.shape[-1], device=x.device)
            out_win = torch.zeros((x.shape[0], x.shape[2]), device=x.device)
            for i in range(win//2, x.shape[-1] - win//2):
                collection[i - win//2:i+win//2] += 1
                pred = net(x[:, :, i - win//2:i+win//2], metrics)
                out_win[:, i - win//2:i+win//2] += pred
            mask = collection != 0
            out_win[:, mask] /= collection[mask]
            out += out_win / len(self.window_sizes)
        return out



def load_convenient(model, state_dict):
    own_state = model.state_dict()
    loaded_modules = 0
    total_modules = len(state_dict.items())
    for name, param in state_dict.items():
        if name not in own_state:
            if name.replace('module.', '') in own_state:
                name = name.replace('module.', '')
            else:
                continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if own_state[name].data.shape == param.shape:
            own_state[name].copy_(param)
            loaded_modules += 1
    print(f"Loaded {loaded_modules} pretrained modules out of {total_modules}")


def main():
    window_size = 13
    use_test = True
    df_train, df_test, df_test_final = get_data(use_test)
    X_test, y_test, errors_test, metrcis_test = utils.split_data_on_windows(df_test, window_size=window_size, generate_metrics=True)
    X_train, y_train, errors_train, metrcis_train = utils.split_data_on_windows(df_train, window_size=window_size, filter_train=False,
                                                                                generate_metrics=True)
    X_true_test, _, errors_true_test, metrcis_true_test = utils.split_data_on_windows(df_test_final, window_size=window_size,
                                                                                      no_gt=True, generate_metrics=True)
    X = X_factory(X_train)
    X_test_tensor = X_factory(X_test)
    X_true_test_tensor = X_factory(X_true_test)
    metrcis_train = metrics_factory(metrcis_train)
    metrcis_true_test = metrics_factory(metrcis_true_test)
    metrcis_test = metrics_factory(metrcis_test)

    y = torch.from_numpy(y_train).float().cuda()
    model_name = 'Combined_Net'  # Combined_LSTM_Net Combined_Net EncoderDecoder
    net = getattr(sys.modules[__name__], model_name)(window_size).float()
    load_convenient(net, torch.load(f'{model_name}_weights.pth', map_location='cpu'))
    net = net.cuda()
    net = net.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
    f_scores = []
    loss_function = Loss()
    mask = torch.sum(X[:, 0, :], dim=1) != 0
    X_train = X[mask]
    y_train = y[mask]
    metrics_train_train = metrcis_train[mask] 
    tqdm_range = trange(2000, desc='f1 score: 0', leave=True)
    for i in tqdm_range:
        out = net(X_train, metrics_train_train)
        optimizer.zero_grad()
        loss = loss_function(out, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        f_scores.append(test(net, X_test_tensor, y_test, metrcis_test))
        tqdm_range.set_description(f"f1 score: {round(np.mean(f_scores[-5:]), 3)}")
        tqdm_range.refresh()  
    net.eval()
    with torch.no_grad():
        pred = net(X_test_tensor, metrcis_test)
        pred_train = net(X, metrcis_train)
        pred_true_test = net(X_true_test_tensor, metrcis_true_test)
    pred = pred.cpu().numpy()
    pred_train = pred_train.cpu().numpy()
    pred_true_test = pred_true_test.cpu().numpy()
    metrcis_true_test = metrcis_true_test.cpu().numpy()
    metrcis_train = metrcis_train.cpu().numpy()
    metrcis_test = metrcis_test.cpu().numpy()
    honest_predicts, honest_predicts_continous, metrcis_test = utils.window_predicts_to_honest_predicts(df_test, pred, 
                                                                                          errors_test,
                                                                                          metrcis_test,
                                                                                          window_size=window_size, 
                                                                                          threshold=0.31,
                                                                                          get_continious=True)
    honest_predicts_final, honest_predicts_continous_final, metrcis_true_test = utils.window_predicts_to_honest_predicts(df_test_final, pred_true_test, 
                                                                                          errors_true_test,
                                                                                          metrcis_true_test,
                                                                                          window_size=window_size, 
                                                                                          threshold=0.31,
                                                                                          get_continious=True)
    honest_predicts_train, honest_predicts_train_continious, metrcis_train = utils.window_predicts_to_honest_predicts(df_train, 
                                                                                                       pred_train,
                                                                                                       errors_train,
                                                                                                       metrcis_train,
                                                                                                       window_size=window_size,
                                                                                                       threshold=0.31,
                                                                                                       get_continious=True)
    df_test['dummy_net_predict'] = honest_predicts
    df_train['dummy_net_predict'] = honest_predicts_train
    df_test_final['dummy_net_predict'] = honest_predicts_final
    df_test_final['dummy_net_predict_continious'] = honest_predicts_continous_final
    df_test['dummy_net_predict_continious'] = honest_predicts_continous
    df_train['dummy_net_predict_continious'] = honest_predicts_train_continious
    metrics_cols = ['metric_' + str(i) for i in range(metrcis_train.shape[1])]
    for i in range(len(metrics_cols)):
        df_train[metrics_cols[i]] = metrcis_train[:, i]
        df_test[metrics_cols[i]] = metrcis_test[:, i]
        df_test_final[metrics_cols[i]] = metrcis_true_test[:, i]
    f1 = f1_score(honest_predicts, df_test['y'].values)
    print("F1 score not honest:", np.mean(f_scores[-5:]))
    print("F1 score:", f1)
    print(confusion_matrix(honest_predicts, df_test['y'].values))
    if f1 > 0.874:
        exp_id = str(datetime.now())[11:19].replace(':', '_')
        df_test_final.to_csv(f'dummy_net_predict_test_final_{exp_id}.csv', index=None)
        df_test.to_csv(f"dummy_net_predict_test_{exp_id}.csv", index=None)
        df_train.to_csv(f"dummy_net_predict_train_{exp_id}.csv", index=None)
        torch.save(net.state_dict(), f'{model_name}_weights_{exp_id}.pth')
        print("Saved results", exp_id)


if __name__ == '__main__':
    main()