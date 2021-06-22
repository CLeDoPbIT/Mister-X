import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.fft
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter

from dataAnalysis.watch_processing.end2end import utils
from utils import split_data_on_windows, normalize_min_max, split_data_on_windows_with_features


class Args:
    def __init__(self, test=False):
        self.train = train
        self.experiment_id = 1
        self.n_workers = 1
        self.model = 'Unet'
        self.dataset_path = '/data/train_cut.csv'
        self.batch_size = 3000
        self.lr = 1e-3
        self.device_ids = [0]
        self.device = utils.find_device()
        self.num_epochs = 1500
        self.val_step = 400
        self.use_pretrained = False
        self.pretrain_folder = '1_version_570'
        self.best_f1 = 0
        self.seed = 1
        self.dropout = 0.2
        utils.seed_all(self.seed)
        self.y_key = 'cut_y'
        # self.use_lr_finder = False
        # self.start_lr_finder_lr = 1E-06
        self.val_set_fraction = 0.05
        self.window_size = 16
        self.features_list = ['RMSSD', 'pNN50','Mean_Median_Load', 'IMA', 'lfnu', 'hfnu', 'pnni_20']
        # self.train_max = 5000000
        if self.test:
            self.val_set_fraction = 0.5
            self.device_ids = [0]
            self.n_workers = 1
            self.batch_size = 5000
            self.val_step = 10
            self.use_pretrained = True
            self.num_epochs = 1000
        self.comment = ""


def conv_block(input_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(input_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU())


class Unet(nn.Module):
    def __init__(self, args):
        super(Unet, self).__init__()
        self.args = args
        input_channels = len(args.features_list)+1
        # encoder
        self.conv_block1 = conv_block(input_channels, out_channels=8)
        self.drop1 = nn.Dropout(args.dropout)
        self.max_pool = nn.MaxPool1d(2)
        self.conv_block2 = conv_block(input_channels=8, out_channels=16)
        self.drop2 = nn.Dropout(args.dropout)
        self.conv_block3 = conv_block(input_channels=16, out_channels=32)
        self.drop3 = nn.Dropout(args.dropout)
        self.conv_block4 = conv_block(input_channels=32, out_channels=64)
        self.drop4 = nn.Dropout(args.dropout)
        self.conv_block5 = conv_block(input_channels=64, out_channels=128)
        self.drop5 = nn.Dropout(args.dropout)

        # decoder
        self.upconv1 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)
        self.drop6 = nn.Dropout(args.dropout)
        self.conv_block6 = conv_block(input_channels=128+64, out_channels=64)
        self.upconv2 = nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1)
        self.drop7 = nn.Dropout(args.dropout)
        self.conv_block7 = conv_block(input_channels=96, out_channels=32)
        self.upconv3 = nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1)
        self.drop8 = nn.Dropout(args.dropout)
        self.drop9 = nn.Dropout(args.dropout)
        self.conv_block8 = conv_block(input_channels=48, out_channels=16)
        self.upconv4 = nn.ConvTranspose1d(16, 16, kernel_size=4, stride=2, padding=1)
        self.conv_block9 = conv_block(input_channels=24, out_channels=12)
        self.last_conv = nn.Conv1d(12, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):  # x [bs,1, window_len)
        x = x.float()
        x = self.conv_block1(x)
        cross_conn1 = self.drop1(x)
        x = self.max_pool(cross_conn1)  # x [bs,4, window_len/2)
        x = self.conv_block2(x)
        cross_conn2 = self.drop2(x)
        x = self.max_pool(cross_conn2)  # x [bs,8, window_len/4)
        x = self.conv_block3(x)
        cross_conn3 = self.drop3(x)
        x = self.max_pool(cross_conn3)  # x [bs,16, window_len/8)
        x = self.conv_block4(x)
        cross_conn4 = self.drop4(x)
        x = self.max_pool(cross_conn4)  # x [bs,32, window_len/16)
        x = self.conv_block5(x)  # x [bs,8, window_len/16)
        lowest_x = self.drop5(x)
        # print(cross_conn4.shape)
        x = torch.cat([cross_conn4, self.upconv1(lowest_x)], dim=1)
        after_upconv1 = self.conv_block6(x)
        x = self.drop6(after_upconv1)
        x = torch.cat([cross_conn3, self.upconv2(x)], dim=1)
        after_upconv2 = self.conv_block7(x)
        x = self.drop7(after_upconv2)
        x = torch.cat([cross_conn2, self.upconv3(x)], dim=1)
        x = self.drop8(x)
        after_upconv3 = self.conv_block8(x)
        x = torch.cat([cross_conn1, self.upconv4(after_upconv3)], dim=1)
        x = self.drop9(x)
        after_upconv4 = self.conv_block9(x)
        signal_pred = self.last_conv(after_upconv4)
        return torch.squeeze(signal_pred, dim=1)


def train(train_dataset, train_data_loader, test_dataset, test_data_loader, loss_function, model,
          optimizer, scheduler, writer, experiment_folder, args):
    model.train()
    loss = 0
    for epoch in range(args.num_epochs):
        print('Epoch', epoch)
        epoch_loss = 0
        total = 0
        correct = 0
        mean_f1 = 0
        for data in train_data_loader:
            optimizer.zero_grad()
            x, y = data
            x = x.to(args.device, dtype=torch.float)
            y = y.to(args.device)

            sample_signal_std = torch.std(x)
            if sample_signal_std != sample_signal_std:
                print("input is nan", data.shape)
            out = model(x)
            # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(out.data, 1)
            # print('pred shape', predicted.shape)
            total += y.size(0) * y.size(1)
            predicts = (torch.sigmoid(out) > 0.5).int()
            mean_f1 += f1_score(torch.reshape(predicts, (-1,)).cpu().detach().numpy(), torch.reshape(y, (-1,)).cpu().detach().numpy())
            correct += (predicts == y).sum().item()
            loss = loss_function(out, y)
            epoch_loss += loss.item()
            if loss != loss:
                print(data)
                print("Error: loss is nan!")
                break

            assert loss == loss
            loss.backward()
            optimizer.step()
        val_f1 = eval(model, val_data_loader)
        if val_f1 > args.best_f1:
            state_dict = model.state_dict()
            torch.save(state_dict, '/models/unet.pth')

        scheduler.step(val_f1)
        print('val_f1', val_f1)
        model.train()
        print('train_loss', epoch_loss/len(train_data_loader))
        print('mean_f1',mean_f1/len(train_data_loader))
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))


class Segmentation_dataset(torch.utils.data.Dataset):
    def __init__(self, args, is_test):
        self.is_test = is_test
        self.subfolter = 'test' if is_test else 'train'
        df_all = pd.read_csv(args.dataset_path)
        df_all.x = normalize_min_max(df_all.x.values, df_all.corrected.values)
        df_all[args.y_key] = df_all[args.y_key].values.astype(float)
        df_metrics = pd.read_csv(
            '/data/rr_metrics/clear_rr_metrics_16.csv')
        df_train = df_all.loc[
            df_all.id.isin(pd.read_csv('/data/train.csv').id.unique())]
        df_test = df_all.loc[
            df_all.id.isin(pd.read_csv('/data/test.csv').id.unique())]
        if is_test:
            df_metrics = df_metrics.loc[
                df_metrics.id.isin(
                    pd.read_csv('/data/test.csv').id.unique())]
            self.X, self.y, self.errors = split_data_on_windows_with_features(df_test, df_metrics, 0, 0, args.features_list)
        else:
            df_metrics = df_metrics.loc[
                df_metrics.id.isin(
                    pd.read_csv('/data/train.csv').id.unique())]
            self.X, self.y, self.errors = split_data_on_windows_with_features(df_train, df_metrics, 0, 0, args.features_list)
        # self.X = normalize_min_max(self.X, self.errors)
        # self.X = self.X[:, np.newaxis]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def eval(model, val_data_loader):
    model.eval()
    total = 0
    correct = 0
    mean_f1 = 0
    with torch.no_grad():
        for x, y in val_data_loader:
            x = x.to(args.device, dtype=torch.float)
            y = y.to(args.device)
            out = model(x)
            y_pred = (out > 1).int()
            mean_f1 += f1_score(torch.reshape(y_pred, (-1,)).cpu().detach().numpy(),
                                torch.reshape(y, (-1,)).cpu().detach().numpy())

    return mean_f1/len(val_data_loader)


def test(model, y_key):
    def predict_on_dataset(model, df_test, y_key, df_metrics=None):
        if args.y_key in df_test.columns:
            y_all = df_test[args.y_key].values
        else:
            y_all = None
        model.cpu()
        with torch.no_grad():
            all_preds = []
            all_counts = []
            for id in df_test['id'].unique():
                df_id = df_test[df_test['id'].values == id]
                counts = np.zeros(len(df_id))
                y_preds = np.zeros(len(df_id))
                for i in range(0, len(df_id) + 1 - args.window_size):
                    x_chunk = df_id['x'].values[i: i + args.window_size]
                    time_start = df_id['time'].values[i: i + args.window_size][0]

                    if df_metrics is not None:

                        df_metrics_exp = df_metrics[df_metrics.id==id]
                        if time_start not in df_metrics_exp.ts_start.values:
                            y_preds[i:i + args.window_size] = y_preds[i:i + args.window_size] + np.zeros(args.window_size)
                            counts[i:i + args.window_size] = counts[i:i + args.window_size] + np.ones_like(args.window_size)
                            print(time_start, id)
                            continue
                        metrics = df_metrics_exp.iloc[np.argwhere(df_metrics_exp.ts_start.values == time_start)[0][0]][args.features_list].values
                        metrics = np.tile(metrics, 16).reshape((16, len(args.features_list))).T
                        x_chunk = np.concatenate([x_chunk[np.newaxis, :], metrics])
                        x_chunk = x_chunk[np.newaxis, :]
                    else:
                        x_chunk = x_chunk[np.newaxis, np.newaxis, :]
                    y_pred = model(torch.Tensor(x_chunk))
                    # _, y_pred = torch.max(y_pred.data, 1)
                    # y_pred = (y_pred > 0.5).int()
                    y_pred = y_pred.detach().numpy()
                    y_preds[i:i + args.window_size] = y_preds[i:i + args.window_size] + y_pred
                    counts[i:i + args.window_size] = counts[i:i + args.window_size] + np.ones_like(y_pred)
                all_preds.append(y_preds)
                all_counts.append(counts)

            return np.concatenate(all_preds) / np.concatenate(all_counts), y_all
    model.eval()
    total = 0
    correct = 0
    # load honest y
    df_metrics = pd.read_csv(
        '/data/rr_metrics/clear_rr_metrics_16.csv')
    for feature in args.features_list:
        feature_vals = df_metrics[feature].values
        df_metrics[feature] = (feature_vals - feature_vals.min())/(feature_vals.max()-feature_vals.min())
    df_metrics.ts_start = np.round(df_metrics.ts_start.values*1000)
    df_all = pd.read_csv(args.dataset_path)
    df_all[args.y_key] = df_all[args.y_key].values.astype(int)
    df_all['x'] = normalize_min_max(df_all['x'].values, df_all['corrected'].values)
    df_train = df_all.loc[df_all.id.isin(pd.read_csv('/data/train.csv').id.unique())]
    df_test = df_all.loc[df_all.id.isin(pd.read_csv('/data/test.csv').id.unique())]
    df_test_final = pd.read_csv('/data/test_final/test_FINAL_filtered.csv')
    df_test_final['x'] = normalize_min_max(df_test_final['x'].values, df_test_final['corrected'].values)
    preds_train, gt_train = predict_on_dataset(model, df_train, y_key, df_metrics)
    preds_test, gt_test = predict_on_dataset(model, df_test, y_key, df_metrics)
    df_metrics_test = pd.read_csv('/data/rr_metrics/TEST_FINAL_clear_rr_metrics_16.csv')
    df_metrics_test.ts_start = np.round(df_metrics_test.ts_start.values * 1000)
    for feature in args.features_list:
        feature_vals = df_metrics_test[feature].values
        df_metrics_test[feature] = (feature_vals - feature_vals.min())/(feature_vals.max()-feature_vals.min())
    preds_test_final, gt_test_final = predict_on_dataset(model, df_test_final, y_key, df_metrics_test)
    # preds_test_final, gt_test_final = 0,0
    return preds_train, gt_train, preds_test, gt_test, preds_test_final, gt_test_final




if __name__ == '__main__':
    args = Args()
    full_dataset = Segmentation_dataset(args, is_test=False)
    val_dataset = torch.utils.data.Subset(full_dataset,
                                          np.arange(int(len(full_dataset) * (1 - args.val_set_fraction)),
                                                    len(full_dataset)))
    train_dataset = torch.utils.data.Subset(full_dataset,
                                            np.arange(int(len(full_dataset) * (1 - args.val_set_fraction))))
    test_dataset = Segmentation_dataset(args, is_test=True)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.n_workers,
                                                    pin_memory=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.n_workers,
                                                  pin_memory=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.n_workers,
                                                   pin_memory=False)
    model = Unet(args).to(args.device).float()
    if args.train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        experiment_folder = ' /covid_experiments/'
        log_path = os.path.join(experiment_folder, str(datetime.now()))
        writer = SummaryWriter(log_path)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=100,
                                                               threshold=0.001, verbose=True)
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10,]).cuda())

        train(train_dataset, train_data_loader, val_dataset, val_data_loader, loss_function, model, optimizer,
              scheduler, writer, experiment_folder, args)
        preds, gt = test(model, y_key=args.y_key)
        print(np.mean(gt==0))
        print('accuracy', np.mean((preds == preds.max()) == gt))
        print('f1-score', f1_score((preds == preds.max()).astype(int), gt))
        print(np.mean(preds))
    else:
        state_dict = torch.load('/models/unet.pth')
        model.load_state_dict(state_dict)
        preds_train, gt_train, preds_test, gt_test, preds_test_final, gt_test_final = test(model, y_key=args.y_key)
        np.save('/media/shared1/users/dkonovalov/covid_experiments/preds_train.npy', preds_train)
        np.save('/media/shared1/users/dkonovalov/covid_experiments/preds_test.npy', preds_test)
        np.save('/media/shared1/users/dkonovalov/covid_experiments/preds_test_final.npy', preds_test_final)

        print('accuracy_train', np.mean((preds_train > 1) == gt_train))
        print('f1-score_train', f1_score((preds_train > 1).astype(int), gt_train))
        print('accuracy', np.mean((preds_test > 1) == gt_test))
        print('f1-score', f1_score((preds_test > 1).astype(int), gt_test))