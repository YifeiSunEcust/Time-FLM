import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from scipy.spatial.distance import euclidean
warnings.filterwarnings('ignore')
from glob import glob
class FLM_Dataset(Dataset):
    def __init__(self, root_path, flag='train', seq_len=None, pred_len=None,
                 file_pattern='*.csv', scale=True, percent=100):
        # 初始化参数
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_range = 64

        self.flag = flag
        self.scale = scale
        self.percent = percent

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.file_pattern = file_pattern
        self.__read_multiple_files__()
        print(self.__len__())

    def __read_multiple_files__(self):
        self.scalers = {}
        self.datasets = []
        csv_files = glob(os.path.join(self.root_path, self.file_pattern))

        for file_path in csv_files:
            df_raw = pd.read_csv(file_path)
            target_features = [col for col in df_raw.columns if col not in ['batch_id', 'hh', 'stage']]

            if self.scale:
                scaler = StandardScaler()
                df_raw[target_features] = scaler.fit_transform(df_raw[target_features])
                self.scalers[file_path] = scaler
            time_features = ['stage']

            batches = []
            for batch_id in df_raw['batch_id'].unique():
                batch_data = df_raw[df_raw['batch_id'] == batch_id].reset_index(drop=True)
                if len(batch_data) < self.total_range:
                    continue
                batches.append({
                    'data': batch_data,
                    'features': target_features,
                    'time_features': time_features
                })
            # 如果当前文件的所有批次都不满足长度要求，则跳过该文件
            if len(batches) > 0:
                self.datasets.append({
                    'file': file_path,
                    'batches': batches
                })

        # 数据划分
        self.select_batches = []
        for dataset in self.datasets:
            batches = dataset['batches']
            batch_num = len(batches)
            train_end = int(0.7 * batch_num)
            test_end = train_end + int(0.2 * batch_num)

            if self.flag == 'train':
                self.select_batches.extend(batches[:train_end])
            elif self.flag == 'test':
                self.select_batches.extend(batches[train_end:test_end])
            elif self.flag == 'val':
                self.select_batches.extend(batches[test_end:])

    def __getitem__(self, index):
        seq_len = self.seq_len
        pred_len = self.pred_len
        total_len = seq_len + pred_len

        for batch in self.select_batches:
            max_start_idx = len(batch['data']) - total_len + 1
            num_variables = len(batch['features'])
            total_samples = max_start_idx * num_variables

            if index < total_samples:
                variable_index = index % num_variables
                s_begin = index // num_variables
                target_feature = batch['features'][variable_index]
                break
            else:
                index -= total_samples

        s_end = s_begin + seq_len
        r_begin = s_end
        r_end = r_begin + pred_len

        # 截取时间序列和对应的标记
        seq_x = batch['data'].iloc[s_begin:s_end][[target_feature]].values
        seq_y = batch['data'].iloc[r_begin:r_end][[target_feature]].values
        seq_x_mark = batch['data'].iloc[s_begin:s_end][batch['time_features']].values
        seq_y_mark = batch['data'].iloc[r_begin:r_end][batch['time_features']].values

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        total_len = 0
        for batch in self.select_batches:
            batch_len = len(batch['data']) - self.total_range + 1
            if batch_len > 0:
                total_len += batch_len * len(batch['features'])
        return total_len

    def inverse_transform(self, data, file_path, feature_index):
        # 针对某个文件中的特定变量进行逆归一化
        scaler = self.scalers.get(file_path)
        if scaler is None:
            raise ValueError(f"Scaler for file {file_path} not found.")
        return scaler.inverse_transform(data)