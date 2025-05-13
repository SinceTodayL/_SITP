import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class STTNDataset(Dataset):
    def __init__(self, csv_path, history_len=5, predict_len=5):
        """
        :param csv_path: 路径，形如 'E:/_SITP/data/Data.csv'
        :param history_len: 用于建模的历史时间步数
        :param predict_len: 预测时间步数
        """
        super(STTNDataset, self).__init__()

        # 基本配置
        self.history_len = history_len
        self.predict_len = predict_len
        self.total_time = 14
        self.num_nodes = 1127
        self.num_features = 4

        # 读取数据：原始形状为 (1127, 56)
        raw = pd.read_csv(csv_path, skiprows=1, header=None).values.astype(np.float32)
        data = raw.reshape(self.num_nodes, self.total_time, self.num_features)  # -> (1127, 14, 4)
        data = np.transpose(data, (1, 0, 2))  # -> (14, 1127, 4)
        self.data = data

        # 样本数
        self.sample_cnt = self.total_time - history_len - predict_len + 1

    def __len__(self):
        return self.sample_cnt

    def __getitem__(self, idx):
        # 输入片段：[history_len, num_nodes, features]
        x = self.data[idx : idx + self.history_len]          # (history_len, 1127, 4)
        y = self.data[idx + self.history_len : idx + self.history_len + self.predict_len]  # (predict_len, 1127, 4)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
