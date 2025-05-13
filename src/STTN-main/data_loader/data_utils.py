import numpy as np
import pandas as pd
from utils.math_utils import z_score


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data  # dict: train/val/test -> (x, y)
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type][0])  # 返回样本数

    def z_inverse(self, data):
        return data * self.std + self.mean


def load_custom_dataset(csv_path, n_his=10, n_pred=1):
    raw_data = pd.read_csv(csv_path).values  # shape: [1127, 56]
    assert raw_data.shape[1] == 56
    data = raw_data.reshape(1127, 14, 4).transpose(1, 0, 2)  # [14, 1127, 4]

    x, y = [], []
    for t in range(14 - n_his - n_pred + 1):
        x.append(data[t:t+n_his+1])
        y.append(data[t+n_his:t+n_his+n_pred])
    x = np.stack(x)  # [num_sample, n_his, 1127, 4]
    y = np.stack(y)  # [num_sample, n_pred, 1127, 4]

    mean = np.mean(x)
    std = np.std(x)
    x = z_score(x, mean, std)
    y = (y - mean) / std


    n_train = int(len(x) * 0.6)
    n_val = int(len(x) * 0.2)
    n_test = len(x) - n_train - n_val

    data = {
        'train': (x[:n_train], y[:n_train]),
        'val': (x[n_train:n_train+n_val], y[n_train:n_train+n_val]),
        'test': (x[-n_test:], y[-n_test:])
    }
    return Dataset(data, {'mean': mean, 'std': std})


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=True):
    """
    生成训练/测试用的 batch 数据。

    Args:
        inputs: a tuple (x, y)
        batch_size: int, batch 大小
        dynamic_batch: 不切整（保留最后残留 batch）
        shuffle: 是否打乱顺序

    Yields:
        每次迭代返回一个 (x_batch, y_batch)
    """
    x, y = inputs
    length = len(x)

    if shuffle:
        idx = np.random.permutation(length)
        x, y = x[idx], y[idx]

    num_batch = length // batch_size
    for i in range(num_batch):
        s, e = i * batch_size, (i + 1) * batch_size
        yield x[s:e], y[s:e]

    # 动态 batch，最后一批不足的也返回
    if dynamic_batch and length % batch_size != 0:
        yield x[num_batch * batch_size:], y[num_batch * batch_size:]
