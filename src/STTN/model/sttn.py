import torch
import torch.nn as nn
from model.layers import STBlock


class STTN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 num_blocks, Kt, adj, history_len, predict_len):
        """
        :param in_channels: 输入特征数（例如 4）
        :param out_channels: 输出特征数（例如 4）
        :param hidden_channels: 每个 STBlock 的通道数（例如 64）
        :param num_blocks: STBlock 层数
        :param Kt: 时间卷积核大小
        :param adj: 邻接矩阵 Tensor，[N, N]
        :param history_len: 输入时间步数
        :param predict_len: 要预测多少个时间步
        """
        super(STTN, self).__init__()
        self.blocks = nn.ModuleList()
        self.adj = adj

        # 输入变换为 hidden_channels
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1))

        for _ in range(num_blocks):
            self.blocks.append(STBlock(hidden_channels, hidden_channels, Kt, adj))

        # 输出层：预测 predict_len 个时间步
        self.output_layer = nn.Conv2d(hidden_channels, out_channels * predict_len, kernel_size=(1, 1))

        self.predict_len = predict_len
        self.out_channels = out_channels

    def forward(self, x):
        """
        :param x: [B, T, N, F]
        :return: [B, predict_len, N, out_channels]
        """
        x = x.permute(0, 3, 1, 2)  # -> [B, F, T, N]
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_layer(x)  # -> [B, out_channels * predict_len, T, N]
        x = x[:, :, -1, :]  # 只取最后一个时间点的输出

        # reshape 为 [B, predict_len, N, out_channels]
        x = x.reshape(x.size(0), self.predict_len, x.size(-1), self.out_channels)
        return x
