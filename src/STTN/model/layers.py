import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, activation='GLU'):
        super(TemporalConvLayer, self).__init__()
        self.activation = activation
        self.c_out = c_out
        self.conv = nn.Conv2d(in_channels=c_in,
                              out_channels=2 * c_out if activation == 'GLU' else c_out,
                              kernel_size=(kernel_size, 1),
                              padding=(kernel_size - 1, 0))  # padding only on time axis

    def forward(self, x):
        """
        :param x: [B, C_in, T, N]
        :return:  [B, C_out, T, N]
        """
        x_conv = self.conv(x)
        if self.activation == 'GLU':
            lhs, rhs = torch.split(x_conv, self.c_out, dim=1)
            return lhs * torch.sigmoid(rhs)
        else:
            return F.relu(x_conv)


class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(c_in, c_out)

    def forward(self, x, adj):
        """
        :param x: [B, C_in, T, N]
        :param adj: [N, N]
        :return: [B, C_out, T, N]
        """
        B, C, T, N = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, T, N, C]
        x = torch.matmul(adj, x)  # 空间维度图卷积 [B, T, N, C]
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)  # [B, C_out, T, N]
        return x


class STBlock(nn.Module):
    def __init__(self, c_in, c_out, Kt, adj):
        super(STBlock, self).__init__()
        self.temp1 = TemporalConvLayer(c_in, c_out, kernel_size=Kt)
        self.spatial = GraphConvLayer(c_out, c_out)
        self.temp2 = TemporalConvLayer(c_out, c_out, kernel_size=Kt)
        self.residual = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.adj = adj

    def forward(self, x):
        """
        :param x: [B, C, T, N]
        """
        res = self.residual(x)

        x = self.temp1(x)
        x = self.spatial(x, self.adj)
        x = self.temp2(x)

        # 对齐 res 和 x 的时间维度（dim=2 是时间维度）
        if x.size(2) != res.size(2):
            min_t = min(x.size(2), res.size(2))
            x = x[:, :, -min_t:, :]
            res = res[:, :, -min_t:, :]

        return x + res

