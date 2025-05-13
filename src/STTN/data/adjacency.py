import numpy as np
import torch

def generate_linear_adjacency(num_nodes):
    """
    生成线性结构的邻接矩阵，形状为 [num_nodes, num_nodes]
    每个节点仅与前一个和后一个节点相连
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        if i > 0:
            A[i, i - 1] = 1
        if i < num_nodes - 1:
            A[i, i + 1] = 1
    return torch.tensor(A, dtype=torch.float32)
