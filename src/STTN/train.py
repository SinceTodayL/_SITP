import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data.dataset import STTNDataset
from data.adjacency import generate_linear_adjacency
from model.sttn import STTN

# 配置参数
csv_path = "E:/_SITP/data/Data.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

history_len = 8
predict_len = 5
batch_size = 16
epochs = 50
learning_rate = 0.001

# 加载数据集
dataset = STTNDataset(csv_path, history_len, predict_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 构建邻接矩阵
adj = generate_linear_adjacency(num_nodes=1127).to(device)

# 初始化模型
model = STTN(in_channels=4,
             out_channels=4,
             hidden_channels=64,
             num_blocks=3,
             Kt=3,
             adj=adj,
             history_len=history_len,
             predict_len=predict_len).to(device)

# 损失函数 & 优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)          # [B, T, N, F]
        y = y.to(device)          # [B, predict_len, N, F]

        optimizer.zero_grad()
        output = model(x)         # [B, predict_len, N, F]
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "best_model.pt")
