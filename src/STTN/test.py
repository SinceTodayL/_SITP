import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.dataset import STTNDataset
from data.adjacency import generate_linear_adjacency
from model.sttn import STTN

# 配置
csv_path = "E:/_SITP/data/Data.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
history_len = 5
predict_len = 5
batch_size = 1  # 只预测一个样本，便于可视化

# 数据加载
dataset = STTNDataset(csv_path, history_len, predict_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 邻接矩阵
adj = generate_linear_adjacency(num_nodes=1127).to(device)

# 模型初始化（应与你训练时一致）
model = STTN(in_channels=4,
             out_channels=4,
             hidden_channels=64,
             num_blocks=3,
             Kt=3,
             adj=adj,
             history_len=history_len,
             predict_len=predict_len).to(device)

# 加载训练好的权重（如果已保存）
model.load_state_dict(torch.load("best_model.pt"))

model.eval()


# ======= 可视化参数设置 =======
start = 100
end = 200
feat_id = 0  # 你想看的特征（0-3）
colors = ['r', 'g', 'b', 'orange', 'purple']
labels = [f"t+{i+1}" for i in range(predict_len)]

# 取一个样本
x, y_true = next(iter(dataloader))
x, y_true = x.to(device), y_true.to(device)

with torch.no_grad():
    y_pred = model(x)  # [1, predict_len, N, F]

# 转换为 NumPy
y_true = y_true[0].cpu().numpy()  # [T, N, F]
y_pred = y_pred[0].cpu().numpy()  # [T, N, F]

# ======= 开始画图：真实 vs 预测 =======
plt.figure(figsize=(14, 6))

for t in range(predict_len):
    true_vals = y_true[t, start:end, feat_id]
    pred_vals = y_pred[t, start:end, feat_id]
    node_range = range(start, end)

    plt.plot(node_range, true_vals, color=colors[t], label=f"True {labels[t]}", linestyle='-')
    plt.plot(node_range, pred_vals, color=colors[t], label=f"Pred {labels[t]}", linestyle='--')

plt.title(f"轨道点 {start}~{end} 的特征 {feat_id}：真实值 vs 预测值")
plt.xlabel("轨道点 ID")
plt.ylabel("波长值")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
