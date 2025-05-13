import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import evaluate

def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Standard Label')
    plt.show()

file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
df = pd.read_excel(file_path)
data = df.values

# 数据标准化
scaler = StandardScaler()
data = scaler.fit_transform(data)
data_tensor = torch.tensor(data, dtype=torch.float32)

# GAN模型
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 模型初始化
input_dim = 28
latent_dim = 20
generator = Generator(input_dim=latent_dim, output_dim=input_dim)
discriminator = Discriminator(input_dim=input_dim)

optimizer_G = optim.Adam(generator.parameters(), lr=0.00005)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005)
loss_function = nn.BCELoss()

# 训练参数
num_epochs = 1000
batch_size = 2
real_labels = torch.ones(batch_size, 1)
fake_labels = torch.zeros(batch_size, 1)

dataloader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

# 用于存储每个epoch的F1-score
f1_scores_record1 = []
f1_scores_record2 = []
epochs = []

# GAN训练
for epoch in range(num_epochs):
    for real_samples, in dataloader:
        # 训练判别器
        optimizer_D.zero_grad()
        outputs = discriminator(real_samples)
        d_loss_real = loss_function(outputs, real_labels[:len(real_samples)])

        noise = torch.randn(len(real_samples), latent_dim)
        fake_samples = generator(noise)
        outputs = discriminator(fake_samples.detach())
        d_loss_fake = loss_function(outputs, fake_labels[:len(real_samples)])

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        noise = torch.randn(len(real_samples), latent_dim)
        fake_samples = generator(noise)
        outputs = discriminator(fake_samples)

        g_loss = loss_function(outputs, real_labels[:len(real_samples)])
        g_loss.backward()
        optimizer_G.step()

    # 每隔10个epoch计算一次F1-score
    if epoch % 1 == 0:
        with torch.no_grad():
            noise = torch.randn(data_tensor.size(0), latent_dim)
            reconstructed_data = generator(noise)

        reconstruction_error = torch.mean((data_tensor - reconstructed_data) ** 2, dim=1)
        threshold = torch.quantile(reconstruction_error, 0.5883)
        anomalies = reconstruction_error > threshold
        anomalous_indices = torch.where(anomalies)[0].tolist()

        predictions = [1] * len(data)
        for index in anomalous_indices:
            predictions[index] = -1

        file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\new_label.xlsx"
        df_labels = pd.read_excel(file_path, header=None).to_numpy().flatten()

        record_1_array = []
        record_2_array = []

        for i in range(0, len(df_labels)):
            if df_labels[i] == 0:
                record_1_array.append(1)
                record_2_array.append(1)
            elif df_labels[i] == 1:
                record_1_array.append(-1)
                record_2_array.append(1)
            elif df_labels[i] == 2:
                record_1_array.append(-1)
                record_2_array.append(-1)

        f1_record1 = f1_score(record_1_array, predictions)
        f1_record2 = f1_score(record_2_array, predictions)

        f1_scores_record1.append(f1_record1)
        f1_scores_record2.append(f1_record2)
        epochs.append(epoch)

        print(f'Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | F1 Record 1: {f1_record1:.4f} | F1 Record 2: {f1_record2:.4f}')

# 绘制 num_epochs 与 F1-score 的关系图
plt.figure(figsize=(10, 6))
plt.plot(epochs, f1_scores_record1, label='F1 Score Record 1')
# plt.plot(epochs, f1_scores_record2, label='F1 Score Record 2')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 score and training epoches')
plt.legend()
plt.grid(True)
plt.show()
