import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import evaluate
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
data = df.values  # 将数据转换为NumPy数组

# Step 2: PCA降维
pca = PCA(n_components=7)  # 降维到10个主成分，可以根据需要调整
data_pca = pca.fit_transform(data)
data_tensor = torch.tensor(data_pca, dtype=torch.float32)  # 转换为PyTorch张量


# Step 3: 构建GAN模型
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(0.3),  # 添加Dropout
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),  # 添加Dropout
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
            nn.Dropout(0.3),  # 添加Dropout
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # 添加Dropout
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 模型初始化
input_dim = data_pca.shape[1]  # 输入数据的维度（PCA降维后的维度）
latent_dim = data_pca.shape[1]  # 隐空间的维度，保持与输入维度一致
weight_decay = 1e-4  # L2正则化项的系数

generator = Generator(input_dim=latent_dim, output_dim=input_dim)
discriminator = Discriminator(input_dim=input_dim)

# 优化器和损失函数
optimizer_G = optim.Adam(generator.parameters(), lr=0.0005, weight_decay=weight_decay)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0005, weight_decay=weight_decay)
loss_function = nn.BCELoss()

# Step 4: 训练GAN
num_epochs = 1000
batch_size = 10

dataloader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for real_samples, in dataloader:
        # 训练判别器
        optimizer_D.zero_grad()
        real_labels = torch.ones(len(real_samples), 1)
        fake_labels = torch.zeros(len(real_samples), 1)

        outputs = discriminator(real_samples)
        d_loss_real = loss_function(outputs, real_labels)

        noise = torch.randn(len(real_samples), latent_dim)
        fake_samples = generator(noise)
        outputs = discriminator(fake_samples.detach())
        d_loss_fake = loss_function(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        noise = torch.randn(len(real_samples), latent_dim)
        fake_samples = generator(noise)
        outputs = discriminator(fake_samples)

        g_loss = loss_function(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}')

# Step 5: 计算重构误差
with torch.no_grad():
    reconstructed_data = generator(data_tensor)

# 计算重构误差（使用均方误差）
reconstruction_error = torch.mean((data_tensor - reconstructed_data) ** 2, dim=1)

# Step 6: 判定异常点
threshold = torch.quantile(reconstruction_error, 0.583)  # 异常点检测比例可以调整
anomalies = reconstruction_error > threshold
anomalous_indices = torch.where(anomalies)[0].tolist()

print(f'Anomalous samples indices: {[index + 1 for index in anomalous_indices]}')


# Step 7: 检测准确率
predictions = [1] * 1127
for index in anomalous_indices:
    predictions[index] = -1

file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\new_label.xlsx"
df = pd.read_excel(file_path, header=None).to_numpy().flatten()
record_1_array = []
record_2_array = []

for i in range(0, 1127):
    if df[i] == 0:
        record_1_array.append(1)
        record_2_array.append(1)
    elif df[i] == 1:
        record_1_array.append(-1)
        record_2_array.append(1)
    elif df[i] == 2:
        record_1_array.append(-1)
        record_2_array.append(-1)

print("for record_1: ")
evaluate.evaluate_indicator(record_1_array, predictions)
print("for record_2: ")
evaluate.evaluate_indicator(record_2_array, predictions)

plot_confusion_matrix(record_1_array, predictions)
plot_confusion_matrix(record_2_array, predictions)
