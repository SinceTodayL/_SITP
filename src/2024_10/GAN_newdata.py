import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam

# Step 1: 读取原始数据
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy_label.xlsx"  # 替换为你的文件路径
data = pd.read_excel(file_path)

# Step 2: 分离特征和标签
X = data.iloc[:, :-1].values  # 所有行，除最后一列外的特征
y = data.iloc[:, -1].values   # 标签

# Step 3: 划分测试集和训练集，确保测试数据不参与GAN训练
X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: 定义GAN模型（生成器和判别器）
def build_generator(latent_dim, data_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(data_dim, activation='tanh'))  # 输出与特征维度相同
    return model

def build_discriminator(data_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

# Step 5: 初始化GAN模型
latent_dim = 100  # 噪声维度
data_dim = X_train_orig.shape[1]  # 训练数据的特征维度

generator = build_generator(latent_dim, data_dim)
discriminator = build_discriminator(data_dim)

discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Step 6: 训练GAN生成更多数据
def train_gan(X, epochs=100, batch_size=32):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # 训练判别器
        idx = np.random.randint(0, X.shape[0], batch_size)
        real_data = X[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, real)
        d_loss_fake = discriminator.train_on_batch(generated_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)

        if epoch % 10 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

# 训练GAN
train_gan(X_train_orig)

# Step 7: 使用生成器生成更多数据
def generate_data(generator, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    gen_data = generator.predict(noise)
    return gen_data

# 生成比原数据多一倍的数据
generated_data = generate_data(generator, 10*X_train_orig.shape[0])

# Step 8: 将生成的数据与训练集结合
X_train_combined = np.vstack((X_train_orig, generated_data))
y_train_combined = np.hstack((y_train_orig, np.ones(generated_data.shape[0])))  # 假设生成的都是正常数据

# Step 9: 使用随机森林进行训练，确保测试集来自于原始数据
model = KNeighborsClassifier(n_neighbors=100)
# model = MLPClassifier(random_state=42, max_iter=100)
# model = RandomForestClassifier(random_state=42)
model.fit(X_train_combined, y_train_combined)

# 进行预测
y_pred = model.predict(X_test)

# Step 10: 评估模型
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"precision: {accuracy:.6f}")
print(f"recall: {recall:.6f}")
print(f"F1 score: {f1:.6f}")

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
plt.ylabel('real label')
plt.xlabel('predicted label')
plt.title('confusion matrix')
plt.show()
