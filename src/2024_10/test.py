from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# 定义判别器模型
discriminator = Sequential([
    Dense(128, activation='relu', input_dim=28),  # 根据你的数据维度修改
    Dense(1, activation='sigmoid')
])

# 编译判别器
discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

# 创建训练数据
real_data = np.random.randn(32, 28)  # 用你自己的真实数据
real_labels = np.ones((32, 1))       # 真实标签
fake_data = np.random.randn(32, 28)  # 生成的假数据
fake_labels = np.zeros((32, 1))      # 假标签

# 训练判别器
d_loss_real = discriminator.train_on_batch(real_data, real_labels)
d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

print(f"Real Loss: {d_loss_real}, Fake Loss: {d_loss_fake}")
