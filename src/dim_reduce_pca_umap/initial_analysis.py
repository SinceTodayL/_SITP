import pandas as pd
import numpy as np

# 读取原始Excel文件
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data.xlsx"
df = pd.read_excel(file_path)

# 提取和转换数据为三维矩阵
matrix_list = []
for i in range(4):
    sub_df = df.iloc[:, i + 1:35:5]
    matrix_list.append(sub_df.values.T)

# 合并所有二维矩阵为一个三维矩阵
data_3d = np.stack(matrix_list)
print(data_3d.shape)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 随机选择一个切片进行可视化
slice_index = data_3d.shape[0] // 2
slice_data = data_3d[slice_index, :, :]

plt.imshow(slice_data, cmap='gray')
plt.title("2D Slice of 3D Data")
plt.show()

# 三维可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = np.indices((5, 8, 1128))
ax.voxels(x, y, z, data_3d, edgecolor='k')
plt.title("3D Data Visualization")
plt.show()

import tensorflow as tf
from tensorflow.keras import layers, models


# 创建一个简单的3D卷积神经网络
model = models.Sequential()
model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=data_3d.shape[1:]))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

# 添加全连接层
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))  # 假设有10个类别

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 查看模型架构
model.summary()

from sklearn.model_selection import train_test_split

# 假设数据标签为labels
labels = np.random.randint(0, 10, size=data_3d.shape[0])  # 生成随机标签作为示例

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data_3d, labels, test_size=0.2, random_state=42)

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")
