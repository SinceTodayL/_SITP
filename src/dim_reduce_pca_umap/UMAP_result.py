import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
df = pd.read_excel(file_path)

# 提取数据并转换为numpy数组
data = df.iloc[:, 0:28].values

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 使用UMAP进行降维到3维
umap_model = umap.UMAP(n_neighbors=5, min_dist=0, n_components=2, random_state=42)
data_umap = umap_model.fit_transform(data_scaled)

# 使用孤立森林进行异常检测
iso_forest = IsolationForest(contamination=0.03)
labels = iso_forest.fit_predict(data)

# 提取异常点和正常点
anomalies = data_umap[labels == -1]
normal = data_umap[labels == 1]
'''
# 可视化结果
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制正常点和异常点
ax.scatter(normal[:, 0], normal[:, 1], normal[:, 2], c='blue', label='Normal', edgecolor='k', s=5)
ax.scatter(anomalies[:, 0], anomalies[:, 1], anomalies[:, 2], c='red', label='Anomalies', edgecolor='k', s=25)

ax.set_title('3D UMAP Projection with Anomaly Detection')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.legend()
plt.show()
'''
# 输出异常点的索引
anomalies_indices = np.where(labels == -1)[0]
print(f"Detected {len(anomalies_indices)} anomalies.")
print("Anomalies indices:", anomalies_indices)
result=pd.DataFrame(anomalies_indices)
file_path="C:\\Users\\LiuZh\\Desktop\\SITP\\tmp.xlsx"
result.to_excel(file_path)