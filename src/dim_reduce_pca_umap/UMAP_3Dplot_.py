import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import umap
from sklearn.ensemble import IsolationForest
# 读取数据
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
df = pd.read_excel(file_path)

# 提取数据并转换为numpy数组
data = df.iloc[:, 0:28].values

# 使用UMAP将数据降维到3维
trans = umap.UMAP(
    n_neighbors=100,
    min_dist=0,

    n_components=3,
    metric="euclidean",
)
embedding = trans.fit_transform(data)

"""# 可视化降维后的数据
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],c='b', s=1, marker='o')
ax.set_title('UMAP Projection of Data into 3D')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_zlabel('UMAP Dimension 3')
plt.show()
"""

# 使用 Isolation Forest 识别异常
iso_forest = IsolationForest(contamination=0.41)  # 假设污染率为 2%
anomaly_labels = iso_forest.fit_predict(embedding)
anomalies = np.where(anomaly_labels == -1)[0]

# 创建一个新的 Matplotlib 图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制正常点
normal_mask = anomaly_labels == 1
ax.scatter(embedding[normal_mask, 0], embedding[normal_mask, 1], embedding[normal_mask, 2],
           c='blue',label='normal', s=5)

# 绘制异常点（标记为红色）
ax.scatter(embedding[anomalies, 0], embedding[anomalies, 1], embedding[anomalies, 2],
           c='red', label='outliers', s=10)

# 设置标签和标题
ax.set_xlabel('UMAP dimension 1')
ax.set_ylabel('UMAP dimension 2')
ax.set_zlabel('UMAP dimension 3')
ax.set_title('UMAP')

# 显示图例
ax.legend()

# 显示图形
plt.show()

# 打印异常点的索引
result=[]
for value in anomalies :
    result.append(value+1)
print("异常点的索引:", [result])
