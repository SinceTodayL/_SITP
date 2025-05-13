import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
df = pd.read_excel(file_path)
data = df.iloc[:, 0:28].values

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 使用UMAP进行降维到1维
umap_model = umap.UMAP(n_neighbors=100, min_dist=0.1, n_components=1, random_state=42)
data_umap = umap_model.fit_transform(data_scaled)

result = pd.DataFrame(data_umap, columns=["UMAP_1D result"])
output_file = "C:\\Users\\LiuZh\\Desktop\\SITP\\UMAP_.xlsx"
# 将结果保存到Excel文件
result.to_excel(output_file, index=False)

"""# 使用孤立森林进行异常检测
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(data_umap)
labels = iso_forest.predict(data_umap)

# 可视化结果
plt.figure(figsize=(12, 8))
plt.scatter(range(len(data_umap)), data_umap, c=labels, cmap='coolwarm', edgecolor='k')
plt.title('1D UMAP Projection with Anomaly Detection')
plt.xlabel('Sample Index')
plt.ylabel('UMAP 1')
plt.colorbar()
plt.show()

# 输出异常点的索引
anomalies = np.where(labels == -1)[0]
print(f"Detected {len(anomalies)} anomalies.")
print("Anomalies indices:", anomalies)
"""