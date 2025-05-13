import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# 从 Excel 文件中加载数据
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
df = pd.read_excel(file_path)

# 提取前28列作为特征数据
data = df.iloc[:, 0:28].values

# 使用PCA进行降维
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# 将降维后的数据转换为DataFrame
data_pca_df = pd.DataFrame(data_pca)

# 保存到新的Excel文件
output_file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\pca_2d_result.xlsx"
data_pca_df.to_excel(output_file_path)

print("降维结果已成功保存到:", output_file_path)

# 使用 Isolation Forest 识别异常
iso_forest = IsolationForest(contamination=0.029)
anomaly_labels = iso_forest.fit_predict(data_pca)
anomalies = np.where(anomaly_labels == -1)[0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

normal_mask = anomaly_labels == 1
ax.scatter(data_pca[normal_mask, 0], data_pca[normal_mask, 1], data_pca[normal_mask, 2],
           c='blue',label='normal', s=5)

ax.scatter(data_pca[anomalies, 0], data_pca[anomalies, 1], data_pca[anomalies, 2],
           c='red', label='outliers', s=10)

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('PCA')

ax.legend()

plt.show()

result=[]
for value in anomalies :
    result.append(value+1)
print("异常点的索引:", [result])
