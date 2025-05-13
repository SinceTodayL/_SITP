import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data.xlsx"
df = pd.read_excel(file_path)
matrix_list = []

for i in range(4):
    sub_df = df.iloc[:, i + 1:35:5]
    matrix_list.append(sub_df.values.T)
# 合并所有二维矩阵为一个三维矩阵
data_3d = np.stack(matrix_list)

first_umap_results = []

for matrix in data_3d:
    _umap = umap.UMAP(n_neighbors=50, min_dist=0, n_components=1)
    transformed_matrix = _umap.fit_transform(matrix.T)
    first_umap_results.append(transformed_matrix[:, 0])


first_umap_results = np.array(first_umap_results)

# 对1127x4矩阵进行PCA，得到1127x1的向量
_umap = umap.UMAP(n_neighbors=15, min_dist=1, n_components=1)
final_umap = _umap.fit_transform(first_umap_results.T)

result=pd.DataFrame(final_umap)
file_path="C:\\Users\\LiuZh\\Desktop\\SITP\\UMAP_.xlsx"
result.to_excel(file_path)

