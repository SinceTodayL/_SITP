import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
# 读取原始Excel文件
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data.xlsx"
df = pd.read_excel(file_path)
matrix_list = []

for i in range(4):
    sub_df = df.iloc[:, i + 1:35:5]
    matrix_list.append(sub_df.values.T)
# 合并所有二维矩阵为一个三维矩阵
data_3d = np.stack(matrix_list)

first_pca_results = []

for matrix in data_3d:
    pca = PCA(n_components=1)
    transformed_matrix = pca.fit_transform(matrix.T)
    first_pca_results.append(transformed_matrix[:, 0])


# 将结果转换为1127x1的向量
first_pca_results = np.array(first_pca_results)

# 对1127x4矩阵进行PCA，得到1127x1的向量
final_pca = PCA(n_components=2)
pca_data = final_pca.fit_transform(first_pca_results.T)

# Use Isolation Forest to identify outliers
iso_forest = IsolationForest(contamination=0.02)  # Adjust the contamination as needed
outliers = iso_forest.fit_predict(pca_data)
# Get indices of outliers
outlier_indices = [index for index, value in enumerate(outliers) if value == -1]

for value in outlier_indices:
    value += 1

# Print outlier indices
print("Indices of outliers:", outlier_indices)


# Plot the PCA-transformed data and highlight outliers
plt.figure(figsize=(10, 8))
plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7, label='Normal Data', s=10)
plt.scatter(pca_data[outliers == -1, 0], pca_data[outliers == -1, 1], color='r', label='Outliers', s=20)
plt.title('PCA result of 2D Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

"""result = pd.DataFrame(final_result, columns=['PCA Component'])
# 指定保存的Excel文件路径
output_file = "C:\\Users\\LiuZh\\Desktop\\SITP\\pca_result_1.xlsx"
# 将结果保存到Excel文件
result.to_excel(output_file, index=False)"""