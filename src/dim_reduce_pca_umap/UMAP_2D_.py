import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import umap
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
# 读取数据
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
df = pd.read_excel(file_path)
data = df.iloc[:, 0:28].values

# UMAP降维
trans = umap.UMAP(
    n_neighbors=8,
    min_dist=0.5,
    n_components=2,
    metric="euclidean",
)
embedding = trans.fit_transform(data)

# Use Isolation Forest to identify outliers
iso_forest = IsolationForest(contamination=0.025)  # Adjust the contamination as needed
outliers = iso_forest.fit_predict(embedding)
# Get indices of outliers
outlier_indices = [index for index, value in enumerate(outliers) if value == -1]


'''
from sklearn.neighbors import LocalOutlierFactor
# Use Local Outlier Factor to identify outliers
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)  # Adjust parameters as needed
outliers = lof.fit_predict(pca_data)

# Get indices of outliers
outlier_indices = [index for index, value in enumerate(outliers) if value == -1]
'''

for value in outlier_indices:
    value += 1

# Print outlier indices
print("Indices of outliers:", outlier_indices)


# Plot the PCA-transformed data and highlight outliers
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, label='Normal Data', s=10)
plt.scatter(embedding[outliers == -1, 0],embedding[outliers == -1, 1], color='r', label='Outliers', s=20)
plt.title('UMAP result of 2D Data')
plt.xlabel('UMAP dimension 1')
plt.ylabel('UMAP dimension 2')
plt.legend()
plt.grid(True)
plt.show()
