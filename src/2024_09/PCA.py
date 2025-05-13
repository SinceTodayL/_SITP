import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import numpy as np
import check


file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
df = pd.read_excel(file_path)
data = df.iloc[:, 0:28].values

pca = PCA(n_components=7)
pca_data = pca.fit_transform(data)

# Use Isolation Forest to identify outliers
iso_forest = IsolationForest(contamination=0.41)  # Adjust the contamination as needed
outliers = iso_forest.fit_predict(pca_data)
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

outlier_indices = [value + 1 for value in outlier_indices]

# Print outlier indices
print("Indices of outliers:", outlier_indices)
check.check_accuracy(outlier_indices)