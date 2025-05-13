import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import evaluate


def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# 载入并预处理数据
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
df = pd.read_excel(file_path)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df.values)

data = torch.tensor(data_normalized, dtype=torch.float32).unsqueeze(1)
conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=28)
data_reduced_to_1d = conv(data)

# 去掉维度为1的维度，并展平数据
data_reduced_to_1d = data_reduced_to_1d.squeeze().detach().numpy()
data_reduced_to_1d = data_reduced_to_1d.reshape(data_reduced_to_1d.shape[0], -1)

print(f"Shape after conv: {data_reduced_to_1d.shape}")

# Isolation Forest
IF_model = IsolationForest(contamination=0.4117)
IF_model.fit(data_reduced_to_1d)
predictions = IF_model.predict(data_reduced_to_1d)

# 载入标签并评估
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\new_label.xlsx"
df_labels = pd.read_excel(file_path, header=None).to_numpy().flatten()
record_1_array = []
record_2_array = []

for i in range(len(df_labels)):
    if df_labels[i] == 0:
        record_1_array.append(1)
        record_2_array.append(1)
    elif df_labels[i] == 1:
        record_1_array.append(-1)
        record_2_array.append(1)
    elif df_labels[i] == 2:
        record_1_array.append(-1)
        record_2_array.append(-1)

# 确保 record_1_array 和 record_2_array 的长度与预测结果一致
data_length = min(len(record_1_array), len(data_reduced_to_1d))
record_1_array = record_1_array[:data_length]
record_2_array = record_2_array[:data_length]
predictions = predictions[:data_length]

print("Record 1 Evaluation:")
evaluate.evaluate_indicator(record_1_array, predictions)
print()
print("Record 2 Evaluation:")
evaluate.evaluate_indicator(record_2_array, predictions)

plot_confusion_matrix(record_1_array, predictions)
plot_confusion_matrix(record_2_array, predictions)
