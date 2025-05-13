import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import check
import evaluate
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Standard Label')
    plt.show()


file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
df = pd.read_excel(file_path)

data = torch.tensor(df.values, dtype=torch.float32)
data = data.unsqueeze(1)

conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=28)
data_reduced_to_1d = conv(data)
data_reduced_to_1d.squeeze()
data_reduced_to_1d = data_reduced_to_1d.detach().numpy()
data_reduced_to_1d = data_reduced_to_1d.reshape(-1, 1)

IF_model = IsolationForest(contamination=0.4117)   # the contamination value can be changed
IF_model.fit(data_reduced_to_1d)
predictions = IF_model.predict(data_reduced_to_1d)

file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\new_label.xlsx"
df = pd.read_excel(file_path, header=None).to_numpy().flatten()
record_1_array = []
record_2_array = []

for i in range(0, 1127):
    if df[i] == 0:
        record_1_array.append(1)
        record_2_array.append(1)
    elif df[i] == 1:
        record_1_array.append(-1)
        record_2_array.append(1)
    elif df[i] == 2:
        record_1_array.append(-1)
        record_2_array.append(-1)

print("for record_1: ")
print(record_1_array)
print(list(predictions))
evaluate.evaluate_indicator(record_1_array, predictions)
print()
print(record_2_array)
print(list(predictions))
print("for record_2: ")
evaluate.evaluate_indicator(record_2_array, predictions)
plot_confusion_matrix(record_1_array, predictions)
plot_confusion_matrix(record_2_array, predictions)


"""anomalies = np.array(np.where(predictions == -1))
anomalies += 1
anomalies = anomalies.tolist()[0]"""

"""print("anomalies index :", anomalies)
print()
check.check_accuracy(anomalies)"""

"""unusual = []
for i in range(0, data_reduced_to_1d.size):
    if data_reduced_to_1d[i][0] > 3 or data_reduced_to_1d[i][0] < -3:
        unusual.append(i + 1)

print(unusual)"""

"""df = pd.DataFrame(data_reduced_to_1d)
new_file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\convolution_result_3D_data.xlsx"
df.to_excel(new_file_path, index=False, header=False)
"""