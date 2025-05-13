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


def channels_4():
    file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy.xlsx"
    df = pd.read_excel(file_path)
    data = df.values

    matrix1 = data[:, 0::4]
    matrix2 = data[:, 1::4]
    matrix3 = data[:, 2::4]
    matrix4 = data[:, 3::4]
    combined_matrix = np.stack((matrix1, matrix2, matrix3, matrix4), axis=-1)

    tensor_data = torch.tensor(combined_matrix, dtype=torch.float32)
    tensor_data = tensor_data.permute(0, 2, 1)

    conv = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7)
    output = conv(tensor_data)

    output = output.squeeze(-1)
    output = output.detach().numpy()
    print("output result: ", output)
    print("len of output: ", len(output))
    IF_model = IsolationForest(contamination=0.417)  # the contamination value can be changed
    IF_model.fit(output)
    predictions = IF_model.predict(output)

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


def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Standard Label')
    plt.show()


channels_4()

