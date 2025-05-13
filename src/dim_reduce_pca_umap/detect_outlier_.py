import pandas as pd
from scipy import stats
import numpy as np


file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\UMAP_.xlsx"
df = pd.read_excel(file_path)
data = df.iloc[:,1].values

import numpy as np
import matplotlib.pyplot as plt

def visualize_outliers(data):
    # 计算平均值和标准差
    mean = np.mean(data)
    std_dev = np.std(data)

    # 计算每个数据点的 Z-score
    z_scores = [(x - mean) / std_dev for x in data]

    # 定义异常点阈值
    threshold = 1.7

    # 找出异常点的索引
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]


    result=[]
    for value in outlier_indices:
        result.append(value+1)

    print("异常点索引：", [result])
    print(len(outlier_indices))
    print(data[73])
    # 可视化数据和异常点
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Data', linewidth=0.5)
    plt.scatter(outlier_indices, [data[i] for i in outlier_indices], color='red', label='Outliers')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Visualization of Outliers in the Dataset')
    plt.legend()
    plt.show()

# 调用函数进行可视化
visualize_outliers(data)
