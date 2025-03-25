import os
import pandas as pd
import numpy as np

# 输入和输出文件夹路径
file_path = r"E:\_SITP\data"  # 请替换为实际的文件夹路径
output_path = r"E:\_SITP\data\Label"  # 结果文件夹
os.makedirs(output_path, exist_ok=True)

# 初始化标签统计数组
label_sum = None
file_count = 0

# 遍历所有 .xlsx 文件
for filename in os.listdir(file_path):
    if filename.endswith(".xlsx"):
        file_count += 1
        file_full_path = os.path.join(file_path, filename)
        
        # 读取 Excel 文件
        df = pd.read_excel(file_full_path, header=0)  # 假设数据没有列名
        labels = df.iloc[:, -1].values  # 取最后一列作为标签
        
        # 初始化 label_sum 数组
        if label_sum is None:
            label_sum = np.zeros_like(labels, dtype=int)
        
        # 累加标签
        label_sum += labels

# 统计标签分布
unique, counts = np.unique(label_sum, return_counts=True)
label_distribution = dict(zip(unique, counts))

# 打印标签分布
print("标签分布:")
for label, count in label_distribution.items():
    print(f"标签值 {label}: {count} 个样本")

# 保存统计结果
output_file = os.path.join(output_path, "label_summary.xlsx")
pd.DataFrame(label_sum, columns=["Label Sum"]).to_excel(output_file, index=False)
print(f"统计结果已保存至 {output_file}")
