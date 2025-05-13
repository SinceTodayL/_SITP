import os
import pandas as pd

# 定义文件路径
label_file = r'C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx'
data_file = r'C:\Users\LiuZh\Desktop\SITP\Data.xlsx'
output_file = r'C:\Users\LiuZh\Desktop\SITP\UpdatedDataLabel.xlsx'

# 读取原始DataLabel文件
sum_labels = pd.read_excel(label_file)

# 将sum_label进行分类映射
updated_labels = sum_labels['sum_label'].apply(lambda x: 0 if x == 0 else (1 if 1 <= x <= 3 else 2))

# 保存新的标签到一个新的Excel文件
updated_labels.to_excel(output_file, index=False, header=['UpdatedLabel'])

# 反馈完成情况
print(f"标签已经成功更新并保存为：{output_file}")
