import os
import pandas as pd
import numpy as np

# 输入和输出文件路径
output_path = r"E:\_SITP\data\Label"  # 结果文件夹
output_file = os.path.join(output_path, r"label_summary.xlsx")
output_file1 = r"E:\_SITP\data\Label\label_2class.xlsx"

# 读取 label_summary.xlsx
if os.path.exists(output_file):
    df = pd.read_excel(output_file)
    label_sum = df["Label Sum"].values
    
    # 标签映射：0 -> 0, 1-4 -> 1, 其他 -> 2
    label_sum = np.where(label_sum <= 1, 0, 1)
    
    # 统计转换后标签分布
    unique, counts = np.unique(label_sum, return_counts=True)
    label_distribution = dict(zip(unique, counts))
    
    # 打印标签分布
    print("转换后标签分布:")
    for label, count in label_distribution.items():
        print(f"标签值 {label}: {count} 个样本")
    
    # 保存转换后的标签数据
    pd.DataFrame(label_sum, columns=["Label Sum"]).to_excel(output_file1, index=False)
    print(f"转换后的统计结果已保存至 {output_file1}")
else:
    print(f"错误: {output_file} 文件不存在，请先运行统计代码。")