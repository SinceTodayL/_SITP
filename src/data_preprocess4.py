import os
import pandas as pd

# 设置文件夹路径
file_path = r'E:\_SITP\data'  # 请替换为实际路径
save_path = r'E:\_SITP\data'  # 保存 .py 文件的路径

# 读取所有 Excel 文件名，按文件名排序
files = sorted([f for f in os.listdir(file_path) if f.endswith('.xlsx')])

# 检查是否为偶数数量的文件
if len(files) % 2 != 0:
    raise ValueError("文件数不是偶数，无法两两配对")

# 初始化一个空的 DataFrame 用于合并数据
data_all = pd.DataFrame()

# 每两个文件为一组（一个时间点），合并其 value_left 和 value_right 列
year_count = len(files) // 2
for i in range(0, len(files), 2):
    # 读取一组文件
    file_left = os.path.join(file_path, files[i])
    file_right = os.path.join(file_path, files[i + 1])
    df_left = pd.read_excel(file_left)[['value_left', 'value_right']].reset_index(drop=True)
    df_right = pd.read_excel(file_right)[['value_left', 'value_right']].reset_index(drop=True)

    # 重命名列为当前年份的 4 个特征
    year_index = i // 2 + 1
    df_left.columns = [f'year{year_index}_left1', f'year{year_index}_right1']
    df_right.columns = [f'year{year_index}_left2', f'year{year_index}_right2']

    # 按列拼接为一个 1127×4 的数据块
    df_combined = pd.concat([df_left, df_right], axis=1)

    # 添加到总表中
    data_all = pd.concat([data_all, df_combined], axis=1)

# 打印输出信息
data_all.info()

# 保存为 .csv 文件，或者你也可以改为 .xlsx
output_csv = os.path.join(save_path, 'merged_56_columns.csv')
data_all.to_csv(output_csv, index=False)
print(f"数据已保存为：{output_csv}")