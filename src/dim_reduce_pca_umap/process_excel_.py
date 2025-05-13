import pandas as pd

# 读取原始Excel文件
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\data.xlsx"
df = pd.read_excel(file_path)

# 初始化新的DataFrame用于存放合并后的数据
new_df = pd.DataFrame()

# 每1128行为一组，遍历整个数据
for i in range(0, len(df), 1127*2):
    # 计算每组的起始索引和结束索引
    start_idx = i + 0
    end_idx = i + 1127

    # 提取第2行至1128行和1129行至2255行的E、F列数据
    part1 = df.iloc[start_idx:end_idx, [4, 5]]  # 第2行至1128行的E、F列
    part2 = df.iloc[end_idx:end_idx + 1127, [4, 5]]  # 第1129至2256行的E、F列

    # 合并两部分数据，确保行号对齐
    merged_df = pd.concat([part1.reset_index(drop=True), part2.reset_index(drop=True)], axis=1)

    # 将合并后的数据添加到新的DataFrame中
    if not new_df.empty:
        new_df = pd.concat([new_df, pd.DataFrame(columns=[' '])], axis=1)
    new_df = pd.concat([new_df, merged_df], ignore_index=True, axis=1)

# 重新设置新DataFrame的列名
# column_names = [f'Column {i}' for i in range(1, new_df.shape[1] + 1)]
# new_df.columns = column_names

# 保存到新的Excel文件
output_file_path = 'C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data.xlsx'
new_df.to_excel(output_file_path, index=False)
