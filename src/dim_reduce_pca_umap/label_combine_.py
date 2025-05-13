import pandas as pd
import os

# 设置文件夹路径和输出文件路径
folder_path = 'C:\\Users\\LiuZh\\Desktop\\SITP\\label_data'  # 替换为你的文件夹路径
output_file = 'C:\\Users\\LiuZh\\Desktop\\SITP\\merged_label_data.xlsx'  # 输出文件名

# 创建一个空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 遍历文件夹中的所有Excel文件
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        if not filename.startswith('~$'):  # 跳过临时文件
            file_path = os.path.join(folder_path, filename)
            try:
                # 读取Excel文件中的E列数据
                df = pd.read_excel(file_path, usecols='E')
                # 将数据添加到all_data中
                all_data = pd.concat([all_data, df], ignore_index=True, axis=1)
            except PermissionError:
                print(f'无法访问文件: {file_path}')
            except Exception as e:
                print(f'处理文件 {file_path} 时出错: {e}')

# 将合并后的数据写入新的Excel文件
try:
    all_data.to_excel(output_file, index=False)
    print(f'所有E列数据已合并到 {output_file}')
except PermissionError:
    print(f'无法写入文件: {output_file}')
except Exception as e:
    print(f'保存文件时出错: {e}')


print(all_data.shape)
outliers=[]
for i in range(1127):
    if sum(all_data.T[i])>=4:
        outliers.append(i+1)

print(outliers)