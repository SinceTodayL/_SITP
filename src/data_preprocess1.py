import os 
import pandas as pd 
import torch 
import numpy as np
from sklearn.decomposition  import PCA 
 
# 获取当前目录下所有xlsx文件并按文件名排序 
all_files = sorted([f for f in os.listdir(r"E:\\_SITP\\data") if f.endswith('.xlsx')]) 
 
# 确保目标文件夹存在 
output_dir = r"E:\\_SITP\\data\\pca_to_1dim"
assert os.path.exists(output_dir),  "目标文件夹不存在"
 
# 每两个文件为一组处理 
for i in range(0, len(all_files), 2):
    group = all_files[i:i+2]
    
    # 验证文件分组规范 
    prefix = group[0][:2]
    if not all(f.startswith(prefix)  for f in group):
        print(f"警告：文件 {group} 不符合前两字符相同规范，已跳过")
        continue 
 
    # 读取并合并数据 
    dfs = []
    for file in group:
        df = pd.read_excel("E:\\_SITP\\data\\" + file) 
        dfs.append(df[['value_left',  'value_right']].values)
    
    # 合并为4列张量 (n_samples, 4)
    combined = torch.tensor(np.concatenate(dfs,  axis=1), dtype=torch.float32) 
    
    # PCA降维 
    pca = PCA(n_components=1)
    reduced = pca.fit_transform(combined.numpy()) 
    
    # 创建结果DataFrame 
    result_df = pd.DataFrame(reduced, columns=['pca_value'])
    
    # 保存文件 
    output_path = os.path.join(output_dir,  f"{prefix}.xlsx")
    result_df.to_excel(output_path,  index=False)
 
print(f"处理完成，共生成{len(all_files)//2}个文件于 {output_dir} 目录")