import numpy as np
import pandas as pd

file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\merged_label_data.xlsx"
df = pd.read_excel(file_path, usecols="O", header=None)
df = df.to_numpy()

label = []
label_size = [0]*3
record_0 = []
record_1 = []
record_2 = []
for i in range(0, df.size):
    if df[i] == 0:
        label.append(0)
        label_size[0] += 1
        record_0.append(i + 1)
    elif 1 <= df[i] < 4:
        label.append(1)
        label_size[1] += 1
        record_1.append(i + 1)
    else:
        label.append(2)
        label_size[2] += 1
        record_2.append(i + 1)

print("record_0:", record_0)
print("record_1:", record_1)
print("record_2:", record_2)

"""former_data = pd.read_excel(file_path, header=None)
df_former_data = pd.DataFrame(former_data)
result = pd.concat([df_former_data, df_label], axis=1)

file_path_ = "C:\\Users\\LiuZh\\Desktop\\SITP\\merged_label_data_.xlsx"
label_file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\new_label.xlsx"
result.to_excel(file_path_, index=False, header=False)
df_label.to_excel(label_file_path, index=False, header=False)"""




