import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"E:\\_SITP\\data\\pca_to_1dim"
a = 14
b = 75 

num_year_groups = 14 // a  
num_sample_groups = 1127 // b 

X = []
files = sorted([f for f in os.listdir(file_path) if f.endswith('.xlsx')])
for file in files:
    df = pd.read_excel(os.path.join(file_path, file))
    X.append(df.values.flatten()) 
X = np.array(X)

for year_group in range(num_year_groups):
    start_year_idx = year_group * a  
    end_year_idx = start_year_idx + a  

    for sample_group in range(num_sample_groups):
        start_sample_idx = sample_group * b  
        end_sample_idx = start_sample_idx + b  

        fig, ax = plt.subplots(figsize=(20, 12))

        colors = plt.cm.jet(np.linspace(0, 1, len(files)))

        for i in range(start_year_idx, end_year_idx):
            if i in [0, 5, 6, 8, 9]:
                continue
            z = X[i, start_sample_idx:end_sample_idx]
            y = np.full(b, i - start_year_idx + 1)
            x = range(b)
            ax.plot(x, z, label=f'File {i+1}')

        ax.set_xlabel('Row Index')
        ax.set_ylabel('Value') 

        plt.title('2D Plot of 1D Excel Data', fontsize=16)
        plt.legend() 
        plt.show()
