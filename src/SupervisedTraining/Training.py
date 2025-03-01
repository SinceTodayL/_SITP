import pandas as pd
import numpy as np

from template import SupervisedTraining, SupervisedTraining_ByStep_Order1
from dim_reducer import DimensionReducer

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes  import GaussianNB 

from sklearn.preprocessing  import PowerTransformer

import matplotlib.pyplot as plt

train_models = [KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1),
                SVC(kernel='rbf', C=1),
                GaussianNB(var_smoothing=1e-9),
                MLPClassifier(alpha=0.0001, learning_rate_init=0.0001, hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42),
                RandomForestClassifier(n_estimators=300, random_state=42, max_depth=10, min_samples_split=2),
                DecisionTreeClassifier(random_state=42, max_depth=10, criterion='entropy'), 
            ]

with_smote = []
without_smote = []
with_pca = []
with_umap = []
with_no = []
class_names = ["Normal", "Mild", "Sereve"]

if __name__ == "__main__":

    data_path = r"E:\_SITP\data\Data.xlsx"
    label_path = r"E:\_SITP\data\DataLabel.xlsx"

    data = pd.read_excel(data_path)
    label = pd.read_excel(label_path)

    X = data.iloc[ : ,  : ].values
    y = label.iloc[ : ].values.squeeze()

    X_res_pca = DimensionReducer(method='pca', n_components=7).fit_transform(X)
    X_res_umap = DimensionReducer(method='umap', n_components=4).fit_transform(X)

    for model in train_models:
        final_result = SupervisedTraining(X=X_res_pca, y=y, train_model=model, 
                       IfStandard=True, IfSMOTE=True, IfVisualize=False)
        final_report = classification_report(y, final_result, target_names=class_names, output_dict=True)
        with_pca.append(final_report['macro avg']['f1-score'])
    
    for model in train_models:
        final_result = SupervisedTraining(X=X_res_umap, y=y, train_model=model, 
                       IfStandard=True, IfSMOTE=True, IfVisualize=False)
        final_report = classification_report(y, final_result, target_names=class_names, output_dict=True)
        with_umap.append(final_report['macro avg']['f1-score'])

    for model in train_models:
        final_result = SupervisedTraining(X=X, y=y, train_model=model, 
                       IfStandard=True, IfSMOTE=True, IfVisualize=False)
        final_report = classification_report(y, final_result, target_names=class_names, output_dict=True)
        with_no.append(final_report['macro avg']['f1-score'])

    
    
    plt.figure(figsize=(10,  6))

    x = ['KNN', 'SVC', 'Naive Bayes', 'MLP', 'RF', 'DecisionTree']

    plt.plot(x, with_pca, color='blue', marker='o', linestyle='-', label='with pca')
    plt.plot(x, with_umap, color='red', marker='^', linestyle='-', label='with umap')
    plt.plot(x, with_no, color='green', marker='s', linestyle='-', label='normal')

    plt.title('dim reduce')
    plt.xlabel('model') 
    plt.ylabel('f1-Score') 
    plt.legend() 
    plt.grid(True)
    plt.tight_layout()   

    import os

    save_dir = r"E:\_SITP\src\SupervisedTraining"  
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,  "dim reduce.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight') 

    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    X_trans = pt.fit_transform(X)  

'''
    for model in train_models:
        final_result = SupervisedTraining(X=X, y=y, train_model=model, 
                       IfStandard=True, IfSMOTE=True, IfVisualize=False)
        final_report = classification_report(y, final_result, target_names=class_names, output_dict=True)
        with_smote.append(final_report['macro avg']['f1-score'])
    
    for model in train_models:
        final_result = SupervisedTraining(X=X, y=y, train_model=model, 
                       IfStandard=True, IfSMOTE=False, IfVisualize=False)
        final_report = classification_report(y, final_result, target_names=class_names, output_dict=True)
        without_smote.append(final_report['macro avg']['f1-score'])
    
    plt.figure(figsize=(10,  6))

    x = ['KNN', 'SVC', 'Naive Bayes', 'MLP', 'RF', 'DecisionTree']

    plt.plot(x, with_smote, color='blue', marker='o', linestyle='-', label='with smote')
    plt.plot(x, without_smote, color='red', marker='^', linestyle='-', label='without smote')

    plt.title('SMOTE vs Without SMOTE')
    plt.xlabel('model') 
    plt.ylabel('f1-Score') 
    plt.legend() 
    plt.grid(True)
    plt.tight_layout()   

    import os

    save_dir = r"E:\_SITP\src\SupervisedTraining"  
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,  "SMOTE vs Without SMOTE.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight') 
'''

#    SupervisedTraining(X=X_res, y=y, train_model=SVC(kernel='linear', probability=True), 
#                      IfStandard=True, IfSMOTE=True, IfVisualize=True) 


#    SupervisedTraining_ByStep_Order1(X=X, y=y, 
#   train_model1=MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=2000, random_state=42),
#   train_model2=MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=300, random_state=42),
#   IfStandard=True, IfSMOTE1=True, IfSMOTE2=True, IfVisualize=True)

#   SupervisedTraining_ByStep_Order1(X=X, y=y, 
#   train_model1=RandomForestClassifier(n_estimators=400, random_state=42),
#   train_model2=RandomForestClassifier(n_estimators=100, random_state=42),
#   IfStandard=True, IfSMOTE1=True, IfSMOTE2=True, IfVisualize=True)




'''
    TODO:
        1. threshold optimization 
        2. dimension reduction 
        3. 改为分步检测
        4. 模型堆叠

'''
