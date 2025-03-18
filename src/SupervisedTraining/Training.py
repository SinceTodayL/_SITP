import pandas as pd
import numpy as np

from template import SupervisedTraining, SupervisedTraining_ByStep_Order1, SupervisedTrainingWithoutKFold
from dim_reducer import DimensionReducer

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes  import GaussianNB 


import seaborn as sns

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

    data_path1 = r"E:\_SITP\data\Data.xlsx"
    label_path1 = r"E:\_SITP\data\DataLabel.xlsx"

    data_path2 = r"E:\_SITP\data\all_data\test_files\TestData.xlsx"
    label_path2 = r"E:\_SITP\data\all_data\test_files\TestLabel.xlsx"

    data1 = pd.read_excel(data_path1)
    label1 = pd.read_excel(label_path1)

    data2 = pd.read_excel(data_path2)
    label2 = pd.read_excel(label_path2)

    X1 = data1.iloc[ : ,  : ].values
    y1 = label1.iloc[ : ].values.squeeze()

    X2 = data2.iloc[ : ,  : ].values
    y2 = label2.iloc[ : ].values.squeeze()

    # X_res_pca = DimensionReducer(method='pca', n_components=7).fit_transform(X)
    # X_res_umap = DimensionReducer(method='umap', n_components=4).fit_transform(X)

    model_trained, scaler_trained = SupervisedTrainingWithoutKFold(X=X1, y=y1, train_model=RandomForestClassifier(n_estimators=300, random_state=123, max_depth=10, min_samples_split=2),
                    IfStandard=True, IfSMOTE=False, IfVisualize=True)
    
    X2 = scaler_trained.fit_transform(X2)

    # X1 = scaler_trained.transform(X1)
    res = model_trained.predict_proba(X2)

    from collections import Counter
    counts = Counter(y2)
    total = len(y2)
    proportions = {label: count / total for label, count in counts.items()} 


    n = res.shape[0] 
    labels = np.zeros(n,  dtype=int)
    k = int(n * proportions[2])
    if k > 0:
        sorted_indices_c = np.argsort(-res[:,  2])
        labels[sorted_indices_c[:k]] = 2

    '''
    # 处理第二个维度
    remaining = labels == 0
    l = int(n * proportions[1])
    sorted_indices_b = np.argsort(-res[remaining,  1])
    for i in range(0, 1127):
        if remaining[i] and i in sorted_indices_b[:l]:
            labels[i] = 1
    '''

    remaining = labels == 0
    for i in range(1127):
        if remaining[i]:    
            if res[i][0] > res[i][1]:
                labels[i] = 0
            else:
                labels[i] = 1

    final_report = classification_report(y2, labels, target_names=class_names, output_dict=True)
    print(final_report)

    cm = confusion_matrix(y2, labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
            xticklabels=class_names, 
            yticklabels=class_names)
    plt.title("Result")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    y_true = y2
    y_prob = res

    y_true_bin = np.where(y_true  == 0, 1, 0)  # 正类为0，负类为1/2
    y_score = y_prob[:, 0]    # 取0类的预测概率作为正类得分

    # 计算PR曲线数据
    precision, recall, pr_thresholds = precision_recall_curve(y_true_bin, y_score)
    pr_auc = auc(recall, precision)

    # 计算ROC曲线数据
    fpr, tpr, roc_thresholds = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)

    # 创建画布
    plt.figure(figsize=(12,  5))

    # 绘制PR曲线
    plt.subplot(1,  2, 1)
    plt.plot(recall,  precision, color='darkorange', lw=2, 
            label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall') 
    plt.ylabel('Precision') 
    plt.title('Precision-Recall  Curve (Class 0 vs Rest)')
    plt.legend(loc='lower left')
    plt.grid(True) 

    # 绘制ROC曲线
    plt.subplot(1,  2, 2)
    plt.plot(fpr,  tpr, color='navy', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,  1], [0, 1], 'k--', lw=1)
    plt.xlabel('False  Positive Rate')
    plt.ylabel('True  Positive Rate')
    plt.title('ROC  Curve (Class 0 vs Rest)')
    plt.legend(loc='lower right')
    plt.grid(True) 

    plt.tight_layout() 
    plt.show() 
        
        
    # final_report = classification_report(y, final_result, target_names=class_names, output_dict=True)
    # with_pca.append(final_report['macro avg']['f1-score'])

    """
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
    """

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
