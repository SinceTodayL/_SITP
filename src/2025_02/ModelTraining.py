import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

# ----------------------------------
# 数据加载
# ----------------------------------
data = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\Data.xlsx", header=0)
labels = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx", header=0)
X = data.values
y = labels.values.flatten()

# 存储评估指标
results_v1 = []  # 用于存储第一种方法的评估结果
results_v2 = []  # 用于存储第二种方法的评估结果

# ----------------------------------
# 分层分类函数（正常->异常->严重）
# ----------------------------------
def hierarchical_classification_v1(model_layer1, model_layer2, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    y_proba_all = np.zeros((len(y), 3))
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 第一层：正常(0) vs 异常(1,2)
        y_train_layer1 = np.where(y_train == 0, 0, 1)  # 正常=0，异常=1
        model_layer1.fit(X_train, y_train_layer1)
        y_pred_layer1 = model_layer1.predict(X_test)
        y_pred_layer1_proba = model_layer1.predict_proba(X_test)[:, 1]  # 异常概率
        
        # 第二层：普通异常(1) vs 明显异常(2)
        mask_abnormal = (y_pred_layer1 == 1)  # 异常样本
        X_test_abnormal = X_test[mask_abnormal]
        y_test_abnormal = y_test[mask_abnormal]
        
        if len(X_test_abnormal) > 0:
            y_train_abnormal = y_train[y_train != 0]  # 训练集中的异常样本
            X_train_abnormal = X_train[y_train != 0]
            y_train_layer2 = np.where(y_train_abnormal == 1, 0, 1)  # 普通异常=0，明显异常=1
            model_layer2.fit(X_train_abnormal, y_train_layer2)
            y_pred_abnormal = model_layer2.predict(X_test_abnormal)
            y_pred_abnormal_proba = model_layer2.predict_proba(X_test_abnormal)[:, 1]  # 明显异常概率
        else:
            y_pred_abnormal = np.array([])
            y_pred_abnormal_proba = np.array([])
        
        # 合并预测结果
        y_pred = np.zeros_like(y_test)
        y_pred[mask_abnormal] = np.where(y_pred_abnormal == 0, 1, 2)  # 普通异常=1，明显异常=2
        y_pred[~mask_abnormal] = 0  # 正常=0
        
        # 收集结果
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Mild", "Severe"],
                yticklabels=["Normal", "Mild", "Severe"])
    plt.title(f'{model_name} Confusion Matrix (Approach 1)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # 分类报告
    report = classification_report(y_true_all, y_pred_all, target_names=["Normal", "Mild", "Severe"], output_dict=True)
    for label in ["Normal", "Mild", "Severe"]:
        results_v1.append({
            "Algorithm": model_name,
            "Task": f"Class {label}",
            "Precision": report[label]['precision'],
            "Recall": report[label]['recall'],
            "F1-Score": report[label]['f1-score']
        })

# ----------------------------------
# 分层分类函数（严重异常->正常/普通）
# ----------------------------------
def hierarchical_classification_v2(model_layer1, model_layer2, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    y_proba_all = np.zeros((len(y), 3))
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 第一层：明显异常(2) vs 其他
        y_train_layer1 = np.where(y_train == 2, 1, 0)  # 明显异常=1，其他=0
        model_layer1.fit(X_train, y_train_layer1)
        y_pred_layer1 = model_layer1.predict(X_test)
        y_pred_layer1_proba = model_layer1.predict_proba(X_test)[:, 1]  # 明显异常概率
        
        # 第二层：正常(0) vs 普通异常(1)
        mask_non_severe = (y_pred_layer1 == 0)  # 非明显异常样本
        X_test_non_severe = X_test[mask_non_severe]
        y_test_non_severe = y_test[mask_non_severe]
        
        if len(X_test_non_severe) > 0:
            y_train_non_severe = y_train[y_train != 2]  # 训练集中的非明显异常样本
            X_train_non_severe = X_train[y_train != 2]
            y_train_layer2 = np.where(y_train_non_severe == 0, 0, 1)  # 正常=0，普通异常=1
            model_layer2.fit(X_train_non_severe, y_train_layer2)
            y_pred_non_severe = model_layer2.predict(X_test_non_severe)
            y_pred_non_severe_proba = model_layer2.predict_proba(X_test_non_severe)[:, 1]  # 普通异常概率
        else:
            y_pred_non_severe = np.array([])
            y_pred_non_severe_proba = np.array([])
        
        # 合并预测结果
        y_pred = np.zeros_like(y_test)
        y_pred[mask_non_severe] = np.where(y_pred_non_severe == 0, 0, 1)  # 正常=0，普通异常=1
        y_pred[~mask_non_severe] = 2  # 明显异常=2
        
        # 收集结果
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=["Normal", "Mild", "Severe"],
                yticklabels=["Normal", "Mild", "Severe"])
    plt.title(f'{model_name} Confusion Matrix (Approach 2)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # 分类报告
    report = classification_report(y_true_all, y_pred_all, target_names=["Normal", "Mild", "Severe"], output_dict=True)
    for label in ["Normal", "Mild", "Severe"]:
        results_v2.append({
            "Algorithm": model_name,
            "Task": f"Class {label}",
            "Precision": report[label]['precision'],
            "Recall": report[label]['recall'],
            "F1-Score": report[label]['f1-score']
        })

# ----------------------------------
# 模型配置与执行
# ----------------------------------
models = {

    # "KNN": (KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=3)),
    # "Random Forest": (RandomForestClassifier(n_estimators=35, max_depth=15), RandomForestClassifier(n_estimators=35, max_depth=15)),
    "MLP": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=10000), MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000))
}

# 执行第一种分层分类
for name, (model1, model2) in models.items():
    hierarchical_classification_v1(model1, model2, name)

# 执行第二种分层分类
for name, (model1, model2) in models.items():
    hierarchical_classification_v2(model1, model2, name)

# 打印第一种方法的评估结果
results_v1_df = pd.DataFrame(results_v1)
print("\n========== 第一种方法评估结果 ==========")
print(results_v1_df.to_markdown(index=False))

# 打印第二种方法的评估结果
results_v2_df = pd.DataFrame(results_v2)
print("\n========== 第二种方法评估结果 ==========")
print(results_v2_df.to_markdown(index=False))