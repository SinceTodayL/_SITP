import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import time

# 数据加载
data = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\Data.xlsx", header=0)
labels = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx", header=0)
X = data.values
y = labels.values.flatten()
class_names = ["Normal", "Mild", "Severe"]

# 创建可视化目录
plt.rcParams['font.size'] = 12  # 统一字体大小

# ========================
# 通用可视化函数
# ========================
def plot_confusion_matrix(cm, title):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc_curve(y_true, y_probs, title):
    """绘制ROC曲线"""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(10, 8))
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# ========================
# 层次化分类函数
# ========================
def hierarchical_classification(model_layer1, model_layer2, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    y_probs_all = np.zeros((len(y), 3))  # 初始化概率矩阵

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # SMOTE过采样（仅对训练数据扩充）
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # 第一层：正常(0) vs 异常(1,2)
        y_train_layer1 = np.where(y_train_res == 0, 0, 1)
        model_layer1.fit(X_train_res, y_train_layer1)
        y_pred_layer1 = model_layer1.predict(X_test)
        y_pred_layer1_proba = model_layer1.predict_proba(X_test)[:, 1]

        # 第二层：普通异常(1) vs 明显异常(2)
        mask_abnormal = (y_pred_layer1 == 1)
        X_test_abnormal = X_test[mask_abnormal]
        y_test_abnormal = y_test[mask_abnormal]

        if len(X_test_abnormal) > 0:
            y_train_abnormal = y_train_res[y_train_res != 0]
            X_train_abnormal = X_train_res[y_train_res != 0]
            y_train_layer2 = np.where(y_train_abnormal == 1, 0, 1)
            model_layer2.fit(X_train_abnormal, y_train_layer2)
            y_pred_abnormal = model_layer2.predict(X_test_abnormal)
            y_pred_abnormal_proba = model_layer2.predict_proba(X_test_abnormal)[:, 1]
        else:
            y_pred_abnormal = np.array([])
            y_pred_abnormal_proba = np.array([])

        # 合并预测结果
        y_pred = np.zeros_like(y_test)
        y_pred[mask_abnormal] = np.where(y_pred_abnormal == 0, 1, 2)
        y_pred[~mask_abnormal] = 0

        # 收集结果
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        # 修复概率分配逻辑
        abnormal_counter = 0
        for i, idx in enumerate(test_idx):
            if mask_abnormal[i]:
                if abnormal_counter < len(y_pred_abnormal_proba):
                    # 异常样本的概率分配
                    prob_severe = y_pred_abnormal_proba[abnormal_counter]
                    prob_mild = 1 - prob_severe
                    y_probs_all[idx] = [0, prob_mild, prob_severe]
                    abnormal_counter += 1
                else:
                    y_probs_all[idx] = [0, 0, 0]
            else:
                # 正常样本的概率分配
                prob_normal = 1 - y_pred_layer1_proba[i]
                y_probs_all[idx] = [prob_normal, 0, 0]

    # 混淆矩阵
    cm = confusion_matrix(y_true_all, y_pred_all)
    plot_confusion_matrix(cm, f'{model_name} Confusion Matrix')

    # ROC曲线
    plot_roc_curve(y_true_all, y_probs_all, f'{model_name} ROC Curves')

    # 分类报告
    report = classification_report(y_true_all, y_pred_all, target_names=class_names)
    print(f"\n{model_name} Classification Report:\n{report}")

# ========================
# 模型配置与执行
# ========================
models = {
    "Decision Tree + Random Forest": (
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(n_estimators=35, random_state=42)
    ),
    "SVM + Random Forest": (
        SVC(kernel='rbf', probability=True, random_state=89),
        RandomForestClassifier(n_estimators=35, random_state=3)
    ),
    "Random Forest + SVM": (
        RandomForestClassifier(n_estimators=35, random_state=42),
        SVC(kernel='rbf', probability=True, random_state=42)
    )
}

for model_name, (model1, model2) in models.items():
    hierarchical_classification(model1, model2, model_name)