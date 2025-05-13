from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# 创建保存图片的目录
os.makedirs('visualizations', exist_ok=True)

# 数据加载与预处理
data = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\Data.xlsx", header=0)
labels = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx", header=0)
X = data.values
y = labels.values.flatten()

# 可视化函数
def plot_feature_distribution(X, y, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.8)
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_score, n_classes, model_name):
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 6))
    for i, color in zip(range(n_classes), ['blue', 'green', 'red']):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def hierarchical_classification(model_layer1, model_layer2, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_pred = np.zeros_like(y)
    layer1_preds = []
    layer2_preds = []
    layer1_trues = []
    layer2_trues = []
    
    # 可视化原始数据分布
    plot_feature_distribution(X, y, "Original Data Distribution")
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train_orig, X_test_orig = X[train_index], X[test_index]
        y_train_orig, y_test_orig = y[train_index], y[test_index]
        
        # 应用SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_orig, y_train_orig)
        
        # ========== 第一层分类（区分正常与异常） ========== 
        y_train_layer1 = np.where(y_train_res == 0, 0, 1)
        model_layer1.fit(X_train_res, y_train_layer1)
        y_pred_layer1 = model_layer1.predict(X_test_orig)
        layer1_preds.extend(y_pred_layer1)
        layer1_trues.extend(np.where(y_test_orig == 0, 0, 1))
        
        # ========== 第二层分类（异常中区分普通异常与明显异常） ========== 
        mask_abnormal = (y_pred_layer1 == 1)
        X_test_abnormal = X_test_orig[mask_abnormal]
        y_test_abnormal = y_test_orig[mask_abnormal]
        
        if len(X_test_abnormal) > 0:
            mask_train_abnormal = (y_train_res != 0)
            X_train_abnormal = X_train_res[mask_train_abnormal]
            y_train_abnormal = y_train_res[mask_train_abnormal]
            y_train_layer2 = np.where(y_train_abnormal == 1, 0, 1)  # Normal -> Mild, Abnormal -> Severe
            
            model_layer2.fit(X_train_abnormal, y_train_layer2)
            y_pred_abnormal = model_layer2.predict(X_test_abnormal)
            
            layer2_preds.extend(y_pred_abnormal)
            layer2_trues.extend(np.where(y_test_abnormal == 1, 0, 1))
            
            y_pred_abnormal_original = np.where(y_pred_abnormal == 0, 1, 2)  # 转换为 Mild 和 Severe
            final_pred[test_index[mask_abnormal]] = y_pred_abnormal_original
        
        final_pred[test_index[~mask_abnormal]] = 0
    
    # ========== 评估指标 ========== 
    layer1_report = classification_report(layer1_trues, layer1_preds, target_names=["Normal", "Abnormal"], output_dict=True)
    if len(layer2_trues) > 0:
        layer2_report = classification_report(layer2_trues, layer2_preds, target_names=["Mild", "Severe"], output_dict=True)
    else:
        layer2_report = {"Mild": {"precision": 0, "recall": 0, "f1-score": 0}, 
                         "Severe": {"precision": 0, "recall": 0, "f1-score": 0}}
    
    final_report = classification_report(y, final_pred, target_names=["Normal", "Mild", "Severe"], output_dict=True)
    macro_avg = final_report['macro avg']
    
    # 输出报告
    print(f"{model_name} Classification Report")
    print(final_report)
    
    # 混淆矩阵
    cm = confusion_matrix(y, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Mild", "Severe"], yticklabels=["Normal", "Mild", "Severe"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    # ROC曲线
    y_probs = model_layer1.predict_proba(X)
    plot_roc_curve(y, y_probs, n_classes=3, model_name=model_name)

# 使用SVM和Random Forest进行分类
model_layer1_svm = SVC(kernel='linear', probability=True)
model_layer2_svm = SVC(kernel='linear', probability=True)
model_layer1_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_layer2_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# SVM层次分类
print("SVM Classification Results:")
hierarchical_classification(model_layer1_svm, model_layer2_svm, "SVM")

# Random Forest层次分类
print("Random Forest Classification Results:")
hierarchical_classification(model_layer1_rf, model_layer2_rf, "Random Forest")
