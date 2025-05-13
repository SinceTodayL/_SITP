import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import os
from itertools import cycle

# 创建保存图片的目录
os.makedirs('visualizations2', exist_ok=True)

# 数据加载与预处理
data = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\Data.xlsx", header=0)
labels = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx", header=0)
X = data.values
y = labels.values.flatten()

# 可视化函数改进版
def plot_feature_distribution(X, y, title, filename):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.8)
    plt.title(title)
    plt.savefig(f'visualizations2/{filename}.png')
    plt.close()

def plot_roc_curve(y_true, y_score, n_classes, model_name):
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= n_classes

    # Plot
    plt.figure(figsize=(10, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'visualizations2/{model_name}_ROC.png')
    plt.close()

def hierarchical_classification(model_layer1, model_layer2, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_pred = np.zeros_like(y)
    y_probs = np.zeros((len(y), 3))  # 存储三类概率
    
    # 可视化原始数据分布
    plot_feature_distribution(X, y, "Original Data Distribution", "original_distribution")

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train_orig, X_test_orig = X[train_index], X[test_index]
        y_train_orig, y_test_orig = y[train_index], y[test_index]
        
        # 应用SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_orig, y_train_orig)
        
        # ========== 第一层分类 ==========
        y_train_layer1 = np.where(y_train_res == 0, 0, 1)
        model_layer1.fit(X_train_res, y_train_layer1)
        y_pred_layer1 = model_layer1.predict(X_test_orig)
        layer1_probs = model_layer1.predict_proba(X_test_orig)
        
        # ========== 第二层分类 ==========
        mask_abnormal = (y_pred_layer1 == 1)
        X_test_abnormal = X_test_orig[mask_abnormal]
        y_test_abnormal = y_test_orig[mask_abnormal]
        
        if len(X_test_abnormal) > 0:
            mask_train_abnormal = (y_train_res != 0)
            X_train_abnormal = X_train_res[mask_train_abnormal]
            y_train_abnormal = y_train_res[mask_train_abnormal]
            y_train_layer2 = np.where(y_train_abnormal == 1, 0, 1)  # 1->0, 2->1

            model_layer2.fit(X_train_abnormal, y_train_layer2)
            y_pred_abnormal = model_layer2.predict(X_test_abnormal)
            layer2_probs = model_layer2.predict_proba(X_test_abnormal)
            
            # 合并概率结果
            y_probs[test_index[mask_abnormal], 1] = layer2_probs[:, 0] * layer1_probs[mask_abnormal, 1]
            y_probs[test_index[mask_abnormal], 2] = layer2_probs[:, 1] * layer1_probs[mask_abnormal, 1]
            y_probs[test_index, 0] = layer1_probs[:, 0]
            
            # 记录预测结果
            final_pred[test_index[mask_abnormal]] = np.where(y_pred_abnormal == 0, 1, 2)
        
        # 处理正常样本
        final_pred[test_index[~mask_abnormal]] = 0

    # ========== 评估指标 ==========
    # 生成分类报告
    report = classification_report(y, final_pred, target_names=["Normal", "Mild", "Severe"], output_dict=True)
    print(f"\n{'='*40}")
    print(f"{model_name} 分类结果报告")
    print(f"{'='*40}")
    print(f"正常类 (Normal):")
    print(f"  Precision: {report['Normal']['precision']:.4f}")
    print(f"  Recall:    {report['Normal']['recall']:.4f}")
    print(f"  F1-score:  {report['Normal']['f1-score']:.4f}")
    print(f"\n普通异常类 (Mild):")
    print(f"  Precision: {report['Mild']['precision']:.4f}")
    print(f"  Recall:    {report['Mild']['recall']:.4f}")
    print(f"  F1-score:  {report['Mild']['f1-score']:.4f}")
    print(f"\n明显异常类 (Severe):")
    print(f"  Precision: {report['Severe']['precision']:.4f}")
    print(f"  Recall:    {report['Severe']['recall']:.4f}")
    print(f"  F1-score:  {report['Severe']['f1-score']:.4f}")
    print(f"\n总加权平均:")
    print(f"  Precision: {report['weighted avg']['precision']:.4f}")
    print(f"  Recall:    {report['weighted avg']['recall']:.4f}")
    print(f"  F1-score:  {report['weighted avg']['f1-score']:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Mild", "Severe"],
                yticklabels=["Normal", "Mild", "Severe"])
    plt.title(f"{model_name} 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.savefig(f'visualizations2/{model_name}_ConfusionMatrix.png')
    plt.close()

    # ROC曲线
    plot_roc_curve(y, y_probs, 3, model_name)

# 测试不同核函数的SVM
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

for kernel in kernels:
    print(f"\n{'#'*40}")
    print(f"正在训练 {kernel} 核SVM...")
    print(f"{'#'*40}")
    
    model_layer1 = SVC(kernel=kernel, probability=True, random_state=42)
    model_layer2 = SVC(kernel=kernel, probability=True, random_state=42)
    
    hierarchical_classification(
        model_layer1=model_layer1,
        model_layer2=model_layer2,
        model_name=f"SVM-{kernel}"
    )