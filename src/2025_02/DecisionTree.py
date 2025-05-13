from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
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
    plt.savefig(f'visualizations/{title}.png')  # 保存图片
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
    plt.savefig(f'visualizations/{model_name}_ROC_Curve.png')  # 保存图片
    plt.show()

def decision_tree_classification(model, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_pred = np.zeros_like(y)  # 根据 y 的大小初始化 final_pred
    y_probs = np.zeros((len(y), 3))  # 用于存储每个类别的概率

    # 可视化原始数据分布
    plot_feature_distribution(X, y, "Original Data Distribution")

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 应用SMOTE进行过采样（仅对训练集）
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # 训练模型
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)  # 获取概率

        # 存储预测结果和概率
        final_pred[test_index] = y_pred
        y_probs[test_index] = y_prob

    # 评估指标
    report = classification_report(y, final_pred, target_names=["Normal", "Mild", "Severe"], output_dict=True)
    macro_avg = report['macro avg']

    print(f"{model_name} Classification Report")
    print(classification_report(y, final_pred, target_names=["Normal", "Mild", "Severe"]))

    # 混淆矩阵
    cm = confusion_matrix(y, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Mild", "Severe"], yticklabels=["Normal", "Mild", "Severe"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f'visualizations/{model_name}_Confusion_Matrix.png')  # 保存图片
    plt.show()

    # ROC曲线
    plot_roc_curve(y, y_probs, n_classes=3, model_name=model_name)

# 使用决策树进行分类
model_dt = DecisionTreeClassifier(max_depth=10, random_state=42)  # 限制树的深度，避免过拟合
# 使用概率校准器对决策树进行校准
calibrated_model_dt = CalibratedClassifierCV(model_dt, method='sigmoid', cv=5)  # 使用交叉验证进行校准
decision_tree_classification(calibrated_model_dt, "Decision Tree")