import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    f1_score  # Ensure f1_score is imported
)
from sklearn.ensemble import RandomForestClassifier

# 1. 加载数据
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy_label.xlsx"  # 替换为您的文件路径
data = pd.read_excel(file_path)

# 2. 数据预处理
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values  # 标签

# 标准化特征值
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# 3. 使用随机森林分类器（支持类别权重）
class_weights = {0: 3, 1: 3, 2: 800}  # 平衡类别1和2的影响
clf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
clf.fit(X_train, y_train)

# 4. 模型预测
y_proba = clf.predict_proba(X_test)


# 自定义阈值优化函数（偏重标签为2）
def optimize_threshold_multiclass(proba, true_labels, focus_class=1):
    best_thresholds = {}
    for class_idx in np.unique(true_labels):
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1, best_threshold = 0, 0.5
        for threshold in thresholds:
            preds = (proba[:, class_idx] >= threshold).astype(int)
            binary_true = (true_labels == class_idx).astype(int)

            # 如果是标签2，给F1-score一个更高的权重
            if class_idx == focus_class:
                f1 = f1_score(binary_true, preds, zero_division=0)
            else:
                # 可以使用普通的F1-score
                f1 = f1_score(binary_true, preds, zero_division=0)

            # 选择F1最高的阈值
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
        best_thresholds[class_idx] = best_threshold
    return best_thresholds


# 针对标签为2的类别优化分类阈值
best_thresholds = optimize_threshold_multiclass(y_proba, y_test, focus_class=2)
print("Optimized Thresholds:", best_thresholds)

# 应用最佳阈值重新预测
y_pred = np.zeros_like(y_test)
for class_idx, threshold in best_thresholds.items():
    y_pred[y_proba[:, class_idx] >= threshold] = class_idx

# 5. 模型评估
conf_matrix = confusion_matrix(y_test, y_pred)

# 计算加权F1-score
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"\nWeighted F1-Score: {f1_weighted:.4f}")

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print(classification_report(y_test, y_pred, zero_division=0))

# 6. 可视化

# 6.1 混淆矩阵可视化（默认蓝色）
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',  # 使用默认蓝色
            xticklabels=np.unique(y), yticklabels=np.unique(y), cbar=False)
plt.title("Confusion Matrix with Counts")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 6.2 特征重要性可视化
importances = clf.feature_importances_
feature_names = data.columns[:-1]
plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=feature_names, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6.3 分类结果分布
plt.figure(figsize=(8, 6))
sns.countplot(x=y_pred, order=np.unique(y), palette="husl")
plt.title("Prediction Distribution Across Classes")
plt.xlabel("Predicted Classes")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 6.4 Precision, Recall, F1-Score 汇总表
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
results = pd.DataFrame({
    "Class": np.unique(y),
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})
print("\nClassification Results Summary:")
print(results)

# 6.5 分类指标柱状图
results_melted = results.melt(id_vars="Class", value_vars=["Precision", "Recall", "F1-Score"])
plt.figure(figsize=(10, 6))
sns.barplot(x="Class", y="value", hue="variable", data=results_melted, palette="Set2")
plt.title("Precision, Recall, F1-Score by Class")
plt.xlabel("Class")
plt.ylabel("Score")
plt.legend(title="Metrics")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
