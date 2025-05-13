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
y = data.iloc[:, -1].values   # 标签

# 标准化特征值
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# 3. 使用随机森林分类器（支持类别权重）
class_weights = {0: 1, 1: 1, 2: 4}  # 平衡类别1和2的影响
clf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
clf.fit(X_train, y_train)

# 4. 模型预测
y_pred = clf.predict(X_test)

# 计算加权F1-score
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"\nWeighted F1-Score: {f1_weighted:.4f}")

# 5. 模型评估
conf_matrix = confusion_matrix(y_test, y_pred)

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
