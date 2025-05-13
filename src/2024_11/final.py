import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

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

# 3. 初始检测模型 - 随机森林
rf_clf = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 1, 2: 4}, random_state=42)
rf_clf.fit(X_train, y_train)
rf_proba = rf_clf.predict_proba(X_test)

# 4. 初始检测模型 - 多层感知器
mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_clf.fit(X_train, y_train)
mlp_proba = mlp_clf.predict_proba(X_test)

# 5. 融合模型预测
combined_proba = (rf_proba + mlp_proba) / 2

# 自定义阈值优化函数
def optimize_threshold_multiclass(proba, true_labels):
    best_thresholds = {}
    for class_idx in np.unique(true_labels):
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1, best_threshold = 0, 0.5
        for threshold in thresholds:
            preds = (proba[:, class_idx] >= threshold).astype(int)
            binary_true = (true_labels == class_idx).astype(int)
            f1 = f1_score(binary_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
        best_thresholds[class_idx] = best_threshold
    return best_thresholds

# 优化分类阈值
best_thresholds = optimize_threshold_multiclass(combined_proba, y_test)
print("Optimized Thresholds for Combined Model:", best_thresholds)

# 应用优化阈值
y_pred_combined = np.zeros_like(y_test)
for class_idx, threshold in best_thresholds.items():
    y_pred_combined[combined_proba[:, class_idx] >= threshold] = class_idx

# 6. 数据扩充
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 输出扩充后的数据分布
print("Original Training Data Distribution:", np.bincount(y_train))
print("Resampled Training Data Distribution:", np.bincount(y_train_resampled))

# 7. 进一步筛选 - KNN
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_resampled, y_train_resampled)
y_pred_knn = knn_clf.predict(X_test)

# 最终模型预测结果
final_pred = y_pred_knn

# 8. 模型评估
conf_matrix = confusion_matrix(y_test, final_pred)
print("\nConfusion Matrix for Final Model:")
print(conf_matrix)

print("\nClassification Report for Final Model:")
print(classification_report(y_test, final_pred, zero_division=0))

# 9. 可视化
# 混淆矩阵可视化
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(y), yticklabels=np.unique(y), cbar=False)
plt.title("Confusion Matrix for Final Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 分类结果分布可视化
plt.figure(figsize=(8, 6))
sns.countplot(x=final_pred, order=np.unique(y), palette="husl")
plt.title("Final Prediction Distribution Across Classes")
plt.xlabel("Predicted Classes")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 扩充数据的分布可视化
print("Resampled Training Data Distribution:", np.bincount(y_train_resampled))
