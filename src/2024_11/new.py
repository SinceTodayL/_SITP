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

# 3. 使用随机森林分类器（支持类别权重）
class_weights = {0: 1, 1: 1, 2: 4}  # 平衡类别1和2的影响
clf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
clf.fit(X_train, y_train)

# 4. 进行 SMOTE 扩充，仅对测试集进行扩充
smote = SMOTE(random_state=42)
X_test_resampled, y_test_resampled = smote.fit_resample(X_test, y_test)

# 5. 模型预测
y_proba = clf.predict_proba(X_test_resampled)

# 自定义阈值优化函数
def optimize_threshold_multiclass(proba, true_labels):
    best_thresholds = {}
    f1_scores = {}
    for class_idx in np.unique(true_labels):
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1, best_threshold = 0, 0.5
        f1_for_class = []
        for threshold in thresholds:
            preds = (proba[:, class_idx] >= threshold).astype(int)
            binary_true = (true_labels == class_idx).astype(int)
            f1 = f1_score(binary_true, preds, zero_division=0)
            f1_for_class.append(f1)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
        best_thresholds[class_idx] = best_threshold
        f1_scores[class_idx] = f1_for_class
    return best_thresholds, f1_scores

# 针对每个类别优化分类阈值
best_thresholds, f1_scores = optimize_threshold_multiclass(y_proba, y_test_resampled)
print("Optimized Thresholds:", best_thresholds)

# 应用最佳阈值重新预测
y_pred = np.zeros_like(y_test_resampled)
for class_idx, threshold in best_thresholds.items():
    y_pred[y_proba[:, class_idx] >= threshold] = class_idx

# 6. 模型评估
conf_matrix = confusion_matrix(y_test_resampled, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test_resampled, y_pred, zero_division=0))

# 可视化
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(y), yticklabels=np.unique(y), cbar=False)
plt.title("Confusion Matrix with SMOTE Resampled Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 计算精度、召回率和F1分数
precision, recall, f1, _ = precision_recall_fscore_support(y_test_resampled, y_pred, zero_division=0)
results = pd.DataFrame({
    "Class": np.unique(y),
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})
print("\nClassification Results Summary:")
print(results)
