import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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

# **第一阶段：类别2检测**
# 使用SMOTE扩充类别2的数据
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 定义类别权重
class_weights = {0: 1, 1: 1, 2: 4}

# 训练随机森林分类器
clf_2 = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
clf_2.fit(X_train_resampled, y_train_resampled)

# 类别2的预测
y_proba_2 = clf_2.predict_proba(X_test)
threshold_2 = 0.5  # 默认阈值
y_pred_2 = (y_proba_2[:, 2] >= threshold_2).astype(int) * 2  # 预测为类别2的样本

# 从测试集中剔除类别2的样本
mask_01 = (y_test != 2)  # 只保留类别0和1
X_test_01 = X_test[mask_01]
y_test_01 = y_test[mask_01]

# 从训练集中剔除类别2的样本
mask_train_01 = (y_train != 2)  # 只保留类别0和1
X_train_01 = X_train[mask_train_01]
y_train_01 = y_train[mask_train_01]

# **第二阶段：类别0和1检测**
# SMOTE扩充类别0和1的数据
smote_01 = SMOTE(random_state=42)
X_train_01_resampled, y_train_01_resampled = smote_01.fit_resample(X_train_01, y_train_01)

# 定义类别权重
class_weights_01 = {0: 1, 1: 1}  # 提高类别1的权重

# 训练随机森林分类器
clf_01 = RandomForestClassifier(n_estimators=100, class_weight=class_weights_01, random_state=42)
clf_01.fit(X_train_01_resampled, y_train_01_resampled)

# 类别0和1的预测
y_proba_01 = clf_01.predict_proba(X_test_01)
threshold_01 = 0.5  # 默认阈值
y_pred_01 = np.where(y_proba_01[:, 1] >= threshold_01, 1, 0)  # 使用阈值预测类别0和1

# **合并预测结果**
y_pred_final = y_pred_2.copy()  # 初始化为类别2的预测
y_pred_final[mask_01] = y_pred_01  # 替换为类别0和1的预测

# **评估最终模型性能**
conf_matrix = confusion_matrix(y_test, y_pred_final)
print("\nFinal Confusion Matrix:")
print(conf_matrix)

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_final, zero_division=0))

# 计算加权平均F1-Score
f1_weighted = f1_score(y_test, y_pred_final, average='weighted', zero_division=0)
print("\nFinal Weighted F1-Score:", f1_weighted)

# **可视化**
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(y), yticklabels=np.unique(y), cbar=False)
plt.title("Final Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
