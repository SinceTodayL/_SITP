import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载和预处理
# 假设数据集格式为：特征列 + 标签列，其中标签列包含0（正常）、1和2（异常）
file_path = 'C:\\Users\\LiuZh\\Desktop\\SITP\\after_SMOTE_dataset.xlsx'  # 替换为数据集的路径
data = pd.read_excel(file_path)

# 特征和标签
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 标签（0 = 正常, 1和2为异常）

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. 使用加权Random Forest进行训练
# 给不同的标签设置权重（较高权重给标签2，因为它是异常标签）
class_weights = {0: 1, 1: 1, 2: 5}  # 标签2的权重较大，重点关注异常点

# 创建并训练Random Forest模型
clf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
clf.fit(X_train, y_train)

# 3. 预测
y_pred = clf.predict(X_test)

# 4. 模型评估
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 6. 可视化特征的重要性
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices, rotation=90)
plt.title("Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
