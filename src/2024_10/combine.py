import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy_label.xlsx"  # 请将路径替换为你的Excel文件路径
data = pd.read_excel(file_path)

# 将特征和标签分开
X = data.iloc[:, :-1]  # 所有行，除最后一列外的所有列
y = data.iloc[:, -1]  # 所有行，最后一列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义基础模型
model_rf = RandomForestClassifier(random_state=42)
model_knn = KNeighborsClassifier(n_neighbors=15)
model_mlp = MLPClassifier(random_state=42, max_iter=100)

# 训练基础模型并获取预测
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

model_knn.fit(X_train, y_train)
pred_knn = model_knn.predict(X_test)

model_mlp.fit(X_train, y_train)
pred_mlp = model_mlp.predict(X_test)

# 组合模型的预测结果
test_meta_features = np.column_stack((pred_rf, pred_knn, pred_mlp))

# 训练元模型（逻辑回归）
meta_model = LogisticRegression()
meta_model.fit(test_meta_features, y_test)

# 使用元模型进行最终预测
final_predictions = meta_model.predict(test_meta_features)

# 计算混淆矩阵、准确率、召回率和 F1 分数
cm = confusion_matrix(y_test, final_predictions)
accuracy = accuracy_score(y_test, final_predictions)
recall = recall_score(y_test, final_predictions, average='weighted')
f1 = f1_score(y_test, final_predictions, average='weighted')

# 输出结果
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score (Final Predictions): {f1:.4f}")

# 打印每个模型的 F1 分数
f1_rf = f1_score(y_test, pred_rf, average='weighted')
f1_knn = f1_score(y_test, pred_knn, average='weighted')
f1_mlp = f1_score(y_test, pred_mlp, average='weighted')

print(f"F1 Score (Random Forest): {f1_rf:.4f}")
print(f"F1 Score (KNN): {f1_knn:.4f}")
print(f"F1 Score (MLP): {f1_mlp:.4f}")

# 打印分类报告
print("\nClassification Report:")
print(classification_report(y_test, final_predictions))

# 绘制混淆矩阵图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
