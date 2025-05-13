import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

# 读取Excel文件
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy_label.xlsx"  # 请将路径替换为你的Excel文件路径
data = pd.read_excel(file_path)

# 将特征和标签分开
X = data.iloc[:, :-1]  # 所有行，除最后一列外的所有列
y = data.iloc[:, -1]   # 所有行，最后一列

# 使用整个数据集进行训练
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 使用整个数据集进行预测
y_pred = model.predict(X)

# 打印混淆矩阵
cm = confusion_matrix(y, y_pred)
print("混淆矩阵:")
print(cm)

# 计算准确率、召回率和 F1 分数
accuracy = accuracy_score(y, y_pred)
recall = recall_score(y, y_pred, average='weighted')  # 计算加权召回率
f1 = f1_score(y, y_pred, average='weighted')  # 计算加权 F1 分数

print(f"precision: {accuracy:.6f}")
print(f"recall: {recall:.6f}")
print(f"F1 score: {f1:.6f}")

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.title('混淆矩阵')
plt.show()
