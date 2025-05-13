import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

# 读取Excel文件
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy_label.xlsx"  # 请将路径替换为你的Excel文件路径
data = pd.read_excel(file_path)

# 将特征和标签分开
X = data.iloc[:, :-1]  # 所有行，除最后一列外的所有列
y = data.iloc[:, -1]   # 所有行，最后一列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型（使用随机森林分类器作为示例）
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 打印混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

# 计算准确率、召回率和 F1 分数
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')  # 计算加权召回率
f1 = f1_score(y_test, y_pred, average='weighted')  # 计算加权 F1 分数

print(f"precision: {accuracy:.6f}")
print(f"recall: {recall:.6f}")
print(f"F1 score: {f1:.6f}")

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
plt.ylabel('real label')
plt.xlabel('predicted label')
plt.title('confusion matrix')
plt.show()
