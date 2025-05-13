import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score

# 读取Excel文件
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\after_SMOTE_dataset.xlsx"  # 请将路径替换为你的Excel文件路径
data = pd.read_excel(file_path)

# 将特征和标签分开
X = data.iloc[:, :-1]  # 所有行，除最后一列外的所有列
y = data.iloc[:, -1]   # 所有行，最后一列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 超参数调整：使用网格搜索找到最佳 K 值
param_grid = {'n_neighbors': range(10, 200)}  # 搜索 1 到 20 的 K 值
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 使用最佳参数训练模型
best_k = grid_search.best_params_['n_neighbors']
print(best_k)
model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')  # 使用加权 KNN
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 打印混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)

# 计算准确率、召回率和 F1 分数
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')  # 计算加权召回率
f1 = f1_score(y_test, y_pred, average='weighted')  # 计算加权 F1 分数

print(f"accuracy: {accuracy:.6f}")
print(f"recall: {recall:.6f}")
print(f"F1 score: {f1:.6f}")

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
plt.ylabel('true')
plt.xlabel('predicted')
plt.title('confusion matrix')
plt.show()
