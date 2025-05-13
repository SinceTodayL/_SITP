import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_excel("C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy_label.xlsx")

# 2. 分离特征和标签
X = data.iloc[:, :-1].values  # 特征：前28列
y = data.iloc[:, -1].values   # 标签：最后一列

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# 5. 使用 SVM 进行训练，设定类别权重
svm = SVC(kernel='rbf', class_weight={0: 2, 1: 3, 2: 100}, random_state=42)
svm.fit(X_train, y_train)

# 6. 预测测试集结果
y_pred = svm.predict(X_test)

# 7. 输出分类报告
print("分类报告：")
print(classification_report(y_test, y_pred))

# 8. 计算并可视化混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

