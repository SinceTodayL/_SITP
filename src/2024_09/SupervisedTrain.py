import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据集
iris = load_iris()
X = iris.data  # 特征（输入）
y = iris.target  # 标签（输出）

# 2. 划分数据集，70%用于训练，30%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建决策树模型
model = DecisionTreeClassifier()

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测测试集
y_pred = model.predict(X_test)

# 6. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 7. 预测新的数据
new_data = [[5.0, 3.5, 1.5, 0.2]]  # 假设新的样本
predicted_class = model.predict(new_data)
print(f"预测的新样本类别: {predicted_class[0]}")
