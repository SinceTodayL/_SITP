import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, accuracy_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
)
import numpy as np

# 1. 读取数据
data = pd.read_excel("C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy_label.xlsx")

# 2. 分离特征和标签
X = data.iloc[:, :-1].values  # 前 28 列为特征
y = data.iloc[:, -1].values   # 最后一列为标签

# 3. 使用 SMOTE 对数据进行过采样
# 定义 SMOTE 对象（生成少数类样本）
smote = SMOTE(random_state=61)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. 转换为 DataFrame，按标签分组保存到新的 Excel
resampled_data = pd.DataFrame(X_resampled, columns=data.columns[:-1])
resampled_data['Label'] = y_resampled
resampled_data.sort_values(by='Label', inplace=True)  # 按标签排序
resampled_data.to_excel("C:\\Users\\LiuZh\\Desktop\\SITP\\after_SMOTE_dataset3.xlsx", index=False)
print("SMOTE 处理后的数据已保存")

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=61, stratify=y_resampled
)

# 6. 训练带权随机森林分类器
class_weights = {0: 1, 1: 1, 2: 4}
rf_model = RandomForestClassifier(random_state=61, class_weight=class_weights)
rf_model.fit(X_train, y_train)

# 7. 阈值优化函数
def optimize_threshold(model, X, y_true):
    """
    优化分类阈值，以获得最佳 F1 分数。
    """
    y_prob = model.predict_proba(X)[:, 1]  # 获取正类的预测概率
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob, pos_label=1)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"最佳阈值: {best_threshold:.4f}, 对应 F1 分数: {np.max(f1_scores):.4f}")
    return best_threshold

# 优化分类阈值
best_threshold = optimize_threshold(rf_model, X_test, y_test)

# 使用优化后的阈值重新评估模型
y_prob = rf_model.predict_proba(X_test)[:, 1]
y_pred_optimized = (y_prob >= best_threshold).astype(int)

# 8. 模型评估
print("分类报告（优化阈值后）：")
print(classification_report(y_test, y_pred_optimized))

# 计算准确率、召回率和 F1 分数
accuracy = accuracy_score(y_test, y_pred_optimized)
recall = recall_score(y_test, y_pred_optimized, average='weighted')  # 计算加权召回率
f1 = f1_score(y_test, y_pred_optimized, average='weighted')  # 计算加权 F1 分数

print(f"accuracy: {accuracy:.6f}")
print(f"recall: {recall:.6f}")
print(f"F1 score: {f1:.6f}")

# 打印混淆矩阵
cm = confusion_matrix(y_test, y_pred_optimized)
print("confusion matrix:")
print(cm)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
plt.ylabel('true')
plt.xlabel('predicted')
plt.title('confusion matrix')
plt.show()
