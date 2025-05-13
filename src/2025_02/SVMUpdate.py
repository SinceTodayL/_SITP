import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# 数据加载
data = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\Data.xlsx", header=0)
labels = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx", header=0)
X = data.values
y = labels.values.flatten()

# 初始化参数
n_splits = 5
class_names = ["Normal", "Mild", "Severe"]

# 创建 K 折交叉验证
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 存储预测结果
final_preds = np.zeros_like(y)

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n{'='*40}")
    print(f"Fold {fold_idx + 1}/{n_splits}")
    print(f"{'='*40}")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 数据标准化（仅在训练集上拟合）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ========== 第一层分类：正常 vs 异常 ==========
    print("\nStage 1: Normal vs Abnormal (Decision Tree)")
    y_train_layer1 = np.where(y_train == 0, 0, 1)  # 正常 -> 0，异常 -> 1
    
    # 使用 SMOTE 过采样
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_layer1)
    
    # 训练决策树模型
    model_layer1 = DecisionTreeClassifier(random_state=42, max_depth=5)
    model_layer1.fit(X_train_res, y_train_res)
    
    # 预测第一层
    y_pred_layer1 = model_layer1.predict(X_test_scaled)
    
    # ========== 第二层分类：Mild vs Severe ==========
    print("\nStage 2: Mild vs Severe (Random Forest)")
    abnormal_mask = (y_pred_layer1 == 1)  # 异常样本的掩码
    X_test_abnormal = X_test_scaled[abnormal_mask]
    y_test_abnormal = y_test[abnormal_mask]
    
    if len(X_test_abnormal) > 0:
        # 训练集异常样本
        abnormal_mask_train = (y_train != 0)
        X_train_abnormal = X_train_scaled[abnormal_mask_train]
        y_train_abnormal = y_train[abnormal_mask_train]
        y_train_layer2 = np.where(y_train_abnormal == 1, 0, 1)  # Mild -> 0, Severe -> 1
        
        # 使用 SMOTE 过采样
        X_train_abnormal_res, y_train_abnormal_res = smote.fit_resample(X_train_abnormal, y_train_layer2)
        
        # 训练随机森林模型
        model_layer2 = RandomForestClassifier(random_state=42, n_estimators=35, max_depth=5)
        model_layer2.fit(X_train_abnormal_res, y_train_abnormal_res)
        
        # 预测第二层
        y_pred_abnormal = model_layer2.predict(X_test_abnormal)
        y_pred_abnormal_original = np.where(y_pred_abnormal == 0, 1, 2)  # 转换回原始标签
        
        # 合并预测结果
        final_preds[test_idx[abnormal_mask]] = y_pred_abnormal_original
    
    # 处理正常样本
    final_preds[test_idx[~abnormal_mask]] = 0
    
    # 打印当前 fold 结果
    print(f"\nFold {fold_idx + 1} Classification Report:")
    print(classification_report(y_test, final_preds[test_idx], target_names=class_names))

# 最终评估
print("\n\n" + "="*40)
print("Final Evaluation")
print("="*40)

# 分类报告
report = classification_report(y, final_preds, target_names=class_names, output_dict=True)
print(f"\nClassification Report:")
print(f"Normal:   Precision: {report['Normal']['precision']:.4f}, Recall: {report['Normal']['recall']:.4f}, F1: {report['Normal']['f1-score']:.4f}")
print(f"Mild:     Precision: {report['Mild']['precision']:.4f}, Recall: {report['Mild']['recall']:.4f}, F1: {report['Mild']['f1-score']:.4f}")
print(f"Severe:   Precision: {report['Severe']['precision']:.4f}, Recall: {report['Severe']['recall']:.4f}, F1: {report['Severe']['f1-score']:.4f}")
print(f"\nWeighted Avg F1: {report['weighted avg']['f1-score']:.4f}")

# 混淆矩阵
cm = confusion_matrix(y, final_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
           xticklabels=class_names, 
           yticklabels=class_names)
plt.title("Confusion Matrix (Decision Tree + Random Forest)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()