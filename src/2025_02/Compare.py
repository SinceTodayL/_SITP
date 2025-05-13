import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

# ----------------------------------
# 1. 数据加载与预处理
# ----------------------------------
data = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\Data.xlsx", header=0)
labels = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx", header=0)
X = data.values
y = labels.values.flatten()

# 存储评估指标
results = []

# ----------------------------------
# 2. 分层分类函数（先分严重异常点，再分普通异常点）
# ----------------------------------
def hierarchical_classification(model_layer1, model_layer2, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_pred = np.zeros_like(y)
    y_true_all, y_pred_all = [], []
    y_true_layer1_all, y_pred_layer1_all = [], []
    y_true_layer2_all, y_pred_layer2_all = [], []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 第一层模型：严重异常 vs 其他
        y_train_layer1 = np.where(y_train == 2, 1, 0)  # 严重异常=1，其他=0
        model_layer1.fit(X_train, y_train_layer1)
        y_pred_layer1 = model_layer1.predict(X_test)
        y_pred_layer1_proba = model_layer1.predict_proba(X_test)[:, 1]  # 获取概率值
        
        # 第二层模型：正常 vs 普通异常
        mask_not_severe = (y_pred_layer1 == 0)  # 非严重异常样本
        X_test_not_severe = X_test[mask_not_severe]
        if len(X_test_not_severe) > 0:
            y_train_not_severe = y_train[y_train != 2]
            X_train_not_severe = X_train[y_train != 2]
            y_train_not_severe_remapped = np.where(y_train_not_severe == 0, 0, 1)  # 正常=0，普通异常=1
            model_layer2.fit(X_train_not_severe, y_train_not_severe_remapped)
            y_pred_not_severe = model_layer2.predict(X_test_not_severe)
            y_pred_not_severe_original = np.where(y_pred_not_severe == 0, 0, 1)  # 正常=0，普通异常=1
        else:
            y_pred_not_severe_original = np.array([])

        # 合并预测结果
        final_pred[test_index] = np.where(y_pred_layer1 == 1, 2, y_pred_not_severe_original)  # 严重异常=2，其他根据第二层模型
        y_true_all.extend(y_test)
        y_pred_all.extend(final_pred[test_index])
        y_true_layer1_all.extend(y_test)
        y_pred_layer1_all.extend(y_pred_layer1)
        y_true_layer2_all.extend(y_test[mask_not_severe])
        y_pred_layer2_all.extend(y_pred_not_severe_original)
    
    # 输出二分类任务（第一层：严重异常 vs 其他）的评估指标
    y_true_layer1_binary = np.where(np.array(y_true_layer1_all) == 2, 1, 0)  # 严重异常=1，其他=0
    y_pred_layer1_binary = np.where(np.array(y_pred_layer1_all) == 1, 1, 0)  # 严重异常=1，其他=0
    report_layer1 = classification_report(y_true_layer1_binary, y_pred_layer1_binary, target_names=["Other", "Severe"], output_dict=True)
    results.append({
        "Algorithm": model_name,
        "Task": "Layer 1 (Severe vs Other)",
        "Precision": report_layer1['weighted avg']["precision"],
        "Recall": report_layer1['weighted avg']["recall"],
        "F1-Score": report_layer1['weighted avg']["f1-score"]
    })
    
    # 输出二分类任务（第二层：正常 vs 普通异常）的评估指标
    if len(y_true_layer2_all) > 0:
        y_true_layer2_binary = np.where(np.array(y_true_layer2_all) == 0, 0, 1)  # 正常=0，普通异常=1
        y_pred_layer2_binary = np.where(np.array(y_pred_layer2_all) == 0, 0, 1)  # 正常=0，普通异常=1
        report_layer2 = classification_report(y_true_layer2_binary, y_pred_layer2_binary, target_names=["Normal", "Mild"], output_dict=True)
        results.append({
            "Algorithm": model_name,
            "Task": "Layer 2 (Normal vs Mild)",
            "Precision": report_layer2['weighted avg']["precision"],
            "Recall": report_layer2['weighted avg']["recall"],
            "F1-Score": report_layer2['weighted avg']["f1-score"]
        })
    
    # 输出三分类任务的评估指标（加权平均）
    report_final = classification_report(y_true_all, y_pred_all, target_names=["Normal", "Mild", "Severe"], output_dict=True)
    results.append({
        "Algorithm": model_name,
        "Task": "Final (Weighted Average)",
        "Precision": report_final['weighted avg']["precision"],
        "Recall": report_final['weighted avg']["recall"],
        "F1-Score": report_final['weighted avg']["f1-score"]
    })
    
    # 混淆矩阵可视化
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Mild", "Severe"],
                yticklabels=["Normal", "Mild", "Severe"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    # ROC 曲线可视化（使用概率值）
    plt.figure(figsize=(8, 6))
    y_true_bin = label_binarize(y_true_all, classes=[0, 1, 2])
    y_proba_all = np.zeros((len(y_true_all), 3))  # 初始化概率矩阵
    
    # 获取每个类别的概率值
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 第一层模型：严重异常 vs 其他
        y_train_layer1 = np.where(y_train == 2, 1, 0)
        model_layer1.fit(X_train, y_train_layer1)
        y_pred_layer1_proba = model_layer1.predict_proba(X_test)[:, 1]
        
        # 第二层模型：正常 vs 普通异常
        mask_not_severe = (y_pred_layer1_proba < 0.5)  # 非严重异常样本
        X_test_not_severe = X_test[mask_not_severe]
        if len(X_test_not_severe) > 0:
            y_train_not_severe = y_train[y_train != 2]
            X_train_not_severe = X_train[y_train != 2]
            y_train_not_severe_remapped = np.where(y_train_not_severe == 0, 0, 1)
            model_layer2.fit(X_train_not_severe, y_train_not_severe_remapped)
            y_pred_not_severe_proba = model_layer2.predict_proba(X_test_not_severe)
            y_proba_all[test_index[mask_not_severe], 0] = y_pred_not_severe_proba[:, 0]  # Normal
            y_proba_all[test_index[mask_not_severe], 1] = y_pred_not_severe_proba[:, 1]  # Mild
        y_proba_all[test_index, 2] = y_pred_layer1_proba  # Severe
    
    # 绘制每个类别的 ROC 曲线
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba_all[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

# ----------------------------------
# 3. 运行所有模型
# ----------------------------------
models = {
    "KNN": (KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=3)),
    "Random Forest": (RandomForestClassifier(n_estimators=100), RandomForestClassifier(n_estimators=50)),
    "MLP": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000))
}

for name, (model1, model2) in models.items():
    hierarchical_classification(model1, model2, name)

# ----------------------------------
# 4. 输出汇总表格
# ----------------------------------
results_df = pd.DataFrame(results)
print("\n========== Final Metrics ==========")
print(results_df.to_markdown(index=False))