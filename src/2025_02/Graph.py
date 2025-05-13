import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------
# 数据加载
# ----------------------------------
data = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\Data.xlsx", header=0)
labels = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx", header=0)
X = data.values
y = labels.values.flatten()

# ----------------------------------
# 分层分类函数（正常->异常->严重）
# ----------------------------------
def hierarchical_classification_v1(model_layer1, model_layer2):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 第一层：正常(0) vs 异常(1,2)
        y_train_layer1 = np.where(y_train == 0, 0, 1)
        model_layer1.fit(X_train, y_train_layer1)
        y_pred_layer1 = model_layer1.predict(X_test)
        
        # 第二层：普通异常(1) vs 明显异常(2)
        mask_abnormal = (y_pred_layer1 == 1)
        X_test_abnormal = X_test[mask_abnormal]
        y_test_abnormal = y_test[mask_abnormal]
        
        if len(X_test_abnormal) > 0:
            y_train_abnormal = y_train[y_train != 0]
            X_train_abnormal = X_train[y_train != 0]
            y_train_layer2 = np.where(y_train_abnormal == 1, 0, 1)
            model_layer2.fit(X_train_abnormal, y_train_layer2)
            y_pred_abnormal = model_layer2.predict(X_test_abnormal)
        else:
            y_pred_abnormal = np.array([])
        
        # 合并预测结果
        y_pred = np.zeros_like(y_test)
        y_pred[mask_abnormal] = np.where(y_pred_abnormal == 0, 1, 2)
        y_pred[~mask_abnormal] = 0
        
        # 收集结果
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    # 混淆矩阵
    '''
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Mild", "Severe"],
                yticklabels=["Normal", "Mild", "Severe"])
    plt.title("Random Forest Confusion Matrix (Approach 1)")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    '''
    # 分类报告
    report = classification_report(y_true_all, y_pred_all, target_names=["Normal", "Mild", "Severe"], output_dict=True)
    return report

# ----------------------------------
# 分层分类函数（严重异常->正常/普通）
# ----------------------------------
def hierarchical_classification_v2(model_layer1, model_layer2):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 第一层：明显异常(2) vs 其他
        y_train_layer1 = np.where(y_train == 2, 1, 0)
        model_layer1.fit(X_train, y_train_layer1)
        y_pred_layer1 = model_layer1.predict(X_test)
        
        # 第二层：正常(0) vs 普通异常(1)
        mask_non_severe = (y_pred_layer1 == 0)
        X_test_non_severe = X_test[mask_non_severe]
        y_test_non_severe = y_test[mask_non_severe]
        
        if len(X_test_non_severe) > 0:
            y_train_non_severe = y_train[y_train != 2]
            X_train_non_severe = X_train[y_train != 2]
            y_train_layer2 = np.where(y_train_non_severe == 0, 0, 1)
            model_layer2.fit(X_train_non_severe, y_train_layer2)
            y_pred_non_severe = model_layer2.predict(X_test_non_severe)
        else:
            y_pred_non_severe = np.array([])
        
        # 合并预测结果
        y_pred = np.zeros_like(y_test)
        y_pred[mask_non_severe] = np.where(y_pred_non_severe == 0, 0, 1)
        y_pred[~mask_non_severe] = 2
        
        # 收集结果
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    # 混淆矩阵
    '''
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=["Normal", "Mild", "Severe"],
                yticklabels=["Normal", "Mild", "Severe"])
    plt.title("Random Forest Confusion Matrix (Approach 2)")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    '''
    # 分类报告
    report = classification_report(y_true_all, y_pred_all, target_names=["Normal", "Mild", "Severe"], output_dict=True)
    return report

# ----------------------------------
# 参数调优可视化（n_estimators 和 max_depth）
# ----------------------------------
def parameter_tuning_visualization():
    # 参数范围
    n_estimators_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100]
    max_depth_list = [15, 20, 25, 30]
    
    # 存储结果
    metrics = {'n_estimators': [], 'max_depth': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    
    # 遍历参数组合
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            # 初始化模型
            model1 = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model2 = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            
            # 运行分层分类方法1
            report = hierarchical_classification_v1(model1, model2)
            
            # 记录指标（以"Severe"类为例）
            metrics['n_estimators'].append(n_estimators)
            metrics['max_depth'].append(max_depth)
            metrics['Precision'].append(report['Severe']['precision'])
            metrics['Recall'].append(report['Severe']['recall'])
            metrics['F1-Score'].append(report['Severe']['f1-score'])
    
    # 转换为DataFrame
    df_metrics = pd.DataFrame(metrics)

    # 可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.lineplot(data=df_metrics, x='n_estimators', y='Precision', hue='max_depth', marker='o')
    plt.title('Precision vs n_estimators')
    
    plt.subplot(1, 3, 2)
    sns.lineplot(data=df_metrics, x='n_estimators', y='Recall', hue='max_depth', marker='o')
    plt.title('Recall vs n_estimators')
    
    plt.subplot(1, 3, 3)
    sns.lineplot(data=df_metrics, x='n_estimators', y='F1-Score', hue='max_depth', marker='o')
    plt.title('F1-Score vs n_estimators')
    
    plt.tight_layout()
    plt.show()

# ----------------------------------
# 主程序
# ----------------------------------
if __name__ == "__main__":
    # 执行分层分类方法1
    model1 = RandomForestClassifier(n_estimators=100, max_depth=10)
    model2 = RandomForestClassifier(n_estimators=50, max_depth=5)
    report_v1 = hierarchical_classification_v1(model1, model2)
    
    # 执行分层分类方法2
    model1 = RandomForestClassifier(n_estimators=100, max_depth=10)
    model2 = RandomForestClassifier(n_estimators=50, max_depth=5)
    report_v2 = hierarchical_classification_v2(model1, model2)
    
    # 参数调优可视化
    parameter_tuning_visualization()