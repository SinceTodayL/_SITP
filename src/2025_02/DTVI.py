# 环境设置
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, learning_curve, validation_curve
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.inspection import permutation_importance, DecisionBoundaryDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import time

# 创建可视化目录
os.makedirs('advanced_visualizations', exist_ok=True)
plt.rcParams['font.size'] = 12  # 统一字体大小

# 数据加载
data = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\Data.xlsx", header=0)
labels = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx", header=0)
X = data.values
y = labels.values.flatten()
class_names = ["Normal", "Mild", "Severe"]

# ========================
# 通用可视化函数
# ========================
def plot_pca_comparison(X, y, title):
    """对比原始数据与SMOTE处理后的PCA分布"""
    pca = PCA(n_components=2)
    
    plt.figure(figsize=(12, 6))
    
    # 原始数据
    plt.subplot(1, 2, 1)
    X_pca = pca.fit_transform(X)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.8)
    plt.title(f"{title} (Original)")
    
    # SMOTE处理后
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_pca_res = pca.fit_transform(X_res)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_pca_res[:, 0], y=X_pca_res[:, 1], hue=y_res, palette='viridis', alpha=0.8)
    plt.title(f"{title} (After SMOTE)")
    
    plt.tight_layout()
    plt.savefig(f'advanced_visualizations/PCA_Comparison_{title}.png', dpi=300)
    plt.show()

def plot_decision_boundary(model, X, y, title):
    """绘制决策边界（使用PCA降维到2D）"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    model.fit(X_pca, y)
    
    disp = DecisionBoundaryDisplay.from_estimator(
        model, X_pca, response_method="predict",
        xlabel='PCA Component 1', ylabel='PCA Component 2',
        alpha=0.5, grid_resolution=50,
        cmap=plt.cm.coolwarm,
    )
    
    scatter = disp.ax_.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    disp.ax_.set_title(title)
    legend = disp.ax_.legend(*scatter.legend_elements(), title="Classes")
    disp.ax_.add_artist(legend)
    plt.savefig(f'advanced_visualizations/Decision_Boundary_{title}.png', dpi=300)
    plt.show()

# ========================
# SVM 专项可视化
# ========================
def svm_analysis(X, y):
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建带SMOTE和校准的Pipeline
    model = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    
    # 交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_probs = np.zeros((len(y), 3))
    final_pred = np.zeros_like(y)
    
    # 存储每个fold的时间
    fit_times = []
    
    print("\n" + "="*40)
    print("SVM 模型分析中...")
    print("="*40)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_times.append(time.time() - start_time)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        final_pred[test_idx] = y_pred
        y_probs[test_idx] = y_prob
    
    # 输出平均训练时间
    print(f"平均训练时间: {np.mean(fit_times):.2f} ± {np.std(fit_times):.2f} 秒")
    
    # 核心可视化
    # 1. 学习曲线
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Cross-Val Score")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.title("SVM Learning Curve")
    plt.legend()
    plt.savefig('advanced_visualizations/SVM_Learning_Curve.png', dpi=300)
    plt.show()
    
    # 2. 验证曲线（调整C参数）
    param_range = np.logspace(-2, 3, 6)
    train_scores, test_scores = validation_curve(
        model.named_steps['svm'], X_scaled, y, 
        param_name="svm__C", param_range=param_range,
        cv=3, scoring="accuracy", n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(param_range, np.mean(train_scores, axis=1), 'o-', label="Training Score")
    plt.semilogx(param_range, np.mean(test_scores, axis=1), 'o-', label="Cross-Val Score")
    plt.xlabel("C (Regularization Parameter)")
    plt.ylabel("Accuracy")
    plt.title("SVM Validation Curve (C Parameter)")
    plt.legend()
    plt.savefig('advanced_visualizations/SVM_Validation_Curve_C.png', dpi=300)
    plt.show()
    
    # 3. 特征重要性（排列重要性）
    result = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=data.columns[sorted_idx])
    plt.title("SVM Permutation Importance")
    plt.tight_layout()
    plt.savefig('advanced_visualizations/SVM_Feature_Importance.png', dpi=300)
    plt.show()
    
    # 4. 决策边界可视化
    plot_decision_boundary(model.named_steps['svm'], X_scaled, y, "SVM Decision Boundary")
    
    # 5. ROC曲线
    y_test_bin = label_binarize(y, classes=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                 label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('advanced_visualizations/SVM_ROC_Curves.png', dpi=300)
    plt.show()
    
    # 6. 混淆矩阵
    cm = confusion_matrix(y, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("SVM Confusion Matrix")
    plt.savefig('advanced_visualizations/SVM_Confusion_Matrix.png', dpi=300)
    plt.show()

# ========================
# 决策树专项可视化
# ========================
def decision_tree_analysis(X, y):
    # 创建带SMOTE的模型
    model = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('tree', DecisionTreeClassifier(max_depth=3, random_state=42))
    ])
    
    print("\n" + "="*40)
    print("决策树模型分析中...")
    print("="*40)
    
    # 交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_probs = np.zeros((len(y), 3))
    final_pred = np.zeros_like(y)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        final_pred[test_idx] = y_pred
        y_probs[test_idx] = y_prob
    
    # 核心可视化
    # 1. 树结构可视化
    plt.figure(figsize=(20, 10))
    plot_tree(model.named_steps['tree'], 
              feature_names=data.columns, 
              class_names=class_names,
              filled=True, rounded=True,
              impurity=False, proportion=True)
    plt.title("Decision Tree Structure")
    plt.savefig('advanced_visualizations/Decision_Tree_Structure.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 特征重要性
    importances = model.named_steps['tree'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), data.columns[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('advanced_visualizations/Decision_Tree_Feature_Importance.png', dpi=300)
    plt.show()
    
    # 3. 学习曲线
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Cross-Val Score")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Learning Curve")
    plt.legend()
    plt.savefig('advanced_visualizations/Decision_Tree_Learning_Curve.png', dpi=300)
    plt.show()
    
    # 4. 验证曲线（调整max_depth）
    param_range = np.arange(1, 11)
    train_scores, test_scores = validation_curve(
        model.named_steps['tree'], X, y, 
        param_name="max_depth", param_range=param_range,
        cv=3, scoring="accuracy", n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', label="Training Score")
    plt.plot(param_range, np.mean(test_scores, axis=1), 'o-', label="Cross-Val Score")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Validation Curve (Max Depth)")
    plt.legend()
    plt.savefig('advanced_visualizations/Decision_Tree_Validation_Curve.png', dpi=300)
    plt.show()
    
    # 5. PR曲线
    y_test_bin = label_binarize(y, classes=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    for i in range(3):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_probs[:, i])
        avg_precision = average_precision_score(y_test_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, lw=2,
                 label=f'Class {class_names[i]} (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Decision Tree Precision-Recall Curves')
    plt.legend(loc="upper right")
    plt.savefig('advanced_visualizations/Decision_Tree_PR_Curves.png', dpi=300)
    plt.show()

# ========================
# 执行分析
# ========================
if __name__ == "__main__":
    # 数据分布对比
    plot_pca_comparison(X, y, "Data Distribution")
    
    # SVM分析
    svm_analysis(X, y)
    
    # 决策树分析
    decision_tree_analysis(X, y)