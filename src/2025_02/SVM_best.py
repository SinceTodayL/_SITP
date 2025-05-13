from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# 创建保存图片的目录
os.makedirs('visualizations', exist_ok=True)

# 数据加载与预处理
data = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\Data.xlsx", header=0)
labels = pd.read_excel(r"C:\Users\LiuZh\Desktop\SITP\DataLabel.xlsx", header=0)
X = data.values
y = labels.values.flatten()

# 可视化函数
def plot_feature_distribution(X, y, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.8)
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_score, n_classes, model_name):
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 6))
    for i, color in zip(range(n_classes), ['blue', 'green', 'red']):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_f1_vs_params(param_name, param_values, f1_scores, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, f1_scores, marker='o', color='b', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel(param_name)
    plt.ylabel('F1-Score')
    plt.title(f'{model_name} F1-Score vs {param_name}')
    plt.grid(True)
    plt.savefig(f'visualizations/{model_name}_f1_vs_{param_name}.png')
    plt.show()

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv=None, n_jobs=None):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

def svm_classification_with_tuning(model, model_name):
    # 使用5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 设置参数范围进行调参
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['linear', 'rbf']
    }

    # 使用GridSearchCV进行调参
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='f1_macro', n_jobs=-1)
    
    # 记录每次调参结果
    param_results = []
    
    # 训练并调参
    grid_search.fit(X, y)
    
    # 获取最佳参数
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")
    print(f"Best F1-Score: {grid_search.best_score_}")

    # 输出每次调参的结果
    print("\nAll Parameter Combinations and their F1-Scores:")
    for params, mean_score, _ in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
        param_results.append(params)
        print(f"Params: {params}, F1-Score: {mean_score:.4f}, Std: {np.std(mean_score):.4f}")
    
    # 保存每次调参的结果到CSV
    param_results_df = pd.DataFrame(grid_search.cv_results_)
    param_results_df.to_csv(f'visualizations/{model_name}_param_results.csv', index=False)
    
    # 获取调参过程中各个参数的F1-Score变化
    param_name = 'C'
    f1_scores = []
    param_values = param_grid[param_name]
    for value in param_values:
        model.set_params(**{param_name: value})
        model.fit(X, y)
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred, average='macro')
        f1_scores.append(f1)
    
    # 可视化F1-Score与C参数的关系
    plot_f1_vs_params(param_name, param_values, f1_scores, model_name)

    # 绘制学习曲线
    plot_learning_curve(grid_search.best_estimator_, f"{model_name} Learning Curve", X, y, cv=kf, n_jobs=-1)

    # 绘制验证曲线
    plot_validation_curve(grid_search.best_estimator_, f"{model_name} Validation Curve", X, y, param_name="C", param_range=param_grid['C'], cv=kf, n_jobs=-1)

    # 绘制最终的分类报告
    final_pred = grid_search.predict(X)
    report = classification_report(y, final_pred, target_names=["Normal", "Mild", "Severe"], output_dict=True)
    print(f"\n{model_name} Final Classification Report:")
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(y, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Mild", "Severe"], yticklabels=["Normal", "Mild", "Severe"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f'visualizations/{model_name}_confusion_matrix.png')
    plt.show()

    # ROC曲线
    y_probs = grid_search.predict_proba(X)
    plot_roc_curve(y, y_probs, n_classes=3, model_name=model_name)

# 使用SVM进行分类和调参
model_svm = SVC(probability=True)
svm_classification_with_tuning(model_svm, "SVM")