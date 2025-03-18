import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from imblearn.metrics import geometric_mean_score

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import seaborn as sns

from threshold_opt import ThresholdOptimization_F1Score, ThresholdOptimization_Gmean

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes  import GaussianNB 


class_names = ["Normal", "Mild", "Sereve"]

def SupervisedTrainingWithoutKFold(train_model, 
                       IfStandard, IfSMOTE, IfVisualize,
                       threshold_opt=False,
                       IfSave=False,X=None, y=None):

    class_names = ["Normal", "Mild", "Sereve"]

    data_path1 = r"E:\_SITP\data\Data.xlsx"
    label_path1 = r"E:\_SITP\data\DataLabel.xlsx"

    data_path2 = r"E:\_SITP\data\all_data\test_files\TestData.xlsx"
    label_path2 = r"E:\_SITP\data\all_data\test_files\TestLabel.xlsx"

    data1 = pd.read_excel(data_path1)
    label1 = pd.read_excel(label_path1)
    data2 = pd.read_excel(data_path2)
    label2 = pd.read_excel(label_path2)

    X1 = data1.iloc[ : ,  : ].values
    y1 = label1.iloc[ : ].values.squeeze()

    X2 = data2.iloc[ : ,  : ].values
    y2 = label2.iloc[ : ].values.squeeze()

    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

    if IfSMOTE:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if IfStandard:
        scaler = StandardScaler()   
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X2 = scaler.transform(X2)

    train_model.fit(X_train, y_train)

    test_res = train_model.predict_proba(X_train)
    test_res1 = train_model.predict(X_train)

    test_res2 = train_model.predict(X_test)
    test_res3 = train_model.predict_proba(X_test)

    if IfVisualize:
        final_report = classification_report(y_test, test_res2, target_names=class_names, output_dict=True)
        print(final_report)
        cm = confusion_matrix(y_test, test_res2)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                xticklabels=class_names, 
                yticklabels=class_names)
        plt.title("Result")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

       
        y_true = y_test
        y_prob = test_res3
        print(final_report)
        y_true_bin = np.where(y_true  == 0, 1, 0)  # 正类为0，负类为1/2
        y_score = y_prob[:, 0]    # 取0类的预测概率作为正类得分

        # 计算PR曲线数据
        precision, recall, pr_thresholds = precision_recall_curve(y_true_bin, y_score)
        pr_auc = auc(recall, precision)

        # 计算ROC曲线数据
        fpr, tpr, roc_thresholds = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)

        # 创建画布
        plt.figure(figsize=(12,  5))

        # 绘制PR曲线
        plt.subplot(1,  2, 1)
        plt.plot(recall,  precision, color='darkorange', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall') 
        plt.ylabel('Precision') 
        plt.title('Precision-Recall  Curve (Class 0 vs Rest)')
        plt.legend(loc='lower left')
        plt.grid(True) 

        # 绘制ROC曲线
        plt.subplot(1,  2, 2)
        plt.plot(fpr,  tpr, color='navy', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,  1], [0, 1], 'k--', lw=1)
        plt.xlabel('False  Positive Rate')
        plt.ylabel('True  Positive Rate')
        plt.title('ROC  Curve (Class 0 vs Rest)')
        plt.legend(loc='lower right')
        plt.grid(True) 

        plt.tight_layout() 
        plt.show() 
        

    return train_model, scaler





def SupervisedTraining(X, y, train_model, 
                       IfStandard, IfSMOTE, IfVisualize,
                       threshold_opt=False,
                       IfSave=False):

    class_name = ["Normal", "Mild", "Sereve"]
    final_preds = np.zeros_like(y)
    final_preds_proba = np.zeros_like(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for flod_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if IfSMOTE:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        if IfStandard:
            scaler = StandardScaler()   
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        train_model.fit(X_train, y_train)

        y_pred = train_model.predict(X_test)
        final_preds[test_idx] = y_pred

        if IfVisualize:
            print(f"\nFlod {flod_idx + 1} Classification Report:")
            print(classification_report(y_test, final_preds[test_idx], target_names=class_name))

    if IfVisualize:
        # confusion matrix
        cm = confusion_matrix(y, final_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                xticklabels=class_name, 
                yticklabels=class_name)
        plt.title("Result")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

        '''
        # ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(class_name)
        y_onehot = pd.get_dummies(y).values
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], final_preds == i)
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {class_name[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        '''

    print("\nOverall Report:")
    overall_report = classification_report(y, final_preds, target_names=class_name, output_dict=True)
    gmean_test = geometric_mean_score(y, final_preds)
    print(classification_report(y, final_preds, target_names=class_name))
        
    macro_f1 = overall_report['macro avg']['f1-score']
    weighted_f1 = overall_report['weighted avg']['f1-score']
    print(f"\nmacro f1-score F1-score: {macro_f1:.4f}")
    print(f"weighted F1-score: {weighted_f1:.4f}")
    print(f"Gmean: {gmean_test: .4f}")

    return final_preds, train_model



'''
    先分出 Normal and Abnormal
    再分出 Abnormal 中的 Mild and Sereve
    转化为 二分类 问题
'''
def SupervisedTraining_ByStep_Order1(X, y, train_model1, train_model2, 
                       IfStandard, IfSMOTE1, IfSMOTE2, IfVisualize,
                       threshold_opt=False,
                       IfSave=False):
    class_name1 = ["Normal", "Abnormal"]
    class_name = ["Normal", "Mild", "Sereve"]
    final_preds = np.zeros_like(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for flod_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if IfStandard:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)      

        '''
            First Layer : Normal vs Abnormal
        '''
        y_train_layer1 = np.where(y_train == 0, 0, 1)
        
        if IfSMOTE1:
            smote = SMOTE(random_state=42)
            X_train_layer1, y_train_layer1 = smote.fit_resample(X_train, y_train_layer1)

        if not threshold_opt:
            train_model1.fit(X_train_layer1, y_train_layer1)                    
            y_pred_layer1 = train_model1.predict(X_test)     
        else:
            train_model1.fit(X_train_layer1, y_train_layer1) 
            y_prob_layer1 = train_model1.predict_proba(X_test)[:, 1]  

            best_threshold1 = ThresholdOptimization_F1Score(y_test, y_prob_layer1, {0:1, 1:1})  
            y_pred_layer1 = (y_prob_layer1 >= best_threshold1).astype(int)  

        if IfVisualize:
            y_test_transform = np.where(y_test == 0, 0, 1)
            print(f"\nFlod {flod_idx + 1} First Classification Report:")
            print(classification_report(y_test_transform, y_pred_layer1, target_names=class_name1))

        '''
            Second Layer : Mild vs Sereve
        '''
        abnormal_mask = (y_pred_layer1 == 1)
        X_test_abnormal = X_test[abnormal_mask]
        y_test_abnormal = y_test[abnormal_mask]

        if len(X_test_abnormal) > 0:
            abnormal_mask_train = (y_train != 0)
            X_train_abnormal = X_train[abnormal_mask_train]
            y_train_abnormal = y_train[abnormal_mask_train]
            y_train_layer2 = np.where(y_train_abnormal == 1, 0, 1)
            y_train_layer2_copy = y_train_layer2

            if IfSMOTE2:
                X_train_abnormal, y_train_layer2 = smote.fit_resample(X_train_abnormal, y_train_layer2_copy)
        
            if not threshold_opt:
                train_model2.fit(X_train_abnormal, y_train_layer2)  
                y_pred_abnormal = train_model2.predict(X_test_abnormal)
            else:
                train_model1.fit(X_train_layer1, y_train_layer1) 
                y_prob_layer1 = train_model1.predict_proba(X_test)[:, 1]  

                best_threshold1 = ThresholdOptimization_F1Score(y_test, y_prob_layer1, {0:1, 1:1})  
                y_pred_layer1 = (y_prob_layer1 >= best_threshold1).astype(int)

            y_pred_abnormal_original = np.where(y_pred_abnormal == 0, 1, 2)

            final_preds[test_idx[abnormal_mask]] = y_pred_abnormal_original

        final_preds[test_idx[~abnormal_mask]] = 0

        if IfVisualize:
            print(f"\nFlod {flod_idx + 1} Classification Report:")
            print(classification_report(y_test, final_preds[test_idx], target_names=class_name))

    if IfVisualize:
        # confusion matrix
        cm = confusion_matrix(y, final_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                xticklabels=class_name, 
                yticklabels=class_name)
        plt.title("Result")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

    print("\nOverall Report:")
    overall_report = classification_report(y, final_preds, target_names=class_name, output_dict=True)
    gmean_test = geometric_mean_score(y, final_preds)
    print(classification_report(y, final_preds, target_names=class_name))
        
    macro_f1 = overall_report['macro avg']['f1-score']
    weighted_f1 = overall_report['weighted avg']['f1-score']
    print(f"\nmacro f1-score F1-score: {macro_f1:.4f}")
    print(f"weighted F1-score: {weighted_f1:.4f}")
    print(f"Gmean: {gmean_test: .4f}")


if __name__ == "__main__":
    SupervisedTrainingWithoutKFold(train_model=RandomForestClassifier(n_estimators=300, random_state=123, max_depth=10, min_samples_split=2, class_weight='balanced'),
                    IfStandard=True, IfSMOTE=True, IfVisualize=True)