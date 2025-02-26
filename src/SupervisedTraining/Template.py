import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.metrics import geometric_mean_score

import matplotlib.pyplot as plt
import seaborn as sns



def SupervisedTraining(X, y, train_model, 
                       IfStandard, IfSMOTE, IfVisualize,
                       threshold_opt=False,
                       IfSave=False):

    class_name = ["Normal", "Mild", "Sereve"]
    final_preds = np.zeros_like(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for flod_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if IfSMOTE:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        if IfStandard:
            scaler = StandardScaler()   
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        train_model.fit(X_train, y_train)

        if not threshold_opt:
            y_pred = train_model.predict(X_test)
            final_preds[test_idx] = y_pred
    #   else:
    #       TODO

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


'''
    先分出 Normal and Abnormal
    再分出 Abnormal 中的 Mild and Sereve
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
    #   else:
    #       TODO

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
        #   else:
        #       TODO

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
