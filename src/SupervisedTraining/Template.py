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


    print("\nOverall Report:")
    overall_report = classification_report(y, final_preds, target_names=class_name, output_dict=True)
    gmean_test = geometric_mean_score(y, final_preds)
    print(classification_report(y, final_preds, target_names=class_name))
        
    macro_f1 = overall_report['macro avg']['f1-score']
    weighted_f1 = overall_report['weighted avg']['f1-score']
    print(f"\nmacro f1-score F1-score: {macro_f1:.4f}")
    print(f"weighted F1-score: {weighted_f1:.4f}")
    print(f"Gmean: {gmean_test: .4f}")
