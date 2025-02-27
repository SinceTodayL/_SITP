from template import SupervisedTraining, SupervisedTraining_ByStep_Order1

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from imblearn.metrics import geometric_mean_score

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


if __name__ == "__main__":
    data_path = r"E:\_SITP\data\Data.xlsx"
    label_path = r"E:\_SITP\data\DataLabel.xlsx"

    data = pd.read_excel(data_path)
    label = pd.read_excel(label_path)

    X = data.iloc[ : ,  : ].values
    y = label.iloc[ : ].values.squeeze()

    class_name = ["Normal", "Mild", "Sereve"]

    train_models = [MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=2000, random_state=42),
                RandomForestClassifier(n_estimators=300, random_state=42),
                KNeighborsClassifier(n_neighbors=15),
                SVC(kernel='rbf'),
                DecisionTreeClassifier(random_state=42, max_depth=10)]
 
    # all_predictions = np.zeros((len(X),  len(train_models)), dtype=np.int64) 
    final_result = np.zeros_like(y)
    
    for i, model in enumerate(train_models):
        pred = SupervisedTraining(X=X, y=y, train_model=model,
                                IfStandard=True, IfSMOTE=True, 
                                IfVisualize=False, threshold_opt=True)
        # all_predictions[:, i] = pred.astype(np.int64)   
        # pred = np.where(pred  == 2, 5, pred)  
        # pred = np.where(pred  == 1, 2, pred) 
        final_result += pred
    
    value, counts = np.unique(final_result, return_counts=True)
    print(dict(zip(value, counts)))

    final_result = np.where(final_result  <= 2, 0, 
                        np.where(final_result  <= 8, 1, 2))
    
    # final_result = stats.mode(all_predictions,  axis=1)[0].flatten()


    # Visualize
    print("\nOverall Report:")
    overall_report = classification_report(y, final_result, target_names=class_name, output_dict=True)
    gmean_test = geometric_mean_score(y, final_result)
    print(classification_report(y, final_result, target_names=class_name))
        
    macro_f1 = overall_report['macro avg']['f1-score']
    weighted_f1 = overall_report['weighted avg']['f1-score']
    print(f"\nmacro f1-score F1-score: {macro_f1:.4f}")
    print(f"weighted F1-score: {weighted_f1:.4f}")
    print(f"Gmean: {gmean_test: .4f}")

    # confusion matrix
    cm = confusion_matrix(y, final_result)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                xticklabels=class_name, 
                yticklabels=class_name)
    plt.title("Result")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

