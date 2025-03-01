import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold
from sklearn.metrics  import confusion_matrix, classification_report
from imblearn.metrics import geometric_mean_score

from sklearn.preprocessing  import StandardScaler
from imblearn.over_sampling  import SMOTE

from sklearn.ensemble  import StackingClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.svm  import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot  as plt
import seaborn as sns



def load_and_preprocess_data():
    data = pd.read_excel(r"E:\_SITP\data\Data.xlsx") 
    label = pd.read_excel(r"E:\_SITP\data\DataLabel.xlsx").squeeze() 
    return data.values,  label

all_model = [   ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=2000, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=300, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=15)),
                ('svc', SVC(kernel='rbf')),
                ('dt', DecisionTreeClassifier(random_state=42, max_depth=10))]

meta_model = LogisticRegression(random_state=42)

g_mean_result = []
macro_f1_result = []
weighted_f1_result = []
sereve_class_f1score = []

def train(base_models, meta_model, index):
    X, y = load_and_preprocess_data()
    
    final_pred = np.zeros_like(y)
    class_name = ["Normal", "Mild", "Sereve"]

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=KFold(n_splits=3, shuffle=True, random_state=124),
        n_jobs=-1,
    )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for flod_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()   
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        stacking_model.fit(X_train,  y_train)       
        y_pred = stacking_model.predict(X_test)
        final_pred[test_idx] = y_pred

        print(f"\nFlod {flod_idx + 1} Classification Report:")
        print(classification_report(y_test, final_pred[test_idx], target_names=class_name))


    print("\nOverall Report:")
    overall_report = classification_report(y, final_pred, target_names=class_name, output_dict=True)
    gmean_test = geometric_mean_score(y, final_pred)
    print(classification_report(y, final_pred, target_names=class_name))
        
    macro_f1 = overall_report['macro avg']['f1-score']
    weighted_f1 = overall_report['weighted avg']['f1-score']
    sereve_f1score = overall_report['Sereve']['f1-score']

    print(f"\nmacro f1-score F1-score: {macro_f1:.4f}")
    print(f"weighted F1-score: {weighted_f1:.4f}")
    print(f"Gmean: {gmean_test: .4f}")
    print(f"Sereve Class F1Score: {sereve_f1score: .4f}")

    g_mean_result.append(gmean_test)
    macro_f1_result.append(macro_f1)
    weighted_f1_result.append(weighted_f1)
    sereve_class_f1score.append(sereve_f1score)

    model_names = str()
    for model in base_models:
        name = str(model[1].__class__.__name__)
        model_names = model_names + " " + name

    cm = confusion_matrix(y, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
            xticklabels=class_name, 
            yticklabels=class_name)
    plt.title(model_names + " Stack")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    save_dir = r"E:\_SITP\src\SupervisedTraining\model_stack_images"  
    os.makedirs(save_dir,  exist_ok=True)
    save_path = os.path.join(save_dir,  f"{model_names} Stack.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
   

from sklearn.ensemble  import GradientBoostingClassifier 
from sklearn.linear_model  import RidgeClassifier 
import xgboost as xgb
import lightgbm as lgb 

base_models_alt = [
    ('GBDT', GradientBoostingClassifier(n_estimators=300, random_state=42)),
    ('XGB', xgb.XGBClassifier(n_estimators=300, objective='multi:softmax', num_class=3, random_state=42)),
    ('LGBM', lgb.LGBMClassifier(n_estimators=300, objective='multiclass', num_class=3, random_state=42))
]

meta_model_alt = LogisticRegression(random_state=42)

stacking_model_alt = StackingClassifier(
    estimators=base_models_alt,
    final_estimator=meta_model_alt,
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=-1
)

import itertools

def get_subsets(lst):
    subsets = []
    for r in range(2, len(lst) + 1):
        for combo in itertools.combinations(lst,  r):
            subsets.append(list(combo)) 
    return subsets

def main():

    all_model_subset = get_subsets(all_model)
    index = 0
    
    for model_subset in all_model_subset:
        index += 1
        train(model_subset, meta_model, index)

    '''
    base_models_alt_subset = get_subsets(base_models_alt)
    for model_subset in base_models_alt_subset:
        index += 1
        train(model_subset, meta_model_alt, index)
    '''

    x = range(len(g_mean_result))  
    y1 = g_mean_result 
    y2 = macro_f1_result 
    y3 = weighted_f1_result 

    print("g_mean_result: ")
    print(g_mean_result)
    print("macro_f1_result")
    print(macro_f1_result)
    print("weighted_f1_result")
    print(weighted_f1_result)

    plt.figure(figsize=(10,  6))

    plt.plot(x, y1, color='blue', marker='o', linestyle='-', label='G-Mean')
    plt.plot(x, y2, color='red', marker='^', linestyle='-', label='Macro F1')
    plt.plot(x, y3, color='green', marker='s', linestyle='-', label='Weighted F1')

    plt.title('Performance Metrics Visualization')
    plt.xlabel('Index') 
    plt.ylabel('Score') 
    plt.legend() 
    plt.grid(True)
    plt.tight_layout()   

    save_dir = r"E:\_SITP\src\SupervisedTraining\model_stack_images"  
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,  "Performance Metrics Visualization.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight') 

    y4 = sereve_class_f1score

    plt.figure(figsize=(10, 6))

    plt.plot(x,  y4, color='blue', marker='o',linestyle='-', label='sereve_class_f1score')
  
    plt.title('Sereve Class F1 Score')
    plt.xlabel('Index') 
    plt.ylabel('Score') 
    plt.legend() 
    plt.grid(True)   
    plt.tight_layout()   

    save_path = os.path.join(save_dir,  "Sereve Class F1 Score.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight') 



if __name__ == "__main__":
    main()