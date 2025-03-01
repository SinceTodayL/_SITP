import pandas as pd
import numpy as np
import openpyxl
import os

from sklearn.model_selection  import GridSearchCV
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics  import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes  import GaussianNB 

from imblearn.pipeline  import Pipeline 
from imblearn.over_sampling  import SMOTE 
from imblearn.metrics import geometric_mean_score

import matplotlib.pyplot as plt

model_configs = [
    {
        'name': 'KNN',
        'model': Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', KNeighborsClassifier())
        ]),
        'params': {
            'smote__k_neighbors': [3,5,7,9],
            'clf__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['euclidean', 'manhattan'] 
        }
    },

    {
        'name': 'RandomForest',
        'model': Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', RandomForestClassifier())
        ]),
        'params': {
            'smote__k_neighbors': [3, 5, 7, 9],  
            'clf__n_estimators': [100, 200, 300, 400],
            'clf__max_depth': [None, 5, 10, 20],
            'clf__min_samples_split': [2, 5]
        }
    },

    {
        'name': 'MLP',
        'model': Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', MLPClassifier(max_iter=3000))
        ]),
        'params': {
            'smote__sampling_strategy': ['auto', 0.8],  
            'clf__hidden_layer_sizes': [(100,), (50,50)],
            'clf__activation': ['relu', 'tanh'],
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__learning_rate_init': [0.0001, 0.001, 0.01] 
        }
    },

    {
        'name': 'SVM',
        'model': Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', SVC())
        ]),
        'params': {
            'smote__k_neighbors': [3,5],
            'clf__C': [0.1, 1, 5, 10, 20],  
            'clf__kernel': ['linear', 'rbf', 'poly'], 
            'clf__gamma': ['scale', 'auto'] 
        }
    },

    {
        'name': 'DecisionTree',
        'model': Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', DecisionTreeClassifier())
        ]),
        'params': {
            'smote__k_neighbors': [3,5],
            'clf__criterion': ['gini', 'entropy', 'log_loss'],  
            'clf__max_depth': [None, 5, 10, 20, 30],
            'clf__min_impurity_decrease': [0.0, 0.1]  
        }
    },

    {
        'name': 'GaussianNB',
        'model': Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', GaussianNB())
        ]),
        'params': {
            'smote__k_neighbors': [3,5],
            'clf__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
        }
    }
]

def main():

    data_path = r"E:\_SITP\data\Data.xlsx"
    label_path = r"E:\_SITP\data\DataLabel.xlsx"

    data = pd.read_excel(data_path)
    label = pd.read_excel(label_path)

    X = data.iloc[ : ,  : ].values
    y = label.iloc[ : ].values.squeeze()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    results = []

    for config in model_configs:
        
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid.fit(X,  y)
        
        class_name = ["Normal", "Mild", "Sereve"]

        best_model = grid.best_estimator_ 
        y_pred = best_model.predict(X) 
        gmean = geometric_mean_score(y, y_pred)
        overall_report = classification_report(y, y_pred, target_names=class_name, output_dict=True)
        
        res = { 
            'model': config['name'],
            'best_params': grid.best_params_, 
            'best_score': grid.best_score_, 
            'f1_score': overall_report['weighted avg']['f1-score'],
            'gmean' : gmean
        }

        print(res)

        results.append(res)

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10,  6))
    plt.bar(results_df['model'],  results_df['best_score'], color='blue', label='f1-score')
    plt.title('f1-score  Performance')
    plt.xlabel('Index') 
    plt.ylabel('Score') 
    plt.legend() 
    plt.grid(True) 
    plt.tight_layout() 
    save_path = os.path.join(r'E:\_SITP\src\SupervisedTraining',  "f1-score Performance.png") 
    plt.savefig(save_path,  dpi=400, bbox_inches='tight')
    plt.close() 

    '''
    print("\n=== Final Results ===")
    print(results_df[['model', 'best_score', 'best_params']])
    results_df[['model', 'best_score', 'best_params']].to_excel(r'E:\_SITP\src\SupervisedTraining\GridSearch_Results_WeightedF1Score_opt.xlsx',  index=False)
    '''


if __name__ == "__main__":
    main()