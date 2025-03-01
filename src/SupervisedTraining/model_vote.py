from template import SupervisedTraining

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes  import GaussianNB 

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from imblearn.metrics import geometric_mean_score

import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import os

# best parameters
train_models = [KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1),
                SVC(kernel='rbf', C=1),
                GaussianNB(var_smoothing=1e-9),
                MLPClassifier(alpha=0.0001, learning_rate_init=0.0001, hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=2000, random_state=42),
                RandomForestClassifier(n_estimators=300, random_state=42, max_depth=10, min_samples_split=2),
                DecisionTreeClassifier(random_state=42, max_depth=10, criterion='entropy'), 
            ]


g_mean_result = []
macro_f1_result = []
weighted_f1_result = []
sereve_class_f1score = []
name_list = []

def get_subsets(lst):
    subsets = []
    for r in range(3, len(lst) + 1):
        for combo in itertools.combinations(lst,  r):
            subsets.append(list(combo)) 
    return subsets

def train_vote_model(models):
    data_path = r"E:\_SITP\data\Data.xlsx"
    label_path = r"E:\_SITP\data\DataLabel.xlsx"

    data = pd.read_excel(data_path)
    label = pd.read_excel(label_path)

    X = data.iloc[ : ,  : ].values
    y = label.iloc[ : ].values.squeeze()

    class_name = ["Normal", "Mild", "Sereve"]
 
    # all_predictions = np.zeros((len(X),  len(train_models)), dtype=np.int64) 
    final_result = np.zeros_like(y)
    
    for i, model in enumerate(models):
        pred = SupervisedTraining(X=X, y=y, train_model=model,
                                IfStandard=True, IfSMOTE=True, 
                                IfVisualize=False, threshold_opt=True)
        # all_predictions[:, i] = pred.astype(np.int64)   
        # pred = np.where(pred  == 2, 5, pred)  
        # pred = np.where(pred  == 1, 2, pred) 
        final_result += pred
    
    value, counts = np.unique(final_result, return_counts=True)
    print(dict(zip(value, counts)))

    size = len(models)

    final_result = np.where(final_result  <= size / 2.3, 0, 
                        np.where(final_result  <= size + 1, 1, 2))
    
    # final_result = stats.mode(all_predictions,  axis=1)[0].flatten()


    # Visualize
    print("\nOverall Report:")
    overall_report = classification_report(y, final_result, target_names=class_name, output_dict=True)
    gmean_test = geometric_mean_score(y, final_result)
    print(classification_report(y, final_result, target_names=class_name))
        
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
    for model in models:
        name = str(model.__class__.__name__)
        model_names = model_names + " " + name
    
    name_list.append(model_names + " Vote")

    # confusion matrix
    cm = confusion_matrix(y, final_result)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                xticklabels=class_name, 
                yticklabels=class_name)
    plt.title(model_names + " Vote")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    save_dir = r"E:\_SITP\src\SupervisedTraining\model_Vote_images1"  
    os.makedirs(save_dir,  exist_ok=True)
    save_path = os.path.join(save_dir,  f"{model_names} Vote.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    

def main():

    all_model_subset = get_subsets(train_models)
    index = 0
    
    for model_subset in all_model_subset:
        index += 1
        train_vote_model(model_subset)

    x = range(len(g_mean_result))  
    y1 = g_mean_result 
    y2 = macro_f1_result 
    y3 = weighted_f1_result 
    y4 = sereve_class_f1score

    data = {
    'Name': name_list,
    'G-Mean': g_mean_result,
    'Macro F1': macro_f1_result,
    'Weighted F1': weighted_f1_result,
    'Severe Class F1': sereve_class_f1score
    }

    df = pd.DataFrame(data)
    df.to_excel(r'E:\_SITP\src\SupervisedTraining\model_Vote_images1\vote_results.xlsx',  index=False)

    save_dir = r"E:\_SITP\src\SupervisedTraining\model_Vote_images1"  
    os.makedirs(save_dir, exist_ok=True)


    plt.figure(figsize=(10,  6))
    plt.bar(x,  y1, color='blue', label='G-Mean')
    plt.title('G-Mean  Performance')
    plt.xlabel('Index') 
    plt.ylabel('Score') 
    plt.legend() 
    plt.tight_layout() 
    save_path = os.path.join(save_dir,  "G-Mean Performance.png") 
    plt.savefig(save_path,  dpi=400, bbox_inches='tight')
    plt.close() 

    plt.figure(figsize=(10,  6))
    plt.bar(x,  y2, color='red', label='Macro F1')
    plt.title('Macro  F1 Performance')
    plt.xlabel('Index') 
    plt.ylabel('Score') 
    plt.legend() 
    plt.tight_layout() 
    save_path = os.path.join(save_dir,  "Macro F1 Performance.png") 
    plt.savefig(save_path,  dpi=400, bbox_inches='tight')
    plt.close() 

    plt.figure(figsize=(10,  6))
    plt.bar(x,  y3, color='green', label='Weighted F1')
    plt.title('Weighted  F1 Performance')
    plt.xlabel('Index') 
    plt.ylabel('Score') 
    plt.legend() 
    plt.tight_layout() 
    save_path = os.path.join(save_dir,  "Weighted F1 Performance.png") 
    plt.savefig(save_path,  dpi=400, bbox_inches='tight')
    plt.close() 

    plt.figure(figsize=(10,  6))
    plt.bar(x,  y4, color='purple', label='Sereve Class F1')
    plt.title('Sereve  Class F1 Score')
    plt.xlabel('Index') 
    plt.ylabel('Score') 
    plt.legend() 
    plt.tight_layout() 
    save_path = os.path.join(save_dir,  "Sereve Class F1 Score.png") 
    plt.savefig(save_path,  dpi=400, bbox_inches='tight')
    plt.close() 


    '''
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

    save_dir = r"E:\_SITP\src\SupervisedTraining\model_Vote_images1"  
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
    ''' 



if __name__ == "__main__":
    main()