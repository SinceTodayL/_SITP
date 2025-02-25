import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, classification_report
from imblearn.metrics import geometric_mean_score

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 
from sklearn.preprocessing  import StandardScaler 
from sklearn.decomposition  import PCA, TruncatedSVD 
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis 
from sklearn.manifold  import TSNE 
import umap 
from functools import lru_cache 
 
def dimensionality_reduction(X, method='pca', n_components=None, y=None, random_state=42):
    
    """
    Unified interface for multidimensional data projection 
    
    Args:
        X (np.ndarray):  Input data matrix of shape (n_samples, n_features)
        method (str): Algorithm selector ['pca', 'lda', 'tsne', 'umap', 'svd', 'ae']
        n_components (int/float): Target dimensions or explained variance ratio (PCA specific)
        y (np.ndarray):  Class labels (required for LDA)
        random_state (int): Seed for reproducible results 
        
    Returns:
        np.ndarray:  Reduced data matrix (n_samples, n_components)
        
    Raises:
        ValueError: For invalid method names or missing labels in supervised mode 
    """
    
    # Validate input method 
    valid_methods = ['pca', 'lda', 'tsne', 'umap', 'svd', 'ae']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Valid options: {valid_methods}")
    
    # Standardize data 
    X_scaled = StandardScaler().fit_transform(X)
    
    # Method dispatcher 
    if method == 'pca':
        return _pca_reduction(X_scaled, n_components, random_state)
    elif method == 'lda':
        return _lda_reduction(X_scaled, y, random_state)
    elif method == 'tsne':
        return _tsne_reduction(X_scaled, n_components, random_state)
    elif method == 'umap':
        return _umap_reduction(X_scaled, n_components, random_state)
    elif method == 'svd':
        return _svd_reduction(X_scaled, n_components, random_state)
    elif method == 'ae':
        return _autoencoder_reduction(X_scaled, n_components, random_state)
 
def _pca_reduction(X, n_components, seed):
    """Principal Component Analysis with variance control"""
    model = PCA(n_components=_resolve_pca_components(n_components), 
               random_state=seed)
    return model.fit_transform(X) 
 
def _resolve_pca_components(param):
    """Auto-select components covering 95% variance if float provided"""
    return param if isinstance(param, int) else 0.95 
 
def _lda_reduction(X, y, seed):
    """Supervised Linear Discriminant Analysis"""
    if y is None:
        raise ValueError("LDA requires class labels through 'y' parameter")
    model = LinearDiscriminantAnalysis(n_components=min(X.shape[1],  len(np.unique(y))-1)) 
    return model.fit_transform(X,  y)
 
def _tsne_reduction(X, n_components, seed):
    """t-SNE for non-linear visualization (perplexity=30 optimized for 1k+ samples)"""
    return TSNE(n_components=n_components or 2, 
                perplexity=30, 
                random_state=seed).fit_transform(X)
 
def _svd_reduction(X, n_components, seed):  
    """Truncated SVD for sparse data or text features"""  
    model = TruncatedSVD(n_components=n_components or 10,  
                        random_state=seed)  
    return model.fit_transform(X)   

def _umap_reduction(X, n_components, seed):
    """UMAP with topological preservation (15 neighbors for local structure)"""
    return umap.UMAP(n_components=n_components or 2, 
                    n_neighbors=15, 
                    min_dist=0.1, 
                    random_state=seed).fit_transform(X)
 
@lru_cache(maxsize=None)
def _autoencoder_reduction(X, latent_dim, seed):
    from tensorflow.keras.layers  import Input, Dense 
    from tensorflow.keras.models  import Model 
    
    np.random.seed(seed)     
    input_dim = X.shape[1] 
    
    # Architecture 
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(max(4, int(latent_dim*0.8)), activation='relu')(input_layer)
    encoded = Dense(latent_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Training configuration 
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam',  loss='mse')
    autoencoder.fit(X,  X, epochs=100, batch_size=32, verbose=0)
    
    return Model(input_layer, encoded).predict(X)



def SupervisedTraining(train_model, IfStandard, IfSMOTE, IfVisualize, dim_reduce=False,threshold_opt=False):

    data_path = r"E:\_SITP\Data.xlsx"
    label_path = r"E:\_SITP\DataLabel.xlsx"

    data = pd.read_excel(data_path, header=0)
    label = pd.read_excel(label_path, header=0)

    X = data.iloc[ : ,  : ].values
    y = label.iloc[ : ].values.squeeze()

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
