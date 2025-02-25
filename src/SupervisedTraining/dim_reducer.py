import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition  import PCA, TruncatedSVD 
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis 
from sklearn.manifold  import TSNE 
from umap import UMAP
from functools import lru_cache

import tensorflow as tf
 
class DimensionReducer:
    def __init__(self, method, n_components=None, random_state=42):
        self.method = method 
        self.n_components = n_components 
        self.random_state = random_state 
        self.model = None 
        self.IfStandard = None 
 
    def fit_transform(self, X, y=None):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X) 
            
        if self.method == 'pca':
            self.model = PCA(n_components=self._resolve_pca_components(X), 
                            random_state=self.random_state) 
            
        elif self.method == 'lda':
            self.model = LinearDiscriminantAnalysis(n_components=min(X.shape[1],  len(np.unique(y))-1)) 

        elif self.method == 'tsne':
            self.model = TSNE(n_components=self.n_components or 2, 
                             perplexity=30, 
                             random_state=self.random_state) 
            
        elif self.method == 'umap':
            self.model = UMAP(n_components=self.n_components or 2,
                             n_neighbors=20,
                             min_dist=0.1,
                             random_state=self.random_state) 
            
        elif self.method == 'svd':
            self.model = TruncatedSVD(n_components=self.n_components or 10,
                                     random_state=self.random_state) 
            
        elif self.method == 'ae':
            self._build_autoencoder(X.shape[1]) 
            self.model.fit(X,  X, epochs=100, batch_size=32, verbose=0)
            
        return self.model.fit_transform(X, y) if y is not None else self.model.fit_transform(X) 
 
    def transform(self, X):
        if self.scaler: 
            X = self.scaler.transform(X) 
            
        if self.method in ['tsne', 'umap']:
            raise ValueError(f"{self.method}  does not support transform on new data")
            
        return self.model.transform(X) 
    
    def _resolve_pca_components(self, X):
        if isinstance(self.n_components, float):
            temp_pca = PCA(n_components=self.n_components)
            temp_pca.fit(X) 
            return temp_pca.n_components_
        return self.n_components or X.shape[1] 
    
    def _build_autoencoder(self, input_dim):
        tf.random.set_seed(self.random_state) 
        input_layer = tf.keras.Input(shape=(input_dim,)) 
        encoded = tf.keras.layers.Dense(max(4,  int(self.n_components*0.8)), activation='relu')(input_layer)
        encoded = tf.keras.layers.Dense(self.n_components,  activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(input_dim,  activation='sigmoid')(encoded)
        self.model  = tf.keras.Model(input_layer,  decoded)
        self.model.compile(optimizer='adam',  loss='mse')



