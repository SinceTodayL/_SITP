import pandas as pd
import numpy as np

from template import SupervisedTraining, SupervisedTraining_ByStep_Order1
from dim_reducer import DimensionReducer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

if __name__ == "__main__":

    data_path = r"E:\_SITP\data\Data.xlsx"
    label_path = r"E:\_SITP\data\DataLabel.xlsx"

    data = pd.read_excel(data_path)
    label = pd.read_excel(label_path)

    X = data.iloc[ : ,  : ].values
    y = label.iloc[ : ].values.squeeze()

    X_res = DimensionReducer(method='pca', n_components=7).fit_transform(X)


#    SupervisedTraining(X=X_res, y=y, train_model=SVC(kernel='linear', probability=True), 
#                      IfStandard=True, IfSMOTE=True, IfVisualize=True, 


    SupervisedTraining_ByStep_Order1(X=X, y=y, 
    train_model1=MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=2000, random_state=42),
    train_model2=MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=300, random_state=42),
    IfStandard=True, IfSMOTE1=True, IfSMOTE2=True, IfVisualize=True)

#   SupervisedTraining_ByStep_Order1(X=X, y=y, 
#   train_model1=RandomForestClassifier(n_estimators=400, random_state=42),
#   train_model2=RandomForestClassifier(n_estimators=100, random_state=42),
#   IfStandard=True, IfSMOTE1=True, IfSMOTE2=True, IfVisualize=True)




'''
    TODO:
        1. threshold optimization 
        2. dimension reduction 
        3. 改为分步检测
        4. 模型堆叠

'''
