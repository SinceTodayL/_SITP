import pandas as pd
import numpy as np

from template import SupervisedTraining
from dim_reducer import DimensionReducer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":

    data_path = r"E:\_SITP\data\Data.xlsx"
    label_path = r"E:\_SITP\data\DataLabel.xlsx"

    data = pd.read_excel(data_path)
    label = pd.read_excel(label_path)

    X = data.iloc[ : ,  : ].values
    y = label.iloc[ : ].values.squeeze()

    X_res = DimensionReducer(method='ae', n_components=3).fit_transform(X)

    SupervisedTraining(X=X_res, y=y, train_model=RandomForestClassifier(n_estimators=300, random_state=42), 
                       IfStandard=True, IfSMOTE=True, IfVisualize=True, 
                       )
