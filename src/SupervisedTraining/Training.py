from Template import SupervisedTraining

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    SupervisedTraining(RandomForestClassifier(n_estimators=5, random_state=42), 
                       IfStandard=True, 
                       IfSMOTE=True, 
                       IfVisualize=True, 
                       dim_reduce='pca',
                       n_component=4)
