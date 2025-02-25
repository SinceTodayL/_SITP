from Template import SupervisedTraining

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    SupervisedTraining(RandomForestClassifier(random_state=7, n_estimators=500, max_depth=10, class_weight='balanced'), True, True, True)
    # SupervisedTraining(MLPClassifier(random_state=41), True, True, True)

