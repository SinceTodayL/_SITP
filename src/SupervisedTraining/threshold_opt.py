import pandas as pd
import numpy as np

from sklearn.metrics  import f1_score, confusion_matrix


def ThresholdOptimization_F1Score(y_pred_prob, y_true, weight=None):

    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    best_f1 = 0
    if weight == None:
        weight = {0:1, 1:1}
    for thresh in thresholds:
        preds = (y_pred_prob >= thresh).astype(int)
        f1 = f1_score(y_true, preds, average='weighted', sample_weight=weight)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold


def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return np.sqrt(sensitivity  * specificity)


def ThresholdOptimization_Gmean(y_pred_prob, y_true):

    best_gmean = 0
    thresholds = np.linspace(0,  1, 100)
    best_threshold = 0.5
    for thresh in thresholds:
        preds = (y_pred_prob >= thresh).astype(int)
        gm = g_mean(y_true, preds)
        if gm > best_gmean:
            best_gmean = gm
            best_threshold = thresh

    return best_threshold