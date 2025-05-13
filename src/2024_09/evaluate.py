from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def evaluate_indicator(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
