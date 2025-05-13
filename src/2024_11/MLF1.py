from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import numpy as np

# Step 1: Load the dataset
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy_label.xlsx"  # Replace with your file path
data = pd.read_excel(file_path)

# Step 2: Separate features and labels
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Use SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 5: Initialize classifiers
rf_model = RandomForestClassifier(random_state=42)
mlp_model = MLPClassifier(random_state=42, max_iter=1000)

# Step 6: Train the models
print("Training the Random Forest model...")
rf_model.fit(X_train_res, y_train_res)
print("Random Forest training complete.")

print("Training the MLP model...")
mlp_model.fit(X_train_res, y_train_res)
print("MLP training complete.")

# Step 7: Evaluate the models
print("Evaluating the models...")

# Random Forest Predictions
y_pred_rf = rf_model.predict(X_test)
# MLP Predictions
y_pred_mlp = mlp_model.predict(X_test)

# Confusion Matrices
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)

# Classification Reports
class_report_rf = classification_report(y_test, y_pred_rf)
class_report_mlp = classification_report(y_test, y_pred_mlp)

# Weighted F1 Scores
f1_score_rf = f1_score(y_test, y_pred_rf, average="weighted")
f1_score_mlp = f1_score(y_test, y_pred_mlp, average="weighted")

print("\nRandom Forest Confusion Matrix:")
print(conf_matrix_rf)
print("\nRandom Forest Classification Report:")
print(class_report_rf)
print(f"Random Forest Weighted F1 Score: {f1_score_rf:.4f}")

print("\nMLP Confusion Matrix:")
print(conf_matrix_mlp)
print("\nMLP Classification Report:")
print(class_report_mlp)
print(f"MLP Weighted F1 Score: {f1_score_mlp:.4f}")

# Step 8: Visualize the Confusion Matrices
plt.figure(figsize=(14, 6))

# Random Forest Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# MLP Confusion Matrix
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=mlp_model.classes_, yticklabels=mlp_model.classes_)
plt.title("MLP Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()

# Step 9: Visualize Feature Importance (Random Forest only)
plt.figure(figsize=(8, 6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), indices)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# Step 10: Visualize the decision boundaries (Only for 2D data)
if X_train.shape[1] == 2:  # Only for 2D data
    # Plot decision boundaries for Random Forest
    plt.figure(figsize=(8, 6))
    X_combined = np.vstack([X_train_res, X_test])
    y_combined = np.hstack([y_train_res, y_test])

    x_min, x_max = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
    y_min, y_max = X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z_rf = rf_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_rf = Z_rf.reshape(xx.shape)

    plt.contourf(xx, yy, Z_rf, alpha=0.8)
    plt.scatter(X_train_res[:, 0], X_train_res[:, 1], c=y_train_res, edgecolors='k', marker='o', s=50, cmap=plt.cm.Paired)
    plt.title("Random Forest Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()

    # Plot decision boundaries for MLP
    Z_mlp = mlp_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_mlp = Z_mlp.reshape(xx.shape)

    plt.contourf(xx, yy, Z_mlp, alpha=0.8)
    plt.scatter(X_train_res[:, 0], X_train_res[:, 1], c=y_train_res, edgecolors='k', marker='o', s=50, cmap=plt.cm.Paired)
    plt.title("MLP Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()

else:
    print("Feature set is not 2D, cannot visualize decision boundaries.")
