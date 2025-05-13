import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter

# Read the original dataset
file_path = "C:\\Users\\LiuZh\\Desktop\\SITP\\3D_data_copy_label.xlsx"  # Replace with your original dataset path
data = pd.read_excel(file_path)

# Separate features and labels
X = data.iloc[:, :-1]  # All rows, all columns except the last one
y = data.iloc[:, -1]   # All rows, the last column

# Split the original dataset: 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# SMOTE oversampling for training data
smote = SMOTE(sampling_strategy="not majority", random_state=42)  # Initial oversampling
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Calculate the maximum class size and resample to 3x the max class size
max_class_size = max(Counter(y_train_resampled).values())
smote_final = SMOTE(sampling_strategy={label: max_class_size for label in set(y_train_resampled)}, random_state=42)
X_train_resampled, y_train_resampled = smote_final.fit_resample(X_train_resampled, y_train_resampled)

# Print the new class distribution
print("Class distribution after SMOTE:", Counter(y_train_resampled))

# Standardize the data
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# KNN model with additional techniques
# Adjusting weights: give higher weight to 0, 1, and 2 classes to reduce confusion
class_weights = {0: 2, 1: 3, 2: 8}  # Higher weight for classes prone to confusion

# Optimize KNN hyperparameters (focus on metrics sensitive to imbalanced data)
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_neighbors': range(5, 50, 5),  # Search over a range of k values
    'weights': ['uniform', 'distance'],  # Test both uniform and distance weights
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Experiment with distance metrics
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_weighted')  # Focus on F1 score
grid_search.fit(X_train_resampled, y_train_resampled)

# Best parameters from GridSearchCV
best_k = grid_search.best_params_['n_neighbors']
best_weights = grid_search.best_params_['weights']
best_metric = grid_search.best_params_['metric']
print(f"Best parameters: k={best_k}, weights={best_weights}, metric={best_metric}")

# Train the final model
model = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, metric=best_metric)
model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1 score: {f1:.6f}")

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Additional visualizations: classification report as a bar plot
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot precision, recall, and F1 scores
report_df.iloc[:-3, :-1].plot(kind='bar', figsize=(10, 6))
plt.title('Classification Report')
plt.ylabel('Score')
plt.xlabel('Class')
plt.legend(loc='lower right')
plt.grid(axis='y')
plt.show()
