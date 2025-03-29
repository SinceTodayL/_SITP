import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import math




"""
    Model
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: Dimension of the model (embedding size)
            max_len: Maximum sequence length
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # Prevents the tensor from being updated by the optimizer

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            x + positional encoding of shape (seq_len, batch_size, d_model)
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].unsqueeze(1)

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=4, d_model=128, nhead=8, num_layers=6, dim_feedforward=256, mask_ratio=0.1):
        """
        Args:
            input_dim: Number of input features per sample (4)
            d_model: Transformer model dimension (128)
            nhead: Number of attention heads (8)
            num_layers: Number of Transformer encoder layers (6)
            dim_feedforward: Dimension of the feedforward network (256)
            mask_ratio: Ratio of time steps to be masked (default: 10%)
        """
        super(TransformerPredictor, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.output_fc = nn.Linear(d_model, input_dim)
        self.mask_ratio = mask_ratio

    def generate_time_mask(self, seq_len):
        """
        Generates a time mask to randomly mask certain time steps.
        
        Args:
            seq_len: Length of the input sequence
        Returns:
            mask: Tensor of shape (seq_len, seq_len) with 0s at masked positions
        """
        mask = torch.ones(seq_len, seq_len)
        num_masked = int(seq_len * self.mask_ratio)
        masked_indices = torch.randperm(seq_len)[:num_masked]
        mask[:, masked_indices] = 0  # Masking specific time steps
        return mask

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (seq_len, batch_size, feature_dim)
                 Here, seq_len is the number of years used for prediction,
                 batch_size is the number of samples (1127), and feature_dim is 4.
        Returns:
            Tensor of shape (seq_len, batch_size, input_dim)
        """
        src = self.input_fc(src)  # (seq_len, batch_size, d_model)
        src = self.positional_encoding(src)
        
        # Generate time mask
        seq_len = src.size(0)
        time_mask = self.generate_time_mask(seq_len).to(src.device)
        
        output = self.transformer_encoder(src, mask=time_mask)
        output = self.output_fc(output) 
        return output








"""
    Data
"""

file_path = r'E:\_SITP\data'
all_files = sorted([f for f in os.listdir(file_path) if f.endswith('.xlsx')])
data_all = []

for i in range(0, len(all_files), 2):
    group = all_files[i:i+2]
    if (i + 2) // 2 in {}:
        continue
    data = []
    for file in group:
        df = pd.read_excel(os.path.join(file_path, file), header=0)
        data.append(df[['value_left', 'value_right']].values)
    data_combined = np.hstack((data[0], data[1]))
    data_all.append(data_combined)
data_all = torch.tensor(data_all, dtype=torch.float32)
'''
# Normalize (0-1) (choice)
mean_vals = data_all.mean(dim=0, keepdim=True)
std_vals = data_all.std(dim=0, keepdim=True)
std_vals[std_vals == 0] = 1 
data_all = (data_all - mean_vals) / std_vals
'''





"""
    Train
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
loss_history = []
train_time_step = len(data_all)

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(1, train_time_step):
        # Use the first i years as the source.
        # Since data_all is (time, batch, feature) and already in the correct shape,
        src = data_all[:i].to(device)  # Shape: (i, 1127, 4)
        # The target is the (i+1)th year.
        target = data_all[i].to(device)  # Shape: (1127, 4)

        optimizer.zero_grad()
        output = model(src)  # Output shape: (i, 1127, 4)
        # We only use the last time step of the output to compare with the target.
        loss = criterion(output[-1, :, :], target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (data_all.size(0) - 2)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# Plot train loss
plt.plot(loss_history, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.show()







"""
    Test (Choice)
"""

# In testing, we also use all available past years to predict the next year.
for i in range(0):  
    with torch.no_grad():
        test_src = data_all[:i + 10].to(device)
        predicted_output = model(test_src).cpu().numpy() 
        true_output = data_all[i + 10].cpu().numpy() 

    plt.figure(figsize=(20, 12))
    for feature in range(4):  # For each of the 4 features
        plt.subplot(2, 2, feature + 1)
        plt.plot(true_output[:, feature], label="True")
        plt.plot(predicted_output[-1, :, feature], label="Predicted", linestyle="dashed")
        plt.xlabel("Sample Index")
        plt.ylabel(f"Feature {feature+1}")
        plt.legend()
    plt.suptitle(f"Transformer Prediction vs True Data for Year {i+10}")
    plt.show()







"""
    predict next years
    1. predict
    2. compare
"""
from mpl_toolkits.mplot3d import Axes3D

model.eval()  # Set model to evaluation mode
num_future = 10
predicted_years = []  

# Start with the full history (all 14 years) as the initial input.
# data_all shape: (14, 1127, 4)
history = data_all.clone().to(device)

# Recursively predict future years.
for i in range(num_future):
    with torch.no_grad():
        output = model(history)  # Output shape: (current_seq_len, 1127, 4)
        next_year = output[-1, :, :]  # Use the last time step as the prediction (shape: (1127, 4))
    predicted_years.append(next_year.cpu())
    history = torch.cat([history, next_year.unsqueeze(0)], dim=0)

# ----------------------------
# 2D Visualization of Future Predictions
# ----------------------------
plt.figure(figsize=(20, 12))
pca = PCA(n_components=1)
# For each future year, plot the predicted data (mean over features) across all samples.
for i in range(num_future):
    pred_year = predicted_years[i].numpy()  # shape: (1127, 4)
    pred_pca = pca.fit_transform(pred_year).flatten()    
    plt.plot(np.arange(1127), pred_pca, label=f'Future Year {i+1}')

plt.xlabel("Sample Index (0-based)")
plt.ylabel("Predicted Mean Value")
plt.title(f"{num_future}-Year Future Predictions (Mean over Features) per Sample")
plt.legend()
plt.grid(True)
plt.show()







"""
    Compare with Standard Label
"""

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

file_path = "E:\_SITP\data\Label\label.xlsx"  # Replace with your actual file path
real_labels_df = pd.read_excel(file_path, header=0)
real_labels = real_labels_df.values.flatten()  

unique, counts = np.unique(real_labels, return_counts=True)
class_proportions = counts / len(real_labels)  # Proportions of 0, 1, and 2
print(f"Class Proportions in Real Labels: 0 -> {class_proportions[0]:.4f}, 1 -> {class_proportions[1]:.4f}, 2 -> {class_proportions[2]:.4f}")
num_samples = len(real_labels)
num_class_0 = int(class_proportions[0] * num_samples)
num_class_1 = int(class_proportions[1] * num_samples)
num_class_2 = num_samples - num_class_0 - num_class_1  

pred_data = np.array([predicted_years[i].numpy() for i in range(num_future)])  # shape: (num_future, 1127, 4)



'''
    Evaluate What is Abnormal :
    Regreesio Index?
'''

'''
# ----------------------------
# Compute the regression index:
# ----------------------------
# 'predicted_years'  : (shape: (predict_year_num, 1127, 4))
# Compute the regression index for each sample (j)
regression_index = np.sum(
    np.abs(np.sum(pred_data[1:] - pred_data[:-1], axis=2)),  # sum over features
    axis=0                                                   # sum over future years
)  # shape: (1127,)
predicted_labels = np.zeros_like(real_labels)
sorted_indices = np.argsort(regression_index)
predicted_labels[sorted_indices[:num_class_0]] = 0  
predicted_labels[sorted_indices[num_class_0:num_class_0 + num_class_1]] = 1  
predicted_labels[sorted_indices[num_class_0 + num_class_1:]] = 2  
'''


value_size = np.sum(
    pred_data, 
    axis=(0, 2)                                                  
)  # shape: (1127,)
predicted_labels = np.zeros_like(real_labels)
sorted_indices = np.argsort(value_size)
predicted_labels[sorted_indices[:num_class_0]] = 0  
predicted_labels[sorted_indices[num_class_0:num_class_0 + num_class_1]] = 1  
predicted_labels[sorted_indices[num_class_0 + num_class_1:]] = 2  




# ----------------------------
# Compute Precision, Recall, and F1 Score, plot confusion matrix
# ----------------------------
precision_macro = precision_score(real_labels, predicted_labels, average='macro', labels=[0, 1, 2])
recall_macro = recall_score(real_labels, predicted_labels, average='macro', labels=[0, 1, 2])
f1_macro = f1_score(real_labels, predicted_labels, average='macro', labels=[0, 1, 2])
precision_weighted = precision_score(real_labels, predicted_labels, average='weighted', labels=[0, 1, 2])
recall_weighted = recall_score(real_labels, predicted_labels, average='weighted', labels=[0, 1, 2])
f1_weighted = f1_score(real_labels, predicted_labels, average='weighted', labels=[0, 1, 2])
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro): {recall_macro:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Precision (Weighted): {precision_weighted:.4f}")
print(f"Recall (Weighted): {recall_weighted:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
conf_matrix = confusion_matrix(real_labels, predicted_labels, labels=[0, 1, 2])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Minor Anomaly", "Severe Anomaly"], yticklabels=["Normal", "Minor Anomaly", "Severe Anomaly"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



