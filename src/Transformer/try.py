import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=4, d_model=128, nhead=8, num_layers=4, dim_feedforward=256):
        super(TransformerPredictor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_fc = nn.Linear(input_dim, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        self.output_fc = nn.Linear(d_model, input_dim)
    
    def forward(self, src):
        # src 形状: (batch, seq_len=1127, input_dim=4)
        if src.dim() == 2:
            src = src.unsqueeze(0)

        src = self.input_fc(src)  # (batch, seq_len, d_model)
        
        # Transformer 需要 seq_len 维度在第二维
        src = src.permute(1, 0, 2)  # (seq_len, batch, d_model)
        
        # 目标序列用输入序列的最后一个时间步作为起点
        tgt = src[-1:].repeat(1127, 1, 1)  # (1127, batch, d_model)
        
        output = self.transformer(src, tgt)  # (seq_len, batch, d_model)
        output = self.output_fc(output)  # (seq_len, batch, input_dim)
        
        return output.permute(1, 0, 2).squeeze(0)  # 还原回 (batch, seq_len, input_dim)
    
time_steps = 8
batch_size = 1 
file_path = r'E:\_SITP\data'
all_files = sorted([f for f in os.listdir(file_path) if f.endswith('.xlsx')]) 
data_all = []
for i in range(0, len(all_files), 2):
    group = all_files[i:i+2]
    data = []
    for file in group:
        df = pd.read_excel(file_path + r'\\' + file, header=0)
        data.append(df[['value_left', 'value_right']].values)
    data = np.hstack((data[0], data[1]))
    data_all.append(data)
data_all = torch.tensor(data_all, dtype=torch.float32)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerPredictor().to(device)
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100  
loss_history = []

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(time_steps - 1):
        src = data_all[i].to(device)  
        target = data_all[i + 1].to(device) 
        
        optimizer.zero_grad()
        output = model(src) 
        loss_mse = criterion_mse(output, target)
        loss_mae = criterion_mae(output, target)
        loss = loss_mse + loss_mae  
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    loss_history.append(total_loss / (time_steps - 1))
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / (time_steps - 1):.6f}")


plt(loss_history, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.show()

for i in range(5):
    with torch.no_grad():
        test_input = data_all[i + 8].to(device) 
        predicted_output = model(test_input).cpu().numpy()
        true_output = data_all[i + 9].cpu().numpy()

    plt.figure(figsize=(20, 12))
    for feature in range(4):
        plt.subplot(2, 2, feature + 1)
        plt.plot(true_output[ 200:400, feature], label="True")
        plt.plot(predicted_output[ 200:400, feature], label="Predicted", linestyle="dashed")
        plt.xlabel("Time Step")
        plt.ylabel(f"Feature {feature+1}")
        plt.legend()
    plt.suptitle("Transformer Prediction vs True Data")
    plt.show()
