import torch 
import torch.nn  as nn 
from torch.utils.data  import Dataset, DataLoader 
import pandas as pd 
import numpy as np 
from sklearn.metrics  import mean_absolute_error, mean_squared_error 
from sklearn.preprocessing  import MinMaxScaler 
 
# 1. 数据预处理类 
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data  = data 
        self.seq_length  = seq_length 
        
    def __len__(self):
        return len(self.data)  - self.seq_length  
    
    def __getitem__(self, idx):
        x = self.data[idx:idx  + self.seq_length] 
        y = self.data[idx  + self.seq_length] 
        return x, y 
 
# 2. Transformer 模型定义 
class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(Transformer, self).__init__()
        self.d_model = d_model 
        
        # 输入嵌入层 
        self.enc_embedding  = nn.Linear(input_dim, d_model)
        self.dec_embedding  = nn.Linear(input_dim, d_model)
        
        # 定义位置编码 
        self.positional_encoding  = nn.Parameter(torch.randn(1,  28, d_model))
        
        # 定义 Transformer 编码器和解码器 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        
        self.transformer_encoder  = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.transformer_decoder  = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出层 
        self.output_layer  = nn.Linear(d_model, input_dim)
 
    def forward(self, src, tgt):
        # 添加位置编码 
        src = self.enc_embedding(src)  + self.positional_encoding[:,  :src.size(1)] 
        tgt = self.dec_embedding(tgt)  + self.positional_encoding[:,  :tgt.size(1)] 
        
        # 前向传播 
        memory = self.transformer_encoder(src.permute(1,  0, 2))
        output = self.transformer_decoder(tgt.permute(1,  0, 2), memory).permute(1, 0, 2)
        
        # 生成预测结果 
        output = self.output_layer(output[:,  -1:])
        return output 
 
# 3. 主函数 
def main():
    # 3.1 加载数据 
    df = pd.read_excel(r'E:\_SITP\data\Data.xlsx') 
    data = df.values  
    
    # 3.2 数据归一化 
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data) 
    
    # 3.3 定义超参数 
    input_dim = 4  # 每年4个特征 
    d_model = 64 
    nhead = 8 
    num_encoder_layers = 3 
    num_decoder_layers = 3 
    seq_length = 4  # 每年4个特征 
    
    # 3.4 创建数据集和数据加载器 
    dataset = TimeSeriesDataset(data_normalized, seq_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3.5 初始化模型、优化器和损失函数 
    model = Transformer(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),  lr=0.001)
    
    # 3.6 训练模型 
    num_epochs = 50 
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0 
        
        for i, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.float() 
            y_batch = y_batch.float() 
            
            # 将输入和目标对齐 
            src = x_batch.view(-1,  seq_length, input_dim)
            tgt = y_batch.view(-1,  1, input_dim)
            
            optimizer.zero_grad() 
            outputs = model(src, tgt)
            loss = criterion(outputs, tgt)
            loss.backward() 
            optimizer.step() 
            
            total_loss += loss.item() 
            
        avg_loss = total_loss / len(loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # 3.7 逐步预测并计算误差 
    model.eval() 
    historical_data = data_normalized.copy()   # 复制原始归一化数据用于逐步预测 
    errors_mae = []
    errors_mse = []
    
    with torch.no_grad(): 
        for year in range(6):  # 预测第2年到第7年（共6次预测）
            current_year_idx = year * seq_length 
            input_seq = historical_data[current_year_idx:current_year_idx + seq_length]
            input_seq = torch.FloatTensor(input_seq).unsqueeze(0)
            
            # 预测下一年的数据 
            predicted_next_year = model(input_seq, input_seq)
            predicted_next_year = predicted_next_year.squeeze().numpy() 
            
            # 计算误差（如果存在真实数据）
            if year < 6:  # 最后一年（第7年）没有后续数据用于验证 
                true_next_year = data_normalized[current_year_idx + seq_length]
                mae = mean_absolute_error(true_next_year, predicted_next_year)
                mse = mean_squared_error(true_next_year, predicted_next_year)
                
                errors_mae.append(mae) 
                errors_mse.append(mse) 
                
                print(f'Year {year+1} -> Year {year+2}:')
                print(f'MAE: {mae:.4f}, MSE: {mse:.4f}')
                print('-' * 50)
            
            # 将预测结果添加到历史数据中（用于下一步预测）
            historical_data = np.vstack([historical_data,  predicted_next_year])
    
    print("\nAll Errors:")
    print(f"MAE: {np.mean(errors_mae):.4f}") 
    print(f"MSE: {np.mean(errors_mse):.4f}") 
 
if __name__ == '__main__':
    main()