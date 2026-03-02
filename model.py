import warnings
warnings.filterwarnings('ignore')

# PyTorch库
import torch
import torch.nn as nn


class LSTMDemandModel(nn.Module):
    """
    PyTorch LSTM需求预测模型
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 forecast_horizon=7, dropout=0.2, n_skus=None, embedding_dim=50):
        super(LSTMDemandModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.use_sku_embedding = n_skus is not None and n_skus > 0

        # SKU嵌入层
        if self.use_sku_embedding:
            self.sku_embedding = nn.Embedding(n_skus, embedding_dim)
            self.lstm_input_size = input_size + embedding_dim
        else:
            self.lstm_input_size = input_size

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, forecast_horizon)
        )

    def forward(self, x, sku_ids=None):
        batch_size, seq_len, _ = x.shape

        # 如果有SKU嵌入，将其拼接到每个时间步
        if self.use_sku_embedding and sku_ids is not None:
            sku_emb = self.sku_embedding(sku_ids)  # (batch_size, embedding_dim)
            sku_emb = sku_emb.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, embedding_dim)
            x = torch.cat([x, sku_emb], dim=-1)

        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)

        # 取最后一个时间步的隐藏状态
        last_hidden = lstm_out[:, -1, :]

        # 全连接层
        output = self.fc_layers(last_hidden)

        return output


class SimpleLSTMModel(nn.Module):
    """
    简化版LSTM模型
    """

    def __init__(self, input_size, hidden_size=64, forecast_horizon=7, dropout=0.2):
        super(SimpleLSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size // 2, 32)
        self.fc2 = nn.Linear(32, forecast_horizon)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一层LSTM
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)

        # 第二层LSTM
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)

        # 取最后一个时间步
        last_out = lstm2_out[:, -1, :]

        # 全连接层
        x = self.relu(self.fc1(last_out))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


