import warnings

import numpy as np

warnings.filterwarnings('ignore')

# PyTorch库
import torch
import torch.nn as nn
from tqdm import tqdm
class DemandPredictor:
    """
    需求预测器
    """

    def __init__(self, model, device, scaler_dict=None):
        self.model = model
        self.device = device
        self.model.eval()
        self.scaler_dict = scaler_dict or {}

    def predict(self, X, sku_ids=None):
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            if sku_ids is not None:
                sku_tensor = torch.LongTensor(sku_ids).to(self.device)
                predictions = self.model(X_tensor, sku_tensor)
            else:
                predictions = self.model(X_tensor)

            return predictions.cpu().numpy()

    def predict_future(self, df, sku_id, sequence_length, feature_cols, n_future=7):
        """
        预测单个SKU的未来销量
        """
        sku_data = df[df['sku_ID'] == sku_id].sort_values('date')

        if len(sku_data) < sequence_length:
            return None

        # 获取最后sequence_length天的数据
        last_sequence = sku_data[feature_cols].values[-sequence_length:]

        # 归一化
        if sku_id in self.scaler_dict:
            last_sequence = self.scaler_dict[sku_id].transform(last_sequence)

        # 转换为张量并预测
        X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(X, sku_id)

        return pred.cpu().numpy().flatten()


def evaluate_model(model, test_loader, device, criterion=nn.L1Loss()):
    """
    评估模型性能
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            if len(batch) == 3:
                X, y, sku_ids = batch
                X, y, sku_ids = X.to(device), y.to(device), sku_ids.to(device)
                outputs = model(X, sku_ids)
            elif len(batch) == 2:
                X, y = batch
                X, y = X.to(device), y.to(device)
                outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # 合并所有预测和真实值
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # 计算各种指标
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    # MAPE (避免除0)
    mask = all_targets > 0.01
    if mask.sum() > 0:
        mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
    else:
        mape = float('inf')

    print(f"评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.1f}%")

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': all_preds,
        'targets': all_targets
    }