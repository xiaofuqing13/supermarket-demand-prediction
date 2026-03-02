import warnings
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from tqdm import tqdm

warnings.filterwarnings('ignore')


class ARIMABaseline:
    """
    ARIMA基线模型 - 用于与LSTM对比
    ARIMA只能捕捉线性趋势和季节性，无法学习非线性关系
    """

    def __init__(self, forecast_horizon=7):
        self.forecast_horizon = forecast_horizon
        self.models = {}

    def fit_predict(self, df, test_start_idx, sequence_length=30, max_skus=20):
        """
        对每个SKU拟合ARIMA模型并预测
        """
        print("开始ARIMA基线模型拟合...")

        skus = df['sku_ID'].unique()
        # 限制SKU数量避免耗时太长
        if len(skus) > max_skus:
            # 选择销量最大的SKU
            top_skus = df.groupby('sku_ID')['quantity'].sum().nlargest(max_skus).index
            skus = top_skus

        all_preds = []
        all_targets = []
        sku_names = []

        for sku in tqdm(skus, desc='ARIMA拟合'):
            sku_data = df[df['sku_ID'] == sku].sort_values('date')

            if len(sku_data) < sequence_length + self.forecast_horizon:
                continue

            # 时间序列
            sales = sku_data['quantity'].values

            # 划分训练/测试 - 与LSTM保持相同比例
            split_idx = int(len(sales) * 0.85)
            train_data = sales[:split_idx]
            test_data = sales[split_idx:]

            if len(test_data) < self.forecast_horizon:
                continue

            try:
                # 使用auto_arima自动选参
                model = auto_arima(
                    train_data,
                    start_p=0, max_p=3,
                    start_q=0, max_q=3,
                    d=None, max_d=2,
                    seasonal=True, m=7,
                    start_P=0, max_P=2,
                    start_Q=0, max_Q=2,
                    D=None, max_D=1,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    n_fits=20
                )

                self.models[sku] = model

                # 滚动预测
                n_windows = min(5, (len(test_data) - self.forecast_horizon) + 1)
                for i in range(n_windows):
                    actual = test_data[i:i + self.forecast_horizon]
                    if len(actual) < self.forecast_horizon:
                        break
                    # 预测
                    pred = model.predict(n_periods=self.forecast_horizon)
                    pred = np.maximum(pred, 0)  # 销量不能为负

                    all_preds.append(pred)
                    all_targets.append(actual)
                    sku_names.append(sku)

                    # 更新模型
                    model.update(test_data[i:i + 1])

            except Exception:
                continue

        if all_preds:
            preds = np.array(all_preds)
            targets = np.array(all_targets)
        else:
            preds = np.array([]).reshape(0, self.forecast_horizon)
            targets = np.array([]).reshape(0, self.forecast_horizon)

        # 计算指标
        if len(preds) > 0:
            mae = np.mean(np.abs(preds - targets))
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            mape = np.mean(np.abs((targets - preds) / (targets + 1))) * 100
        else:
            mae = rmse = mape = float('inf')

        results = {
            'predictions': preds,
            'targets': targets,
            'sku_names': sku_names,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

        print(f"\nARIMA评估结果:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.1f}%")

        return results
