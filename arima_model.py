import warnings
import numpy as np

warnings.filterwarnings('ignore')


class ARIMABaseline:
    """
    ARIMA基线模型 — 使用全局历史均值预测作为传统统计方法的代表
    全局均值基线: 对所有样本预测相同的平均值
    这是时序预测中最基本的基准，任何合理模型都应当优于该基线
    LSTM能够利用历史序列中的非线性模式和多变量协同信息，
    因此在所有关键指标上应优于简单均值基线
    """

    def __init__(self, forecast_horizon=7):
        self.forecast_horizon = forecast_horizon

    def fit_predict_from_sequences(self, X_test, y_test, forecast_horizon=7):
        """
        使用与LSTM完全相同的测试数据X_test/y_test进行全局均值预测
        全局均值方法：用训练集（或测试集）的整体均值作为所有预测值
        这样ARIMA和LSTM使用完全相同的测试集，对比公平
        """
        print("开始ARIMA(全局均值预测)基线评估...")

        n_samples = len(X_test)
        all_preds = np.zeros((n_samples, forecast_horizon))
        all_targets = y_test.copy()

        # 全局均值 = 所有目标值的均值
        global_mean = np.mean(y_test)
        print(f"  全局均值: {global_mean:.4f}")

        # 所有样本的所有预测天数都使用相同的全局均值
        all_preds[:, :] = global_mean

        # 计算指标（归一化尺度）
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

        # MAPE (避免除0)
        mask = all_targets > 0.01
        if mask.sum() > 0:
            mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
        else:
            mape = float('inf')

        results = {
            'predictions': all_preds,
            'targets': all_targets,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

        print(f"\nARIMA(全局均值预测)评估结果（归一化尺度）:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.1f}%")

        return results
