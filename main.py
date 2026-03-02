import os
import warnings

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

matplotlib.use('Agg')
from model import LSTMDemandModel, SimpleLSTMModel
from predictor import DemandPredictor, evaluate_model
from trainer import DemandTrainer
from arima_model import ARIMABaseline
from visualizer import ResultVisualizer
from inventory_decision import InventoryDecisionSystem

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from preprocessor import *


def main():
    print("=" * 60)
    print("超市多品类商品协同需求预测系统")
    print("（含互补性/替代性协同分析 + ARIMA对比 + 库存决策）")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 确保输出目录
    os.makedirs('run', exist_ok=True)

    # 1. 数据预处理
    print("\n【步骤1】数据预处理")
    preprocessor = DataPreprocessor(800)

    excel_file = "data/交易数据.xlsx"
    inventory_file = "data/库存数据.xlsx"
    replenishment_file = "data/补货数据.xlsx"
    product_file = "data/产品数据.xlsx"
    display_file = "data/展示数据.xlsx"

    data = preprocessor.prepare_data(
        excel_file, inventory_file, replenishment_file,
        product_file, display_file
    )

    # 初始化可视化器
    viz = ResultVisualizer(save_dir='run')

    # 2. 品类分析可视化
    print("\n【步骤2】品类分析可视化")
    viz.plot_category_analysis(data)

    # 3. 协同特征可视化
    print("\n【步骤3】协同特征可视化（互补性与替代性）")
    if hasattr(preprocessor, 'corr_matrix'):
        viz.plot_complementarity_heatmap(preprocessor.corr_matrix)
    if hasattr(preprocessor, 'complement_dict') and hasattr(preprocessor, 'substitute_dict'):
        viz.plot_synergy_network(
            preprocessor.complement_dict,
            preprocessor.substitute_dict,
            preprocessor.corr_matrix
        )

    # 4. 准备序列数据
    print("\n【步骤4】准备序列数据")
    feature_cols = ['quantity', 'original_unit_price', 'beginning_inventory',
                    'complement_sales', 'substitute_sales']
    data['dayofweek'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    feature_cols.extend(['dayofweek', 'month', 'is_weekend'])

    sequence_length = 30
    forecast_horizon = 7

    X, y, sku_ids = prepare_sequences(
        data,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_cols=feature_cols
    )

    print(f"X形状: {X.shape}")
    print(f"y形状: {y.shape}")

    # 5. 划分数据集
    print("\n【步骤5】划分数据集")
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    if sku_ids is not None:
        sku_train = sku_ids[:train_size]
        sku_val = sku_ids[train_size:train_size + val_size]
        sku_test = sku_ids[train_size + val_size:]
    else:
        sku_train = sku_val = sku_test = None

    print(f"训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")
    print(f"测试集: {X_test.shape}")

    # 6. 创建数据加载器
    print("\n【步骤6】创建数据加载器")
    batch_size = 32

    if sku_train is not None:
        train_dataset = TimeSeriesDataset(X_train, y_train, sku_train)
        val_dataset = TimeSeriesDataset(X_val, y_val, sku_val)
        test_dataset = TimeSeriesDataset(X_test, y_test, sku_test)
    else:
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 7. 构建LSTM模型
    print("\n【步骤7】构建LSTM模型")
    input_size = X.shape[2]
    n_unique_skus = len(data['sku_encoded'].unique()) if sku_ids is not None else 0
    use_sku_embedding = n_unique_skus > 10 and sku_ids is not None

    if use_sku_embedding:
        print("使用带SKU嵌入的LSTM模型")
        model = LSTMDemandModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            forecast_horizon=forecast_horizon,
            dropout=0.2,
            n_skus=n_unique_skus,
            embedding_dim=min(50, max(10, n_unique_skus // 5))
        )
    else:
        print("使用简化版LSTM模型")
        model = SimpleLSTMModel(
            input_size=input_size,
            hidden_size=64,
            forecast_horizon=forecast_horizon,
            dropout=0.2
        )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 8. 训练LSTM模型
    print("\n【步骤8】训练LSTM模型")
    model_path = 'best_model.pth'

    # 如果旧模型存在则加载，否则重新训练
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("已加载已有模型，跳过训练...")
        except:
            os.remove(model_path)
            print("旧模型架构不匹配，重新训练...")

    trainer = DemandTrainer(model, device, learning_rate=0.0005)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=80, early_stop_patience=15
    )

    # 训练曲线可视化
    viz.plot_training_curve(train_losses, val_losses)

    # 9. 评估LSTM模型
    print("\n【步骤9】评估LSTM模型")
    lstm_results = evaluate_model(model, test_loader, device)

    # 10. ARIMA基线模型（朴素预测，使用与LSTM相同的测试集）
    print("\n【步骤10】ARIMA基线模型（朴素预测，用于对比）")
    arima = ARIMABaseline(forecast_horizon=forecast_horizon)
    arima_results = arima.fit_predict_from_sequences(
        X_test, y_test, forecast_horizon=forecast_horizon
    )

    # 11. 模型对比可视化
    print("\n【步骤11】LSTM vs ARIMA 模型对比可视化")
    viz.plot_model_comparison(lstm_results, arima_results)
    viz.plot_summary_table(lstm_results, arima_results)

    # 预测样本对比
    arima_pred_display = arima_results['predictions'] if len(arima_results['predictions']) > 0 else None
    arima_target_display = arima_results['targets'] if len(arima_results['targets']) > 0 else None
    viz.plot_prediction_samples(
        lstm_results['predictions'], lstm_results['targets'],
        arima_pred_display, arima_target_display,
        forecast_horizon=forecast_horizon
    )

    # 误差按步长分析
    viz.plot_error_by_horizon(lstm_results['predictions'], lstm_results['targets'], forecast_horizon)

    # 12. 库存决策应用
    print("\n【步骤12】库存决策应用")
    # 计算scale_factor：各SKU的平均最大日销量
    sku_max_sales = data.groupby('sku_ID')['quantity'].max()
    scale_factor = max(sku_max_sales.median(), 50)  # 至少50
    print(f"  销量换算系数 (scale_factor): {scale_factor:.0f}")
    inv_system = InventoryDecisionSystem(safety_factor=1.5, lead_time=3,
                                          service_level=0.95, scale_factor=scale_factor)
    decisions_df = inv_system.plot_inventory_dashboard(
        lstm_results['predictions'], lstm_results['targets'],
        forecast_horizon=forecast_horizon, save_dir='run'
    )

    # 保存决策表
    decisions_df.to_csv('run/inventory_decisions.csv', index=False, encoding='utf-8-sig')
    print("  库存决策表已保存: run/inventory_decisions.csv")

    # 13. 输出汇总
    print("\n" + "=" * 60)
    print("所有结果已保存到 run/ 目录:")
    print("=" * 60)
    for f in sorted(os.listdir('run')):
        fpath = os.path.join('run', f)
        size = os.path.getsize(fpath) / 1024
        print(f"  📊 {f} ({size:.0f} KB)")

    print("\n" + "=" * 60)
    print("LSTM vs ARIMA 结果摘要:")
    print(f"  LSTM  - MAE: {lstm_results['mae']:.4f}, RMSE: {lstm_results['rmse']:.4f}")
    print(f"  ARIMA - MAE: {arima_results['mae']:.4f}, RMSE: {arima_results['rmse']:.4f}")
    improvement = (arima_results['mae'] - lstm_results['mae']) / arima_results['mae'] * 100
    print(f"  LSTM较ARIMA改进: {improvement:.1f}%")
    print("=" * 60)

    return model, trainer


if __name__ == "__main__":
    model, trainer = main()
