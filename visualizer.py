import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """可视化模块 - 生成论文级别的精美图表"""

    def __init__(self, save_dir='run'):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
        # 专业配色方案
        self.colors = {
            'lstm': '#2196F3',
            'arima': '#FF9800',
            'actual': '#4CAF50',
            'complement': '#E91E63',
            'substitute': '#9C27B0',
            'accent1': '#00BCD4',
            'accent2': '#FF5722',
            'bg': '#FAFAFA',
        }

    def plot_training_curve(self, train_losses, val_losses):
        """训练损失曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(train_losses) + 1)

        # 线性坐标
        axes[0].plot(epochs, train_losses, color=self.colors['lstm'], linewidth=2, label='训练损失', marker='o', markersize=3)
        axes[0].plot(epochs, val_losses, color=self.colors['arima'], linewidth=2, label='验证损失', marker='s', markersize=3)
        axes[0].fill_between(epochs, train_losses, alpha=0.1, color=self.colors['lstm'])
        axes[0].fill_between(epochs, val_losses, alpha=0.1, color=self.colors['arima'])
        axes[0].set_title('LSTM模型训练过程', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[0].set_ylabel('损失值 (Loss)', fontsize=12)
        axes[0].legend(fontsize=11, loc='upper right')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_facecolor(self.colors['bg'])

        # 对数坐标
        axes[1].plot(epochs, train_losses, color=self.colors['lstm'], linewidth=2, label='训练损失', marker='o', markersize=3)
        axes[1].plot(epochs, val_losses, color=self.colors['arima'], linewidth=2, label='验证损失', marker='s', markersize=3)
        axes[1].set_title('LSTM训练过程（对数尺度）', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[1].set_ylabel('损失值 (Log Scale)', fontsize=12)
        axes[1].set_yscale('log')
        axes[1].legend(fontsize=11, loc='upper right')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_facecolor(self.colors['bg'])

        plt.tight_layout()
        path = f'{self.save_dir}/training_curve.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

    def plot_complementarity_heatmap(self, corr_matrix, top_n=15):
        """商品互补性热力图"""
        # 选取相关性最显著的商品
        mean_abs_corr = corr_matrix.abs().mean()
        top_skus = mean_abs_corr.nlargest(top_n).index
        sub_corr = corr_matrix.loc[top_skus, top_skus]

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)
        sns.heatmap(sub_corr, mask=mask, annot=True, fmt='.2f',
                    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                    linewidths=0.5, ax=ax,
                    cbar_kws={'label': '相关系数', 'shrink': 0.8})
        ax.set_title('商品间销量相关性热力图\n（绿色=互补关系 | 红色=替代关系）',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.tight_layout()
        path = f'{self.save_dir}/complementarity_heatmap.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

    def plot_synergy_network(self, complement_dict, substitute_dict, corr_matrix, top_n=10):
        """互补/替代关系网络图"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # 互补关系
        skus = list(complement_dict.keys())[:top_n]
        ax = axes[0]
        pairs = []
        for sku in skus:
            for comp in complement_dict[sku][:2]:
                if comp in corr_matrix.index and sku in corr_matrix.index:
                    val = corr_matrix.loc[sku, comp]
                    if val > 0.1:
                        pairs.append((str(sku)[:8], str(comp)[:8], val))
        if pairs:
            pair_df = pd.DataFrame(pairs, columns=['商品A', '商品B', '相关系数']).head(15)
            bar_colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(pair_df)))
            ax.barh(range(len(pair_df)),
                    pair_df['相关系数'],
                    color=bar_colors, edgecolor='white', linewidth=0.5)
            ax.set_yticks(range(len(pair_df)))
            ax.set_yticklabels([f"{a} ↔ {b}" for a, b in zip(pair_df['商品A'], pair_df['商品B'])], fontsize=9)
            ax.set_xlabel('相关系数', fontsize=12)
        ax.set_title('互补商品关系强度\n（经常一起购买的商品对）', fontsize=13, fontweight='bold')
        ax.set_facecolor(self.colors['bg'])
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')

        # 替代关系
        ax = axes[1]
        pairs = []
        for sku in skus:
            for sub in substitute_dict[sku][:2]:
                if sub in corr_matrix.index and sku in corr_matrix.index:
                    val = corr_matrix.loc[sku, sub]
                    if val < 0:
                        pairs.append((str(sku)[:8], str(sub)[:8], abs(val)))
        if pairs:
            pair_df = pd.DataFrame(pairs, columns=['商品A', '商品B', '|相关系数|']).head(15)
            bar_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(pair_df)))
            ax.barh(range(len(pair_df)),
                    pair_df['|相关系数|'],
                    color=bar_colors, edgecolor='white', linewidth=0.5)
            ax.set_yticks(range(len(pair_df)))
            ax.set_yticklabels([f"{a} ↔ {b}" for a, b in zip(pair_df['商品A'], pair_df['商品B'])], fontsize=9)
            ax.set_xlabel('|相关系数|', fontsize=12)
        else:
            ax.text(0.5, 0.5, '未发现显著替代关系', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('替代商品关系强度\n（此消彼长的商品对）', fontsize=13, fontweight='bold')
        ax.set_facecolor(self.colors['bg'])
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')

        plt.tight_layout()
        path = f'{self.save_dir}/synergy_network.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

    def plot_model_comparison(self, lstm_results, arima_results):
        """LSTM vs ARIMA 模型对比"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # 1. 指标柱状图
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['MAE', 'RMSE']
        lstm_vals = [lstm_results['mae'], lstm_results['rmse']]
        arima_vals = [arima_results['mae'], arima_results['rmse']]

        x = np.arange(len(metrics))
        w = 0.35
        bars1 = ax1.bar(x - w/2, lstm_vals, w, label='LSTM', color=self.colors['lstm'],
                        edgecolor='white', linewidth=1.5, zorder=3)
        bars2 = ax1.bar(x + w/2, arima_vals, w, label='ARIMA', color=self.colors['arima'],
                        edgecolor='white', linewidth=1.5, zorder=3)

        # 标注数值
        for bar in bars1:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        for bar in bars2:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, fontsize=13)
        ax1.set_title('预测精度对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('误差值', fontsize=12)
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_facecolor(self.colors['bg'])
        ax1.set_axisbelow(True)

        # 2. 改进率雷达图
        ax2 = fig.add_subplot(gs[0, 1])
        improvement_mae = (arima_results['mae'] - lstm_results['mae']) / arima_results['mae'] * 100
        improvement_rmse = (arima_results['rmse'] - lstm_results['rmse']) / arima_results['rmse'] * 100

        categories = ['MAE\n改进率', 'RMSE\n改进率']
        values = [max(improvement_mae, 0), max(improvement_rmse, 0)]
        bar_colors = [self.colors['lstm'] if v > 0 else '#EF5350' for v in values]
        bars = ax2.bar(categories, values, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)
        for bar, v in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold',
                     color=self.colors['lstm'])
        ax2.set_title('LSTM相对ARIMA的改进率', fontsize=14, fontweight='bold')
        ax2.set_ylabel('改进百分比 (%)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_facecolor(self.colors['bg'])
        ax2.set_axisbelow(True)

        # 3. 预测值散点图对比
        ax3 = fig.add_subplot(gs[1, 0])
        n = min(100, len(lstm_results['predictions'].flatten()))
        lstm_flat = lstm_results['predictions'].flatten()[:n]
        target_flat = lstm_results['targets'].flatten()[:n]
        ax3.scatter(target_flat, lstm_flat, alpha=0.6, s=30, c=self.colors['lstm'], label='LSTM', edgecolors='white', linewidth=0.3)
        if len(arima_results['predictions']) > 0:
            arima_flat = arima_results['predictions'].flatten()[:n]
            arima_target = arima_results['targets'].flatten()[:n]
            ax3.scatter(arima_target, arima_flat, alpha=0.5, s=30, c=self.colors['arima'], label='ARIMA', marker='D', edgecolors='white', linewidth=0.3)

        max_val = max(target_flat.max(), lstm_flat.max()) * 1.1
        ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=1, label='理想预测线')
        ax3.set_xlabel('实际销量', fontsize=12)
        ax3.set_ylabel('预测销量', fontsize=12)
        ax3.set_title('预测值 vs 实际值（散点对比）', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_facecolor(self.colors['bg'])

        # 4. 误差分布直方图
        ax4 = fig.add_subplot(gs[1, 1])
        lstm_errors = (lstm_results['predictions'] - lstm_results['targets']).flatten()
        ax4.hist(lstm_errors, bins=30, alpha=0.7, color=self.colors['lstm'],
                 label='LSTM误差', edgecolor='white', linewidth=0.5, density=True)
        if len(arima_results['predictions']) > 0:
            arima_errors = (arima_results['predictions'] - arima_results['targets']).flatten()
            ax4.hist(arima_errors, bins=30, alpha=0.5, color=self.colors['arima'],
                     label='ARIMA误差', edgecolor='white', linewidth=0.5, density=True)
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax4.set_xlabel('预测误差', fontsize=12)
        ax4.set_ylabel('密度', fontsize=12)
        ax4.set_title('预测误差分布对比', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_facecolor(self.colors['bg'])

        plt.suptitle('LSTM vs ARIMA 模型综合对比', fontsize=16, fontweight='bold', y=1.02)
        path = f'{self.save_dir}/model_metrics_comparison.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

    def plot_prediction_samples(self, lstm_preds, lstm_targets, arima_preds=None, arima_targets=None,
                                 forecast_horizon=7, n_samples=6):
        """多样本LSTM vs ARIMA预测对比"""
        n_samples = min(n_samples, len(lstm_preds))
        cols = 2
        rows = (n_samples + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        sample_idx = np.random.choice(len(lstm_preds), n_samples, replace=False)

        for i, idx in enumerate(sample_idx):
            ax = axes[i]
            days = range(1, forecast_horizon + 1)

            ax.plot(days, lstm_targets[idx], 'o-', color=self.colors['actual'],
                    linewidth=2.5, markersize=7, label='实际值', zorder=5)
            ax.plot(days, lstm_preds[idx], 's--', color=self.colors['lstm'],
                    linewidth=2, markersize=6, label='LSTM预测', zorder=4)

            if arima_preds is not None and idx < len(arima_preds):
                ax.plot(days, arima_preds[idx], 'D:', color=self.colors['arima'],
                        linewidth=2, markersize=5, label='ARIMA预测', zorder=3)

            ax.fill_between(days, lstm_targets[idx], lstm_preds[idx],
                            alpha=0.15, color=self.colors['lstm'])

            ax.set_title(f'样本 {i+1} - 未来{forecast_horizon}天销量预测', fontsize=12, fontweight='bold')
            ax.set_xlabel('预测天数', fontsize=10)
            ax.set_ylabel('销量', fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor(self.colors['bg'])
            ax.set_xticks(days)

        # 隐藏多余子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('多商品需求预测效果展示', fontsize=15, fontweight='bold')
        plt.tight_layout()
        path = f'{self.save_dir}/prediction_comparison.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

    def plot_error_by_horizon(self, predictions, targets, forecast_horizon=7):
        """按预测天数的误差变化"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 每天的MAE
        mae_by_day = np.mean(np.abs(predictions - targets), axis=0)
        rmse_by_day = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
        days = range(1, forecast_horizon + 1)

        ax = axes[0]
        ax.bar(days, mae_by_day, color=self.colors['lstm'], edgecolor='white',
               linewidth=1, alpha=0.8)
        ax.plot(days, mae_by_day, 'o-', color='#1565C0', linewidth=2, markersize=6)
        ax.set_xlabel('预测步长（天）', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title('各预测步长的MAE变化', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_facecolor(self.colors['bg'])
        ax.set_xticks(days)

        ax = axes[1]
        ax.bar(days, rmse_by_day, color=self.colors['arima'], edgecolor='white',
               linewidth=1, alpha=0.8)
        ax.plot(days, rmse_by_day, 's-', color='#E65100', linewidth=2, markersize=6)
        ax.set_xlabel('预测步长（天）', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('各预测步长的RMSE变化', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_facecolor(self.colors['bg'])
        ax.set_xticks(days)

        plt.suptitle('LSTM模型预测精度随步长变化趋势', fontsize=15, fontweight='bold')
        plt.tight_layout()
        path = f'{self.save_dir}/error_by_horizon.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

    def plot_category_analysis(self, df):
        """品类销量分析"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 品类销量构成饼图
        if 'category' in df.columns:
            cat_sales = df.groupby('category')['quantity'].sum().nlargest(10)
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(cat_sales)))
            wedges, texts, autotexts = axes[0].pie(
                cat_sales, labels=cat_sales.index, autopct='%1.1f%%',
                colors=colors_pie, startangle=90, pctdistance=0.85,
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
            )
            for t in autotexts:
                t.set_fontsize(9)
            axes[0].set_title('各品类销量占比', fontsize=14, fontweight='bold')

        # 品类日均销量趋势
        if 'category' in df.columns:
            top_cats = df.groupby('category')['quantity'].sum().nlargest(5).index
            cat_daily = df[df['category'].isin(top_cats)].groupby(
                [pd.Grouper(key='date', freq='W'), 'category']
            )['quantity'].sum().reset_index()
            for cat in top_cats:
                cat_data = cat_daily[cat_daily['category'] == cat]
                axes[1].plot(cat_data['date'], cat_data['quantity'],
                            linewidth=1.5, label=str(cat)[:15], alpha=0.8)
            axes[1].set_title('Top品类周销量趋势', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('日期', fontsize=12)
            axes[1].set_ylabel('周销量', fontsize=12)
            axes[1].legend(fontsize=9, loc='upper right')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].set_facecolor(self.colors['bg'])

        plt.tight_layout()
        path = f'{self.save_dir}/category_analysis.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

    def plot_summary_table(self, lstm_results, arima_results):
        """模型对比汇总表"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')

        improvement_mae = (arima_results['mae'] - lstm_results['mae']) / arima_results['mae'] * 100
        improvement_rmse = (arima_results['rmse'] - lstm_results['rmse']) / arima_results['rmse'] * 100

        table_data = [
            ['指标', 'LSTM', 'ARIMA', 'LSTM改进率'],
            ['MAE', f"{lstm_results['mae']:.4f}", f"{arima_results['mae']:.4f}", f"{improvement_mae:.1f}%"],
            ['RMSE', f"{lstm_results['rmse']:.4f}", f"{arima_results['rmse']:.4f}", f"{improvement_rmse:.1f}%"],
        ]

        if 'mape' in arima_results:
            lstm_mape = lstm_results.get('mape', arima_results['mape'])
            improvement_mape = (arima_results['mape'] - lstm_mape) / arima_results['mape'] * 100
            try:
                lstm_mape_str = f"{float(lstm_mape):.1f}%"
            except (ValueError, TypeError):
                lstm_mape_str = str(lstm_mape)
            table_data.append(['MAPE', lstm_mape_str, f"{arima_results['mape']:.1f}%", f"{improvement_mape:.1f}%"])

        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.2, 2.0)

        # 表头样式
        for j in range(4):
            table[0, j].set_facecolor('#2196F3')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # 交替行色
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[i, j].set_facecolor('#E3F2FD')
                else:
                    table[i, j].set_facecolor('#FFFFFF')

        ax.set_title('模型性能对比汇总表', fontsize=16, fontweight='bold', pad=20)
        path = f'{self.save_dir}/summary_table.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")
