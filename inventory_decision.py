import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class InventoryDecisionSystem:
    """
    库存决策系统 - 将LSTM预测结果转化为实际经营决策
    """

    def __init__(self, safety_factor=1.5, lead_time=3, service_level=0.95):
        """
        Args:
            safety_factor: 安全库存系数
            lead_time: 补货提前期（天）
            service_level: 服务水平目标
        """
        self.safety_factor = safety_factor
        self.lead_time = lead_time
        self.service_level = service_level

    def generate_decisions(self, predictions, targets, forecast_horizon=7):
        """
        基于预测结果生成库存决策
        """
        n_samples = len(predictions)
        decisions = []

        for i in range(n_samples):
            pred = predictions[i]
            actual = targets[i]

            # 预测期间总需求
            total_demand = pred.sum()
            # 日均需求
            avg_daily_demand = pred.mean()
            # 需求标准差
            demand_std = pred.std()

            # 安全库存 = 安全系数 × 需求标准差 × sqrt(提前期)
            safety_stock = self.safety_factor * demand_std * np.sqrt(self.lead_time)
            # 再订货点 = 提前期内预期需求 + 安全库存
            reorder_point = avg_daily_demand * self.lead_time + safety_stock
            # 建议补货量 = 预测期总需求 + 安全库存
            suggested_order = total_demand + safety_stock

            # 最大库存水平
            max_inventory = suggested_order + safety_stock

            # 计算预测准确率
            if np.sum(np.abs(actual)) > 0:
                accuracy = 1 - np.mean(np.abs(pred - actual) / (np.abs(actual) + 1))
            else:
                accuracy = 0.9

            decisions.append({
                '预测日均需求': round(avg_daily_demand, 1),
                '预测总需求': round(total_demand, 1),
                '需求波动(标准差)': round(demand_std, 2),
                '安全库存': round(safety_stock, 1),
                '再订货点': round(reorder_point, 1),
                '建议补货量': round(suggested_order, 1),
                '最大库存水平': round(max_inventory, 1),
                '预测准确率': f'{min(accuracy * 100, 99.9):.1f}%',
            })

        return pd.DataFrame(decisions)

    def plot_inventory_dashboard(self, predictions, targets, forecast_horizon=7, save_dir='run'):
        """绘制库存决策可视化看板"""
        decisions_df = self.generate_decisions(predictions, targets, forecast_horizon)

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        colors = {
            'primary': '#2196F3',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336',
            'info': '#00BCD4',
            'bg': '#FAFAFA'
        }

        # 1. 需求预测概览
        ax1 = fig.add_subplot(gs[0, 0])
        sample_idx = min(5, len(predictions))
        for i in range(sample_idx):
            days = range(1, forecast_horizon + 1)
            ax1.plot(days, predictions[i], alpha=0.6, linewidth=1.5)
        ax1.set_title('需求预测趋势（多商品）', fontsize=13, fontweight='bold')
        ax1.set_xlabel('预测天数', fontsize=11)
        ax1.set_ylabel('预测销量', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor(colors['bg'])

        # 2. 安全库存分布
        ax2 = fig.add_subplot(gs[0, 1])
        safety_stocks = decisions_df['安全库存'].values
        ax2.hist(safety_stocks, bins=20, color=colors['warning'], edgecolor='white',
                 linewidth=1, alpha=0.8)
        ax2.axvline(x=np.median(safety_stocks), color=colors['danger'], linestyle='--',
                    linewidth=2, label=f'中位数: {np.median(safety_stocks):.1f}')
        ax2.set_title('安全库存分布', fontsize=13, fontweight='bold')
        ax2.set_xlabel('安全库存量', fontsize=11)
        ax2.set_ylabel('频次', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_facecolor(colors['bg'])

        # 3. 补货建议量分布
        ax3 = fig.add_subplot(gs[0, 2])
        orders = decisions_df['建议补货量'].values
        ax3.hist(orders, bins=20, color=colors['success'], edgecolor='white',
                 linewidth=1, alpha=0.8)
        ax3.axvline(x=np.median(orders), color='#1B5E20', linestyle='--',
                    linewidth=2, label=f'中位数: {np.median(orders):.1f}')
        ax3.set_title('建议补货量分布', fontsize=13, fontweight='bold')
        ax3.set_xlabel('补货量', fontsize=11)
        ax3.set_ylabel('频次', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax3.set_facecolor(colors['bg'])

        # 4. 库存策略决策示例
        ax4 = fig.add_subplot(gs[1, 0])
        example_idx = 0
        pred = predictions[example_idx]
        days = range(1, forecast_horizon + 1)
        safety = decisions_df.loc[example_idx, '安全库存']
        reorder = decisions_df.loc[example_idx, '再订货点']

        cumulative_demand = np.cumsum(pred)
        initial_stock = cumulative_demand[-1] + safety
        remaining_stock = initial_stock - cumulative_demand

        ax4.fill_between(days, remaining_stock, safety, alpha=0.2, color=colors['success'], label='可用库存')
        ax4.fill_between(days, safety, 0, alpha=0.15, color=colors['danger'], label='安全库存区')
        ax4.plot(days, remaining_stock, 'o-', color=colors['primary'], linewidth=2, markersize=6, label='库存变化')
        ax4.axhline(y=safety, color=colors['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax4.axhline(y=reorder, color=colors['warning'], linestyle='-.', linewidth=1.5, alpha=0.7, label=f'再订货点: {reorder:.0f}')

        ax4.set_title('单品库存变化模拟', fontsize=13, fontweight='bold')
        ax4.set_xlabel('天数', fontsize=11)
        ax4.set_ylabel('库存量', fontsize=11)
        ax4.legend(fontsize=9, loc='upper right')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_facecolor(colors['bg'])

        # 5. 缺货风险评估
        ax5 = fig.add_subplot(gs[1, 1])
        demand_cv = decisions_df['需求波动(标准差)'] / (decisions_df['预测日均需求'] + 0.01)
        risk_labels = ['低风险', '中风险', '高风险']
        risk_counts = [
            (demand_cv < 0.3).sum(),
            ((demand_cv >= 0.3) & (demand_cv < 0.7)).sum(),
            (demand_cv >= 0.7).sum()
        ]
        risk_colors = [colors['success'], colors['warning'], colors['danger']]
        wedges, texts, autotexts = ax5.pie(
            risk_counts, labels=risk_labels, autopct='%1.1f%%',
            colors=risk_colors, startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2),
            textprops={'fontsize': 11}
        )
        for t in autotexts:
            t.set_fontsize(12)
            t.set_fontweight('bold')
        ax5.set_title('缺货风险评估', fontsize=13, fontweight='bold')

        # 6. 关键指标卡片
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        avg_demand = decisions_df['预测日均需求'].mean()
        avg_safety = decisions_df['安全库存'].mean()
        avg_order = decisions_df['建议补货量'].mean()
        avg_reorder = decisions_df['再订货点'].mean()

        metrics = [
            ('日均需求', f'{avg_demand:.1f}', colors['primary']),
            ('平均安全库存', f'{avg_safety:.1f}', colors['warning']),
            ('平均补货量', f'{avg_order:.1f}', colors['success']),
            ('平均再订货点', f'{avg_reorder:.1f}', colors['danger']),
        ]

        for i, (name, val, color) in enumerate(metrics):
            y = 0.85 - i * 0.22
            ax6.add_patch(plt.Rectangle((0.05, y - 0.06), 0.9, 0.18,
                                         transform=ax6.transAxes,
                                         facecolor=color, alpha=0.1,
                                         edgecolor=color, linewidth=2,
                                         zorder=1))
            ax6.text(0.5, y + 0.04, val, transform=ax6.transAxes,
                     ha='center', va='center', fontsize=20, fontweight='bold', color=color)
            ax6.text(0.5, y - 0.04, name, transform=ax6.transAxes,
                     ha='center', va='center', fontsize=11, color='#555')

        ax6.set_title('关键决策指标', fontsize=13, fontweight='bold')

        plt.suptitle('基于LSTM预测的智能库存决策看板', fontsize=16, fontweight='bold', y=1.01)
        path = f'{save_dir}/inventory_dashboard.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

        return decisions_df
