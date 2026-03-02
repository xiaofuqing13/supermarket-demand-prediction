import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# PyTorch库
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class DataPreprocessor:
    def __init__(self, active_sku_num):
        self.sku_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.subcategory_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()
        self.scaler_dict = {}  # 为每个特征保存单独的scaler
        self.active_sku_num = active_sku_num

    def load_excel_data(self, file_path):
        """
        从Excel文件加载所有sheet的数据
        """
        print(f"正在从 {file_path} 加载数据...")

        # 获取所有sheet名称
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names

        print(f"找到sheet: {sheet_names}")

        # 交易数据 - 合并所有月份的sheet
        trans_dfs = []
        for sheet in sheet_names:
            # 假设交易数据的sheet是月份名称（如January, February等）
            df = pd.read_excel(file_path, sheet_name=sheet)
            trans_dfs.append(df)

        trans = pd.concat(trans_dfs, ignore_index=True)
        print(f"交易数据合并完成，共 {len(trans)} 条记录")

        return trans

    def prepare_data(self, excel_file_path, inventory_file, replenishment_file,
                     product_file, display_file):
        """
        主数据处理函数
        """
        print("开始数据预处理...")

        # 1. 加载交易数据
        trans = self.load_excel_data(excel_file_path)

        # 加载其他Excel文件（假设都是单个sheet）
        inventory = pd.read_excel(inventory_file)
        replenishment = pd.read_excel(replenishment_file)
        product = pd.read_excel(product_file)
        display = pd.read_excel(display_file)

        # 转换日期列
        trans['date'] = pd.to_datetime(trans['date'], format='%d/%m/%Y', errors='coerce')
        inventory['date'] = pd.to_datetime(inventory['date'], format='%m/%Y', errors='coerce')
        display['date'] = pd.to_datetime(display['date'], format='%m/%Y', errors='coerce')
        replenishment['order_date'] = pd.to_datetime(replenishment['order_date'], format='%m/%d/%Y', errors='coerce')
        replenishment['arrival_date'] = pd.to_datetime(replenishment['arrival_date'], format='%m/%d/%Y',
                                                       errors='coerce')

        # 2.1 处理交易数据 - 按天汇总
        trans_daily = trans.groupby(['date', 'sku_ID']).agg({
            'quantity': 'sum',
            'original_unit_price': 'mean',
            'sales_revenue': 'sum',
            'category': 'first',
            'subcategory': 'first',
            'brand_ID': 'first'
        }).reset_index()

        print(f"交易数据按天汇总后: {len(trans_daily)} 条记录")

        # 2.2 处理产品数据
        product['stop_year'] = product['stop_year'].astype(str)
        product['stop_date'] = pd.to_datetime(product['stop_year'].astype(str) + '-12-31', errors='coerce')
        product['introduction_date'] = pd.to_datetime(product['introduction_year'].astype(str) + '-01-01',
                                                      errors='coerce')

        # 2.3 生成完整的日期范围
        all_dates = pd.date_range(
            start=trans_daily['date'].min(),
            end=trans_daily['date'].max(),
            freq='D'
        )

        # 2.4 获取所有活跃的SKU
        active_skus = product[pd.isna(product['stop_date'])]['sku_ID'].unique()[:self.active_sku_num]
        print(f"活跃SKU数量: {len(active_skus)}")

        # 创建完整的(SKU, Date)组合
        full_index = pd.MultiIndex.from_product(
            [active_skus, all_dates],
            names=['sku_ID', 'date']
        )
        full_df = pd.DataFrame(index=full_index).reset_index()

        # 2.5 合并交易数据
        data = full_df.merge(trans_daily, on=['sku_ID', 'date'], how='left')

        # 填充缺失值
        data['quantity'] = data['quantity'].fillna(0)
        data['sales_revenue'] = data['sales_revenue'].fillna(0)

        # 价格向前填充
        data = data.sort_values(['sku_ID', 'date'])
        data['original_unit_price'] = data.groupby('sku_ID')['original_unit_price'].ffill()

        # 2.6 合并产品信息
        data = data.merge(product[['sku_ID', 'category', 'subcategory', 'brand_ID',
                                   'introduction_year', 'operation_mode']], on='sku_ID', how='left')

        # 2.7 处理库存数据（月度扩展到每日）
        inventory['year'] = inventory['date'].dt.year
        inventory['month'] = inventory['date'].dt.month

        inventory_expanded = []
        for sku in tqdm(active_skus, desc="处理活跃的SKUS库存数据（月度扩展到每日）"):
            sku_inv = inventory[inventory['sku_ID'] == sku]
            if len(sku_inv) > 0:
                for _, row in sku_inv.iterrows():
                    month_dates = pd.date_range(
                        start=row['date'],
                        periods=pd.Timestamp(row['date']).days_in_month,
                        freq='D'
                    )
                    for d in month_dates:
                        inventory_expanded.append({
                            'sku_ID': sku,
                            'date': d,
                            'beginning_inventory': row['beginning_inventory'],
                            'on_order_inventory': row['on-order_inventory'],
                            'stock_value': row['stock_value']
                        })

        inv_daily = pd.DataFrame(inventory_expanded) if inventory_expanded else pd.DataFrame()

        if not inv_daily.empty:
            data = data.merge(inv_daily, on=['sku_ID', 'date'], how='left')

        # 2.8 处理展示数据
        display['year'] = display['date'].dt.year
        display['month'] = display['date'].dt.month

        display_expanded = []
        for sku in tqdm(active_skus, desc="处理活跃的SKUS库存数据（月度扩展到每日）"):  # 先测试前10个SKU
            sku_disp = display[display['sku_ID'] == sku]
            if len(sku_disp) > 0:
                for _, row in sku_disp.iterrows():
                    month_dates = pd.date_range(
                        start=row['date'],
                        periods=pd.Timestamp(row['date']).days_in_month,
                        freq='D'
                    )
                    for d in month_dates:
                        display_expanded.append({
                            'sku_ID': sku,
                            'date': d,
                            'facing_number': row['facing_number'],
                            'shelf_capacity': row['shelf_capacity']
                        })

        disp_daily = pd.DataFrame(display_expanded) if display_expanded else pd.DataFrame()

        if not disp_daily.empty:
            data = data.merge(disp_daily, on=['sku_ID', 'date'], how='left')

        # 填充缺失的数值列为0
        numeric_cols = ['beginning_inventory', 'on_order_inventory', 'stock_value',
                        'facing_number', 'shelf_capacity']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = data[col].fillna(0)

        # 2.9 特征工程
        data = self._create_features(data)

        print(f"数据预处理完成！最终数据形状: {data.shape}")
        return data

    def _create_features(self, df):
        """
        创建各种特征
        """
        print("开始特征工程...")

        df = df.sort_values(['sku_ID', 'date'])

        # 时间特征
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # 历史销量特征
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}_sales'] = df.groupby('sku_ID')['quantity'].shift(lag)

        # 滚动窗口统计
        for window in [7, 14]:
            df[f'rolling_mean_{window}'] = df.groupby('sku_ID')['quantity'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby('sku_ID')['quantity'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )

        # ========== 协同特征：互补性与替代性 ==========
        print("  计算商品间互补性与替代性特征...")
        # 构建SKU日销量透视表
        pivot = df.pivot_table(index='date', columns='sku_ID', values='quantity', aggfunc='sum', fill_value=0)
        # 计算SKU间的相关系数矩阵
        corr_matrix = pivot.corr()
        self.corr_matrix = corr_matrix  # 保存供可视化使用

        # 为每个SKU找到互补商品（正相关top-3）和替代商品（负相关top-3）
        sku_list = corr_matrix.columns.tolist()
        complement_dict = {}  # 互补商品
        substitute_dict = {}  # 替代商品
        for sku in sku_list:
            corr_vals = corr_matrix[sku].drop(sku)
            # 互补：正相关最高的3个
            top_pos = corr_vals.nlargest(3).index.tolist()
            complement_dict[sku] = top_pos
            # 替代：负相关最强的3个（相关系数最小的）
            top_neg = corr_vals.nsmallest(3).index.tolist()
            substitute_dict[sku] = top_neg

        self.complement_dict = complement_dict
        self.substitute_dict = substitute_dict

        # 计算互补商品聚合销量和替代商品聚合销量
        complement_sales = []
        substitute_sales = []
        for _, row in df.iterrows():
            sku = row['sku_ID']
            date = row['date']
            # 互补销量
            comp_skus = complement_dict.get(sku, [])
            comp_val = pivot.loc[date, comp_skus].sum() if date in pivot.index and len(comp_skus) > 0 else 0
            complement_sales.append(comp_val)
            # 替代销量
            sub_skus = substitute_dict.get(sku, [])
            sub_val = pivot.loc[date, sub_skus].sum() if date in pivot.index and len(sub_skus) > 0 else 0
            substitute_sales.append(sub_val)

        df['complement_sales'] = complement_sales
        df['substitute_sales'] = substitute_sales

        # 互补性强度指标（与互补商品的平均相关系数）
        df['complement_strength'] = df['sku_ID'].map(
            lambda sku: corr_matrix.loc[sku, complement_dict.get(sku, [])].mean()
            if sku in corr_matrix.index and len(complement_dict.get(sku, [])) > 0 else 0
        )
        # 替代性强度指标
        df['substitute_strength'] = df['sku_ID'].map(
            lambda sku: corr_matrix.loc[sku, substitute_dict.get(sku, [])].mean()
            if sku in corr_matrix.index and len(substitute_dict.get(sku, [])) > 0 else 0
        )

        # 跨品类协同指数：同品类全体销量
        if 'category' in df.columns:
            cat_sales = df.groupby(['date', 'category'])['quantity'].sum().reset_index()
            cat_sales.columns = ['date', 'category', 'category_total_sales']
            df = df.merge(cat_sales, on=['date', 'category'], how='left')
        if 'subcategory' in df.columns:
            subcat_sales = df.groupby(['date', 'subcategory'])['quantity'].sum().reset_index()
            subcat_sales.columns = ['date', 'subcategory', 'subcat_total_sales']
            df = df.merge(subcat_sales, on=['date', 'subcategory'], how='left')
            df['subcat_sales_share'] = df['quantity'] / (df['subcat_total_sales'] + 1)

        print(f"  互补性/替代性协同特征构建完成")

        # 库存相关特征
        if 'beginning_inventory' in df.columns:
            df['inventory_level'] = df['beginning_inventory']
            df['low_inventory_flag'] = (df['beginning_inventory'] < df['rolling_mean_7'] * 2).astype(int)

        # 价格特征
        if 'original_unit_price' in df.columns:
            df['price_change'] = df.groupby('sku_ID')['original_unit_price'].pct_change()

        # 商品上市时长
        if 'introduction_year' in df.columns:
            df['product_age'] = df['date'].dt.year - df['introduction_year']

        # 填充NaN值
        df = df.fillna(0)

        # 编码分类变量
        df['sku_encoded'] = self.sku_encoder.fit_transform(df['sku_ID'])

        if 'category' in df.columns:
            df['category_encoded'] = self.category_encoder.fit_transform(df['category'].astype(str))
        if 'subcategory' in df.columns:
            df['subcategory_encoded'] = self.subcategory_encoder.fit_transform(df['subcategory'].astype(str))
        if 'brand_ID' in df.columns:
            df['brand_encoded'] = self.brand_encoder.fit_transform(df['brand_ID'].fillna('unknown').astype(str))

        print(f"特征工程完成，共有 {df.shape[1]} 个特征")
        return df


class TimeSeriesDataset(Dataset):
    """
    PyTorch数据集类
    """

    def __init__(self, X, y, sku_ids=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sku_ids = torch.LongTensor(sku_ids) if sku_ids is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.sku_ids is not None:
            return self.X[idx], self.y[idx], self.sku_ids[idx]
        return self.X[idx], self.y[idx]


def prepare_sequences(df, sequence_length=30, forecast_horizon=7, feature_cols=None):
    """
    为模型准备序列数据
    """
    if feature_cols is None:
        feature_cols = ['quantity', 'original_unit_price', 'rolling_mean_7',
                        'rolling_std_7', 'is_weekend', 'month', 'dayofweek',
                        'lag_1_sales', 'lag_2_sales', 'lag_7_sales']

    # 只保留存在的列
    feature_cols = [col for col in feature_cols if col in df.columns]

    print("开始创建序列数据...")

    X_list, y_list, sku_list = [], [], []
    skus = df['sku_ID'].unique()

    for sku in tqdm(skus, desc='处理SKU', unit="sku"):
        sku_data = df[df['sku_ID'] == sku].sort_values('date')

        if len(sku_data) <= sequence_length + forecast_horizon:
            continue

        # 获取特征数据
        sku_features = sku_data[feature_cols].values

        # 归一化
        scaler = MinMaxScaler()
        sku_scaled = scaler.fit_transform(sku_features)

        # 创建序列
        for i in range(len(sku_scaled) - sequence_length - forecast_horizon + 1):
            X_list.append(sku_scaled[i:i + sequence_length])
            y_list.append(sku_data['quantity'].values[i + sequence_length:i + sequence_length + forecast_horizon])
            sku_list.append(sku_data['sku_encoded'].iloc[i])

    if X_list:
        return np.array(X_list), np.array(y_list), np.array(sku_list)
    else:
        return None, None, None