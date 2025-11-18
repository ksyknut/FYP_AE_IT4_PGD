import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# 设置路径
index_path = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\sp500_index.csv"
components_dir = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\components"

# 时间边界
earliest_date = pd.to_datetime("2015-01-02")
latest_date = pd.to_datetime("2025-10-10")

def load_and_clean_data(file_path):
    """
    加载CSV文件并清理数据
    数据格式：前两行是元数据，第3行开始是实际数据
    """
    df = pd.read_csv(file_path)
    
    # 跳过前两行（元数据行）
    df = df.iloc[2:].reset_index(drop=True)
    
    # 重命名列
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # 转换数据类型
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # 按日期排序
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

print("=" * 80)
print("读取指数数据...")
print("=" * 80)

# 读取指数数据
index_df = load_and_clean_data(index_path)

print(f"指数数据shape: {index_df.shape}")
print(f"指数数据时间范围: {index_df['Date'].min()} 到 {index_df['Date'].max()}")

# 计算指数收益
index_df['index_return'] = index_df['Close'].pct_change()
index_df = index_df[['Date', 'index_return']].copy()
index_df = index_df.rename(columns={'index_return': 'index'})

print(f"指数收益数据shape: {index_df.shape}")

print("\n" + "=" * 80)
print("读取成分股数据...")
print("=" * 80)

# 获取所有成分股CSV文件
stock_files = [f for f in os.listdir(components_dir) if f.endswith('.csv')]
print(f"找到 {len(stock_files)} 个成分股文件")

# 存储所有股票数据
all_stocks_data = {}
valid_stocks = []
filtered_stocks = []

for i, stock_file in enumerate(stock_files):
    stock_symbol = stock_file.replace('.csv', '')
    stock_path = os.path.join(components_dir, stock_file)
    
    try:
        stock_df = load_and_clean_data(stock_path)
        
        # 获取股票的上市和退市日期
        first_date = stock_df['Date'].min()
        last_date = stock_df['Date'].max()
        
        # 筛选条件：
        # - 上市日期 <= 2015-01-02（2015-01-02之前或在该日上市）
        # - 退市日期 >= 2025-10-10（2025-10-10之后或在该日退市）
        if first_date <= earliest_date and last_date >= latest_date:
            
            # 计算收益率
            stock_df['return'] = stock_df['Close'].pct_change()
            stock_df = stock_df[['Date', 'return']].copy()
            stock_df = stock_df.rename(columns={'return': stock_symbol})
            
            all_stocks_data[stock_symbol] = stock_df
            valid_stocks.append(stock_symbol)
        else:
            reason = []
            if first_date > earliest_date:
                reason.append(f"IPO晚于2015-01-02 ({first_date.date()})")
            if last_date < latest_date:
                reason.append(f"退市早于2025-10-10 ({last_date.date()})")
            
            filtered_stocks.append({
                'symbol': stock_symbol,
                'first_date': first_date,
                'last_date': last_date,
                'reason': ' & '.join(reason)
            })
        
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(stock_files)} 文件，有效股票: {len(valid_stocks)}, 已过滤: {len(filtered_stocks)}")
    
    except Exception as e:
        print(f"处理 {stock_symbol} 出错: {str(e)}")
        filtered_stocks.append({'symbol': stock_symbol, 'first_date': None, 'last_date': None, 'reason': f'读取错误: {str(e)}'})
        continue

print(f"\n总共有效股票数: {len(valid_stocks)}")
print(f"已过滤股票数: {len(filtered_stocks)}")
print(f"\n有效股票列表（前30个）: {sorted(valid_stocks)[:30]}")

print("\n" + "=" * 80)
print("合并数据...")
print("=" * 80)

# 获取所有日期的并集（不是交集）
all_dates = set(index_df['Date'])
for stock_symbol in valid_stocks:
    all_dates = all_dates.union(set(all_stocks_data[stock_symbol]['Date']))

all_dates = sorted(list(all_dates))
print(f"总共日期数: {len(all_dates)}")
print(f"日期范围: {all_dates[0]} 到 {all_dates[-1]}")

# 创建基础DataFrame（包含所有日期）
master_df = pd.DataFrame({'Date': all_dates})

# 合并指数数据
index_merged = index_df.copy()
master_df = master_df.merge(index_merged, on='Date', how='left')

print(f"合并指数后shape: {master_df.shape}")

# 合并股票数据
for j, stock_symbol in enumerate(sorted(valid_stocks)):
    stock_data = all_stocks_data[stock_symbol].copy()
    master_df = master_df.merge(stock_data, on='Date', how='left')
    
    if (j + 1) % 50 == 0:
        print(f"已合并 {j + 1}/{len(valid_stocks)} 股票，当前shape: {master_df.shape}")

print(f"\n合并后数据shape: {master_df.shape}")
print(f"数据列数: {len(master_df.columns)}")
print(f"数据行数: {len(master_df)}")
print(f"股票数: {len(valid_stocks)}")

print("\n" + "=" * 80)
print("处理缺失值...")
print("=" * 80)

# 统计缺失值
missing_before = master_df.isnull().sum().sum()
print(f"处理前缺失值总数: {missing_before}")

if missing_before > 0:
    # 用上下两个值的平均值填充缺失值
    for col in master_df.columns:
        if col != 'Date':
            # 迭代填充缺失值
            max_iterations = 10
            iteration = 0
            
            while master_df[col].isnull().any() and iteration < max_iterations:
                mask = master_df[col].isnull()
                indices = master_df[mask].index.tolist()
                
                for idx in indices:
                    if idx > 0 and idx < len(master_df) - 1:
                        upper_val = master_df.loc[idx - 1, col]
                        lower_val = master_df.loc[idx + 1, col]
                        
                        if pd.notna(upper_val) and pd.notna(lower_val):
                            master_df.loc[idx, col] = (upper_val + lower_val) / 2
                        elif pd.notna(upper_val):
                            master_df.loc[idx, col] = upper_val
                        elif pd.notna(lower_val):
                            master_df.loc[idx, col] = lower_val
                    elif idx == 0 and idx < len(master_df) - 1:
                        lower_val = master_df.loc[idx + 1, col]
                        if pd.notna(lower_val):
                            master_df.loc[idx, col] = lower_val
                    elif idx == len(master_df) - 1 and idx > 0:
                        upper_val = master_df.loc[idx - 1, col]
                        if pd.notna(upper_val):
                            master_df.loc[idx, col] = upper_val
                
                iteration += 1
            
            # 如果还有缺失值但无法填充，则使用前向填充
            if master_df[col].isnull().any():
                master_df[col] = master_df[col].fillna(method='ffill').fillna(method='bfill')

missing_after = master_df.isnull().sum().sum()
print(f"处理后缺失值总数: {missing_after}")

# 设置Date为索引
master_df = master_df.set_index('Date')

print("\n" + "=" * 80)
print("最终矩阵信息")
print("=" * 80)
print(f"矩阵shape: {master_df.shape}")
print(f"行数（日期）: {master_df.shape[0]}")
print(f"列数（股票+指数）: {master_df.shape[1]}")
print(f"股票数: {master_df.shape[1] - 1}")

print(f"\n列名（前20列）: {master_df.columns.tolist()[:20]}")
print(f"最后5列: {master_df.columns.tolist()[-5:]}")

print(f"\n数据统计:")
print(f"日期范围: {master_df.index.min()} 到 {master_df.index.max()}")

print(f"\n前5行:")
print(master_df.head())

print(f"\n后5行:")
print(master_df.tail())

print(f"\n数据描述统计（前10列）:")
print(master_df.iloc[:, :10].describe())

# 保存结果
output_path = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\matrix_data.csv"
master_df.to_csv(output_path)
print(f"\n✓ 矩阵已保存到: {output_path}")

# 保存列名信息
column_info_path = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\column_info.txt"
with open(column_info_path, 'w') as f:
    f.write("列名顺序（矩阵的X列和Y）:\n")
    f.write("=" * 80 + "\n")
    for i, col in enumerate(master_df.columns):
        if col == 'index':
            f.write(f"{i}: {col} [Y - 指数表现]\n")
        else:
            f.write(f"{i}: {col} [X - 股票收益]\n")

print(f"✓ 列名信息已保存到: {column_info_path}")

# 保存被过滤的股票信息
if filtered_stocks:
    filtered_path = r"F:\University\FYP\FYP_AE_IT4_PGD\sp500_data\filtered_stocks.txt"
    with open(filtered_path, 'w') as f:
        f.write(f"被过滤的股票总数: {len(filtered_stocks)}\n")
        f.write("=" * 80 + "\n")
        for stock_info in sorted(filtered_stocks, key=lambda x: x['symbol']):
            if stock_info['first_date'] is not None:
                f.write(f"{stock_info['symbol']}: IPO={stock_info['first_date'].date()}, 退市={stock_info['last_date'].date()}, 原因={stock_info['reason']}\n")
            else:
                f.write(f"{stock_info['symbol']}: 原因={stock_info['reason']}\n")
    
    print(f"✓ 被过滤股票信息已保存到: {filtered_path}")

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)