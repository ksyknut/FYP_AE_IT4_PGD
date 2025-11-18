import yfinance as yf
import os
from datetime import datetime

def fix_failed_tickers(start_date='2015-01-01', end_date=None):
    """
    补全下载失败的BRK.B和BF.B数据（替换为BRK-B和BF-B格式）
    """
    # 失败的股票代码及修正后的代码
    failed_tickers = {
        'BRK.B': 'BRK-B',   # 伯克希尔·哈撒韦B类股
        'BF.B': 'BF-B'      # 布朗-福曼B类股
    }
    
    # 数据保存目录（与主程序一致）
    components_dir = os.path.join("sp500_data", "components")
    os.makedirs(components_dir, exist_ok=True)  # 确保目录存在
    
    # 若未指定结束日期，默认使用当前日期
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"开始补全失败的股票数据（{start_date}至{end_date}）...")
    success = []
    still_failed = []
    
    for original_ticker, fixed_ticker in failed_tickers.items():
        # 检查目标文件是否已存在（避免重复下载）
        output_path = os.path.join(components_dir, f"{fixed_ticker}.csv")
        if os.path.exists(output_path):
            print(f"{fixed_ticker} 已存在，跳过下载")
            success.append(fixed_ticker)
            continue
        
        try:
            print(f"尝试下载修正后的代码 {fixed_ticker}（原代码 {original_ticker}）...")
            # 下载数据
            stock_data = yf.download(fixed_ticker, start=start_date, end=end_date)
            
            if stock_data.empty:
                print(f"警告：{fixed_ticker} 无数据返回")
                still_failed.append(fixed_ticker)
                continue
            
            # 保存数据
            stock_data.to_csv(output_path)
            print(f"{fixed_ticker} 数据已保存至 {output_path}")
            success.append(fixed_ticker)
        
        except Exception as e:
            print(f"{fixed_ticker} 下载失败: {str(e)}")
            still_failed.append(fixed_ticker)
    
    # 输出结果
    print("\n补全结果：")
    print(f"成功补全：{success}")
    if still_failed:
        print(f"仍失败：{still_failed}（可能需要手动检查）")
    else:
        print("所有失败股票均已补全！")

if __name__ == "__main__":
    # 补全2015-2025年数据（与主程序时间范围一致）
    fix_failed_tickers(start_date='2015-01-01')