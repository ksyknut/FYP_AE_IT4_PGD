import yfinance as yf
import pandas as pd
import os
import time
import requests
from datetime import datetime
from io import StringIO

def get_sp500_tickers():
    """从维基百科获取标普500成分股代码（兼容最新列名'Symbol'）"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 获取网页内容
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]  # 标普500成分股表格
        
        # 打印表格列名（方便调试列名变化）
        print(f"检测到表格列名: {sp500_table.columns.tolist()}")
        
        # 重点：加入'Symbol'作为可能的股票代码列名（根据你的错误信息新增）
        ticker_col = None
        for col in ['Symbol', 'Ticker', 'Ticker Symbol']:  # 优先检查'Symbol'
            if col in sp500_table.columns:
                ticker_col = col
                break
        
        if not ticker_col:
            raise ValueError("未找到股票代码列（可能列名已变更）")
        
        # 提取股票代码并去重（处理可能的空值）
        tickers = sp500_table[ticker_col].dropna().unique().tolist()
        print(f"成功获取 {len(tickers)} 只标普500成分股代码")
        return tickers
    
    except Exception as e:
        print(f"获取成分股列表失败: {e}")
        # 备份列表（前50只，可手动补充完整）
        backup_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'XOM', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'ABBV', 'MRK', 'LLY',
            'PFE', 'BAC', 'KO', 'PEP', 'COST', 'WMT', 'DIS', 'ADBE', 'NFLX', 'CMCSA',
            'CSCO', 'INTC', 'AMD', 'QCOM', 'AMGN', 'AVGO', 'TXN', 'MCD', 'UPS', 'IBM',
            'CAT', 'BA', 'MMM', 'GE', 'WBA', 'T', 'VZ', 'MO', 'PM', 'ABT'
        ]
        print(f"使用备份列表（{len(backup_tickers)}只）")
        return backup_tickers

def download_sp500_data(start_date='2015-01-01', end_date='2025-10-10'):
    """下载标普500指数及成分股数据"""
    tickers = get_sp500_tickers()
    index_ticker = "^GSPC"
    
    # 创建保存目录
    output_dir = "sp500_data"
    os.makedirs(output_dir, exist_ok=True)
    components_dir = os.path.join(output_dir, "components")
    os.makedirs(components_dir, exist_ok=True)
    
    # 下载指数数据
    print(f"\n下载标普500指数数据（{start_date}至{end_date}）...")
    try:
        index_data = yf.download(index_ticker, start=start_date, end=end_date)
        index_data.to_csv(os.path.join(output_dir, "sp500_index.csv"))
        print(f"指数数据已保存至 {output_dir}/sp500_index.csv")
    except Exception as e:
        print(f"指数数据下载失败: {e}")
        index_data = None
    
    # 下载成分股数据
    print(f"\n开始下载 {len(tickers)} 只成分股数据...")
    success_count = 0
    fail_tickers = []
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"({i}/{len(tickers)}) 下载 {ticker}...")
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                stock_data.to_csv(os.path.join(components_dir, f"{ticker}.csv"))
                success_count += 1
            else:
                print(f"警告：{ticker} 无数据")
                fail_tickers.append(ticker)
            if i % 5 == 0:
                time.sleep(1)  # 延迟避免请求限制
        except Exception as e:
            print(f"{ticker} 下载失败: {e}")
            fail_tickers.append(ticker)
    
    print(f"\n下载完成：成功 {success_count} 只，失败 {len(fail_tickers)} 只")
    if fail_tickers:
        print(f"失败的股票代码：{fail_tickers}")
    
    return index_data, components_dir

if __name__ == "__main__":
    index_data, components_path = download_sp500_data(
        start_date='2015-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    print(f"\n所有数据已保存至 {os.path.abspath('sp500_data')}")