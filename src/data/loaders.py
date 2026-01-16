"""
데이터 로더 - yfinance 다운로드
"""
import yfinance as yf
import pandas as pd
from typing import Optional

def download_data(ticker: str, start: str = '2015-01-01', end: str = '2025-01-01') -> Optional[pd.DataFrame]:
    """
    Yahoo Finance에서 데이터 다운로드
    
    Args:
        ticker: 자산 티커 (예: 'GLD', '^VIX')
        start: 시작 날짜
        end: 종료 날짜
    
    Returns:
        DataFrame: OHLCV 데이터 또는 None (실패 시)
    """
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        
        # MultiIndex 컬럼 처리
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return data
    
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None


def download_multiple_assets(tickers: list, start: str = '2015-01-01', end: str = '2025-01-01') -> dict:
    """
    여러 자산 데이터 일괄 다운로드
    
    Args:
        tickers: 티커 리스트
        start: 시작 날짜
        end: 종료 날짜
    
    Returns:
        dict: {ticker: DataFrame} 딕셔너리
    """
    data_dict = {}
    
    for ticker in tickers:
        data = download_data(ticker, start, end)
        if data is not None and len(data) >= 500:
            data_dict[ticker] = data
        else:
            print(f"Warning: {ticker} has insufficient data (len={len(data) if data is not None else 0})")
    
    return data_dict
