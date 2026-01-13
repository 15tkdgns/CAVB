#!/usr/bin/env python3
"""데이터 생성 스크립트"""
import yfinance as yf
import pandas as pd
from pathlib import Path

Path('data/raw').mkdir(parents=True, exist_ok=True)

spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy.to_csv('data/raw/spy_data_2020_2025.csv')
print(f"Created data file with {len(spy)} rows")
