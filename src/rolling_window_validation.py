#!/usr/bin/env python3
"""
Rolling Window 검증
====================

다양한 창 크기로 Robustness 검증:
- 250일, 500일, 750일 창
- 50일 간격으로 이동
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def download_data(ticker, start='2015-01-01', end='2025-01-01'):
    """데이터 다운로드"""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None


def prepare_data(ticker):
    """데이터 준비"""
    asset = download_data(ticker)
    vix = download_data('^VIX')
    
    if asset is None or vix is None or len(asset) < 500:
        return None
    
    df = asset[['Close']].copy()
    df.columns = ['Price']
    df['VIX'] = vix['Close'].reindex(df.index).ffill().bfill()
    df['returns'] = df['Price'].pct_change()
    
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['CAVB_target'] = df['VIX'] - df['RV_future']
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    return df.dropna()


def rolling_window_cv(ticker, asset_name, window_sizes=[250, 500, 750], step=50):
    """
    Rolling window cross-validation
    
    Args:
        window_sizes: 학습 데이터 창 크기 (일 단위)
        step: 창 이동 간격
    
    Returns:
        Dict with robustness statistics
    """
    print(f"\n{'='*70}")
    print(f"Rolling Window: {asset_name} ({ticker})")
    print(f"{'='*70}")
    
    df = prepare_data(ticker)
    if df is None:
        print(f"  ✗ 데이터 로드 실패")
        return None
    
    print(f"  전체 데이터: {len(df)} 행")
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
    
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_cavb = df['CAVB_target'].values
    vix_arr = df['VIX'].values
    
    results = {ws: [] for ws in window_sizes}
    gap = 22
    
    for window_size in window_sizes:
        print(f"\n  [Window Size: {window_size}일]")
        
        for start_idx in range(0, len(df) - window_size - gap - 50, step):
            train_end = start_idx + window_size
            test_start = train_end + gap
            test_end = min(test_start + 50, len(df))
            
            if test_end - test_start < 30:
                continue
            
            X_train = X[start_idx:train_end]
            X_test = X[test_start:test_end]
            y_train = y_rv[start_idx:train_end]
            y_test = y_cavb[test_start:test_end]
            vix_test = vix_arr[test_start:test_end]
            
            # 스케일링
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # 모델 학습
            model = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=SEED, max_iter=2000)
            model.fit(X_train_s, y_train)
            
            cavb_pred = vix_test - model.predict(X_test_s)
            r2 = r2_score(y_test, cavb_pred)
            
            results[window_size].append(r2)
        
        scores = results[window_size]
        if scores:
            print(f"    Window 개수: {len(scores)}")
            print(f"    Mean R²: {np.mean(scores):.4f}")
            print(f"    Std R²:  {np.std(scores):.4f}")
            print(f"    Min R²:  {np.min(scores):.4f}")
            print(f"    Max R²:  {np.max(scores):.4f}")
    
    # Summary
    summary = {}
    for ws, scores in results.items():
        if scores:
            summary[ws] = {
                'mean_r2': float(np.mean(scores)),
                'std_r2': float(np.std(scores)),
                'min_r2': float(np.min(scores)),
                'max_r2': float(np.max(scores)),
                'n_windows': len(scores)
            }
    
    return {
        'asset': ticker,
        'asset_name': asset_name,
        'window_results': summary
    }


def main():
    print("\n" + "🔄" * 35)
    print("Rolling Window 검증 (Robustness Check)")
    print("🔄" * 35)
    
    assets = [
        ('EFA', 'EAFE (선진국)'),
        ('GLD', 'Gold (금)'),
        ('SPY', 'S&P 500'),
    ]
    
    all_results = []
    
    for ticker, name in assets:
        result = rolling_window_cv(ticker, name)
        if result:
            all_results.append(result)
    
    # 전체 요약
    print("\n" + "=" * 70)
    print("전체 요약")
    print("=" * 70)
    
    for asset_result in all_results:
        print(f"\n{asset_result['asset_name']}:")
        for ws, stats in asset_result['window_results'].items():
            print(f"  {ws}일 창: Mean R²={stats['mean_r2']:.3f} ± {stats['std_r2']:.3f}, " +
                  f"Range=[{stats['min_r2']:.3f}, {stats['max_r2']:.3f}], N={stats['n_windows']}")
    
    # 저장
    output = {
        'description': 'Rolling window validation with multiple window sizes',
        'window_sizes': [250, 500, 750],
        'step': 50,
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/rolling_window.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/rolling_window.json")


if __name__ == '__main__':
    main()
