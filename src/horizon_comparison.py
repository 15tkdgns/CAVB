#!/usr/bin/env python3
"""
5일 vs 22일 예측 비교
=====================

22일 예측의 낮은 R²를 개선하기 위해 5일 예측과 비교
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
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


def prepare_data_multi_horizon(ticker):
    """여러 예측 기간의 데이터 준비"""
    asset = download_data(ticker)
    vix = download_data('^VIX')
    
    if asset is None or vix is None or len(asset) < 500:
        return None
    
    df = asset[['Close']].copy()
    df.columns = ['Price']
    df['VIX'] = vix['Close'].reindex(df.index).ffill().bfill()
    df['returns'] = df['Price'].pct_change()
    
    # 변동성
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    
    # CAVB
    df['CAVB'] = df['VIX'] - df['RV_22d']
    
    # **다양한 예측 기간의 타겟**
    df['RV_future_5d'] = df['RV_22d'].shift(-5)   # 5일 후
    df['RV_future_22d'] = df['RV_22d'].shift(-22) # 22일 후
    
    df['CAVB_target_5d'] = df['VIX'] - df['RV_future_5d']
    df['CAVB_target_22d'] = df['VIX'] - df['RV_future_22d']
    
    # 특성
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    return df.dropna()


def compare_horizons(ticker, asset_name):
    """5일 vs 22일 예측 비교"""
    print(f"\n{'='*70}")
    print(f"예측 기간 비교: {asset_name} ({ticker})")
    print(f"{'='*70}")
    
    df = prepare_data_multi_horizon(ticker)
    if df is None:
        print(f"  ✗ 데이터 로드 실패")
        return None
    
    print(f"  데이터: {len(df)} 행")
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
    
    X = df[feature_cols].values
    vix_arr = df['VIX'].values
    
    results = {}
    
    # 5일 예측 vs 22일 예측
    for horizon, gap in [('5d', 5), ('22d', 22)]:
        print(f"\n  [{horizon} 예측 (Gap={gap}일)]")
        
        y_rv_col = f'RV_future_{horizon}'
        y_cavb_col = f'CAVB_target_{horizon}'
        
        y_rv = df[y_rv_col].values
        y_cavb = df[y_cavb_col].values
        
        # 3-Way Split
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train = X[:train_end]
        X_val = X[train_end+gap:val_end]
        X_test = X[val_end+gap:]
        
        y_train = y_rv[:train_end]
        y_val_cavb = y_cavb[train_end+gap:val_end]
        y_test_cavb = y_cavb[val_end+gap:]
        
        vix_val = vix_arr[train_end+gap:val_end]
        vix_test = vix_arr[val_end+gap:]
        
        print(f"    Split: Train={len(X_train)} / Val={len(X_val)} / Test={len(X_test)}")
        
        # 스케일링
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        # ElasticNet 학습
        model = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=SEED, max_iter=2000)
        model.fit(X_train_s, y_train)
        
        # Validation 성능
        cavb_pred_val = vix_val - model.predict(X_val_s)
        r2_val = r2_score(y_val_cavb, cavb_pred_val)
        
        # Test 성능
        cavb_pred_test = vix_test - model.predict(X_test_s)
        r2_test = r2_score(y_test_cavb, cavb_pred_test)
        mae_test = mean_absolute_error(y_test_cavb, cavb_pred_test)
        
        # Naive (Persistence)
        cavb_naive = df['CAVB_lag1'].values[val_end+gap:]
        r2_naive = r2_score(y_test_cavb, cavb_naive)
        
        improvement = r2_test - r2_naive
        
        print(f"    Val R²:        {r2_val:.4f}")
        print(f"    Test R²:       {r2_test:.4f}")
        print(f"    Test MAE:      {mae_test:.2f}")
        print(f"    Naive R²:      {r2_naive:.4f}")
        print(f"    Improvement:   {improvement:+.4f}")
        
        results[horizon] = {
            'horizon': horizon,
            'gap_days': gap,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'r2_val': float(r2_val),
            'r2_test': float(r2_test),
            'r2_naive': float(r2_naive),
            'mae_test': float(mae_test),
            'improvement': float(improvement)
        }
    
    # 비교 요약
    print(f"\n  [비교 요약]")
    print(f"    5일 예측:  Test R² = {results['5d']['r2_test']:.4f}")
    print(f"    22일 예측: Test R² = {results['22d']['r2_test']:.4f}")
    diff = results['5d']['r2_test'] - results['22d']['r2_test']
    print(f"    차이:      {diff:+.4f} ({'5일이 우수' if diff > 0 else '22일이 우수'})")
    
    return {
        'asset': ticker,
        'asset_name': asset_name,
        'horizons': results
    }


def main():
    print("\n" + "⏱️" * 35)
    print("예측 기간 비교: 5일 vs 22일")
    print("⏱️" * 35)
    
    assets = [
        ('EFA', 'EAFE (선진국)'),
        ('TLT', 'Treasury (국채)'),
        ('GLD', 'Gold (금)'),
        ('SPY', 'S&P 500'),
        ('EEM', 'Emerging (신흥국)'),
    ]
    
    all_results = []
    
    for ticker, name in assets:
        result = compare_horizons(ticker, name)
        if result:
            all_results.append(result)
    
    # 전체 요약
    print("\n" + "=" * 70)
    print("전체 자산 비교")
    print("=" * 70)
    
    summary_data = []
    for asset_result in all_results:
        h5 = asset_result['horizons']['5d']
        h22 = asset_result['horizons']['22d']
        diff = h5['r2_test'] - h22['r2_test']
        
        summary_data.append({
            'Asset': asset_result['asset_name'],
            '5d R²': f"{h5['r2_test']:.3f}",
            '22d R²': f"{h22['r2_test']:.3f}",
            'Diff': f"{diff:+.3f}",
            'Better': '5d' if diff > 0 else '22d'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(f"\n{summary_df.to_string(index=False)}")
    
    # 통계 분석
    diffs = [float(row['Diff']) for row in summary_data]
    print(f"\n평균 차이: {np.mean(diffs):+.3f}")
    print(f"5일이 우수한 자산: {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
    
    # 저장
    output = {
        'description': 'Comparison of 5-day vs 22-day volatility prediction',
        'methodology': '3-Way Split (60/20/20), ElasticNet',
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/horizon_comparison.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/horizon_comparison.json")


if __name__ == '__main__':
    main()
