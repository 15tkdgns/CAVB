#!/usr/bin/env python3
"""
Subperiod 분석
===============

위기/정상기별 성능 비교:
- Pre-COVID (2015-2019)
- COVID (2020-2020)
- Post-COVID (2021-2025)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)

# 분석 기간 정의
PERIODS = {
    'Pre-COVID': ('2015-01-01', '2019-12-31'),
    'COVID': ('2020-01-01', '2020-12-31'),
    'Post-COVID': ('2021-01-01', '2024-12-31')
}


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
    
    # 변동성
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    
    # CAVB
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['CAVB_target'] = df['VIX'] - df['RV_future']
    
    # 특성
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    return df.dropna()


def subperiod_performance(ticker, asset_name, periods=PERIODS):
    """
    기간별 성능 비교
    
    Returns:
        DataFrame with period-specific performance + t-test vs Naive
    """
    print(f"\n{'='*70}")
    print(f"Subperiod 분석: {asset_name} ({ticker})")
    print(f"{'='*70}")
    
    df = prepare_data(ticker)
    if df is None:
        print(f"  ✗ 데이터 로드 실패")
        return None
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
    
    results = []
    
    for period_name, (start, end) in periods.items():
        print(f"\n  [{period_name}] {start} ~ {end}")
        
        # 기간 필터링
        mask = (df.index >= start) & (df.index <= end)
        df_period = df[mask]
        
        if len(df_period) < 100:
            print(f"    ✗ 데이터 부족: {len(df_period)} 행")
            continue
        
        print(f"    데이터: {len(df_period)} 행")
        
        # Train/Test 분할 (80/20)
        split = int(len(df_period) * 0.8)
        gap = 22
        
        X = df_period[feature_cols].values
        y_rv = df_period['RV_future'].values
        y_cavb = df_period['CAVB_target'].values
        vix_arr = df_period['VIX'].values
        
        X_train, X_test = X[:split], X[split+gap:]
        y_train = y_rv[:split]
        y_test = y_cavb[split+gap:]
        vix_test = vix_arr[split+gap:]
        
        if len(X_test) < 20:
            print(f"    ✗ 테스트 데이터 부족: {len(X_test)} 행")
            continue
        
        # 스케일링
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # ElasticNet
        model = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=SEED, max_iter=2000)
        model.fit(X_train_s, y_train)
        cavb_pred = vix_test - model.predict(X_test_s)
        
        r2 = r2_score(y_test, cavb_pred)
        mae = mean_absolute_error(y_test, cavb_pred)
        
        # Naive (Persistence)
        naive_pred = df_period['CAVB_lag1'].values[split+gap:]
        r2_naive = r2_score(y_test, naive_pred)
        
        # t-test: ML vs Naive
        # H0: 두 모델의 예측 오차가 같다
        # H1: ML이 더 우수하다
        loss_ml = (y_test - cavb_pred) ** 2
        loss_naive = (y_test - naive_pred) ** 2
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(loss_naive, loss_ml)  # loss_naive > loss_ml이면 ML이 우수
        
        improvement = r2 - r2_naive
        
        print(f"    ElasticNet R² = {r2:.4f}")
        print(f"    Naive R²      = {r2_naive:.4f}")
        print(f"    Improvement   = {improvement:+.4f}")
        print(f"    t-stat        = {t_stat:.3f}")
        print(f"    p-value       = {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
        
        results.append({
            'period': period_name,
            'start': start,
            'end': end,
            'n_samples': len(df_period),
            'n_test': len(X_test),
            'r2_ml': float(r2),
            'r2_naive': float(r2_naive),
            'improvement': float(improvement),
            'mae': float(mae),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        })
    
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n  [요약]")
        print(results_df[['period', 'n_samples', 'r2_ml', 'improvement', 't_statistic', 'p_value']].to_string(index=False))
        
        return {
            'asset': ticker,
            'asset_name': asset_name,
            'periods': results
        }
    else:
        return None


def main():
    print("\n" + "📅" * 35)
    print("Subperiod 분석 (Pre-COVID / COVID / Post-COVID)")
    print("📅" * 35)
    
    assets = [
        ('EFA', 'EAFE (선진국)'),
        ('TLT', 'Treasury (국채)'),
        ('GLD', 'Gold (금)'),
        ('SPY', 'S&P 500'),
        ('EEM', 'Emerging (신흥국)'),
    ]
    
    all_results = []
    
    for ticker, name in assets:
        result = subperiod_performance(ticker, name)
        if result:
            all_results.append(result)
    
    # 전체 요약
    print("\n" + "=" * 70)
    print("전체 요약")
    print("=" * 70)
    
    for asset_result in all_results:
        print(f"\n{asset_result['asset_name']}:")
        for period in asset_result['periods']:
            sig = '***' if period['p_value'] < 0.001 else '**' if period['p_value'] < 0.01 else '*' if period['p_value'] < 0.05 else ''
            print(f"  {period['period']:12s}: R²={period['r2_ml']:.3f}, vs Naive={period['improvement']:+.3f}, " +
                  f"t={period['t_statistic']:.2f}, p={period['p_value']:.4f} {sig}")
    
    # 저장
    output = {
        'description': 'Subperiod analysis: Pre-COVID / COVID / Post-COVID',
        'periods': {k: v for k, v in PERIODS.items()},
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/subperiod_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/subperiod_analysis.json")


if __name__ == '__main__':
    main()
