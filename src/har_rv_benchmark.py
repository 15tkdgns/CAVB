#!/usr/bin/env python3
"""
HAR-RV 벤치마크 비교 (5일 예측)
=================================

변동성 예측 문헌의 표준 모델인 HAR-RV와 CAVB 모델 비교
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
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


def prepare_data_5day(ticker):
    """5일 예측 데이터 준비"""
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
    
    # 타겟 (5일 예측)
    df['RV_future'] = df['RV_22d'].shift(-5)
    df['CAVB_target'] = df['VIX'] - df['RV_future']
    
    # 기본 특성
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    return df.dropna()


def har_rv_benchmark(ticker, asset_name):
    """HAR-RV vs CAVB 모델 비교"""
    print(f"\n{'='*70}")
    print(f"HAR-RV 벤치마크: {asset_name} ({ticker})")
    print(f"{'='*70}")
    
    df = prepare_data_5day(ticker)
    if df is None:
        print(f"  ✗ 데이터 로드 실패")
        return None
    
    print(f"  데이터: {len(df)} 행")
    
    # 3-Way Split
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    gap = 5
    
    vix_arr = df['VIX'].values
    y_rv = df['RV_future'].values
    y_cavb = df['CAVB_target'].values
    
    vix_val = vix_arr[train_end+gap:val_end]
    vix_test = vix_arr[val_end+gap:]
    
    y_train = y_rv[:train_end]
    y_val_cavb = y_cavb[train_end+gap:val_end]
    y_test_cavb = y_cavb[val_end+gap:]
    
    results = {}
    
    # ========== 모델 1: HAR-RV (변동성만) ==========
    print(f"\n  [Model 1: HAR-RV]")
    print(f"    Features: RV_1d, RV_5d, RV_22d (변동성만)")
    
    X_har = df[['RV_1d', 'RV_5d', 'RV_22d']].values
    
    X_train_har = X_har[:train_end]
    X_val_har = X_har[train_end+gap:val_end]
    X_test_har = X_har[val_end+gap:]
    
    scaler_har = StandardScaler()
    X_train_har_s = scaler_har.fit_transform(X_train_har)
    X_val_har_s = scaler_har.transform(X_val_har)
    X_test_har_s = scaler_har.transform(X_test_har)
    
    model_har = LinearRegression()
    model_har.fit(X_train_har_s, y_train)
    
    # CAVB 예측으로 변환
    cavb_pred_har = vix_test - model_har.predict(X_test_har_s)
    r2_har = r2_score(y_test_cavb, cavb_pred_har)
    mae_har = mean_absolute_error(y_test_cavb, cavb_pred_har)
    
    print(f"    Test R² = {r2_har:.4f}")
    print(f"    Test MAE = {mae_har:.2f}")
    
    results['HAR-RV'] = {
        'model': 'HAR-RV (Linear)',
        'features': 'RV_1d, RV_5d, RV_22d',
        'n_features': 3,
        'r2_test': float(r2_har),
        'mae_test': float(mae_har)
    }
    
    # ========== 모델 2: HAR-RV + VIX ==========
    print(f"\n  [Model 2: HAR-RV + VIX]")
    print(f"    Features: HAR + VIX_lag1, VIX_change")
    
    X_har_vix = df[['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_change']].values
    
    X_train_hv = X_har_vix[:train_end]
    X_val_hv = X_har_vix[train_end+gap:val_end]
    X_test_hv = X_har_vix[val_end+gap:]
    
    scaler_hv = StandardScaler()
    X_train_hv_s = scaler_hv.fit_transform(X_train_hv)
    X_val_hv_s = scaler_hv.transform(X_val_hv)
    X_test_hv_s = scaler_hv.transform(X_test_hv)
    
    model_hv = LinearRegression()
    model_hv.fit(X_train_hv_s, y_train)
    
    cavb_pred_hv = vix_test - model_hv.predict(X_test_hv_s)
    r2_hv = r2_score(y_test_cavb, cavb_pred_hv)
    mae_hv = mean_absolute_error(y_test_cavb, cavb_pred_hv)
    
    print(f"    Test R² = {r2_hv:.4f}")
    print(f"    Test MAE = {mae_hv:.2f}")
    print(f"    Improvement over HAR: {(r2_hv - r2_har):+.4f}")
    
    results['HAR-RV+VIX'] = {
        'model': 'HAR-RV + VIX',
        'features': 'RV_1d, RV_5d, RV_22d, VIX_lag1, VIX_change',
        'n_features': 5,
        'r2_test': float(r2_hv),
        'mae_test': float(mae_hv),
        'improvement_over_har': float(r2_hv - r2_har)
    }
    
    # ========== 모델 3: CAVB (ElasticNet, Full) ==========
    print(f"\n  [Model 3: CAVB (ElasticNet)]")
    print(f"    Features: 9개 (RV + VIX + CAVB)")
    
    X_cavb = df[['RV_1d', 'RV_5d', 'RV_22d',
                 'VIX_lag1', 'VIX_lag5', 'VIX_change',
                 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']].values
    
    X_train_cavb = X_cavb[:train_end]
    X_val_cavb = X_cavb[train_end+gap:val_end]
    X_test_cavb = X_cavb[val_end+gap:]
    
    scaler_cavb = RobustScaler()
    X_train_cavb_s = scaler_cavb.fit_transform(X_train_cavb)
    X_val_cavb_s = scaler_cavb.transform(X_val_cavb)
    X_test_cavb_s = scaler_cavb.transform(X_test_cavb)
    
    model_cavb = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=SEED, max_iter=2000)
    model_cavb.fit(X_train_cavb_s, y_train)
    
    cavb_pred_full = vix_test - model_cavb.predict(X_test_cavb_s)
    r2_cavb = r2_score(y_test_cavb, cavb_pred_full)
    mae_cavb = mean_absolute_error(y_test_cavb, cavb_pred_full)
    
    print(f"    Test R² = {r2_cavb:.4f}")
    print(f"    Test MAE = {mae_cavb:.2f}")
    print(f"    Improvement over HAR-RV+VIX: {(r2_cavb - r2_hv):+.4f}")
    
    results['CAVB'] = {
        'model': 'CAVB (ElasticNet)',
        'features': 'RV + VIX + CAVB (9 features)',
        'n_features': 9,
        'r2_test': float(r2_cavb),
        'mae_test': float(mae_cavb),
        'improvement_over_har_vix': float(r2_cavb - r2_hv)
    }
    
    # ========== 통계 검정: CAVB vs HAR-RV+VIX ==========
    print(f"\n  [Statistical Test: CAVB vs HAR-RV+VIX]")
    
    # Diebold-Mariano Test
    loss_cavb = (y_test_cavb - cavb_pred_full) ** 2
    loss_hv = (y_test_cavb - cavb_pred_hv) ** 2
    loss_diff = loss_hv - loss_cavb  # Positive = CAVB better
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(loss_hv, loss_cavb)
    
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    p-value: {p_value:.4f}")
    
    if p_value < 0.001:
        sig = "***"
        result = "CAVB 통계적으로 유의하게 우수"
    elif p_value < 0.01:
        sig = "**"
        result = "CAVB 통계적으로 유의하게 우수"
    elif p_value < 0.05:
        sig = "*"
        result = "CAVB 유의하게 우수"
    else:
        sig = ""
        result = "통계적 차이 없음"
    
    print(f"    결과: {result} {sig}")
    
    results['statistical_test'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'result': result
    }
    
    # 요약
    print(f"\n  [비교 요약]")
    print(f"    HAR-RV:       R² = {r2_har:.4f}")
    print(f"    HAR-RV+VIX:   R² = {r2_hv:.4f} ({(r2_hv-r2_har)/r2_har*100:+.1f}%)")
    print(f"    CAVB (Full):  R² = {r2_cavb:.4f} ({(r2_cavb-r2_hv)/r2_hv*100:+.1f}%)")
    
    return {
        'asset': ticker,
        'asset_name': asset_name,
        'n_samples': len(df),
        'train_size': train_end,
        'val_size': val_end - train_end - gap,
        'test_size': n - val_end - gap,
        'results': results
    }


def main():
    print("\n" + "📈" * 35)
    print("HAR-RV 벤치마크 비교 (5일 예측)")
    print("📈" * 35)
    
    assets = [
        ('GLD', 'Gold (금)'),
        ('EFA', 'EAFE (선진국)'),
        ('TLT', 'Treasury (국채)'),
        ('SPY', 'S&P 500'),
        ('EEM', 'Emerging (신흥국)'),
    ]
    
    all_results = []
    
    for ticker, name in assets:
        result = har_rv_benchmark(ticker, name)
        if result:
            all_results.append(result)
    
    # 전체 요약
    print("\n" + "=" * 70)
    print("전체 자산 비교")
    print("=" * 70)
    
    summary_data = []
    for r in all_results:
        res = r['results']
        summary_data.append({
            'Asset': r['asset_name'],
            'HAR-RV': f"{res['HAR-RV']['r2_test']:.3f}",
            'HAR+VIX': f"{res['HAR-RV+VIX']['r2_test']:.3f}",
            'CAVB': f"{res['CAVB']['r2_test']:.3f}",
            'Improve': f"{res['CAVB']['improvement_over_har_vix']:+.3f}",
            'p-value': f"{res['statistical_test']['p_value']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(f"\n{summary_df.to_string(index=False)}")
    
    # 평균 개선
    improvements = [r['results']['CAVB']['improvement_over_har_vix'] for r in all_results]
    avg_improvement = np.mean(improvements)
    print(f"\nCAVB의 평균 개선 (vs HAR-RV+VIX): {avg_improvement:+.4f}")
    
    # CAVB가 우수한 자산 개수
    n_superior = sum(1 for r in all_results if r['results']['statistical_test']['significant'])
    print(f"통계적으로 유의한 자산: {n_superior}/{len(all_results)}")
    
    # 저장
    output = {
        'description': 'HAR-RV Benchmark Comparison (5-day prediction)',
        'methodology': {
            'models': {
                'HAR-RV': 'Linear(RV_1d, RV_5d, RV_22d)',
                'HAR-RV+VIX': 'Linear(HAR + VIX_lag1, VIX_change)',
                'CAVB': 'ElasticNet(RV + VIX + CAVB, 9 features)'
            },
            'split': '60/20/20',
            'gap': 5,
            'statistical_test': 'Paired t-test'
        },
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/har_rv_benchmark.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/har_rv_benchmark.json")


if __name__ == '__main__':
    main()
