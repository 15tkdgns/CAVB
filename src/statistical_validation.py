#!/usr/bin/env python3
"""
통계적 유의성 검증
====================

SCI 저널 수준의 통계 검증:
- Bootstrap 기반 계수 t-test
- 95% 신뢰구간
- 각 변수의 유의성 검정
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.utils import resample
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


def coefficient_ttest(model, X_train, y_train, feature_names, n_bootstrap=1000):
    """
    Bootstrap 기반 계수 t-test 및 신뢰구간
    
    Args:
        model: 학습된 모델 (ElasticNet)
        X_train: 학습 데이터
        y_train: 타겟
        feature_names: 변수명 리스트
        n_bootstrap: Bootstrap 반복 횟수
    
    Returns:
        DataFrame with t-statistics, p-values, and confidence intervals
    """
    print(f"  Bootstrap (n={n_bootstrap})으로 계수 유의성 검증...")
    
    bootstrap_coefs = []
    
    for i in range(n_bootstrap):
        # Bootstrap 샘플링
        X_boot, y_boot = resample(X_train, y_train, random_state=i)
        
        # 모델 학습
        model_boot = clone(model)
        model_boot.fit(X_boot, y_boot)
        bootstrap_coefs.append(model_boot.coef_)
    
    bootstrap_coefs = np.array(bootstrap_coefs)
    
    # 통계량 계산
    coef_mean = bootstrap_coefs.mean(axis=0)
    coef_std = bootstrap_coefs.std(axis=0)
    
    # t-statistic: coef / SE
    t_stat = coef_mean / (coef_std + 1e-10)
    
    # p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_bootstrap-1))
    
    # 95% CI
    ci_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)
    
    # 결과 DataFrame
    results = pd.DataFrame({
        'Variable': feature_names,
        'Coefficient': coef_mean,
        'Std_Error': coef_std,
        't_statistic': t_stat,
        'p_value': p_value,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'Significant': p_value < 0.05
    })
    
    # 유의성 기호 추가
    def sig_marker(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    
    results['Sig'] = results['p_value'].apply(sig_marker)
    
    return results.sort_values('p_value')


def analyze_asset_statistics(ticker, asset_name):
    """자산별 통계 검증"""
    print(f"\n{'='*70}")
    print(f"통계 검증: {asset_name} ({ticker})")
    print(f"{'='*70}")
    
    df = prepare_data(ticker)
    if df is None:
        print(f"  ✗ 데이터 로드 실패")
        return None
    
    print(f"  데이터: {len(df)} 행")
    
    # 특성
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
    
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_cavb = df['CAVB_target'].values
    vix_arr = df['VIX'].values
    
    # 3-Way Split: Train(60%) / Validation(20%) / Test(20%)
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    gap = 22
    
    X_train = X[:train_end]
    X_val = X[train_end+gap:val_end]
    X_test = X[val_end+gap:]
    
    y_train = y_rv[:train_end]
    y_val_cavb = y_cavb[train_end+gap:val_end]
    y_test_cavb = y_cavb[val_end+gap:]
    
    vix_val = vix_arr[train_end+gap:val_end]
    vix_test = vix_arr[val_end+gap:]
    
    print(f"  Split: Train={len(X_train)} / Val={len(X_val)} / Test={len(X_test)}")
    
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
    
    print(f"\n  [성능]")
    print(f"    Validation R² = {r2_val:.4f}")
    print(f"    Test R²       = {r2_test:.4f}")
    print(f"    Test MAE      = {mae_test:.2f}")
    
    # 계수 t-test
    coef_results = coefficient_ttest(model, X_train_s, y_train, feature_cols, n_bootstrap=500)
    
    print(f"\n  [계수 유의성 검증]")
    print(coef_results[['Variable', 'Coefficient', 'Std_Error', 't_statistic', 
                        'p_value', 'CI_Lower', 'CI_Upper', 'Sig']].to_string(index=False))
    
    # 유의한 변수 개수
    n_sig = coef_results['Significant'].sum()
    print(f"\n  유의한 변수: {n_sig}/{len(feature_cols)} (p < 0.05)")
    
    return {
        'asset': ticker,
        'asset_name': asset_name,
        'n_samples': len(df),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'r2_validation': float(r2_val),
        'r2_test': float(r2_test),
        'mae_test': float(mae_test),
        'coefficients': coef_results.to_dict('records'),
        'n_significant': int(n_sig)
    }


def main():
    print("\n" + "📊" * 35)
    print("통계적 유의성 검증 (SCI 수준)")
    print("📊" * 35)
    
    assets = [
        ('EFA', 'EAFE (선진국)'),
        ('TLT', 'Treasury (국채)'),
        ('GLD', 'Gold (금)'),
        ('SPY', 'S&P 500'),
        ('EEM', 'Emerging (신흥국)'),
    ]
    
    all_results = []
    
    for ticker, name in assets:
        result = analyze_asset_statistics(ticker, name)
        if result:
            all_results.append(result)
    
    # 전체 요약
    print("\n" + "=" * 70)
    print("전체 요약")
    print("=" * 70)
    
    summary_df = pd.DataFrame([{
        'Asset': r['asset_name'],
        'Train': r['train_size'],
        'Val': r['val_size'],
        'Test': r['test_size'],
        'Val R²': f"{r['r2_validation']:.3f}",
        'Test R²': f"{r['r2_test']:.3f}",
        'Sig Vars': f"{r['n_significant']}/9"
    } for r in all_results])
    
    print(f"\n{summary_df.to_string(index=False)}")
    
    # 저장
    output = {
        'description': 'Statistical validation with bootstrap t-tests and 3-way split',
        'methodology': {
            'split': 'Train(60%) / Validation(20%) / Test(20%)',
            'gap': 22,
            'bootstrap_iterations': 500,
            'significance_level': 0.05
        },
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/statistical_validation.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/statistical_validation.json")


if __name__ == '__main__':
    main()
