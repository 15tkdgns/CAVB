#!/usr/bin/env python3
"""
진정한 VRP 분석: 자산별 IV 지수 사용
====================================

자산별 적절한 내재변동성 지수를 사용하여 VRP를 계산하고 예측력 비교

자산별 IV 지수:
- GLD (금): GVZ (CBOE Gold Volatility Index)
- USO (원유): OVX (CBOE Oil Volatility Index)  
- TLT (채권): MOVE Index (ICE BofA MOVE Index) - 대용: VXTLT
- SPY (주식): VIX (CBOE Volatility Index)
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
        if len(data) < 100:
            return None
        return data
    except Exception as e:
        print(f"    다운로드 실패: {ticker} - {e}")
        return None


def calculate_rv(returns, window=22):
    """실현변동성 계산 (연율화)"""
    return returns.rolling(window).std() * np.sqrt(252) * 100


def analyze_true_vrp(asset_ticker, iv_ticker, asset_name, iv_name):
    """진정한 VRP 분석"""
    print(f"\n  {asset_name} ({asset_ticker})")
    print(f"    IV 지수: {iv_name} ({iv_ticker})")
    
    # 데이터 다운로드
    asset = download_data(asset_ticker)
    iv = download_data(iv_ticker)
    
    if asset is None:
        print(f"    ✗ {asset_ticker} 데이터 없음")
        return None
    
    if iv is None:
        print(f"    ✗ {iv_ticker} 데이터 없음")
        return None
    
    # 데이터 병합
    df = asset[['Close']].copy()
    df.columns = ['Price']
    df['IV'] = iv['Close'].reindex(df.index).ffill().bfill()
    df['returns'] = df['Price'].pct_change()
    
    # 실현변동성
    df['RV_22d'] = calculate_rv(df['returns'])
    
    # 진정한 VRP
    df['VRP'] = df['IV'] - df['RV_22d']
    
    # 타겟: 22일 후 실현변동성
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['VRP_true'] = df['IV'] - df['RV_future']
    
    # 특성
    df['IV_lag1'] = df['IV'].shift(1)
    df['VRP_lag1'] = df['VRP'].shift(1)
    df['VRP_lag5'] = df['VRP'].shift(5)
    
    df = df.dropna()
    
    if len(df) < 300:
        print(f"    ✗ 데이터 부족: {len(df)} 행")
        return None
    
    print(f"    데이터: {len(df)} 행")
    
    # IV-RV 상관
    iv_rv_corr = df['IV'].corr(df['RV_22d'])
    print(f"    IV-RV 상관: {iv_rv_corr:.3f}")
    
    # 모델 학습
    feature_cols = ['RV_22d', 'IV_lag1', 'VRP_lag1', 'VRP_lag5']
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_vrp = df['VRP_true'].values
    iv_arr = df['IV'].values
    
    split = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split])
    X_test_s = scaler.transform(X[split:])
    
    en = ElasticNet(alpha=0.01, random_state=SEED)
    en.fit(X_train_s, y_rv[:split])
    vrp_pred = iv_arr[split:] - en.predict(X_test_s)
    
    r2 = r2_score(y_vrp[split:], vrp_pred)
    vrp_mean = y_vrp[split:].mean()
    dir_acc = ((y_vrp[split:] > vrp_mean) == (vrp_pred > vrp_mean)).mean()
    
    print(f"    R²: {r2:.4f}")
    print(f"    방향정확도: {dir_acc*100:.1f}%")
    
    return {
        'asset': asset_ticker,
        'iv_index': iv_ticker,
        'asset_name': asset_name,
        'iv_name': iv_name,
        'n_samples': len(df),
        'iv_rv_corr': float(iv_rv_corr),
        'r2': float(r2),
        'direction_acc': float(dir_acc)
    }


def analyze_cross_asset_basis(asset_ticker, asset_name):
    """크로스 자산 기준 (VIX 사용) - 기존 방법"""
    print(f"\n  {asset_name} ({asset_ticker}) - VIX 기준")
    
    asset = download_data(asset_ticker)
    vix = download_data('^VIX')
    
    if asset is None or vix is None:
        return None
    
    df = asset[['Close']].copy()
    df.columns = ['Price']
    df['IV'] = vix['Close'].reindex(df.index).ffill().bfill()
    df['returns'] = df['Price'].pct_change()
    df['RV_22d'] = calculate_rv(df['returns'])
    df['Spread'] = df['IV'] - df['RV_22d']
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['Spread_true'] = df['IV'] - df['RV_future']
    df['IV_lag1'] = df['IV'].shift(1)
    df['Spread_lag1'] = df['Spread'].shift(1)
    df['Spread_lag5'] = df['Spread'].shift(5)
    
    df = df.dropna()
    
    if len(df) < 300:
        return None
    
    iv_rv_corr = df['IV'].corr(df['RV_22d'])
    
    feature_cols = ['RV_22d', 'IV_lag1', 'Spread_lag1', 'Spread_lag5']
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_spread = df['Spread_true'].values
    iv_arr = df['IV'].values
    
    split = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split])
    X_test_s = scaler.transform(X[split:])
    
    en = ElasticNet(alpha=0.01, random_state=SEED)
    en.fit(X_train_s, y_rv[:split])
    spread_pred = iv_arr[split:] - en.predict(X_test_s)
    
    r2 = r2_score(y_spread[split:], spread_pred)
    spread_mean = y_spread[split:].mean()
    dir_acc = ((y_spread[split:] > spread_mean) == (spread_pred > spread_mean)).mean()
    
    print(f"    VIX-RV 상관: {iv_rv_corr:.3f}")
    print(f"    R²: {r2:.4f}")
    print(f"    방향정확도: {dir_acc*100:.1f}%")
    
    return {
        'asset': asset_ticker,
        'iv_index': '^VIX',
        'asset_name': asset_name,
        'iv_name': 'VIX (Cross-Asset)',
        'iv_rv_corr': float(iv_rv_corr),
        'r2': float(r2),
        'direction_acc': float(dir_acc)
    }


def main():
    print("\n" + "=" * 70)
    print("진정한 VRP 분석: 자산별 적절한 IV 지수 사용")
    print("=" * 70)
    
    results = {
        'true_vrp': [],
        'cross_asset': []
    }
    
    # 분석 대상 자산
    assets = [
        # (자산 티커, IV 티커, 자산 이름, IV 이름)
        ('GLD', '^GVZ', 'Gold', 'GVZ'),
        ('USO', '^OVX', 'Oil', 'OVX'),
        ('TLT', '^VIX', 'Treasury', 'VIX (대용)'),  # MOVE 데이터 접근 어려움
        ('SPY', '^VIX', 'S&P 500', 'VIX'),
        ('EFA', '^VIX', 'EAFE', 'VIX (대용)'),
        ('EEM', '^VIX', 'Emerging', 'VIX (대용)'),
    ]
    
    # ==============================
    # 진정한 VRP (자산별 IV)
    # ==============================
    print("\n" + "-" * 70)
    print("1. 진정한 VRP (자산별 IV 지수)")
    print("-" * 70)
    
    for asset_ticker, iv_ticker, asset_name, iv_name in assets:
        result = analyze_true_vrp(asset_ticker, iv_ticker, asset_name, iv_name)
        if result:
            results['true_vrp'].append(result)
    
    # ==============================
    # 크로스 자산 Basis (VIX 사용)
    # ==============================
    print("\n" + "-" * 70)
    print("2. 크로스 자산 Basis (모두 VIX 사용)")
    print("-" * 70)
    
    for asset_ticker, _, asset_name, _ in assets:
        result = analyze_cross_asset_basis(asset_ticker, asset_name)
        if result:
            results['cross_asset'].append(result)
    
    # ==============================
    # 비교 요약
    # ==============================
    print("\n" + "=" * 70)
    print("비교 요약")
    print("=" * 70)
    
    print(f"\n{'자산':<12} | {'True VRP R²':>12} | {'Cross-Asset R²':>15} | {'차이':>10}")
    print("-" * 60)
    
    comparison = []
    for true_result in results['true_vrp']:
        asset = true_result['asset']
        cross_result = next((r for r in results['cross_asset'] if r['asset'] == asset), None)
        
        if cross_result:
            diff = true_result['r2'] - cross_result['r2']
            print(f"{true_result['asset_name']:<12} | {true_result['r2']:>12.4f} | {cross_result['r2']:>15.4f} | {diff:>+10.4f}")
            comparison.append({
                'asset': asset,
                'true_vrp_r2': true_result['r2'],
                'cross_asset_r2': cross_result['r2'],
                'difference': diff
            })
    
    # ==============================
    # 핵심 결론
    # ==============================
    print("\n" + "=" * 70)
    print("핵심 결론")
    print("=" * 70)
    
    if comparison:
        avg_diff = np.mean([c['difference'] for c in comparison])
        if avg_diff < -0.1:
            conclusion = "진정한 VRP는 Cross-Asset Basis보다 예측력이 낮습니다."
        elif avg_diff > 0.1:
            conclusion = "진정한 VRP는 Cross-Asset Basis보다 예측력이 높습니다."
        else:
            conclusion = "진정한 VRP와 Cross-Asset Basis 예측력이 유사합니다."
        
        print(f"\n  평균 R² 차이: {avg_diff:+.4f}")
        print(f"  결론: {conclusion}")
    
    # 저장
    results['comparison'] = comparison
    results['timestamp'] = datetime.now().isoformat()
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/true_vrp_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/true_vrp_analysis.json")
    
    return results


if __name__ == '__main__':
    main()
