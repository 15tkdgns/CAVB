#!/usr/bin/env python3
"""
VRP (Volatility Risk Premium) 예측 연구 - 간소화 버전
======================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def main():
    print("\n" + "🔬" * 30)
    print("VRP (Volatility Risk Premium) 예측 연구")
    print("🔬" * 30)
    
    # ============================================
    # 1. 데이터 로드
    # ============================================
    print("\n[1/4] 데이터 로드")
    
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    spy['returns'] = spy['Close'].pct_change()
    
    # ============================================
    # 2. 변동성 및 VRP 계산
    # ============================================
    print("\n[2/4] VRP 계산")
    
    # 실현변동성 (연율화 %)
    spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # 현재 VRP = VIX - 과거 RV
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    
    # 미래 22일 실현변동성 계산
    spy['RV_future'] = spy['RV_22d'].shift(-22)  # 22일 후의 RV
    
    # 진정한 VRP = 현재 VIX - 미래 RV
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
    print(f"  VIX 평균: {spy['VIX'].mean():.2f}%")
    print(f"  RV 평균: {spy['RV_22d'].mean():.2f}%")
    print(f"  VRP 평균: {spy['VRP'].mean():.2f}%")
    
    # ============================================
    # 3. 특성 생성
    # ============================================
    print("\n[3/4] 특성 생성")
    
    # HAR-RV 스타일
    spy['RV_1d'] = spy['returns'].abs() * np.sqrt(252) * 100
    
    # VIX 특성
    spy['VIX_lag1'] = spy['VIX'].shift(1)
    spy['VIX_lag5'] = spy['VIX'].shift(5)
    spy['VIX_change'] = spy['VIX'].pct_change()
    spy['VIX_ma20'] = spy['VIX'].rolling(20).mean()
    
    # VRP 래그
    spy['VRP_lag1'] = spy['VRP'].shift(1)
    spy['VRP_lag5'] = spy['VRP'].shift(5)
    spy['VRP_ma5'] = spy['VRP'].rolling(5).mean()
    
    # Regime
    spy['regime_high'] = (spy['VIX'] >= 25).astype(int)
    spy['regime_crisis'] = (spy['VIX'] >= 35).astype(int)
    
    # 수익률
    spy['return_5d'] = spy['returns'].rolling(5).sum()
    spy['return_22d'] = spy['returns'].rolling(22).sum()
    
    # 결측치 제거
    spy = spy.replace([np.inf, -np.inf], np.nan)
    spy = spy.dropna()
    
    print(f"  데이터 샘플: {len(spy)}개")
    
    # ============================================
    # 4. 모델링
    # ============================================
    print("\n[4/4] 모델링")
    
    # 특성 및 타겟
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'regime_crisis', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values  # 미래 RV 예측
    y_vrp = spy['VRP_true'].values  # VRP 예측
    
    # 분할
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_rv_train, y_rv_test = y_rv[:split_idx], y_rv[split_idx:]
    y_vrp_train, y_vrp_test = y_vrp[:split_idx], y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # ---- HAR-RV (벤치마크) ----
    print("\n  🔹 HAR-RV (벤치마크)")
    har_X = spy[['RV_1d', 'RV_5d', 'RV_22d']].values
    har_X_train, har_X_test = har_X[:split_idx], har_X[split_idx:]
    
    har = LinearRegression()
    har.fit(har_X_train, y_rv_train)
    y_pred = har.predict(har_X_test)
    r2 = r2_score(y_rv_test, y_pred)
    results['HAR-RV'] = {'r2': r2, 'target': 'RV'}
    print(f"     R² (RV): {r2:.4f}")
    
    # ---- ElasticNet (RV) ----
    print("\n  🔹 ElasticNet (RV)")
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_rv_train)
    y_pred = en.predict(X_test_s)
    r2 = r2_score(y_rv_test, y_pred)
    results['ElasticNet_RV'] = {'r2': r2, 'target': 'RV'}
    print(f"     R² (RV): {r2:.4f}")
    
    # ---- ElasticNet (VRP) ----
    print("\n  🔹 ElasticNet (VRP)")
    en_vrp = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en_vrp.fit(X_train_s, y_vrp_train)
    y_vrp_pred = en_vrp.predict(X_test_s)
    r2 = r2_score(y_vrp_test, y_vrp_pred)
    results['ElasticNet_VRP'] = {'r2': r2, 'target': 'VRP'}
    print(f"     R² (VRP): {r2:.4f}")
    
    # ---- GradientBoosting (VRP) ----
    print("\n  🔹 GradientBoosting (VRP)")
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                   random_state=SEED)
    gb.fit(X_train_s, y_vrp_train)
    y_pred = gb.predict(X_test_s)
    r2 = r2_score(y_vrp_test, y_pred)
    results['GradientBoosting_VRP'] = {'r2': r2, 'target': 'VRP'}
    print(f"     R² (VRP): {r2:.4f}")
    
    # ============================================
    # 전략 분석
    # ============================================
    print("\n" + "=" * 50)
    print("📊 전략 분석")
    print("=" * 50)
    
    vrp_mean = y_vrp_test.mean()
    
    # 방향 예측 정확도
    direction_actual = (y_vrp_test > vrp_mean).astype(int)
    direction_pred = (y_vrp_pred > vrp_mean).astype(int)
    accuracy = (direction_actual == direction_pred).mean()
    
    print(f"\n  VRP 통계:")
    print(f"     테스트 VRP 평균: {vrp_mean:.2f}%")
    print(f"     테스트 VRP 표준편차: {y_vrp_test.std():.2f}%")
    
    print(f"\n  예측 성능:")
    print(f"     VRP 방향 정확도: {accuracy*100:.1f}%")
    
    # ============================================
    # 결과 요약
    # ============================================
    print("\n" + "=" * 50)
    print("📊 결과 요약")
    print("=" * 50)
    
    print("\n  모델별 성능:")
    for model, data in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
        print(f"     {model:25s}: R² = {data['r2']:.4f} ({data['target']})")
    
    best = max(results.items(), key=lambda x: x[1]['r2'])
    
    print(f"""
    🏆 최고 성능: {best[0]}
       R² = {best[1]['r2']:.4f}
    
    💡 핵심 발견:
       • VRP 예측 R² = {results['ElasticNet_VRP']['r2']:.4f}
       • 방향 예측 정확도 = {accuracy*100:.1f}%
       • VRP 양수 비율 = {(y_vrp_test > 0).mean()*100:.1f}%
    """)
    
    # 저장
    output = {
        'results': {k: {'r2': float(v['r2']), 'target': v['target']} for k, v in results.items()},
        'strategy': {
            'direction_accuracy': float(accuracy),
            'vrp_mean': float(vrp_mean),
            'vrp_positive_ratio': float((y_vrp_test > 0).mean())
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/vrp_prediction_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 결과 저장: paper/vrp_prediction_results.json")


if __name__ == '__main__':
    main()
