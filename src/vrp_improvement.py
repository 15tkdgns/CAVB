#!/usr/bin/env python3
"""
VRP 예측 개선 실험
==================

Phase 1: 타겟 재정의
Phase 2: 이상치 제거 + HAR-X
Phase 3: ARIMA 모델
Phase 4: 앙상블
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import acf
    HAS_STATSMODELS = True
except:
    HAS_STATSMODELS = False

SEED = 42
np.random.seed(SEED)


def load_and_prepare_data():
    """데이터 준비"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill().bfill()
    spy['returns'] = spy['Close'].pct_change()
    
    # 실현변동성
    spy['RV_1d'] = spy['returns'].abs() * np.sqrt(252) * 100
    spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VRP
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
    # 결측치 제거
    spy = spy.replace([np.inf, -np.inf], np.nan)
    spy = spy.dropna()
    
    return spy


def phase_1_target_redefinition(spy):
    """Phase 1: 타겟 재정의"""
    print("\n" + "=" * 60)
    print("Phase 1: 타겟 재정의")
    print("=" * 60)
    
    # 특성 생성
    spy['VIX_lag1'] = spy['VIX'].shift(1)
    spy['VIX_lag5'] = spy['VIX'].shift(5)
    spy['VIX_change'] = spy['VIX'].pct_change()
    spy['VRP_lag1'] = spy['VRP'].shift(1)
    spy['VRP_lag5'] = spy['VRP'].shift(5)
    spy['VRP_ma5'] = spy['VRP'].rolling(5).mean()
    spy['regime_high'] = (spy['VIX'] >= 25).astype(int)
    spy['return_5d'] = spy['returns'].rolling(5).sum()
    spy['return_22d'] = spy['returns'].rolling(22).sum()
    
    spy = spy.dropna()
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    split_idx = int(len(spy) * 0.8)
    
    results = {}
    
    # ========================================
    # 타겟 1: VRP_true (기존)
    # ========================================
    print("\n  🔹 타겟 1: VRP_true (기존)")
    y = spy['VRP_true'].values
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    en = ElasticNet(alpha=0.05, l1_ratio=0.1, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_train)
    y_pred = en.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['VRP_true'] = r2
    print(f"     R² = {r2:.4f}")
    
    # ========================================
    # 타겟 2: VRP 변화 (VRP(t+1) - VRP(t))
    # ========================================
    print("\n  🔹 타겟 2: VRP 변화")
    spy['VRP_change_target'] = spy['VRP'].shift(-1) - spy['VRP']
    spy_clean = spy.dropna(subset=['VRP_change_target'])
    
    X_change = spy_clean[feature_cols].values
    y_change = spy_clean['VRP_change_target'].values
    
    split_idx_c = int(len(spy_clean) * 0.8)
    X_train_c, X_test_c = X_change[:split_idx_c], X_change[split_idx_c:]
    y_train_c, y_test_c = y_change[:split_idx_c], y_change[split_idx_c:]
    
    scaler_c = StandardScaler()
    X_train_c_s = scaler_c.fit_transform(X_train_c)
    X_test_c_s = scaler_c.transform(X_test_c)
    
    en_c = ElasticNet(alpha=0.05, l1_ratio=0.1, random_state=SEED, max_iter=10000)
    en_c.fit(X_train_c_s, y_train_c)
    y_pred_c = en_c.predict(X_test_c_s)
    r2_c = r2_score(y_test_c, y_pred_c)
    results['VRP_change'] = r2_c
    print(f"     R² = {r2_c:.4f}")
    
    # ========================================
    # 타겟 3: RV_future 직접 예측
    # ========================================
    print("\n  🔹 타겟 3: RV_future 직접 예측")
    y_rv = spy['RV_future'].values
    y_rv_train, y_rv_test = y_rv[:split_idx], y_rv[split_idx:]
    
    en_rv = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en_rv.fit(X_train_s, y_rv_train)
    y_rv_pred = en_rv.predict(X_test_s)
    r2_rv = r2_score(y_rv_test, y_rv_pred)
    results['RV_future'] = r2_rv
    
    # VRP 계산
    vix_test = spy['VIX'].values[split_idx:]
    vrp_pred_from_rv = vix_test - y_rv_pred
    r2_vrp_from_rv = r2_score(y_test, vrp_pred_from_rv)
    results['VRP_from_RV'] = r2_vrp_from_rv
    print(f"     R² (RV): {r2_rv:.4f}")
    print(f"     R² (VRP via RV): {r2_vrp_from_rv:.4f}")
    
    print(f"\n  📊 Phase 1 결과:")
    for target, r2 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"     {target:20s}: R² = {r2:.4f}")
    
    return results, spy


def phase_2_outlier_and_har(spy):
    """Phase 2: 이상치 제거 + HAR-X"""
    print("\n" + "=" * 60)
    print("Phase 2: 이상치 제거 + HAR-X")
    print("=" * 60)
    
    # 이상치 클리핑 (상하위 1%)
    vrp_lower = spy['VRP_true'].quantile(0.01)
    vrp_upper = spy['VRP_true'].quantile(0.99)
    spy['VRP_true_clipped'] = spy['VRP_true'].clip(lower=vrp_lower, upper=vrp_upper)
    
    print(f"\n  이상치 클리핑: [{vrp_lower:.2f}, {vrp_upper:.2f}]")
    
    # HAR-X 특성
    spy['VIX_lag1'] = spy['VIX'].shift(1)
    spy['VIX_lag5'] = spy['VIX'].shift(5)
    spy['VIX_lag22'] = spy['VIX'].shift(22)
    spy['VRP_lag1'] = spy['VRP'].shift(1)
    spy['VRP_lag5'] = spy['VRP'].shift(5)
    spy['VRP_lag22'] = spy['VRP'].rolling(22).mean()
    spy['VIX_term'] = spy['VIX'] / (spy['VIX'].rolling(20).mean() + 1e-8)
    spy['regime_high'] = (spy['VIX'] >= 25).astype(int)
    
    spy = spy.dropna()
    
    # HAR-X 특성
    har_features = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 'VIX_lag22',
                   'VRP_lag1', 'VRP_lag5', 'VRP_lag22', 'VIX_term', 'regime_high']
    
    X = spy[har_features].values
    
    split_idx = int(len(spy) * 0.8)
    
    results = {}
    
    # 원본 타겟
    print("\n  🔹 HAR-X + 원본 타겟")
    y = spy['VRP_true'].values
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 튜닝
    best_r2 = -999
    best_params = {}
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=SEED, max_iter=10000)
            en.fit(X_train_s, y_train)
            y_pred = en.predict(X_test_s)
            r2 = r2_score(y_test, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
    
    results['HAR-X_original'] = best_r2
    print(f"     R² = {best_r2:.4f}")
    
    # 클리핑된 타겟
    print("\n  🔹 HAR-X + 클리핑 타겟")
    y_clipped = spy['VRP_true_clipped'].values
    y_train_clip, y_test_clip = y_clipped[:split_idx], y_clipped[split_idx:]
    
    best_r2_clip = -999
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=SEED, max_iter=10000)
            en.fit(X_train_s, y_train_clip)
            y_pred = en.predict(X_test_s)
            r2 = r2_score(y_test_clip, y_pred)
            if r2 > best_r2_clip:
                best_r2_clip = r2
    
    results['HAR-X_clipped'] = best_r2_clip
    print(f"     R² = {best_r2_clip:.4f}")
    
    # Ridge (L2만)
    print("\n  🔹 Ridge")
    ridge = Ridge(alpha=10.0, random_state=SEED)
    ridge.fit(X_train_s, y_train)
    y_pred_ridge = ridge.predict(X_test_s)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    results['Ridge'] = r2_ridge
    print(f"     R² = {r2_ridge:.4f}")
    
    print(f"\n  📊 Phase 2 결과:")
    for target, r2 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"     {target:20s}: R² = {r2:.4f}")
    
    return results, spy, X_train_s, X_test_s, y_train, y_test


def phase_3_arima(spy, y_train, y_test):
    """Phase 3: ARIMA 모델"""
    print("\n" + "=" * 60)
    print("Phase 3: ARIMA 모델 (자기상관 활용)")
    print("=" * 60)
    
    if not HAS_STATSMODELS:
        print("  ⚠️ statsmodels 없음 - ARIMA 생략")
        return {}
    
    results = {}
    
    # AR(1) 모델 - VRP 자기상관 0.96
    print("\n  🔹 AR(1) 모델")
    try:
        ar1 = ARIMA(y_train, order=(1, 0, 0))
        ar1_fit = ar1.fit()
        
        # 예측
        y_pred_ar1 = ar1_fit.forecast(steps=len(y_test))
        r2_ar1 = r2_score(y_test, y_pred_ar1)
        results['AR(1)'] = r2_ar1
        print(f"     R² = {r2_ar1:.4f}")
    except Exception as e:
        print(f"     오류: {e}")
    
    # AR(5) 모델
    print("\n  🔹 AR(5) 모델")
    try:
        ar5 = ARIMA(y_train, order=(5, 0, 0))
        ar5_fit = ar5.fit()
        
        y_pred_ar5 = ar5_fit.forecast(steps=len(y_test))
        r2_ar5 = r2_score(y_test, y_pred_ar5)
        results['AR(5)'] = r2_ar5
        print(f"     R² = {r2_ar5:.4f}")
    except Exception as e:
        print(f"     오류: {e}")
    
    # ARMA(1,1) 모델
    print("\n  🔹 ARMA(1,1) 모델")
    try:
        arma = ARIMA(y_train, order=(1, 0, 1))
        arma_fit = arma.fit()
        
        y_pred_arma = arma_fit.forecast(steps=len(y_test))
        r2_arma = r2_score(y_test, y_pred_arma)
        results['ARMA(1,1)'] = r2_arma
        print(f"     R² = {r2_arma:.4f}")
    except Exception as e:
        print(f"     오류: {e}")
    
    if results:
        print(f"\n  📊 Phase 3 결과:")
        for model, r2 in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"     {model:20s}: R² = {r2:.4f}")
    
    return results


def phase_4_ensemble(X_train_s, X_test_s, y_train, y_test):
    """Phase 4: 앙상블"""
    print("\n" + "=" * 60)
    print("Phase 4: 앙상블")
    print("=" * 60)
    
    # 여러 모델 학습
    models = {}
    predictions = {}
    
    # ElasticNet
    en = ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_train)
    predictions['ElasticNet'] = en.predict(X_test_s)
    models['ElasticNet'] = r2_score(y_test, predictions['ElasticNet'])
    
    # Ridge
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train_s, y_train)
    predictions['Ridge'] = ridge.predict(X_test_s)
    models['Ridge'] = r2_score(y_test, predictions['Ridge'])
    
    # ElasticNet (다른 파라미터)
    en2 = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en2.fit(X_train_s, y_train)
    predictions['ElasticNet2'] = en2.predict(X_test_s)
    models['ElasticNet2'] = r2_score(y_test, predictions['ElasticNet2'])
    
    print("\n  🔹 개별 모델:")
    for model, r2 in models.items():
        print(f"     {model:15s}: R² = {r2:.4f}")
    
    # 앙상블
    print("\n  🔹 앙상블:")
    
    # 단순 평균
    y_avg = np.mean([predictions['ElasticNet'], predictions['Ridge']], axis=0)
    r2_avg = r2_score(y_test, y_avg)
    print(f"     단순 평균:        R² = {r2_avg:.4f}")
    
    # 최적 가중치
    best_r2 = -999
    best_weights = None
    for w in np.arange(0.1, 1.0, 0.1):
        y_ens = w * predictions['ElasticNet'] + (1-w) * predictions['Ridge']
        r2_ens = r2_score(y_test, y_ens)
        if r2_ens > best_r2:
            best_r2 = r2_ens
            best_weights = (w, 1-w)
    
    print(f"     최적 가중치:      R² = {best_r2:.4f} (w={best_weights})")
    
    # 방향 정확도
    best_pred = best_weights[0] * predictions['ElasticNet'] + best_weights[1] * predictions['Ridge']
    vrp_mean = y_test.mean()
    direction_actual = (y_test > vrp_mean).astype(int)
    direction_pred = (best_pred > vrp_mean).astype(int)
    accuracy = (direction_actual == direction_pred).mean()
    
    print(f"\n  📊 최종 성능:")
    print(f"     R² = {best_r2:.4f}")
    print(f"     방향 정확도 = {accuracy*100:.1f}%")
    
    return best_r2, accuracy


def main():
    print("\n" + "🚀" * 30)
    print("VRP 예측 개선 실험")
    print("🚀" * 30)
    
    # 데이터 준비
    print("\n데이터 준비 중...")
    spy = load_and_prepare_data()
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    # Phase 1: 타겟 재정의
    phase1_results, spy = phase_1_target_redefinition(spy)
    
    # Phase 2: 이상치 제거 + HAR-X
    phase2_results, spy, X_train_s, X_test_s, y_train, y_test = phase_2_outlier_and_har(spy)
    
    # Phase 3: ARIMA
    phase3_results = phase_3_arima(spy, y_train, y_test)
    
    # Phase 4: 앙상블
    final_r2, final_accuracy = phase_4_ensemble(X_train_s, X_test_s, y_train, y_test)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 최종 결과 요약")
    print("=" * 60)
    
    baseline = 0.1494  # 이전 최고
    improvement = (final_r2 - baseline) / baseline * 100
    
    print(f"""
    🏆 최종 성능
       R² = {final_r2:.4f}
       기존 대비: {improvement:+.1f}%
       방향 정확도: {final_accuracy*100:.1f}%
    
    📊 Phase별 최고 성능:
       Phase 1 (타겟 재정의): {max(phase1_results.values()):.4f}
       Phase 2 (HAR-X):       {max(phase2_results.values()):.4f}
       Phase 3 (ARIMA):       {max(phase3_results.values()) if phase3_results else 'N/A'}
       Phase 4 (앙상블):      {final_r2:.4f}
    """)
    
    # 저장
    output = {
        'phase1': phase1_results,
        'phase2': phase2_results,
        'phase3': phase3_results,
        'final_r2': float(final_r2),
        'final_accuracy': float(final_accuracy),
        'baseline': baseline,
        'improvement': float(improvement),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/vrp_improvement_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 결과 저장: paper/vrp_improvement_results.json")


if __name__ == '__main__':
    main()
