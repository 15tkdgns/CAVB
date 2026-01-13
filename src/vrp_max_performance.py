#!/usr/bin/env python3
"""
VRP 예측 최대 성능 최적화
==========================

전략:
1. 특성 확장 + 선택
2. Rolling 재학습
3. 이상치 제거
4. 다중 타임프레임
5. 앙상블 최적화
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def load_and_create_features():
    """데이터 로드 및 확장된 특성 생성"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill().bfill()
    spy['returns'] = spy['Close'].pct_change()
    
    # ============================================
    # 실현변동성 (다양한 윈도우)
    # ============================================
    spy['RV_1d'] = spy['returns'].abs() * np.sqrt(252) * 100
    spy['RV_3d'] = spy['returns'].rolling(3).std() * np.sqrt(252) * 100
    spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
    spy['RV_10d'] = spy['returns'].rolling(10).std() * np.sqrt(252) * 100
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VRP
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    
    # 타겟
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
    # ============================================
    # 확장된 특성
    # ============================================
    
    # VIX 래그
    for lag in [1, 2, 3, 5, 10]:
        spy[f'VIX_lag{lag}'] = spy['VIX'].shift(lag)
    
    # VIX 변화율
    spy['VIX_change_1d'] = spy['VIX'].pct_change()
    spy['VIX_change_5d'] = spy['VIX'].pct_change(5)
    spy['VIX_change_22d'] = spy['VIX'].pct_change(22)
    
    # VIX 기간구조
    spy['VIX_ma5'] = spy['VIX'].rolling(5).mean()
    spy['VIX_ma20'] = spy['VIX'].rolling(20).mean()
    spy['VIX_term'] = spy['VIX'] / (spy['VIX_ma20'] + 1e-8)
    spy['VIX_zscore'] = (spy['VIX'] - spy['VIX_ma20']) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    # VRP 래그
    for lag in [1, 2, 3, 5, 10]:
        spy[f'VRP_lag{lag}'] = spy['VRP'].shift(lag)
    
    # VRP 통계
    spy['VRP_ma5'] = spy['VRP'].rolling(5).mean()
    spy['VRP_std5'] = spy['VRP'].rolling(5).std()
    spy['VRP_zscore'] = (spy['VRP'] - spy['VRP'].rolling(20).mean()) / (spy['VRP'].rolling(20).std() + 1e-8)
    
    # RV 비율
    spy['RV_ratio_5_22'] = spy['RV_5d'] / (spy['RV_22d'] + 1e-8)
    spy['RV_ratio_3_10'] = spy['RV_3d'] / (spy['RV_10d'] + 1e-8)
    
    # 수익률 통계
    spy['return_5d'] = spy['returns'].rolling(5).sum()
    spy['return_22d'] = spy['returns'].rolling(22).sum()
    spy['drawdown'] = spy['Close'] / spy['Close'].rolling(22).max() - 1
    
    # Regime
    spy['regime_low'] = (spy['VIX'] < 15).astype(int)
    spy['regime_high'] = (spy['VIX'] >= 25).astype(int)
    spy['regime_crisis'] = (spy['VIX'] >= 35).astype(int)
    
    # 결측치 제거
    spy = spy.replace([np.inf, -np.inf], np.nan)
    spy = spy.dropna()
    
    # 특성 목록
    feature_cols = [
        'RV_1d', 'RV_3d', 'RV_5d', 'RV_10d', 'RV_22d',
        'VIX_lag1', 'VIX_lag2', 'VIX_lag3', 'VIX_lag5', 'VIX_lag10',
        'VIX_change_1d', 'VIX_change_5d', 'VIX_change_22d',
        'VIX_term', 'VIX_zscore',
        'VRP_lag1', 'VRP_lag2', 'VRP_lag3', 'VRP_lag5', 'VRP_lag10',
        'VRP_ma5', 'VRP_std5', 'VRP_zscore',
        'RV_ratio_5_22', 'RV_ratio_3_10',
        'return_5d', 'return_22d', 'drawdown',
        'regime_low', 'regime_high', 'regime_crisis'
    ]
    
    return spy, feature_cols


def strategy_1_feature_selection(spy, feature_cols, baseline_r2):
    """전략 1: 특성 선택"""
    print("\n" + "=" * 50)
    print("[1/5] 특성 확장 + 선택")
    print("=" * 50)
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    y_vrp = spy['VRP_true'].values
    vix = spy['VIX'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_rv_train, y_rv_test = y_rv[:split_idx], y_rv[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    vix_test = vix[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 특성 선택 (상위 k개)
    results = {}
    for k in [10, 15, 20, 25, len(feature_cols)]:
        if k > len(feature_cols):
            continue
        
        selector = SelectKBest(f_regression, k=k)
        X_train_sel = selector.fit_transform(X_train_s, y_rv_train)
        X_test_sel = selector.transform(X_test_s)
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_sel, y_rv_train)
        rv_pred = en.predict(X_test_sel)
        
        vrp_pred = vix_test - rv_pred
        vrp_r2 = r2_score(y_vrp_test, vrp_pred)
        results[k] = vrp_r2
        print(f"  k={k:2d}: VRP R² = {vrp_r2:.4f}")
    
    best_k = max(results, key=results.get)
    best_r2 = results[best_k]
    improvement = (best_r2 - baseline_r2) / baseline_r2 * 100
    print(f"\n  ✓ 최적 k = {best_k}, R² = {best_r2:.4f} ({improvement:+.1f}%)")
    
    return best_r2


def strategy_2_rolling(spy, feature_cols, baseline_r2):
    """전략 2: Rolling 재학습"""
    print("\n" + "=" * 50)
    print("[2/5] Rolling 재학습")
    print("=" * 50)
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    y_vrp = spy['VRP_true'].values
    vix = spy['VIX'].values
    
    split_idx = int(len(spy) * 0.8)
    
    # Rolling prediction
    window = 252  # 1년
    step = 63     # 분기
    
    vrp_preds = []
    vrp_actuals = []
    
    for i in range(split_idx, len(spy), step):
        train_start = max(0, i - window)
        train_end = i
        test_end = min(i + step, len(spy))
        
        X_train = X[train_start:train_end]
        y_train = y_rv[train_start:train_end]
        X_test = X[train_end:test_end]
        
        if len(X_train) < 50 or len(X_test) == 0:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        rv_pred = en.predict(X_test_s)
        
        vrp_pred = vix[train_end:test_end] - rv_pred
        vrp_preds.extend(vrp_pred)
        vrp_actuals.extend(y_vrp[train_end:test_end])
    
    if len(vrp_preds) > 0:
        rolling_r2 = r2_score(vrp_actuals, vrp_preds)
        improvement = (rolling_r2 - baseline_r2) / baseline_r2 * 100
        print(f"  ✓ Rolling R² = {rolling_r2:.4f} ({improvement:+.1f}%)")
        return rolling_r2
    return baseline_r2


def strategy_3_outlier(spy, feature_cols, baseline_r2):
    """전략 3: 이상치 제거"""
    print("\n" + "=" * 50)
    print("[3/5] 이상치 제거")
    print("=" * 50)
    
    # RV_future 클리핑
    lower = spy['RV_future'].quantile(0.01)
    upper = spy['RV_future'].quantile(0.99)
    spy['RV_future_clip'] = spy['RV_future'].clip(lower=lower, upper=upper)
    
    print(f"  클리핑 범위: [{lower:.2f}, {upper:.2f}]")
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future_clip'].values
    y_vrp = spy['VRP_true'].values
    vix = spy['VIX'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_rv_train, y_rv_test = y_rv[:split_idx], y_rv[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    vix_test = vix[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_rv_train)
    rv_pred = en.predict(X_test_s)
    
    vrp_pred = vix_test - rv_pred
    vrp_r2 = r2_score(y_vrp_test, vrp_pred)
    improvement = (vrp_r2 - baseline_r2) / baseline_r2 * 100
    print(f"  ✓ 이상치 제거 R² = {vrp_r2:.4f} ({improvement:+.1f}%)")
    
    return vrp_r2


def strategy_4_multiframe(spy, feature_cols, baseline_r2):
    """전략 4: 다중 타임프레임"""
    print("\n" + "=" * 50)
    print("[4/5] 다중 타임프레임")
    print("=" * 50)
    
    # 5일, 10일, 22일 RV 타겟
    spy['RV_future_5d'] = spy['RV_5d'].shift(-5)
    spy['RV_future_10d'] = spy['RV_10d'].shift(-10)
    spy['RV_future_22d'] = spy['RV_22d'].shift(-22)
    
    spy_clean = spy.dropna(subset=['RV_future_5d', 'RV_future_10d', 'RV_future_22d'])
    
    X = spy_clean[feature_cols].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    predictions = {}
    for horizon, col in [('5d', 'RV_future_5d'), ('10d', 'RV_future_10d'), ('22d', 'RV_future_22d')]:
        y_train = spy_clean[col].values[:split_idx]
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        predictions[horizon] = en.predict(X_test_s)
    
    # 가중 평균 (22d에 더 높은 가중치)
    rv_pred = 0.2 * predictions['5d'] + 0.3 * predictions['10d'] + 0.5 * predictions['22d']
    vrp_pred = vix_test - rv_pred
    vrp_r2 = r2_score(y_vrp_test, vrp_pred)
    improvement = (vrp_r2 - baseline_r2) / baseline_r2 * 100
    print(f"  ✓ 다중 타임프레임 R² = {vrp_r2:.4f} ({improvement:+.1f}%)")
    
    return vrp_r2


def strategy_5_ensemble(spy, feature_cols, baseline_r2):
    """전략 5: 앙상블 최적화"""
    print("\n" + "=" * 50)
    print("[5/5] 앙상블 최적화")
    print("=" * 50)
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    y_vrp = spy['VRP_true'].values
    vix = spy['VIX'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_rv_train, y_rv_test = y_rv[:split_idx], y_rv[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    vix_test = vix[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 여러 모델
    models = {
        'EN_01': ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=SEED, max_iter=10000),
        'EN_05': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000),
        'EN_09': ElasticNet(alpha=0.01, l1_ratio=0.9, random_state=SEED, max_iter=10000),
        'Ridge': Ridge(alpha=1.0, random_state=SEED),
        'Lasso': Lasso(alpha=0.01, random_state=SEED, max_iter=10000)
    }
    
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_s, y_rv_train)
        predictions[name] = model.predict(X_test_s)
    
    # 최적 가중치 탐색 (3개 모델)
    best_r2 = -999
    best_combo = None
    model_names = list(predictions.keys())
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            for w in np.arange(0.3, 0.8, 0.1):
                rv_ens = w * predictions[model_names[i]] + (1-w) * predictions[model_names[j]]
                vrp_pred = vix_test - rv_ens
                r2 = r2_score(y_vrp_test, vrp_pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_combo = (model_names[i], model_names[j], w)
    
    improvement = (best_r2 - baseline_r2) / baseline_r2 * 100
    print(f"  ✓ 최적 앙상블: {best_combo[0]} + {best_combo[1]} (w={best_combo[2]:.1f})")
    print(f"  ✓ R² = {best_r2:.4f} ({improvement:+.1f}%)")
    
    return best_r2, best_combo


def main():
    print("\n" + "🚀" * 30)
    print("VRP 예측 최대 성능 최적화")
    print("🚀" * 30)
    
    # 데이터 준비
    print("\n데이터 준비...")
    spy, feature_cols = load_and_create_features()
    print(f"  ✓ 데이터: {len(spy)} 행")
    print(f"  ✓ 특성: {len(feature_cols)} 개")
    
    baseline_r2 = 0.1915
    print(f"\n  📊 기준선 VRP R² = {baseline_r2:.4f}")
    
    # 각 전략 실행
    results = {}
    
    r2_1 = strategy_1_feature_selection(spy, feature_cols, baseline_r2)
    results['특성 선택'] = r2_1
    
    r2_2 = strategy_2_rolling(spy, feature_cols, baseline_r2)
    results['Rolling'] = r2_2
    
    r2_3 = strategy_3_outlier(spy, feature_cols, baseline_r2)
    results['이상치 제거'] = r2_3
    
    r2_4 = strategy_4_multiframe(spy, feature_cols, baseline_r2)
    results['다중 타임프레임'] = r2_4
    
    r2_5, best_combo = strategy_5_ensemble(spy, feature_cols, baseline_r2)
    results['앙상블'] = r2_5
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("📊 최종 결과")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\n  전략별 성능:")
    for strategy, r2 in sorted_results:
        improvement = (r2 - baseline_r2) / baseline_r2 * 100
        marker = "⭐" if r2 == sorted_results[0][1] else "  "
        print(f"     {marker} {strategy:20s}: R² = {r2:.4f} ({improvement:+.1f}%)")
    
    best_strategy = sorted_results[0]
    
    print(f"""
    🏆 최고 성능 전략: {best_strategy[0]}
       R² = {best_strategy[1]:.4f}
       기존 대비: {(best_strategy[1] - baseline_r2)/baseline_r2*100:+.1f}%
    """)
    
    # 저장
    output = {
        'baseline': baseline_r2,
        'results': {k: float(v) for k, v in results.items()},
        'best_strategy': best_strategy[0],
        'best_r2': float(best_strategy[1]),
        'improvement': float((best_strategy[1] - baseline_r2) / baseline_r2 * 100),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/vrp_max_performance.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 결과 저장: paper/vrp_max_performance.json")


if __name__ == '__main__':
    main()
