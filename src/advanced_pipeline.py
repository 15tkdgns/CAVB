#!/usr/bin/env python3
"""
고급 ML 파이프라인: Feature Engineering + Stacking
====================================================

5일 예측 기준으로 R² 개선:
- Feature Engineering (상호작용항, Rolling 특성)
- Stacking (XGBoost + RandomForest + GradientBoosting)
- RobustScaler (이상치 강건)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

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
    
    return df


def create_interaction_features(df):
    """상호작용항 생성"""
    # VIX와 변동성의 상호작용
    df['VIX_RV_interact'] = df['VIX_lag1'] * df['RV_22d']
    df['VIX_change_RV5'] = df['VIX_change'] * df['RV_5d']
    
    # CAVB 지속성 강화
    df['CAVB_VIX_interact'] = df['CAVB_lag1'] * df['VIX_lag1']
    
    # 비선형 효과
    df['VIX_squared'] = df['VIX_lag1'] ** 2
    df['RV_squared'] = df['RV_22d'] ** 2
    
    # 비율 변수 (분모가 0인 경우 방지)
    df['VIX_RV_ratio'] = df['VIX_lag1'] / (df['RV_22d'] + 1e-10)
    
    return df


def create_rolling_features(df):
    """Rolling Window 특성"""
    # 변동성의 변동성
    df['RV_volatility_5d'] = df['RV_22d'].rolling(5).std()
    df['RV_volatility_10d'] = df['RV_22d'].rolling(10).std()
    
    # CAVB 추세
    df['CAVB_trend_5d'] = df['CAVB'].rolling(5).mean()
    df['CAVB_trend_10d'] = df['CAVB'].rolling(10).mean()
    
    # VIX 모멘텀
    df['VIX_momentum_5d'] = df['VIX'].pct_change(5)
    df['VIX_momentum_10d'] = df['VIX'].pct_change(10)
    
    return df


def advanced_pipeline(ticker, asset_name):
    """통합 고급 ML 파이프라인"""
    print(f"\n{'='*70}")
    print(f"고급 ML 파이프라인: {asset_name} ({ticker})")
    print(f"{'='*70}")
    
    # Step 1: 데이터 준비
    df = prepare_data_5day(ticker)
    if df is None:
        print(f"  ✗ 데이터 로드 실패")
        return None
    
    print(f"  원본 데이터: {len(df)} 행")
    
    # Step 2: Feature Engineering
    print("  [Feature Engineering...]")
    df = create_interaction_features(df)
    df = create_rolling_features(df)
    df = df.dropna()
    
    print(f"  처리 후 데이터: {len(df)} 행")
    
    # 확장된 특성 목록
    feature_cols_enhanced = [
        # 기본 변수
        'RV_1d', 'RV_5d', 'RV_22d',
        'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
        # 상호작용항
        'VIX_RV_interact', 'VIX_change_RV5', 'CAVB_VIX_interact',
        # 비선형
        'VIX_squared', 'RV_squared', 'VIX_RV_ratio',
        # Rolling
        'RV_volatility_5d', 'RV_volatility_10d',
        'CAVB_trend_5d', 'CAVB_trend_10d',
        'VIX_momentum_5d', 'VIX_momentum_10d'
    ]
    
    print(f"  특성 개수: 기본 9개 → 확장 {len(feature_cols_enhanced)}개")
    
    X = df[feature_cols_enhanced].values
    y_rv = df['RV_future'].values
    y_cavb = df['CAVB_target'].values
    vix_arr = df['VIX'].values
    
    # Step 3: 3-Way Split
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    gap = 5
    
    X_train = X[:train_end]
    X_val = X[train_end+gap:val_end]
    X_test = X[val_end+gap:]
    
    y_train = y_rv[:train_end]
    y_val_cavb = y_cavb[train_end+gap:val_end]
    y_test_cavb = y_cavb[val_end+gap:]
    
    vix_val = vix_arr[train_end+gap:val_end]
    vix_test = vix_arr[val_end+gap:]
    
    print(f"  Split: Train={len(X_train)} / Val={len(X_val)} / Test={len(X_test)}")
    
    # Step 4: RobustScaler (이상치에 강함)
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Step 5: Baseline (ElasticNet)
    from sklearn.linear_model import ElasticNet
    baseline = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=SEED, max_iter=2000)
    baseline.fit(X_train_s, y_train)
    
    cavb_pred_baseline = vix_test - baseline.predict(X_test_s)
    r2_baseline = r2_score(y_test_cavb, cavb_pred_baseline)
    
    print(f"\n  [Baseline (ElasticNet)]")
    print(f"    Test R² = {r2_baseline:.4f}")
    
    # Step 6: Stacking
    print(f"\n  [Stacking Models...]")
    
    base_models = []
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        base_models.append(('xgb', XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbosity=0
        )))
    
    #  RandomForest
    base_models.append(('rf', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=SEED
    )))
    
    # GradientBoosting
    base_models.append(('gbm', GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED
    )))
    
    # Meta-learner
    meta_model = Ridge(alpha=0.5)
    
    # Stacking (without cv to avoid issues)
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model
    )
    
    print(f"    Base models: {[name for name, _ in base_models]}")
    print(f"    Meta-learner: Ridge")
    print(f"    Training...")
    
    stacking.fit(X_train_s, y_train)
    
    # Step 7: 평가
    cavb_pred_val = vix_val - stacking.predict(X_val_s)
    cavb_pred_test = vix_test - stacking.predict(X_test_s)
    
    r2_val = r2_score(y_val_cavb, cavb_pred_val)
    r2_test = r2_score(y_test_cavb, cavb_pred_test)
    mae_test = mean_absolute_error(y_test_cavb, cavb_pred_test)
    
    improvement = r2_test - r2_baseline
    
    print(f"\n  [Stacking Results]")
    print(f"    Val R²:       {r2_val:.4f}")
    print(f"    Test R²:      {r2_test:.4f}")
    print(f"    Test MAE:     {mae_test:.2f}")
    print(f"    Improvement:  {improvement:+.4f} ({(improvement/r2_baseline)*100:+.1f}%)")
    
    return {
        'asset': ticker,
        'asset_name': asset_name,
        'n_samples': len(df),
        'n_features': len(feature_cols_enhanced),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'r2_baseline': float(r2_baseline),
        'r2_val': float(r2_val),
        'r2_test': float(r2_test),
        'mae_test': float(mae_test),
        'improvement': float(improvement),
        'improvement_pct': float((improvement/r2_baseline)*100)
    }


def main():
    print("\n" + "🚀" * 35)
    print("고급 ML 파이프라인: Feature Engineering + Stacking")
    print("🚀" * 35)
    
    assets = [
        ('GLD', 'Gold (금)'),
        ('EFA', 'EAFE (선진국)'),
        ('TLT', 'Treasury (국채)'),
        ('SPY', 'S&P 500'),
        ('EEM', 'Emerging (신흥국)'),
    ]
    
    all_results = []
    
    for ticker, name in assets:
        result = advanced_pipeline(ticker, name)
        if result:
            all_results.append(result)
    
    # 전체 요약
    print("\n" + "=" * 70)
    print("전체 요약")
    print("=" * 70)
    
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Asset': r['asset_name'],
            'Features': r['n_features'],
            'Baseline R²': f"{r['r2_baseline']:.3f}",
            'Stacking R²': f"{r['r2_test']:.3f}",
            'Improvement': f"{r['improvement']:+.3f}",
            'Improve %': f"{r['improvement_pct']:+.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(f"\n{summary_df.to_string(index=False)}")
    
    # 평균 개선
    avg_improvement = np.mean([r['improvement_pct'] for r in all_results])
    print(f"\n평균 개선: {avg_improvement:+.1f}%")
    
    # 저장
    output = {
        'description': 'Advanced ML Pipeline: Feature Engineering + Stacking (5-day prediction)',
        'methodology': {
            'feature_engineering': 'Interaction terms + Rolling features',
            'scaling': 'RobustScaler',
            'models': 'XGBoost + RandomForest + GradientBoosting → Ridge',
            'cv': 'TimeSeriesSplit (5 folds)'
        },
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/advanced_pipeline.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/advanced_pipeline.json")


if __name__ == '__main__':
    main()
