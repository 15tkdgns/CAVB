"""
Regime-Adaptive Model - Week 1 Implementation
VIX 기반 Market Regime Indicator 추가
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
from src.data import download_data, prepare_features, extract_features_and_target, three_way_split
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import ELASTIC_NET_PARAMS, SPLIT_RATIOS
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import json

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market Regime Indicator 추가
    
    Args:
        df: prepare_features 결과
    
    Returns:
        Regime 변수가 추가된 DataFrame
    """
    # VIX 기반 Regime 분류
    df['regime'] = pd.cut(
        df['VIX'], 
        bins=[0, 15, 25, 100], 
        labels=['low', 'mid', 'high']
    )
    
    # Dummy variables
    df['regime_low'] = (df['regime'] == 'low').astype(float)
    df['regime_mid'] = (df['regime'] == 'mid').astype(float)
    df['regime_high'] = (df['regime'] == 'high').astype(float)
    
    # VIX Term Structure (slope) - 단순화: VIX 변화율 사용
    df['VIX_slope'] = df['VIX'].pct_change(5)
    
    return df


def analyze_regime_performance(ticker='GLD', horizon=5):
    """
    Regime별 성능 분석
    """
    print(f"\n{'='*60}")
    print(f"Regime Analysis: {ticker}")
    print(f"{'='*60}\n")
    
    # 데이터 로드
    asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    
    # 피처 준비 (Regime 포함)
    df_base = prepare_features(asset, vix, horizon=horizon, include_regime=False)
    df_regime = prepare_features(asset, vix, horizon=horizon, include_regime=True)
    
    # Regime 분포 확인
    print("Regime Distribution:")
    print(df_regime['regime'].value_counts())
    print(f"\nMean VIX by Regime:")
    print(df_regime.groupby('regime')['VIX'].mean())
    print()
    
    # 기본 피처 + Regime    
    # 피처 추출
    X_base, _, y_cavb_base = extract_features_and_target(df_base, horizon=horizon, include_regime=False)
    X_regime, _, y_cavb_regime = extract_features_and_target(df_regime, horizon=horizon, include_regime=True)
    
    # 데이터 길이 확인 및 정렬
    min_len = min(len(X_base), len(X_regime))
    X_base = X_base[:min_len]
    X_regime = X_regime[:min_len]
    y_cavb = y_cavb_base[:min_len]
    
    # Split
    X_train_b, X_val_b, X_test_b, y_train, y_val, y_test = three_way_split(
        X_base, y_cavb,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        gap=horizon
    )
    
    X_train_r, X_val_r, X_test_r, _, _, _ = three_way_split(
        X_regime, y_cavb,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        gap=horizon
    )
    
    # 모델 학습
    scaler_base = RobustScaler()
    X_train_b_s = scaler_base.fit_transform(X_train_b)
    X_test_b_s = scaler_base.transform(X_test_b)
    
    scaler_regime = RobustScaler()
    X_train_r_s = scaler_regime.fit_transform(X_train_r)
    X_test_r_s = scaler_regime.transform(X_test_r)
    
    model_base = ElasticNet(**ELASTIC_NET_PARAMS)
    model_regime = ElasticNet(**ELASTIC_NET_PARAMS)
    
    model_base.fit(X_train_b_s, y_train)
    model_regime.fit(X_train_r_s, y_train)
    
    # 평가
    y_pred_base = model_base.predict(X_test_b_s)
    y_pred_regime = model_regime.predict(X_test_r_s)
    
    r2_base = r2_score(y_test, y_pred_base)
    r2_regime = r2_score(y_test, y_pred_regime)
    
    mae_base = mean_absolute_error(y_test, y_pred_base)
    mae_regime = mean_absolute_error(y_test, y_pred_regime)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline (9 features):")
    print(f"  Test R²:  {r2_base:.4f}")
    print(f"  Test MAE: {mae_base:.4f}")
    print()
    print(f"Regime-Enhanced (12 features):")
    print(f"  Test R²:  {r2_regime:.4f}")
    print(f"  Test MAE: {mae_regime:.4f}")
    print()
    print(f"Improvement:")
    print(f"  ΔR²:  {r2_regime - r2_base:+.4f} ({(r2_regime/r2_base - 1)*100:+.2f}%)")
    print(f"  ΔMAE: {mae_regime - mae_base:+.4f}")
    
    # Regime별 오차 분석
    test_start_idx = len(X_train_b) + horizon + len(X_val_b) + horizon
    test_df = df_regime.iloc[test_start_idx:test_start_idx + len(y_test)].copy()
    test_df['error_base'] = np.abs(y_test - y_pred_base)
    test_df['error_regime'] = np.abs(y_test - y_pred_regime)
    
    print("\n" + "="*60)
    print("Error by Regime")
    print("="*60)
    for regime in ['low', 'mid', 'high']:
        regime_mask = test_df['regime'] == regime
        if regime_mask.sum() > 0:
            err_base = test_df.loc[regime_mask, 'error_base'].mean()
            err_regime = test_df.loc[regime_mask, 'error_regime'].mean()
            print(f"{regime.upper()} Vol ({regime_mask.sum()} samples):")
            print(f"  Baseline MAE:  {err_base:.4f}")
            print(f"  Regime MAE:    {err_regime:.4f}")
            print(f"  Improvement:   {err_regime - err_base:+.4f}")
    
    # Feature Importance
    print("\n" + "="*60)
    print("Top 5 Features (Regime Model)")
    print("="*60)
    
    feature_cols_regime = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                          'VIX_change', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
                          'regime_low', 'regime_high', 'VIX_slope']
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols_regime,
        'coef': np.abs(model_regime.coef_)
    }).sort_values('coef', ascending=False)
    
    print(feature_importance.head().to_string(index=False))
    
    # 결과 저장
    results = {
        'ticker': ticker,
        'baseline': {'r2': float(r2_base), 'mae': float(mae_base)},
        'regime': {'r2': float(r2_regime), 'mae': float(mae_regime)},
        'improvement': {
            'r2_delta': float(r2_regime - r2_base),
            'r2_pct': float((r2_regime/r2_base - 1)*100),
            'mae_delta': float(mae_regime - mae_base)
        },
        'regime_distribution': df_regime['regime'].value_counts().to_dict(),
        'top_features': feature_importance.head(5).to_dict('records')
    }
    
    return results


if __name__ == '__main__':
    # 모든 자산 분석
    all_results = {}
    
    for ticker in ASSETS.keys():
        try:
            results = analyze_regime_performance(ticker, horizon=5)
            all_results[ticker] = results
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue
    
    # 전체 결과 저장
    output_file = 'results/regime_baseline_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 요약
    print("\n" + "="*60)
    print("SUMMARY - All Assets")
    print("="*60)
    
    summary_df = pd.DataFrame([
        {
            'Asset': ticker,
            'Baseline R²': res['baseline']['r2'],
            'Regime R²': res['regime']['r2'],
            'ΔR²': res['improvement']['r2_delta'],
            'Improvement %': res['improvement']['r2_pct']
        }
        for ticker, res in all_results.items()
    ])
    
    print(summary_df.to_string(index=False))
    print(f"\nAverage Improvement: {summary_df['ΔR²'].mean():+.4f} ({summary_df['Improvement %'].mean():+.2f}%)")
    print(f"\nResults saved to {output_file}")
