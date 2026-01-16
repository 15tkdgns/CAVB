"""
Feature Selection Experiment
25개 변수 → Lasso로 자동 선택
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
from src.data import download_data, prepare_features, extract_features_and_target, three_way_split, get_feature_names
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import SPLIT_RATIOS
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import json


def run_feature_selection(ticker='GLD', horizon=5, alpha_lasso=0.001):
    """
    Feature Selection 실험
    
    Args:
        ticker: 자산 티커
        horizon: 예측 시계
        alpha_lasso: Lasso regularization strength
    """
    print(f"\n{'='*60}")
    print(f"Feature Selection: {ticker}")
    print(f"{'='*60}\n")
    
    # 데이터 로드
    asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    
    # 3가지 피처셋 준비
    df_base = prepare_features(asset, vix, horizon=horizon, include_enhanced=False)
    df_enhanced = prepare_features(asset, vix, horizon=horizon, include_enhanced=True)
    
    X_base, _, y_base = extract_features_and_target(df_base, horizon=horizon, include_enhanced=False)
    X_enhanced, _, y_enhanced = extract_features_and_target(df_enhanced, horizon=horizon, include_enhanced=True)
    
    # 데이터 길이 정렬
    min_len = min(len(X_base), len(X_enhanced))
    X_base = X_base[:min_len]
    X_enhanced = X_enhanced[:min_len]
    y = y_base[:min_len]
    
    # Split
    X_train_b, X_val_b, X_test_b, y_train, y_val, y_test = three_way_split(
        X_base, y,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        gap=horizon
    )
    
    X_train_e, X_val_e, X_test_e, _, _, _ = three_way_split(
        X_enhanced, y,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        gap=horizon
    )
    
    # === Baseline (9 features) ===
    print("1. Baseline (9 features):")
    scaler_base = RobustScaler()
    X_train_b_s = scaler_base.fit_transform(X_train_b)
    X_test_b_s = scaler_base.transform(X_test_b)
    
    model_base = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    model_base.fit(X_train_b_s, y_train)
    
    y_pred_base = model_base.predict(X_test_b_s)
    r2_base = r2_score(y_test, y_pred_base)
    mae_base = mean_absolute_error(y_test, y_pred_base)
    
    print(f"  Test R²:  {r2_base:.4f}")
    print(f"  Test MAE: {mae_base:.4f}")
    
    # === Enhanced All (25 features) ===
    print("\n2. Enhanced All (25 features):")
    scaler_enhanced = RobustScaler()
    X_train_e_s = scaler_enhanced.fit_transform(X_train_e)
    X_test_e_s = scaler_enhanced.transform(X_test_e)
    
    model_enhanced = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    model_enhanced.fit(X_train_e_s, y_train)
    
    y_pred_enhanced = model_enhanced.predict(X_test_e_s)
    r2_enhanced = r2_score(y_test, y_pred_enhanced)
    mae_enhanced = mean_absolute_error(y_test, y_pred_enhanced)
    
    print(f"  Test R²:  {r2_enhanced:.4f}")
    print(f"  Test MAE: {mae_enhanced:.4f}")
    
    # === Lasso Feature Selection ===
    print("\n3. Lasso Feature Selection:")
    lasso = Lasso(alpha=alpha_lasso, max_iter=10000)
    lasso.fit(X_train_e_s, y_train)
    
    # 선택된 변수
    feature_names = get_feature_names(include_enhanced=True, include_regime=False)
    selected_mask = np.abs(lasso.coef_) > 1e-6
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
    
    print(f"  Selected: {len(selected_features)}/25 features")
    print(f"  Features: {', '.join(selected_features[:10])}...")
    
    # 선택된 변수로 재학습
    X_train_sel = X_train_e_s[:, selected_mask]
    X_test_sel = X_test_e_s[:, selected_mask]
    
    model_selected = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    model_selected.fit(X_train_sel, y_train)
    
    y_pred_selected = model_selected.predict(X_test_sel)
    r2_selected = r2_score(y_test, y_pred_selected)
    mae_selected = mean_absolute_error(y_test, y_pred_selected)
    
    print(f"\n  Test R²:  {r2_selected:.4f}")
    print(f"  Test MAE: {mae_selected:.4f}")
    
    # 비교
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Baseline (9):    R² {r2_base:.4f}")
    print(f"Enhanced (25):   R² {r2_enhanced:.4f}  ({(r2_enhanced/r2_base-1)*100:+.2f}%)")
    print(f"Selected ({len(selected_features)}):    R² {r2_selected:.4f}  ({(r2_selected/r2_base-1)*100:+.2f}%)")
    
    # Feature Importance
    print("\n" + "="*60)
    print("Top 10 Features (Selected Model)")
    print("="*60)
    
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'coef': np.abs(model_selected.coef_)
    }).sort_values('coef', ascending=False)
    
    print(importance_df.head(10).to_string(index=False))
    
    # 결과 저장
    results = {
        'ticker': ticker,
        'baseline_9': {'r2': float(r2_base), 'mae': float(mae_base), 'n_features': 9},
        'enhanced_25': {'r2': float(r2_enhanced), 'mae': float(mae_enhanced), 'n_features': 25},
        'selected': {'r2': float(r2_selected), 'mae': float(mae_selected), 'n_features': len(selected_features)},
        'selected_features': selected_features,
        'top_10_features': importance_df.head(10).to_dict('records')
    }
    
    return results


if __name__ == '__main__':
    # 모든 자산 분석
    all_results = {}
    
    for ticker in ASSETS.keys():
        try:
            results = run_feature_selection(ticker, horizon=5, alpha_lasso=0.001)
            all_results[ticker] = results
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 전체 결과 저장
    output_file = 'results/feature_selection_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 요약
    print("\n" + "="*60)
    print("SUMMARY - All Assets")
    print("="*60)
    
    summary_df = pd.DataFrame([
        {
            'Asset': ticker,
            'Baseline (9)': res['baseline_9']['r2'],
            'Enhanced (25)': res['enhanced_25']['r2'],
            'Selected': res['selected']['r2'],
            'N_Selected': res['selected']['n_features'],
            'Best Improvement %': max(
                (res['enhanced_25']['r2']/res['baseline_9']['r2'] - 1) * 100,
                (res['selected']['r2']/res['baseline_9']['r2'] - 1) * 100
            )
        }
        for ticker, res in all_results.items()
    ])
    
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {output_file}")
