"""
Week 3a: Cross-Asset Features Experiment
기존 5개 자산 데이터로 즉시 추가 가능한 7개 변수
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
from src.data import download_data, prepare_features, three_way_split
from src.data.cross_asset_features import add_cross_asset_features, add_downside_rv
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import SPLIT_RATIOS
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import json


def run_cross_asset_experiment(horizon=5):
    """
    Cross-Asset 피처 실험
    """
    print(f"\n{'='*60}")
    print("Cross-Asset Features Experiment")
    print(f"{'='*60}\n")
    
    # === 1. 모든 자산의 RV 계산 ===
    print("Loading all assets...")
    all_assets_data = {}
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    
    for ticker in ASSETS.keys():
        asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
        df = prepare_features(asset, vix, horizon=horizon, include_enhanced=True)
        all_assets_data[ticker] = df
    
    # === 2. 각 자산별로 실험 ===
    results = {}
    
    for target_ticker in ASSETS.keys():
        print(f"\n{'='*60}")
        print(f"Target Asset: {target_ticker}")
        print(f"{'='*60}\n")
        
        # 타겟 자산 데이터
        df_target = all_assets_data[target_ticker].copy()
        
        # 모든 자산의 RV 수집
        all_rv = {}
        for ticker in ASSETS.keys():
            all_rv[ticker] = all_assets_data[ticker]['RV_22d']
        
        # Cross-Asset 피처 추가
        cross_features = add_cross_asset_features(all_rv)
        
        # Downside RV 추가
        df_target['RV_downside'] = add_downside_rv(df_target['returns'], window=22)
        
        # 인덱스 정렬 후 결합
        cross_features = cross_features.reindex(df_target.index)
        df_combined = pd.concat([df_target, cross_features], axis=1).dropna()
        
        # === 피처 추출 ===
        
        # Baseline (25 features)
        baseline_features = [
            'RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 'VIX_change',
            'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
            'RV_10d', 'RV_std_22d', 'RV_momentum', 'RV_acceleration',
            'VIX_lag10', 'VIX_lag22', 'VIX_ma5', 'VIX_ma22', 'VIX_zscore',
            'CAVB_percentile', 'CAVB_std_22d', 'CAVB_max_22d', 'CAVB_min_22d',
            'RV_VIX_ratio', 'RV_VIX_product', 'CAVB_VIX_ratio'
        ]
        
        # Cross-Asset (7 features)
        cross_features_list = [
            'rv_spy_gld_spread', 'rv_spy_tlt_spread', 'rv_efa_eem_spread',
            'rv_cross_mean', 'rv_cross_std', 'rv_cross_range', 'RV_downside'
        ]
        
        X_baseline = df_combined[baseline_features].values
        X_cross = df_combined[baseline_features + cross_features_list].values
        y = df_combined[f'CAVB_target_{horizon}d'].values
        
        # Split
        X_train_b, X_val_b, X_test_b, y_train, y_val, y_test = three_way_split(
            X_baseline, y,
            train_ratio=SPLIT_RATIOS['train'],
            val_ratio=SPLIT_RATIOS['val'],
            gap=horizon
        )
        
        X_train_c, X_val_c, X_test_c, _, _, _ = three_way_split(
            X_cross, y,
            train_ratio=SPLIT_RATIOS['train'],
            val_ratio=SPLIT_RATIOS['val'],
            gap=horizon
        )
        
        # === 모델 학습 ===
        
        # Baseline (25)
        scaler_b = RobustScaler()
        X_train_b_s = scaler_b.fit_transform(X_train_b)
        X_test_b_s = scaler_b.transform(X_test_b)
        
        model_b = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
        model_b.fit(X_train_b_s, y_train)
        
        r2_baseline = r2_score(y_test, model_b.predict(X_test_b_s))
        
        # Cross-Asset (32)
        scaler_c = RobustScaler()
        X_train_c_s = scaler_c.fit_transform(X_train_c)
        X_test_c_s = scaler_c.transform(X_test_c)
        
        model_c = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
        model_c.fit(X_train_c_s, y_train)
        
        r2_cross = r2_score(y_test, model_c.predict(X_test_c_s))
        
        # 결과
        print(f"Baseline (25): R² {r2_baseline:.4f}")
        print(f"Cross (32):    R² {r2_cross:.4f}  ({(r2_cross/r2_baseline-1)*100:+.2f}%)")
        
        # Feature importance (Cross-Asset만)
        cross_importance = np.abs(model_c.coef_[-7:])
        cross_ranking = pd.DataFrame({
            'feature': cross_features_list,
            'coef': cross_importance
        }).sort_values('coef', ascending=False)
        
        print("\nCross-Asset Feature Importance:")
        print(cross_ranking.to_string(index=False))
        
        results[target_ticker] = {
            'baseline_25': float(r2_baseline),
            'cross_32': float(r2_cross),
            'improvement_pct': float((r2_cross/r2_baseline - 1) * 100),
            'top_cross_features': cross_ranking.head(3).to_dict('records')
        }
    
    # 저장
    output_file = 'results/cross_asset_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 요약
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    summary_df = pd.DataFrame([
        {
            'Asset': ticker,
            'Baseline (25)': res['baseline_25'],
            'Cross (32)': res['cross_32'],
            'Improvement %': res['improvement_pct']
        }
        for ticker, res in results.items()
    ])
    
    print(summary_df.to_string(index=False))
    print(f"\nAverage Improvement: {summary_df['Improvement %'].mean():+.2f}%")
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == '__main__':
    results = run_cross_asset_experiment(horizon=5)
