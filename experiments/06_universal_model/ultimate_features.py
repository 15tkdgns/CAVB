"""
Ultimate Feature Set Experiment
25 baseline + 7 cross-asset + 14 novel = 46 features
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
from src.data import download_data, prepare_features, three_way_split
from src.data.cross_asset_features import add_cross_asset_features, add_downside_rv
from src.data.novel_features import add_novel_features
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import SPLIT_RATIOS
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import json


def run_ultimate_experiment(target_ticker='GLD', horizon=5):
    """
    궁극의 피처셋 실험
    """
    print(f"\n{'='*60}")
    print(f"Ultimate Feature Set: {target_ticker}")
    print(f"{'='*60}\n")
    
    # === 1. 모든 자산 로드 ===
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    all_assets_data = {}
    
    for ticker in ASSETS.keys():
        asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
        df = prepare_features(asset, vix, horizon=horizon, include_enhanced=True)
        all_assets_data[ticker] = df
    
    # === 2. 타겟 자산 ===
    df_target = all_assets_data[target_ticker].copy()
    
    # === 3. Cross-Asset Features ===
    all_rv = {ticker: data['RV_22d'] for ticker, data in all_assets_data.items()}
    cross_features = add_cross_asset_features(all_rv)
    
    # 인덱스 정렬
    cross_features = cross_features.reindex(df_target.index).fillna(method='ffill').fillna(method='bfill')
    df_target = pd.concat([df_target, cross_features], axis=1)
    
    # Downside RV
    df_target['RV_downside'] = add_downside_rv(df_target['returns'], 22)
    
    # === 4. Novel Features ===
    df_target = add_novel_features(df_target)
    
    # 최종 NaN 제거
    df_final = df_target.dropna()
    
    print(f"Final data shape: {df_final.shape}")
    
    # === 5. Feature Sets ===
    
    # Baseline (25)
    baseline_features = [
        'RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
        'RV_10d', 'RV_std_22d', 'RV_momentum', 'RV_acceleration',
        'VIX_lag10', 'VIX_lag22', 'VIX_ma5', 'VIX_ma22', 'VIX_zscore',
        'CAVB_percentile', 'CAVB_std_22d', 'CAVB_max_22d', 'CAVB_min_22d',
        'RV_VIX_ratio', 'RV_VIX_product', 'CAVB_VIX_ratio'
    ]
    
    # Cross-Asset (7)
    cross_asset_features = [
        'rv_spy_gld_spread', 'rv_spy_tlt_spread', 'rv_efa_eem_spread',
        'rv_cross_mean', 'rv_cross_std', 'rv_cross_range', 'RV_downside'
    ]
    
    # Novel (14)
    novel_features = [
        'RS_skew', 'RK_kurt',
        'good_volatility', 'bad_volatility', 'bad_good_ratio',
        'VRP_persistent', 'VRP_transitory', 'VRP_variance_ratio',
        'tail_risk_demand', 'VIX_slope_short', 'VIX_slope_long',
        'VIX_curvature', 'RV_VIX_correlation', 'info_discreteness'
    ]
    
    # 피처 존재 확인
    available_baseline = [f for f in baseline_features if f in df_final.columns]
    available_cross = [f for f in cross_asset_features if f in df_final.columns]
    available_novel = [f for f in novel_features if f in df_final.columns]
    
    print(f"\nAvailable features:")
    print(f"  Baseline: {len(available_baseline)}/25")
    print(f"  Cross-Asset: {len(available_cross)}/7")
    print(f"  Novel: {len(available_novel)}/14")
    
    all_features = available_baseline + available_cross + available_novel
    print(f"  Total: {len(all_features)} features")
    
    # === 6. 데이터 준비 ===
    X_baseline = df_final[available_baseline].values
    X_ultimate = df_final[all_features].values
    y = df_final[f'CAVB_target_{horizon}d'].values
    
    # Split
    X_train_b, _, X_test_b, y_train, _, y_test = three_way_split(
        X_baseline, y,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        gap=horizon
    )
    
    X_train_u, _, X_test_u, _, _, _ = three_way_split(
        X_ultimate, y,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        gap=horizon
    )
    
    # === 7. 모델 학습 ===
    
    # Baseline
    scaler_b = RobustScaler()
    X_train_b_s = scaler_b.fit_transform(X_train_b)
    X_test_b_s = scaler_b.transform(X_test_b)
    
    model_b = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    model_b.fit(X_train_b_s, y_train)
    
    r2_baseline = r2_score(y_test, model_b.predict(X_test_b_s))
    mae_baseline = mean_absolute_error(y_test, model_b.predict(X_test_b_s))
    
    # Ultimate
    scaler_u = RobustScaler()
    X_train_u_s = scaler_u.fit_transform(X_train_u)
    X_test_u_s = scaler_u.transform(X_test_u)
    
    model_u = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    model_u.fit(X_train_u_s, y_train)
    
    r2_ultimate = r2_score(y_test, model_u.predict(X_test_u_s))
    mae_ultimate = mean_absolute_error(y_test, model_u.predict(X_test_u_s))
    
    # Lasso Feature Selection
    lasso = Lasso(alpha=0.001, max_iter=10000)
    lasso.fit(X_train_u_s, y_train)
    
    selected_mask = np.abs(lasso.coef_) > 1e-6
    selected_features = [f for f, s in zip(all_features, selected_mask) if s]
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Baseline ({len(available_baseline)}):  R² {r2_baseline:.4f}, MAE {mae_baseline:.4f}")
    print(f"Ultimate ({len(all_features)}): R² {r2_ultimate:.4f}, MAE {mae_ultimate:.4f}")
    print(f"Improvement: {(r2_ultimate/r2_baseline-1)*100:+.2f}%")
    print(f"\nLasso selected: {len(selected_features)}/{len(all_features)} features")
    
    # Feature Importance (Top 15)
    importance_df = pd.DataFrame({
        'feature': all_features,
        'coef': np.abs(model_u.coef_)
    }).sort_values('coef', ascending=False)
    
    print(f"\n{'='*60}")
    print("Top 15 Features")
    print(f"{'='*60}")
    print(importance_df.head(15).to_string(index=False))
    
    # Top Novel Features
    novel_importance = importance_df[importance_df['feature'].isin(available_novel)].head(5)
    print(f"\n{'='*60}")
    print("Top 5 Novel Features")
    print(f"{'='*60}")
    print(novel_importance.to_string(index=False))
    
    results = {
        'ticker': target_ticker,
        'baseline': {'r2': float(r2_baseline), 'mae': float(mae_baseline), 'n_features': len(available_baseline)},
        'ultimate': {'r2': float(r2_ultimate), 'mae': float(mae_ultimate), 'n_features': len(all_features)},
        'improvement_pct': float((r2_ultimate/r2_baseline - 1) * 100),
        'selected_features': len(selected_features),
        'top_10_features': importance_df.head(10).to_dict('records'),
        'top_novel_features': novel_importance.to_dict('records')
    }
    
    return results


if __name__ == '__main__':
    all_results = {}
    
    for ticker in ASSETS.keys():
        try:
            results = run_ultimate_experiment(ticker, horizon=5)
            all_results[ticker] = results
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 저장
    output_file = 'results/ultimate_features_results.json'
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
            'Ultimate R²': res['ultimate']['r2'],
            'Improvement %': res['improvement_pct'],
            'N Features': f"{res['baseline']['n_features']}→{res['ultimate']['n_features']}"
        }
        for ticker, res in all_results.items()
    ])
    
    print(summary_df.to_string(index=False))
    print(f"\nAverage Improvement: {summary_df['Improvement %'].mean():+.2f}%")
    print(f"\nResults saved to {output_file}")
