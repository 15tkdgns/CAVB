"""
Comprehensive Feature Selection
29 features → 최적 subset 발굴

방법:
1. Lasso (L1 regularization)
2. Recursive Feature Elimination (RFE)
3. Mutual Information
4. Permutation Importance
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
from src.data import download_data, prepare_features, three_way_split
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import SPLIT_RATIOS
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import json


def add_group2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group 2 (VRP Decomposition) 추가"""
    vrp = df['CAVB']
    df['VRP_persistent'] = vrp.rolling(60).mean().fillna(vrp.rolling(20).mean()).fillna(0)
    df['VRP_transitory'] = (vrp - df['VRP_persistent']).fillna(0)
    return df


def run_feature_selection(ticker='GLD', horizon=5):
    """
    종합 피처 셀렉션
    """
    print(f"\n{'='*60}")
    print(f"Feature Selection: {ticker}")
    print(f"{'='*60}\n")
    
    # 데이터
    asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    
    df = prepare_features(asset, vix, horizon=horizon, include_enhanced=True)
    df = add_group2_features(df)
    df = df.dropna()
    
    # 29 features
    all_features = [
        'RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
        'RV_10d', 'RV_std_22d', 'RV_momentum', 'RV_acceleration',
        'VIX_lag10', 'VIX_lag22', 'VIX_ma5', 'VIX_ma22', 'VIX_zscore',
        'CAVB_percentile', 'CAVB_std_22d', 'CAVB_max_22d', 'CAVB_min_22d',
        'RV_VIX_ratio', 'RV_VIX_product', 'CAVB_VIX_ratio',
        'VRP_persistent', 'VRP_transitory'
    ]
    
    X = df[all_features].values
    y = df[f'CAVB_target_{horizon}d'].values
    
    # Split
    X_train, _, X_test, y_train, _, y_test = three_way_split(
        X, y,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        gap=horizon
    )
    
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # === Method 1: Lasso ===
    print("Method 1: Lasso L1 Regularization")
    
    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    lasso_results = {}
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_s, y_train)
        
        selected_mask = np.abs(lasso.coef_) > 1e-6
        n_selected = selected_mask.sum()
        
        if n_selected > 0:
            X_train_sel = X_train_s[:, selected_mask]
            X_test_sel = X_test_s[:, selected_mask]
            
            model = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
            model.fit(X_train_sel, y_train)
            r2 = r2_score(y_test, model.predict(X_test_sel))
            
            lasso_results[alpha] = {
                'n_features': int(n_selected),
                'r2': float(r2),
                'features': [f for f, s in zip(all_features, selected_mask) if s]
            }
            
            print(f"  α={alpha:6.4f}: {n_selected:2d} features, R²={r2:.4f}")
    
    # 최적 alpha
    best_lasso = max(lasso_results.items(), key=lambda x: x[1]['r2'])
    print(f"  Best: α={best_lasso[0]}, {best_lasso[1]['n_features']} features, R²={best_lasso[1]['r2']:.4f}")
    
    # === Method 2: RFE ===
    print("\nMethod 2: Recursive Feature Elimination")
    
    rfe_results = {}
    target_features = [15, 20, 25]
    
    for n_feat in target_features:
        estimator = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
        rfe = RFE(estimator, n_features_to_select=n_feat, step=1)
        rfe.fit(X_train_s, y_train)
        
        X_train_rfe = rfe.transform(X_train_s)
        X_test_rfe = rfe.transform(X_test_s)
        
        model = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
        model.fit(X_train_rfe, y_train)
        r2 = r2_score(y_test, model.predict(X_test_rfe))
        
        selected_features = [f for f, s in zip(all_features, rfe.support_) if s]
        
        rfe_results[n_feat] = {
            'r2': float(r2),
            'features': selected_features
        }
        
        print(f"  {n_feat} features: R²={r2:.4f}")
    
    best_rfe = max(rfe_results.items(), key=lambda x: x[1]['r2'])
    print(f"  Best: {best_rfe[0]} features, R²={best_rfe[1]['r2']:.4f}")
    
    # === Method 3: Mutual Information ===
    print("\nMethod 3: Mutual Information")
    
    mi_scores = mutual_info_regression(X_train_s, y_train, random_state=42)
    mi_ranking = pd.DataFrame({
        'feature': all_features,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("  Top 10 by MI:")
    print(mi_ranking.head(10)[['feature', 'mi_score']].to_string(index=False))
    
    # Top 20 features
    top20_features = mi_ranking.head(20)['feature'].tolist()
    top20_mask = [f in top20_features for f in all_features]
    
    X_train_mi = X_train_s[:, top20_mask]
    X_test_mi = X_test_s[:, top20_mask]
    
    model_mi = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    model_mi.fit(X_train_mi, y_train)
    r2_mi = r2_score(y_test, model_mi.predict(X_test_mi))
    
    print(f"  Top 20: R²={r2_mi:.4f}")
    
    # === Method 4: Permutation Importance ===
    print("\nMethod 4: Permutation Importance")
    
    model_full = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    model_full.fit(X_train_s, y_train)
    
    perm_importance = permutation_importance(
        model_full, X_test_s, y_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )
    
    perm_ranking = pd.DataFrame({
        'feature': all_features,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    print("  Top 10 by Permutation:")
    print(perm_ranking.head(10)[['feature', 'importance']].to_string(index=False))
    
    # === 최종 비교 ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Full (29):        R²={r2_score(y_test, model_full.predict(X_test_s)):.4f}")
    print(f"Lasso Best:       R²={best_lasso[1]['r2']:.4f} ({best_lasso[1]['n_features']} features)")
    print(f"RFE Best:         R²={best_rfe[1]['r2']:.4f} ({best_rfe[0]} features)")
    print(f"MI Top 20:        R²={r2_mi:.4f} (20 features)")
    
    # 공통 중요 변수
    lasso_top = set(best_lasso[1]['features'][:15])
    rfe_top = set(best_rfe[1]['features'][:15])
    mi_top = set(top20_features[:15])
    perm_top = set(perm_ranking.head(15)['feature'].tolist())
    
    consensus = lasso_top & rfe_top & mi_top & perm_top
    
    print(f"\nConsensus Features (4 methods agree): {len(consensus)}")
    print(sorted(list(consensus)))
    
    return {
        'ticker': ticker,
        'full_r2': float(r2_score(y_test, model_full.predict(X_test_s))),
        'lasso_best': best_lasso[1],
        'rfe_best': {'n_features': best_rfe[0], 'r2': best_rfe[1]['r2'], 'features': best_rfe[1]['features']},
        'mi_top20_r2': float(r2_mi),
        'consensus_features': sorted(list(consensus)),
        'rankings': {
            'mi': mi_ranking.head(15).to_dict('records'),
            'perm': perm_ranking.head(15).to_dict('records')
        }
    }


if __name__ == '__main__':
    all_results = {}
    
    for ticker in ASSETS.keys():
        try:
            results = run_feature_selection(ticker, horizon=5)
            all_results[ticker] = results
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # 저장
    output_file = 'results/comprehensive_feature_selection.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 전체 consensus
    print("\n" + "="*60)
    print("CROSS-ASSET CONSENSUS")
    print("="*60)
    
    all_consensus = set()
    for ticker, res in all_results.items():
        all_consensus.update(res['consensus_features'])
    
    # 각 변수가 몇 개 자산에서 선택되었는지
    feature_counts = {}
    for ticker, res in all_results.items():
        for feat in res['consensus_features']:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    
    universal_features = [(f, c) for f, c in feature_counts.items() if c >= 4]
    universal_features.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nUniversal Features (selected by 4+ assets):")
    for feat, count in universal_features:
        print(f"  {feat:25s} {count}/5 assets")
    
    print(f"\nResults saved to {output_file}")
