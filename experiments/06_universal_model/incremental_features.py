"""
Incremental Feature Engineering - Stable Groups
각 그룹별로 안정성 검증
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
from scipy import stats
from src.data import download_data, prepare_features, three_way_split
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import SPLIT_RATIOS
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import json


def add_group1_higher_moments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 1: Higher Moments (가장 안정적)
    - Realized Skewness
    - Realized Kurtosis
    """
    # Skewness
    df['RS_skew'] = df['returns'].rolling(22).apply(
        lambda x: stats.skew(x) if len(x) > 3 else 0
    ).fillna(0)
    
    # Kurtosis
    df['RK_kurt'] = df['returns'].rolling(22).apply(
        lambda x: stats.kurtosis(x, fisher=False) if len(x) > 4 else 3
    ).fillna(3)
    
    return df


def add_group2_vrp_decomp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 2: VRP Decomposition
    - Persistent component
    - Transitory component
    """
    vrp = df['CAVB']
    
    # Persistent (장기 평균)
    df['VRP_persistent'] = vrp.rolling(60).mean().fillna(vrp.rolling(20).mean()).fillna(0)
    
    # Transitory (단기 변동)
    df['VRP_transitory'] = (vrp - df['VRP_persistent']).fillna(0)
    
    return df


def add_group3_good_bad_vol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 3: Good/Bad Volatility
    """
    # Good volatility (upside)
    df['good_vol'] = df['returns'].rolling(22).apply(
        lambda x: x[x > 0].std() * np.sqrt(252) * 100 if len(x[x > 0]) > 1 else 0
    ).fillna(0)
    
    # Bad volatility (downside)  
    df['bad_vol'] = df['returns'].rolling(22).apply(
        lambda x: abs(x[x < 0].std()) * np.sqrt(252) * 100 if len(x[x < 0]) > 1 else 0
    ).fillna(0)
    
    # Ratio
    df['bad_good_ratio'] = (df['bad_vol'] / (df['good_vol'] + 1e-6)).replace([np.inf, -np.inf], 1).fillna(1)
    
    return df


def add_group4_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 4: VIX Term Structure
    """
    # VIX slopes
    df['VIX_slope_short'] = (df['VIX'].pct_change(5) - df['VIX'].pct_change(1)).fillna(0)
    df['VIX_slope_long'] = (df['VIX'].pct_change(22) - df['VIX'].pct_change(5)).fillna(0)
    df['VIX_curvature'] = (df['VIX_slope_long'] - df['VIX_slope_short']).fillna(0)
    
    return df


def run_incremental_experiment(ticker='GLD', horizon=5):
    """
    단계별 피처 추가 실험
    """
    print(f"\n{'='*60}")
    print(f"Incremental Feature Engineering: {ticker}")
    print(f"{'='*60}\n")
    
    # 데이터 로드
    asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    
    # Baseline features (25)
    df = prepare_features(asset, vix, horizon=horizon, include_enhanced=True)
    
    baseline_features = [
        'RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
        'RV_10d', 'RV_std_22d', 'RV_momentum', 'RV_acceleration',
        'VIX_lag10', 'VIX_lag22', 'VIX_ma5', 'VIX_ma22', 'VIX_zscore',
        'CAVB_percentile', 'CAVB_std_22d', 'CAVB_max_22d', 'CAVB_min_22d',
        'RV_VIX_ratio', 'RV_VIX_product', 'CAVB_VIX_ratio'
    ]
    
    # 각 그룹 추가
    df = add_group1_higher_moments(df)
    df = add_group2_vrp_decomp(df)
    df = add_group3_good_bad_vol(df)
    df = add_group4_term_structure(df)
    
    # NaN 제거
    df = df.dropna()
    
    print(f"Data shape after processing: {df.shape}")
    
    if len(df) < 100:
        print("ERROR: Too few samples after dropna!")
        return None
    
    # 타겟
    y = df[f'CAVB_target_{horizon}d'].values
    
    # 피처 그룹 정의
    groups = {
        'Baseline (25)': baseline_features,
        '+ Group1 (27)': baseline_features + ['RS_skew', 'RK_kurt'],
        '+ Group2 (29)': baseline_features + ['RS_skew', 'RK_kurt', 'VRP_persistent', 'VRP_transitory'],
        '+ Group3 (32)': baseline_features + ['RS_skew', 'RK_kurt', 'VRP_persistent', 'VRP_transitory',
                                               'good_vol', 'bad_vol', 'bad_good_ratio'],
        '+ Group4 (35)': baseline_features + ['RS_skew', 'RK_kurt', 'VRP_persistent', 'VRP_transitory',
                                               'good_vol', 'bad_vol', 'bad_good_ratio',
                                               'VIX_slope_short', 'VIX_slope_long', 'VIX_curvature']
    }
    
    results = {}
    
    for group_name, features in groups.items():
        # 피처 존재 확인
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < len(features):
            print(f"\nWarning: {group_name} - Missing {len(features) - len(available_features)} features")
        
        X = df[available_features].values
        
        # Split
        X_train, _, X_test, y_train, _, y_test = three_way_split(
            X, y,
            train_ratio=SPLIT_RATIOS['train'],
            val_ratio=SPLIT_RATIOS['val'],
            gap=horizon
        )
        
        # 모델
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
        model.fit(X_train_s, y_train)
        
        r2 = r2_score(y_test, model.predict(X_test_s))
        
        results[group_name] = {
            'r2': float(r2),
            'n_features': len(available_features),
            'features': available_features
        }
        
        print(f"{group_name:20s} R² {r2:.4f}")
    
    # 개선도 계산
    baseline_r2 = results['Baseline (25)']['r2']
    
    print(f"\n{'='*60}")
    print("Improvement Analysis")
    print(f"{'='*60}")
    
    for name, res in results.items():
        if name != 'Baseline (25)':
            improvement = (res['r2'] / baseline_r2 - 1) * 100
            print(f"{name:20s} {improvement:+.2f}%")
    
    # 최고 성능
    best_group = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\nBest: {best_group[0]} with R² {best_group[1]['r2']:.4f}")
    
    return {
        'ticker': ticker,
        'results': results,
        'best_group': best_group[0],
        'best_r2': best_group[1]['r2']
    }


if __name__ == '__main__':
    all_results = {}
    
    for ticker in ASSETS.keys():
        try:
            result = run_incremental_experiment(ticker, horizon=5)
            if result:
                all_results[ticker] = result
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # 저장
    output_file = 'results/incremental_features_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 최종 요약
    print("\n" + "="*60)
    print("FINAL SUMMARY - All Assets")
    print("="*60)
    
    for ticker, res in all_results.items():
        baseline_r2 = res['results']['Baseline (25)']['r2']
        best_r2 = res['best_r2']
        improvement = (best_r2 / baseline_r2 - 1) * 100
        
        print(f"{ticker}: Baseline {baseline_r2:.4f} → {res['best_group']} {best_r2:.4f} ({improvement:+.2f}%)")
    
    print(f"\nResults saved to {output_file}")
