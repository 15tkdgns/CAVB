"""
Horizon Comparison Experiment (Refactored)
5일 vs 22일 예측 성능 비교
"""
import sys
sys.path.insert(0, '../..')

from src.data import download_data, prepare_features, extract_features_and_target, three_way_split
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import ELASTIC_NET_PARAMS, SPLIT_RATIOS
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import json

def run_horizon_comparison(ticker='GLD'):
    """
    Horizon 비교 실험 실행
    """
    print(f"\n{'='*60}")
    print(f"Horizon Comparison: {ticker}")
    print(f"{'='*60}\n")
    
    # 데이터 다운로드
    asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    
    results = {}
    
    for horizon in [1, 5, 22]:
        print(f"\nHorizon = {horizon} days:")
        
        # 피처 준비
        df = prepare_features(asset, vix, horizon=horizon)
        X, y_rv, y_cavb = extract_features_and_target(df, horizon=horizon)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(
            X, y_cavb,
            train_ratio=SPLIT_RATIOS['train'],
            val_ratio=SPLIT_RATIOS['val'],
            gap=horizon
        )
        
        # 학습
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = ElasticNet(**ELASTIC_NET_PARAMS)
        model.fit(X_train_s, y_train)
        
        # 평가
        y_pred = model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  Test R²: {r2:.4f}")
        
        results[f'{horizon}d'] = {
            'r2': float(r2),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    # 결과 저장
    output_file = f'results/{ticker}_horizon_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results

if __name__ == '__main__':
    # 모든 자산에 대해 실행
    all_results = {}
    for ticker in ASSETS.keys():
        all_results[ticker] = run_horizon_comparison(ticker)
    
    # 전체 결과 저장
    with open('results/all_horizons.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
