"""
Feature Experiment: Test New Extended Features
새로운 피처들의 VRP 예측 성과 테스트
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression

from src.data.extended_features import add_all_extended_features
from src.data.novel_features import add_novel_features


def load_data(ticker: str = 'SPY', start: str = '2010-01-01', end: str = '2024-12-31'):
    """데이터 로드 및 기본 피처 생성"""
    print(f"Loading data for {ticker}...")
    
    # Price data
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    # Handle multi-index columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # VIX
    vix_data = yf.download('^VIX', start=start, end=end, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)
    vix = vix_data['Close']
    vix = vix.reindex(data.index).ffill()
    
    # Calculate returns and RV
    data['returns'] = data['Close'].pct_change()
    data['RV_1d'] = data['returns'].abs() * np.sqrt(252) * 100
    data['RV_5d'] = data['returns'].rolling(5).std() * np.sqrt(252) * 100
    data['RV_22d'] = data['returns'].rolling(22).std() * np.sqrt(252) * 100
    data['RV_10d'] = data['returns'].rolling(10).std() * np.sqrt(252) * 100
    
    # VIX features
    data['VIX'] = vix
    data['VIX_lag1'] = vix.shift(1)
    data['VIX_lag5'] = vix.shift(5)
    data['VIX_lag10'] = vix.shift(10)
    data['VIX_lag22'] = vix.shift(22)
    data['VIX_change'] = vix.pct_change(5) * 100
    data['VIX_ma5'] = vix.rolling(5).mean()
    data['VIX_ma22'] = vix.rolling(22).mean()
    data['VIX_zscore'] = (vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-6)
    
    # CAVB (VIX - RV)
    data['CAVB'] = data['VIX'] - data['RV_22d']
    data['CAVB_lag1'] = data['CAVB'].shift(1)
    data['CAVB_lag5'] = data['CAVB'].shift(5)
    data['CAVB_ma5'] = data['CAVB'].rolling(5).mean()
    data['CAVB_std_22d'] = data['CAVB'].rolling(22).std()
    data['CAVB_max_22d'] = data['CAVB'].rolling(22).max()
    data['CAVB_min_22d'] = data['CAVB'].rolling(22).min()
    data['CAVB_percentile'] = data['CAVB'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5
    )
    
    # Interaction features
    data['RV_VIX_ratio'] = data['RV_22d'] / (vix + 1e-6)
    data['RV_VIX_product'] = data['RV_22d'] * vix / 100
    data['CAVB_VIX_ratio'] = data['CAVB'] / (vix + 1e-6)
    
    # VRP decomposition
    vrp = vix - data['RV_22d']
    data['VRP_persistent'] = vrp.rolling(60).mean()
    data['VRP_transitory'] = vrp - data['VRP_persistent']
    
    # Target: 5-day forward CAVB
    data['target'] = data['CAVB'].shift(-5)
    
    return data.dropna()


def prepare_features(df: pd.DataFrame) -> tuple:
    """피처셋 준비"""
    # Base features (original 9)
    base_features = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                     'VIX_change', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
    
    # Extended features
    extended_features = [
        'RV_10d', 'VIX_lag10', 'VIX_lag22', 'VIX_ma5', 'VIX_ma22', 'VIX_zscore',
        'CAVB_std_22d', 'CAVB_max_22d', 'CAVB_min_22d', 'CAVB_percentile',
        'RV_VIX_ratio', 'RV_VIX_product', 'CAVB_VIX_ratio',
        'VRP_persistent', 'VRP_transitory'
    ]
    
    # New features (from extended_features.py)
    new_features = []
    for col in df.columns:
        if col not in base_features + extended_features + ['target', 'returns', 'Close', 
                                                            'Open', 'High', 'Low', 'Volume',
                                                            'Adj Close', 'VIX', 'CAVB', 'RV_1d']:
            if col.startswith(('VIX_', 'TLT_', 'USD_', 'RSI_', 'MACD', 'BB_', 
                              'momentum_', 'gold_', 'bond_', 'risk_', 'sector_', 
                              'vol_regime', 'regime_')):
                new_features.append(col)
    
    return base_features, extended_features, new_features


def run_experiment(df: pd.DataFrame, features: list, model_name: str = 'ElasticNet') -> dict:
    """실험 실행"""
    # Filter features that exist
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) == 0:
        return {'error': 'No features available'}
    
    # Prepare data
    X = df[available_features].values
    y = df['target'].values
    
    # Train/Test split (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model
    models = {
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=5000),
        'Lasso': Lasso(alpha=0.01, max_iter=5000),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    model = models.get(model_name, models['ElasticNet'])
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Direction accuracy
    dir_acc = np.mean((y_test > 0) == (y_pred_test > 0))
    
    # Feature importance (for linear models)
    if hasattr(model, 'coef_'):
        importance = dict(zip(available_features, np.abs(model.coef_).tolist()))
    else:
        importance = {}
    
    return {
        'model': model_name,
        'n_features': len(available_features),
        'features': available_features,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'direction_accuracy': dir_acc,
        'importance': importance
    }


def run_feature_importance_analysis(df: pd.DataFrame, all_features: list) -> dict:
    """피처 중요도 분석"""
    available_features = [f for f in all_features if f in df.columns]
    
    X = df[available_features].values
    y = df['target'].values
    
    # Mutual Information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_ranking = sorted(zip(available_features, mi_scores), key=lambda x: x[1], reverse=True)
    
    return {
        'mutual_info': [{'feature': f, 'score': float(s)} for f, s in mi_ranking[:20]]
    }


def main():
    print("=" * 60)
    print("EXTENDED FEATURE EXPERIMENT")
    print("=" * 60)
    
    # Load and prepare data
    df = load_data('SPY', '2010-01-01', '2024-12-31')
    print(f"Base data: {len(df)} rows, {len(df.columns)} columns")
    
    # Add extended features
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    df = add_all_extended_features(df, start_date, end_date)
    print(f"After extended features: {len(df.columns)} columns")
    
    # Drop rows with NaN target
    df = df.dropna(subset=['target'])
    print(f"Final data: {len(df)} rows")
    
    # Prepare feature sets
    base_features, extended_features, new_features = prepare_features(df)
    all_features = base_features + extended_features + new_features
    
    print(f"\nFeature counts:")
    print(f"  Base: {len(base_features)}")
    print(f"  Extended: {len(extended_features)}")
    print(f"  New: {len(new_features)}")
    print(f"  Total: {len(all_features)}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_info': {
            'rows': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}"
        },
        'feature_counts': {
            'base': len(base_features),
            'extended': len(extended_features),
            'new': len(new_features),
            'total': len(all_features)
        },
        'experiments': {}
    }
    
    # Experiment 1: Base features only
    print("\n[1] Base Features Only (ElasticNet):")
    exp1 = run_experiment(df, base_features, 'ElasticNet')
    results['experiments']['base_only'] = exp1
    print(f"    Train R²: {exp1['train_r2']:.4f}")
    print(f"    Test R²: {exp1['test_r2']:.4f}")
    print(f"    Direction Acc: {exp1['direction_accuracy']:.4f}")
    
    # Experiment 2: Base + Extended
    print("\n[2] Base + Extended Features (ElasticNet):")
    exp2 = run_experiment(df, base_features + extended_features, 'ElasticNet')
    results['experiments']['base_extended'] = exp2
    print(f"    Train R²: {exp2['train_r2']:.4f}")
    print(f"    Test R²: {exp2['test_r2']:.4f}")
    print(f"    Direction Acc: {exp2['direction_accuracy']:.4f}")
    
    # Experiment 3: All features
    print("\n[3] All Features (ElasticNet):")
    exp3 = run_experiment(df, all_features, 'ElasticNet')
    results['experiments']['all_features'] = exp3
    print(f"    Train R²: {exp3['train_r2']:.4f}")
    print(f"    Test R²: {exp3['test_r2']:.4f}")
    print(f"    Direction Acc: {exp3['direction_accuracy']:.4f}")
    
    # Experiment 4: Different models with all features
    print("\n[4] Model Comparison (All Features):")
    model_comparison = {}
    for model_name in ['ElasticNet', 'Lasso', 'Ridge', 'RandomForest', 'GradientBoosting', 'MLP']:
        result = run_experiment(df, all_features, model_name)
        model_comparison[model_name] = {
            'test_r2': result['test_r2'],
            'test_mae': result['test_mae'],
            'direction_accuracy': result['direction_accuracy']
        }
        print(f"    {model_name}: R²={result['test_r2']:.4f}, Dir Acc={result['direction_accuracy']:.4f}")
    
    results['experiments']['model_comparison'] = model_comparison
    
    # Feature importance
    print("\n[5] Feature Importance Analysis:")
    importance = run_feature_importance_analysis(df, all_features)
    results['feature_importance'] = importance
    print("    Top 10 features by Mutual Information:")
    for i, item in enumerate(importance['mutual_info'][:10], 1):
        print(f"    {i}. {item['feature']}: {item['score']:.4f}")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'extended_features_results.json')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
