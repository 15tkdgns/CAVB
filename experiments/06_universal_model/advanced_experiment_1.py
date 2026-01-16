"""
Advanced Universal Model Experiments - Fixed Version
Experiment 1: Cross-Asset Spillover + LightGBM Early Stopping

Target: R² 0.80+ (from current 0.769)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb
import json
import warnings
import re
warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'

def clean_col_name(name):
    """Remove special characters from column names"""
    if isinstance(name, tuple):
        name = '_'.join(str(x) for x in name)
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

def load_data():
    """Load and prepare data for all assets"""
    all_data = {}
    for ticker in ASSETS:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == ticker else '_'.join(col) for col in data.columns]
        data['Return'] = data['Close'].pct_change()
        data['RV_1d'] = data['Return'].abs() * np.sqrt(252) * 100
        data['RV_5d'] = data['Return'].rolling(5).std() * np.sqrt(252) * 100
        data['RV_22d'] = data['Return'].rolling(22).std() * np.sqrt(252) * 100
        all_data[ticker] = data
    
    # Load VIX
    vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix = vix_data[('Close', '^VIX')]
    else:
        vix = vix_data['Close']
    return all_data, vix

def create_cross_asset_features(all_data, vix, target_ticker):
    """Create cross-asset spillover features"""
    df = all_data[target_ticker].copy()
    df['VIX'] = vix
    
    # Basic features
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    # Target: 5-day forward CAVB
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    
    # Cross-Asset Spillover Features
    for other_ticker in ASSETS:
        if other_ticker != target_ticker:
            other_rv = all_data[other_ticker]['RV_22d']
            suffix = other_ticker.replace('^', '')
            
            # RV Spread
            df[f'RV_spread_{suffix}'] = df['RV_22d'] - other_rv
            # Rolling correlation
            df[f'RV_corr_{suffix}'] = df['RV_22d'].rolling(22).corr(other_rv)
            # Lagged RV
            df[f'RV_lag1_{suffix}'] = other_rv.shift(1)
    
    # Average cross-asset RV
    other_rvs = [all_data[t]['RV_22d'] for t in ASSETS if t != target_ticker]
    df['RV_avg_others'] = pd.concat(other_rvs, axis=1).mean(axis=1)
    df['RV_vs_avg'] = df['RV_22d'] - df['RV_avg_others']
    df['VIX_RV_ratio'] = df['VIX'] / (df['RV_avg_others'] + 0.01)
    
    return df.dropna()

def train_lightgbm_early_stop(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train LightGBM with early stopping"""
    # Clean column names
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_train.columns = [clean_col_name(c) for c in X_train.columns]
    X_val.columns = [clean_col_name(c) for c in X_val.columns]
    X_test.columns = [clean_col_name(c) for c in X_test.columns]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'verbose': -1
    }
    
    model = lgb.train(
        params, train_data, num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    
    importance = dict(zip(X_train.columns, model.feature_importance()))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return r2, mae, model.best_iteration, top_features

def train_elasticnet(X_train, y_train, X_test, y_test):
    """Train ElasticNet baseline"""
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    model.fit(X_train_scaled, y_train)
    
    pred = model.predict(X_test_scaled)
    return r2_score(y_test, pred), mean_absolute_error(y_test, pred)

def run_experiment():
    """Run the full experiment"""
    print("=" * 60)
    print("Advanced Universal Model Experiment")
    print("B2: Cross-Asset Spillover + D3: LightGBM Early Stopping")
    print("=" * 60)
    
    print("\n[1/4] Loading data...")
    all_data, vix = load_data()
    
    results = {}
    
    for ticker in ASSETS:
        print(f"\n[2/4] Processing {ticker}...")
        df = create_cross_asset_features(all_data, vix, ticker)
        
        feature_cols = [c for c in df.columns if c not in 
                       ['Target', 'Return', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        X = df[feature_cols]
        y = df['Target']
        
        print(f"    Features: {len(feature_cols)}")
        
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        
        print("    [3/4] Training ElasticNet baseline...")
        baseline_r2, baseline_mae = train_elasticnet(X_train, y_train, X_test, y_test)
        
        print("    [4/4] Training LightGBM with early stopping...")
        lgbm_r2, lgbm_mae, best_iter, top_features = train_lightgbm_early_stop(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        improvement = (lgbm_r2 - baseline_r2) / abs(baseline_r2) * 100 if baseline_r2 != 0 else 0
        
        results[ticker] = {
            'baseline_r2': float(baseline_r2),
            'lgbm_r2': float(lgbm_r2),
            'improvement_pct': float(improvement),
            'best_iteration': best_iter,
            'n_features': len(feature_cols),
            'top_features': [(f, int(v)) for f, v in top_features]
        }
        
        print(f"\n    Results for {ticker}:")
        print(f"    - ElasticNet R²: {baseline_r2:.4f}")
        print(f"    - LightGBM R²:   {lgbm_r2:.4f}")
        print(f"    - Improvement:   {improvement:+.2f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_baseline = np.mean([r['baseline_r2'] for r in results.values()])
    avg_lgbm = np.mean([r['lgbm_r2'] for r in results.values()])
    avg_improvement = (avg_lgbm - avg_baseline) / abs(avg_baseline) * 100 if avg_baseline != 0 else 0
    
    print(f"\nAverage ElasticNet R²: {avg_baseline:.4f}")
    print(f"Average LightGBM R²:   {avg_lgbm:.4f}")
    print(f"Average Improvement:   {avg_improvement:+.2f}%")
    
    output = {
        'experiment': 'B2_D3_CrossAsset_LightGBM',
        'results': results,
        'summary': {
            'avg_baseline_r2': float(avg_baseline),
            'avg_lgbm_r2': float(avg_lgbm),
            'avg_improvement_pct': float(avg_improvement)
        }
    }
    
    with open('results/advanced_experiment_1.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/advanced_experiment_1.json")
    return results

if __name__ == '__main__':
    results = run_experiment()
