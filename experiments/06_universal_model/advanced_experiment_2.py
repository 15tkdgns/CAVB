"""
Advanced Universal Model Experiments
Experiment 2: Multi-Task Learning (5 assets simultaneously)

Target: R² 0.80+ using shared representation learning
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'

def load_all_data():
    """Load data for all assets"""
    all_data = {}
    for ticker in ASSETS:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == ticker else '_'.join(str(c) for c in col) for col in data.columns]
        data['Return'] = data['Close'].pct_change()
        data['RV_1d'] = data['Return'].abs() * np.sqrt(252) * 100
        data['RV_5d'] = data['Return'].rolling(5).std() * np.sqrt(252) * 100
        data['RV_22d'] = data['Return'].rolling(22).std() * np.sqrt(252) * 100
        all_data[ticker] = data
    
    vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix = vix_data[('Close', '^VIX')]
    else:
        vix = vix_data['Close']
    return all_data, vix

def create_features(df, vix):
    """Create standard features for an asset"""
    df = df.copy()
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    return df.dropna()

def prepare_multitask_data(all_data, vix):
    """Prepare combined dataset for multi-task learning"""
    combined_dfs = []
    
    for i, ticker in enumerate(ASSETS):
        df = create_features(all_data[ticker], vix)
        df['asset_id'] = i
        df['ticker'] = ticker
        combined_dfs.append(df)
    
    combined = pd.concat(combined_dfs, axis=0)
    combined = combined.sort_index()
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5', 'asset_id']
    
    return combined, feature_cols

def train_single_task_baselines(all_data, vix):
    """Train individual ElasticNet models per asset (baseline)"""
    results = {}
    
    for ticker in ASSETS:
        df = create_features(all_data[ticker], vix)
        
        feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX', 'VIX_lag1', 'VIX_lag5', 
                       'VIX_change', 'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
        X = df[feature_cols]
        y = df['Target']
        
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        
        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)
        
        results[ticker] = {'r2': r2, 'n_samples': len(X_test)}
    
    return results

def train_multitask_model(combined, feature_cols):
    """Train single model on all assets (multi-task)"""
    X = combined[feature_cols]
    y = combined['Target']
    tickers = combined['ticker']
    
    # Time-based split
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    tickers_test = tickers.iloc[val_end:]
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Ridge (shared linear model)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    
    # Model 2: MLP (shared representation)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.01,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    
    # Per-asset performance
    results = {'ridge': {}, 'mlp': {}}
    
    for ticker in ASSETS:
        mask = tickers_test == ticker
        y_true = y_test[mask]
        
        ridge_r2 = r2_score(y_true, ridge_pred[mask])
        mlp_r2 = r2_score(y_true, mlp_pred[mask])
        
        results['ridge'][ticker] = ridge_r2
        results['mlp'][ticker] = mlp_r2
    
    results['ridge']['avg'] = np.mean(list(results['ridge'].values()))
    results['mlp']['avg'] = np.mean(list(results['mlp'].values()))
    
    return results

def run_experiment():
    """Run multi-task learning experiment"""
    print("=" * 60)
    print("Advanced Universal Model Experiment 2")
    print("C1: Multi-Task Learning (5 Assets Simultaneously)")
    print("=" * 60)
    
    print("\n[1/4] Loading data...")
    all_data, vix = load_all_data()
    
    print("\n[2/4] Training single-task baselines...")
    baseline_results = train_single_task_baselines(all_data, vix)
    
    for ticker, res in baseline_results.items():
        print(f"    {ticker}: R² = {res['r2']:.4f}")
    
    avg_baseline = np.mean([r['r2'] for r in baseline_results.values()])
    print(f"    Average: R² = {avg_baseline:.4f}")
    
    print("\n[3/4] Preparing multi-task data...")
    combined, feature_cols = prepare_multitask_data(all_data, vix)
    print(f"    Combined samples: {len(combined)}")
    print(f"    Features: {len(feature_cols)} (including asset_id)")
    
    print("\n[4/4] Training multi-task models...")
    mt_results = train_multitask_model(combined, feature_cols)
    
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    print("\n{:<8} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Asset", "Baseline", "MTL-Ridge", "R vs B", "MTL-MLP", "M vs B"
    ))
    print("-" * 60)
    
    for ticker in ASSETS:
        base = baseline_results[ticker]['r2']
        ridge = mt_results['ridge'][ticker]
        mlp = mt_results['mlp'][ticker]
        
        ridge_imp = (ridge - base) / abs(base) * 100 if base != 0 else 0
        mlp_imp = (mlp - base) / abs(base) * 100 if base != 0 else 0
        
        print("{:<8} {:>12.4f} {:>12.4f} {:>+11.1f}% {:>12.4f} {:>+11.1f}%".format(
            ticker, base, ridge, ridge_imp, mlp, mlp_imp
        ))
    
    print("-" * 60)
    avg_ridge = mt_results['ridge']['avg']
    avg_mlp = mt_results['mlp']['avg']
    ridge_imp = (avg_ridge - avg_baseline) / abs(avg_baseline) * 100 if avg_baseline != 0 else 0
    mlp_imp = (avg_mlp - avg_baseline) / abs(avg_baseline) * 100 if avg_baseline != 0 else 0
    
    print("{:<8} {:>12.4f} {:>12.4f} {:>+11.1f}% {:>12.4f} {:>+11.1f}%".format(
        "Average", avg_baseline, avg_ridge, ridge_imp, avg_mlp, mlp_imp
    ))
    
    # Save results
    output = {
        'experiment': 'C1_MultiTask_Learning',
        'baseline': {k: v['r2'] for k, v in baseline_results.items()},
        'multitask_ridge': mt_results['ridge'],
        'multitask_mlp': mt_results['mlp'],
        'summary': {
            'avg_baseline': avg_baseline,
            'avg_ridge': avg_ridge,
            'avg_mlp': avg_mlp,
            'ridge_improvement': ridge_imp,
            'mlp_improvement': mlp_imp
        }
    }
    
    with open('results/advanced_experiment_2.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/advanced_experiment_2.json")
    return output

if __name__ == '__main__':
    results = run_experiment()
