"""
Advanced Universal Model Experiments
Experiment 8: Comprehensive Final Models

CatBoost, HistGradientBoosting, Polynomial Ridge, Gaussian Process, 
Quantile RF, Optimized Blending
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import (
    HistGradientBoostingRegressor, 
    RandomForestRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import json
import warnings
warnings.filterwarnings('ignore')

# Try CatBoost
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'

def load_data():
    all_data = {}
    for ticker in ASSETS:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
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
    df = df.copy()
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    df['RV_10d'] = df['Return'].rolling(10).std() * np.sqrt(252) * 100
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    return df.dropna()

def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train all remaining models"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    predictions = {}
    
    # 1. ElasticNet baseline
    m = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    m.fit(X_train_s, y_train)
    predictions['elastic'] = m.predict(X_test_s)
    results['elasticnet'] = r2_score(y_test, predictions['elastic'])
    
    # 2. HistGradientBoosting (sklearn's built-in, fast and robust)
    m = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        learning_rate=0.05,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
    m.fit(X_train, y_train)
    predictions['histgb'] = m.predict(X_test)
    results['hist_gradient_boosting'] = r2_score(y_test, predictions['histgb'])
    
    # 3. CatBoost
    if HAS_CATBOOST:
        m = CatBoostRegressor(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            early_stopping_rounds=30,
            verbose=False,
            random_state=42
        )
        m.fit(X_train, y_train, eval_set=(X_val, y_val))
        predictions['catboost'] = m.predict(X_test)
        results['catboost'] = r2_score(y_test, predictions['catboost'])
    else:
        results['catboost'] = None
    
    # 4. Polynomial Features + Ridge (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)
    
    m = Ridge(alpha=10.0)  # Strong regularization for many features
    m.fit(X_train_poly, y_train)
    predictions['poly_ridge'] = m.predict(X_test_poly)
    results['polynomial_ridge'] = r2_score(y_test, predictions['poly_ridge'])
    
    # 5. Gaussian Process (on subset due to O(n^3) complexity)
    sample_size = min(500, len(X_train_s))
    idx = np.random.choice(len(X_train_s), sample_size, replace=False)
    X_gp = X_train_s[idx]
    y_gp = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
    
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    m = GaussianProcessRegressor(kernel=kernel, alpha=0.5, random_state=42, n_restarts_optimizer=2)
    m.fit(X_gp, y_gp)
    predictions['gp'] = m.predict(X_test_s)
    results['gaussian_process'] = r2_score(y_test, predictions['gp'])
    
    # 6. Quantile Random Forest (using standard RF as approximation)
    m = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    m.fit(X_train, y_train)
    predictions['qrf'] = m.predict(X_test)
    results['quantile_rf'] = r2_score(y_test, predictions['qrf'])
    
    # 7. Optimized Blending (find best weights using validation set)
    val_preds = {
        'elastic': ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000).fit(X_train_s, y_train).predict(X_val_s),
        'histgb': HistGradientBoostingRegressor(max_iter=100, max_depth=5, min_samples_leaf=20, random_state=42).fit(X_train, y_train).predict(X_val),
        'ridge': Ridge(alpha=1.0).fit(X_train_s, y_train).predict(X_val_s)
    }
    
    def blend_loss(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        pred = sum(w * val_preds[k] for w, k in zip(weights, val_preds.keys()))
        return -r2_score(y_val, pred)  # Negative because we minimize
    
    # Optimize weights
    result = minimize(blend_loss, x0=[0.33, 0.33, 0.34], method='Nelder-Mead')
    opt_weights = result.x / result.x.sum()
    
    # Apply to test set
    test_preds = {
        'elastic': predictions['elastic'],
        'histgb': predictions['histgb'],
        'ridge': Ridge(alpha=1.0).fit(X_train_s, y_train).predict(X_test_s)
    }
    
    opt_pred = sum(w * test_preds[k] for w, k in zip(opt_weights, test_preds.keys()))
    results['optimized_blend'] = r2_score(y_test, opt_pred)
    results['blend_weights'] = {k: float(w) for k, w in zip(test_preds.keys(), opt_weights)}
    
    return results

def run_experiment():
    print("=" * 65)
    print("Advanced Universal Model Experiment 8")
    print("CatBoost, HistGB, Polynomial Ridge, GP, Quantile RF, Opt Blend")
    print("=" * 65)
    
    print(f"\nCatBoost available: {HAS_CATBOOST}")
    
    print("\n[1/2] Loading data...")
    all_data, vix = load_data()
    
    all_results = {}
    
    for ticker in ASSETS:
        print(f"\n[2/2] Processing {ticker}...")
        df = create_features(all_data[ticker], vix)
        
        feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'RV_10d', 'VIX', 'VIX_lag1', 
                       'VIX_lag5', 'VIX_change', 'VIX_ma22', 'CAVB', 'CAVB_lag1', 
                       'CAVB_lag5', 'CAVB_ma5']
        X = df[feature_cols]
        y = df['Target']
        
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        
        results = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
        all_results[ticker] = results
        
        print(f"    ElasticNet:      {results['elasticnet']:.4f}")
        print(f"    HistGradientGB:  {results['hist_gradient_boosting']:.4f}")
        if results['catboost']:
            print(f"    CatBoost:        {results['catboost']:.4f}")
        print(f"    Polynomial Ridge:{results['polynomial_ridge']:.4f}")
        print(f"    Gaussian Process:{results['gaussian_process']:.4f}")
        print(f"    Quantile RF:     {results['quantile_rf']:.4f}")
        print(f"    Optimized Blend: {results['optimized_blend']:.4f}")
    
    # Summary
    print("\n" + "=" * 65)
    print("AVERAGE R² BY MODEL")
    print("=" * 65)
    
    models = ['elasticnet', 'hist_gradient_boosting', 'catboost', 
              'polynomial_ridge', 'gaussian_process', 'quantile_rf', 'optimized_blend']
    
    baseline = np.mean([all_results[t]['elasticnet'] for t in ASSETS])
    
    for m in models:
        valid = [all_results[t][m] for t in ASSETS if all_results[t][m] is not None]
        if valid:
            avg = np.mean(valid)
            imp = (avg - baseline) / abs(baseline) * 100
            print(f"{m:<20}: {avg:.4f} ({imp:+.1f}%) [{len(valid)}/5]")
        else:
            print(f"{m:<20}: Not available")
    
    # Find best
    best_avg = 0
    best_model = None
    for m in models:
        valid = [all_results[t][m] for t in ASSETS if all_results[t].get(m) is not None]
        if valid:
            avg = np.mean(valid)
            if avg > best_avg:
                best_avg = avg
                best_model = m
    
    print(f"\nBest Model: {best_model} (avg R² = {best_avg:.4f})")
    
    # Save
    output = {
        'experiment': 'Exp8_Comprehensive_Final',
        'catboost_available': HAS_CATBOOST,
        'results': {k: {m: v[m] if not isinstance(v[m], dict) else str(v[m]) 
                       for m in v.keys()} for k, v in all_results.items()},
        'summary': {
            'best_model': best_model,
            'best_avg_r2': best_avg
        }
    }
    
    with open('results/advanced_experiment_8.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/advanced_experiment_8.json")
    return output

if __name__ == '__main__':
    results = run_experiment()
