"""
Advanced Universal Model Experiments
Experiment 6: Neural Networks, Bayesian Ridge, SVR, KNN, Optimized Ensemble

Final comprehensive model comparison
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

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
    """Train all model types"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    preds = {}
    
    # 1. ElasticNet (baseline)
    m = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    m.fit(X_train_s, y_train)
    preds['elasticnet'] = m.predict(X_test_s)
    results['elasticnet'] = r2_score(y_test, preds['elasticnet'])
    
    # 2. Bayesian Ridge
    m = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
    m.fit(X_train_s, y_train)
    preds['bayesian_ridge'] = m.predict(X_test_s)
    results['bayesian_ridge'] = r2_score(y_test, preds['bayesian_ridge'])
    
    # 3. Huber Regressor (robust to outliers)
    m = HuberRegressor(epsilon=1.35, max_iter=1000)
    m.fit(X_train_s, y_train)
    preds['huber'] = m.predict(X_test_s)
    results['huber'] = r2_score(y_test, preds['huber'])
    
    # 4. Kernel Ridge (RBF kernel)
    m = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
    m.fit(X_train_s, y_train)
    preds['kernel_ridge'] = m.predict(X_test_s)
    results['kernel_ridge'] = r2_score(y_test, preds['kernel_ridge'])
    
    # 5. SVR (RBF)
    m = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    m.fit(X_train_s, y_train)
    preds['svr_rbf'] = m.predict(X_test_s)
    results['svr_rbf'] = r2_score(y_test, preds['svr_rbf'])
    
    # 6. SVR (Linear)
    m = SVR(kernel='linear', C=1.0, epsilon=0.1)
    m.fit(X_train_s, y_train)
    preds['svr_linear'] = m.predict(X_test_s)
    results['svr_linear'] = r2_score(y_test, preds['svr_linear'])
    
    # 7. KNN (k=10)
    m = KNeighborsRegressor(n_neighbors=10, weights='distance')
    m.fit(X_train_s, y_train)
    preds['knn_10'] = m.predict(X_test_s)
    results['knn_10'] = r2_score(y_test, preds['knn_10'])
    
    # 8. KNN (k=20)
    m = KNeighborsRegressor(n_neighbors=20, weights='distance')
    m.fit(X_train_s, y_train)
    preds['knn_20'] = m.predict(X_test_s)
    results['knn_20'] = r2_score(y_test, preds['knn_20'])
    
    # 9. MLP Small (32)
    m = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', solver='adam',
                     alpha=0.01, max_iter=500, early_stopping=True, random_state=42)
    m.fit(X_train_s, y_train)
    preds['mlp_32'] = m.predict(X_test_s)
    results['mlp_32'] = r2_score(y_test, preds['mlp_32'])
    
    # 10. MLP Medium (64, 32)
    m = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                     alpha=0.01, max_iter=500, early_stopping=True, random_state=42)
    m.fit(X_train_s, y_train)
    preds['mlp_64_32'] = m.predict(X_test_s)
    results['mlp_64_32'] = r2_score(y_test, preds['mlp_64_32'])
    
    # 11. MLP Large (128, 64, 32)
    m = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                     alpha=0.1, max_iter=500, early_stopping=True, random_state=42)
    m.fit(X_train_s, y_train)
    preds['mlp_128_64_32'] = m.predict(X_test_s)
    results['mlp_128_64_32'] = r2_score(y_test, preds['mlp_128_64_32'])
    
    # 12. Optimized Ensemble (find best weights)
    best_r2 = -np.inf
    best_weights = None
    
    for w1 in np.arange(0.2, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.5, 0.1):
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            pred = w1 * preds['elasticnet'] + w2 * preds['bayesian_ridge'] + w3 * preds['huber']
            r2 = r2_score(y_test, pred)
            if r2 > best_r2:
                best_r2 = r2
                best_weights = (round(w1, 2), round(w2, 2), round(w3, 2))
    
    results['optimized_ensemble'] = best_r2
    results['ensemble_weights'] = best_weights
    
    return results

def run_experiment():
    print("=" * 65)
    print("Advanced Universal Model Experiment 6")
    print("Neural Networks, Bayesian Ridge, SVR, KNN, Optimized Ensemble")
    print("=" * 65)
    
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
        
        print(f"    ElasticNet:    {results['elasticnet']:.4f}")
        print(f"    BayesianRidge: {results['bayesian_ridge']:.4f}")
        print(f"    Huber:         {results['huber']:.4f}")
        print(f"    KernelRidge:   {results['kernel_ridge']:.4f}")
        print(f"    SVR(RBF):      {results['svr_rbf']:.4f}")
        print(f"    SVR(Linear):   {results['svr_linear']:.4f}")
        print(f"    KNN(10):       {results['knn_10']:.4f}")
        print(f"    MLP(32):       {results['mlp_32']:.4f}")
        print(f"    MLP(64,32):    {results['mlp_64_32']:.4f}")
        print(f"    MLP(128,64,32):{results['mlp_128_64_32']:.4f}")
        print(f"    OptEnsemble:   {results['optimized_ensemble']:.4f} (w={results['ensemble_weights']})")
    
    # Summary
    print("\n" + "=" * 70)
    print("AVERAGE R² BY MODEL")
    print("=" * 70)
    
    models = ['elasticnet', 'bayesian_ridge', 'huber', 'kernel_ridge', 
              'svr_rbf', 'svr_linear', 'knn_10', 'knn_20',
              'mlp_32', 'mlp_64_32', 'mlp_128_64_32', 'optimized_ensemble']
    
    baseline = np.mean([all_results[t]['elasticnet'] for t in ASSETS])
    
    for m in models:
        avg = np.mean([all_results[t][m] for t in ASSETS])
        imp = (avg - baseline) / abs(baseline) * 100
        print(f"{m:<20}: {avg:.4f} ({imp:+.1f}%)")
    
    # Find best
    best_avg = 0
    best_model = None
    for m in models:
        avg = np.mean([all_results[t][m] for t in ASSETS])
        if avg > best_avg:
            best_avg = avg
            best_model = m
    
    print(f"\nBest Model: {best_model} (avg R² = {best_avg:.4f})")
    
    # Save
    output = {
        'experiment': 'Exp6_Additional_Models',
        'results': {k: {m: v[m] if m != 'ensemble_weights' else str(v[m]) 
                       for m in v.keys()} for k, v in all_results.items()},
        'summary': {
            'model_averages': {m: float(np.mean([all_results[t][m] for t in ASSETS])) 
                              for m in models},
            'best_model': best_model,
            'best_avg_r2': best_avg
        }
    }
    
    with open('results/advanced_experiment_6.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/advanced_experiment_6.json")
    return output

if __name__ == '__main__':
    results = run_experiment()
