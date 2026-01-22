"""
22-Day Universal Volatility Prediction Model
Comprehensive model development with Optuna HPO
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']
N_TRIALS = 50  # Optuna trials per model

print('='*65)
print('22-Day Universal Model Development')
print('Comprehensive Optuna Optimization')
print('='*65)

# Load data
all_data = {}
for t in ASSETS:
    d = yf.download(t, start='2015-01-01', end='2025-01-01', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d['Ret'] = d['Close'].pct_change()
    d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
    d['RV_10d'] = d['Ret'].rolling(10).std() * np.sqrt(252) * 100
    d['RV_22d'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
    d['RV_66d'] = d['Ret'].rolling(66).std() * np.sqrt(252) * 100
    all_data[t] = d

vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix[('Close', '^VIX')]
else:
    vix = vix['Close']

print('Data loaded\n')

def create_features_22d(df, vix):
    """Create features optimized for 22-day prediction"""
    df = df.copy()
    
    # VIX features (longer lags for 22-day)
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_lag22'] = df['VIX'].shift(22)
    df['VIX_change'] = df['VIX'].pct_change()
    df['VIX_change_22d'] = df['VIX'].pct_change(22)
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['VIX_ma66'] = df['VIX'].rolling(66).mean()
    df['VIX_std22'] = df['VIX'].rolling(22).std()
    
    # CAVB features
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_lag22'] = df['CAVB'].shift(22)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    df['CAVB_ma22'] = df['CAVB'].rolling(22).mean()
    df['CAVB_ma66'] = df['CAVB'].rolling(66).mean()
    
    # Longer horizon RV features
    df['RV_ratio_22_66'] = df['RV_22d'] / (df['RV_66d'] + 0.01)
    df['RV_momentum_22'] = df['RV_22d'].pct_change(22)
    
    # Target: 22-day ahead
    df['Target_22d'] = df['VIX'] - df['RV_22d'].shift(-22)
    
    return df.dropna()

results = {}

for ticker in ASSETS:
    print(f'\nProcessing {ticker}...')
    
    df = create_features_22d(all_data[ticker], vix)
    
    fcols = ['VIX', 'VIX_lag1', 'VIX_lag5', 'VIX_lag22', 'VIX_change', 
             'VIX_change_22d', 'VIX_ma22', 'VIX_ma66', 'VIX_std22',
             'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_lag22',
             'CAVB_ma5', 'CAVB_ma22', 'CAVB_ma66',
             'RV_5d', 'RV_10d', 'RV_22d', 'RV_66d',
             'RV_ratio_22_66', 'RV_momentum_22']
    
    X, y = df[fcols], df['Target_22d']
    n = len(X)
    tr_end, val_end = int(n*0.6), int(n*0.8)
    
    Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
    Xval, yval = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    Xte, yte = X.iloc[val_end:], y.iloc[val_end:]
    
    Xtrval = pd.concat([Xtr, Xval])
    ytrval = pd.concat([ytr, yval])
    
    scaler = StandardScaler()
    Xts = scaler.fit_transform(Xtr)
    Xvs = scaler.transform(Xval)
    Xtes = scaler.transform(Xte)
    
    scaler_full = StandardScaler()
    Xtvs = scaler_full.fit_transform(Xtrval)
    Xtes_full = scaler_full.transform(Xte)
    
    r = {'n_features': len(fcols)}
    
    # 1. Baseline Huber
    hub_base = HuberRegressor(epsilon=1.35, max_iter=1000)
    hub_base.fit(Xts, ytr)
    r['huber_baseline'] = r2_score(yte, hub_base.predict(Xtes))
    print(f'  Huber Baseline: {r["huber_baseline"]:.4f}')
    
    # 2. Baseline Ridge (often better for longer horizon)
    ridge_base = Ridge(alpha=1.0)
    ridge_base.fit(Xts, ytr)
    r['ridge_baseline'] = r2_score(yte, ridge_base.predict(Xtes))
    print(f'  Ridge Baseline: {r["ridge_baseline"]:.4f}')
    
    # 3. Optuna Huber Tuning
    print(f'  Tuning Huber ({N_TRIALS} trials)...')
    def obj_huber(trial):
        epsilon = trial.suggest_float('epsilon', 1.0, 3.0)
        alpha = trial.suggest_float('alpha', 1e-6, 1.0, log=True)
        m = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)
        m.fit(Xts, ytr)
        return r2_score(yval, m.predict(Xvs))
    
    study_hub = optuna.create_study(direction='maximize')
    study_hub.optimize(obj_huber, n_trials=N_TRIALS, show_progress_bar=False)
    
    hub_best = HuberRegressor(**study_hub.best_params, max_iter=1000)
    hub_best.fit(Xtvs, ytrval)
    r['huber_tuned'] = r2_score(yte, hub_best.predict(Xtes_full))
    r['huber_params'] = study_hub.best_params
    print(f'  Huber Tuned:    {r["huber_tuned"]:.4f}')
    
    # 4. Optuna Ridge Tuning
    print(f'  Tuning Ridge ({N_TRIALS} trials)...')
    def obj_ridge(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 100.0, log=True)
        m = Ridge(alpha=alpha)
        m.fit(Xts, ytr)
        return r2_score(yval, m.predict(Xvs))
    
    study_ridge = optuna.create_study(direction='maximize')
    study_ridge.optimize(obj_ridge, n_trials=N_TRIALS, show_progress_bar=False)
    
    ridge_best = Ridge(**study_ridge.best_params)
    ridge_best.fit(Xtvs, ytrval)
    r['ridge_tuned'] = r2_score(yte, ridge_best.predict(Xtes_full))
    r['ridge_params'] = study_ridge.best_params
    print(f'  Ridge Tuned:    {r["ridge_tuned"]:.4f}')
    
    # 5. Optuna ElasticNet Tuning (for long horizon sparsity)
    print(f'  Tuning ElasticNet ({N_TRIALS} trials)...')
    def obj_elastic(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 10.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        m.fit(Xts, ytr)
        return r2_score(yval, m.predict(Xvs))
    
    study_elas = optuna.create_study(direction='maximize')
    study_elas.optimize(obj_elastic, n_trials=N_TRIALS, show_progress_bar=False)
    
    elas_best = ElasticNet(**study_elas.best_params, max_iter=10000)
    elas_best.fit(Xtvs, ytrval)
    r['elasticnet_tuned'] = r2_score(yte, elas_best.predict(Xtes_full))
    r['elasticnet_params'] = study_elas.best_params
    print(f'  ElasticNet Tuned: {r["elasticnet_tuned"]:.4f}')
    
    # 6. Ensemble of tuned models
    pred_hub = hub_best.predict(Xtes_full)
    pred_ridge = ridge_best.predict(Xtes_full)
    pred_elas = elas_best.predict(Xtes_full)
    
    # Simple average
    pred_avg = (pred_hub + pred_ridge + pred_elas) / 3
    r['ensemble_avg'] = r2_score(yte, pred_avg)
    
    # Weighted by validation performance
    weights = np.array([study_hub.best_value, study_ridge.best_value, study_elas.best_value])
    weights = np.maximum(weights, 0)  # Ensure non-negative
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.array([1/3, 1/3, 1/3])
    
    pred_weighted = weights[0]*pred_hub + weights[1]*pred_ridge + weights[2]*pred_elas
    r['ensemble_weighted'] = r2_score(yte, pred_weighted)
    r['ensemble_weights'] = weights.tolist()
    
    print(f'  Ensemble Avg:   {r["ensemble_avg"]:.4f}')
    print(f'  Ensemble Wtd:   {r["ensemble_weighted"]:.4f}')
    
    # Find best model for this asset
    model_scores = {
        'huber_tuned': r['huber_tuned'],
        'ridge_tuned': r['ridge_tuned'],
        'elasticnet_tuned': r['elasticnet_tuned'],
        'ensemble_weighted': r['ensemble_weighted']
    }
    r['best_model'] = max(model_scores, key=model_scores.get)
    r['best_r2'] = model_scores[r['best_model']]
    
    results[ticker] = r

# Summary
print('\n' + '='*65)
print('FINAL SUMMARY - 22-Day Universal Model')
print('='*65)

models = ['huber_baseline', 'ridge_baseline', 'huber_tuned', 'ridge_tuned', 
          'elasticnet_tuned', 'ensemble_avg', 'ensemble_weighted']

baseline = np.mean([results[t]['huber_baseline'] for t in ASSETS])

print('\nModel Performance:')
for m in models:
    vals = [results[t].get(m) for t in ASSETS if results[t].get(m) is not None]
    if vals:
        avg = np.mean(vals)
        imp = (avg - baseline) / abs(baseline) * 100
        print(f'{m:20}: {avg:.4f} ({imp:+.2f}%)')

# Find best overall
best_avg = 0
best_model = None
for m in models:
    vals = [results[t].get(m) for t in ASSETS if results[t].get(m) is not None]
    if vals:
        avg = np.mean(vals)
        if avg > best_avg:
            best_avg = avg
            best_model = m

print(f'\n**Best Model**: {best_model} (avg R² = {best_avg:.4f})')

# Asset-specific results
print('\nAsset-Specific Best:')
for t in ASSETS:
    print(f'  {t}: {results[t]["best_model"]} (R² = {results[t]["best_r2"]:.4f})')

# Save results
def clean_for_json(r):
    cleaned = {}
    for k, v in r.items():
        if isinstance(v, dict):
            cleaned[k] = {str(kk): float(vv) for kk, vv in v.items()}
        elif isinstance(v, (list, np.ndarray)):
            cleaned[k] = [float(x) for x in v]
        elif isinstance(v, str):
            cleaned[k] = v
        elif v is not None:
            cleaned[k] = float(v) if isinstance(v, (int, float, np.floating)) else str(v)
        else:
            cleaned[k] = None
    return cleaned

output = {
    'experiment': '22Day_Universal_Model',
    'horizon': '22 days',
    'n_trials': N_TRIALS,
    'n_features': len(fcols),
    'features': fcols,
    'results': {k: clean_for_json(v) for k, v in results.items()},
    'summary': {
        'best_model': best_model,
        'best_avg_r2': best_avg,
        'model_averages': {m: np.mean([results[t][m] for t in ASSETS if results[t].get(m)]) 
                          for m in models}
    }
}

with open('results/universal_model_22d.json', 'w') as f:
    json.dump(output, f, indent=2)

print('\nResults saved to results/universal_model_22d.json')
