"""
Advanced Universal Model Experiments
Experiment 10: Intensive Hyperparameter Tuning for Top Models

Comprehensive Optuna optimization for Huber and SVR-Linear
with 100 trials per model per asset
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import HuberRegressor, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']
N_TRIALS = 100  # Intensive tuning

print('='*65)
print('Experiment 10: Intensive Hyperparameter Tuning')
print(f'Trials per model: {N_TRIALS}')
print('='*65)

# Load data
all_data = {}
for t in ASSETS:
    d = yf.download(t, start='2015-01-01', end='2025-01-01', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d['Ret'] = d['Close'].pct_change()
    d['RV_1d'] = d['Ret'].abs() * np.sqrt(252) * 100
    d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
    d['RV_10d'] = d['Ret'].rolling(10).std() * np.sqrt(252) * 100
    d['RV_22d'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
    all_data[t] = d

vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix[('Close', '^VIX')]
else:
    vix = vix['Close']

print('Data loaded\n')

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
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    return df.dropna()

results = {}

for ticker in ASSETS:
    print(f'Processing {ticker}...')
    
    df = create_features(all_data[ticker], vix)
    fcols = ['VIX', 'VIX_lag1', 'VIX_lag5', 'VIX_change', 'VIX_ma22',
             'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
             'RV_1d', 'RV_5d', 'RV_10d', 'RV_22d']
    
    X, y = df[fcols], df['Target']
    n = len(X)
    tr_end, val_end = int(n*0.6), int(n*0.8)
    
    Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
    Xval, yval = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    Xte, yte = X.iloc[val_end:], y.iloc[val_end:]
    
    # Combine train+val for final training
    Xtrval = pd.concat([Xtr, Xval])
    ytrval = pd.concat([ytr, yval])
    
    scaler = StandardScaler()
    Xts = scaler.fit_transform(Xtr)
    Xvs = scaler.transform(Xval)
    Xtes = scaler.transform(Xte)
    
    scaler_full = StandardScaler()
    Xtvs = scaler_full.fit_transform(Xtrval)
    Xtes_full = scaler_full.transform(Xte)
    
    r = {}
    
    # 1. Baseline Huber
    hub_base = HuberRegressor(epsilon=1.35, max_iter=1000)
    hub_base.fit(Xts, ytr)
    r['huber_baseline'] = r2_score(yte, hub_base.predict(Xtes))
    print(f'  Huber Baseline: {r["huber_baseline"]:.4f}')
    
    # 2. Baseline SVR-Linear
    svr_base = SVR(kernel='linear', C=1.0, epsilon=0.1)
    svr_base.fit(Xts, ytr)
    r['svr_baseline'] = r2_score(yte, svr_base.predict(Xtes))
    print(f'  SVR Baseline:   {r["svr_baseline"]:.4f}')
    
    # 3. Optuna Huber Tuning
    print(f'  Tuning Huber ({N_TRIALS} trials)...')
    def obj_huber(trial):
        epsilon = trial.suggest_float('epsilon', 1.0, 3.0)
        alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
        m = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)
        m.fit(Xts, ytr)
        return r2_score(yval, m.predict(Xvs))
    
    study_hub = optuna.create_study(direction='maximize')
    study_hub.optimize(obj_huber, n_trials=N_TRIALS, show_progress_bar=False)
    
    hub_best = HuberRegressor(**study_hub.best_params, max_iter=1000)
    hub_best.fit(Xtvs, ytrval)
    r['huber_tuned'] = r2_score(yte, hub_best.predict(Xtes_full))
    r['huber_params'] = study_hub.best_params
    print(f'  Huber Tuned:    {r["huber_tuned"]:.4f} (params: {study_hub.best_params})')
    
    # 4. Optuna SVR-Linear Tuning
    print(f'  Tuning SVR-Linear ({N_TRIALS} trials)...')
    def obj_svr(trial):
        C = trial.suggest_float('C', 0.01, 100.0, log=True)
        epsilon = trial.suggest_float('epsilon', 0.001, 1.0, log=True)
        m = SVR(kernel='linear', C=C, epsilon=epsilon)
        m.fit(Xts, ytr)
        return r2_score(yval, m.predict(Xvs))
    
    study_svr = optuna.create_study(direction='maximize')
    study_svr.optimize(obj_svr, n_trials=N_TRIALS, show_progress_bar=False)
    
    svr_best = SVR(kernel='linear', **study_svr.best_params)
    svr_best.fit(Xtvs, ytrval)
    r['svr_tuned'] = r2_score(yte, svr_best.predict(Xtes_full))
    r['svr_params'] = study_svr.best_params
    print(f'  SVR Tuned:      {r["svr_tuned"]:.4f} (params: {study_svr.best_params})')
    
    # 5. Optuna ElasticNet Tuning (for comparison)
    print(f'  Tuning ElasticNet ({N_TRIALS} trials)...')
    def obj_elastic(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 1.0, log=True)
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
    pred_svr = svr_best.predict(Xtes_full)
    pred_elas = elas_best.predict(Xtes_full)
    
    # Simple average
    pred_avg = (pred_hub + pred_svr + pred_elas) / 3
    r['ensemble_avg'] = r2_score(yte, pred_avg)
    
    # Weighted by validation performance
    weights = np.array([study_hub.best_value, study_svr.best_value, study_elas.best_value])
    weights = weights / weights.sum()
    pred_weighted = weights[0]*pred_hub + weights[1]*pred_svr + weights[2]*pred_elas
    r['ensemble_weighted'] = r2_score(yte, pred_weighted)
    r['ensemble_weights'] = weights.tolist()
    
    print(f'  Ensemble Avg:   {r["ensemble_avg"]:.4f}')
    print(f'  Ensemble Wtd:   {r["ensemble_weighted"]:.4f}')
    
    results[ticker] = r
    print()

# Summary
print('='*65)
print('FINAL SUMMARY')
print('='*65)

models = ['huber_baseline', 'svr_baseline', 'huber_tuned', 'svr_tuned', 
          'elasticnet_tuned', 'ensemble_avg', 'ensemble_weighted']

baseline = np.mean([results[t]['huber_baseline'] for t in ASSETS])

for m in models:
    vals = [results[t].get(m) for t in ASSETS if results[t].get(m)]
    if vals:
        avg = np.mean(vals)
        imp = (avg - baseline) / abs(baseline) * 100
        print(f'{m:20}: {avg:.4f} ({imp:+.2f}%)')

# Find best
best_avg = 0
best_model = None
for m in models:
    vals = [results[t].get(m) for t in ASSETS if results[t].get(m)]
    if vals:
        avg = np.mean(vals)
        if avg > best_avg:
            best_avg = avg
            best_model = m

print(f'\nBest Model: {best_model} (avg R² = {best_avg:.4f})')

# Save results
def clean_for_json(r):
    cleaned = {}
    for k, v in r.items():
        if isinstance(v, dict):
            cleaned[k] = {str(kk): float(vv) for kk, vv in v.items()}
        elif isinstance(v, (list, np.ndarray)):
            cleaned[k] = [float(x) for x in v]
        elif v is not None:
            cleaned[k] = float(v)
        else:
            cleaned[k] = None
    return cleaned

output = {
    'experiment': 'Exp10_Intensive_Tuning',
    'n_trials': N_TRIALS,
    'results': {k: clean_for_json(v) for k, v in results.items()},
    'summary': {
        'best_model': best_model,
        'best_avg_r2': best_avg,
        'model_averages': {m: np.mean([results[t][m] for t in ASSETS if results[t].get(m)]) 
                          for m in models}
    }
}

with open('results/advanced_experiment_10.json', 'w') as f:
    json.dump(output, f, indent=2)

print('\nResults saved to results/advanced_experiment_10.json')
