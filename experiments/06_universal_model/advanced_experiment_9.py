import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except:
    HAS_OPTUNA = False

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']

print('='*60)
print('Exp9: Optuna HPO + AdaBoost')
print(f'Optuna: {HAS_OPTUNA}')
print('='*60)

# Load data
all_data = {}
for t in ASSETS:
    d = yf.download(t, start='2015-01-01', end='2025-01-01', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d['Ret'] = d['Close'].pct_change()
    d['RV22'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
    all_data[t] = d

vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix[('Close', '^VIX')]
else:
    vix = vix['Close']

print('Data loaded')

results = {}
for ticker in ASSETS:
    print(f'\nProcessing {ticker}...')
    df = all_data[ticker].copy()
    df['VIX'] = vix
    df['CAVB'] = df['VIX'] - df['RV22']
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['Target'] = df['VIX'] - df['RV22'].shift(-5)
    df = df.dropna()
    
    X = df[['VIX', 'CAVB', 'RV22', 'VIX_lag1', 'CAVB_lag1']]
    y = df['Target']
    n = len(X)
    tr, val = int(n*0.6), int(n*0.8)
    
    Xtr, ytr = X.iloc[:tr], y.iloc[:tr]
    Xval, yval = X.iloc[tr:val], y.iloc[tr:val]
    Xte, yte = X.iloc[val:], y.iloc[val:]
    
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr)
    Xtes = sc.transform(Xte)
    
    # 1. Baseline Huber
    hub = HuberRegressor(epsilon=1.35, max_iter=1000)
    hub.fit(Xts, ytr)
    r = {'huber_base': r2_score(yte, hub.predict(Xtes))}
    print(f'  Huber Base: {r["huber_base"]:.4f}')
    
    # 2. AdaBoost
    ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=50)
    ada.fit(Xtr, ytr)
    r['adaboost'] = r2_score(yte, ada.predict(Xte))
    print(f'  AdaBoost:   {r["adaboost"]:.4f}')
    
    # 3. Optuna Huber
    if HAS_OPTUNA:
        Xvs = sc.transform(Xval)
        def obj(trial):
            eps = trial.suggest_float('epsilon', 1.0, 2.0)
            alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
            m = HuberRegressor(epsilon=eps, alpha=alpha, max_iter=500)
            m.fit(Xts, ytr)
            return r2_score(yval, m.predict(Xvs))
        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=20, show_progress_bar=False)
        bp = study.best_params
        sc2 = StandardScaler()
        Xf = sc2.fit_transform(pd.concat([Xtr, Xval]))
        Xtes2 = sc2.transform(Xte)
        m = HuberRegressor(**bp, max_iter=1000)
        m.fit(Xf, pd.concat([ytr, yval]))
        r['huber_optuna'] = r2_score(yte, m.predict(Xtes2))
        print(f'  Huber Opt:  {r["huber_optuna"]:.4f}')
    
    # 4. Optuna SVR
    if HAS_OPTUNA:
        def obj_svr(trial):
            C = trial.suggest_float('C', 0.1, 10.0, log=True)
            eps = trial.suggest_float('epsilon', 0.01, 0.5)
            m = SVR(kernel='linear', C=C, epsilon=eps)
            m.fit(Xts, ytr)
            return r2_score(yval, m.predict(Xvs))
        study_svr = optuna.create_study(direction='maximize')
        study_svr.optimize(obj_svr, n_trials=20, show_progress_bar=False)
        bp_svr = study_svr.best_params
        m = SVR(kernel='linear', **bp_svr)
        m.fit(Xf, pd.concat([ytr, yval]))
        r['svr_optuna'] = r2_score(yte, m.predict(Xtes2))
        print(f'  SVR Opt:    {r["svr_optuna"]:.4f}')
    
    results[ticker] = r

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
baseline = np.mean([results[t]['huber_base'] for t in ASSETS])
for m in ['huber_base', 'adaboost', 'huber_optuna', 'svr_optuna']:
    vals = [results[t].get(m) for t in ASSETS if results[t].get(m)]
    if vals:
        avg = np.mean(vals)
        imp = (avg - baseline) / abs(baseline) * 100
        print(f'{m:15}: {avg:.4f} ({imp:+.1f}%)')

# Save
with open('results/advanced_experiment_9.json', 'w') as f:
    json.dump({'exp': 'Exp9_Optuna_AdaBoost', 'results': results}, f, indent=2)
print('\nSaved to results/advanced_experiment_9.json')
