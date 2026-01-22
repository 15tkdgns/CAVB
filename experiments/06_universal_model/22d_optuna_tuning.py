"""
Experiment 19: 22-Day Model Hyperparameter Tuning
Focus on best performers from Exp 18:
- Ensemble_Specialist (avg R² = 0.153)
- VRP_Adjusted (avg R² = 0.148)
- HAR_Extended (avg R² = 0.079)

Using Optuna for systematic hyperparameter optimization
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']

print('='*70)
print('Experiment 19: 22-Day Model Hyperparameter Tuning')
print('Optuna Optimization for Best Average Performance')
print('='*70)

# ==========================================
# 1. Load Data
# ==========================================
print('\n[1/5] Loading data...')

all_data = {}
for t in ASSETS:
    d = yf.download(t, start='2010-01-01', end='2025-01-01', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d['Ret'] = d['Close'].pct_change()
    d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
    d['RV_22d'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
    d['RV_66d'] = d['Ret'].rolling(66).std() * np.sqrt(252) * 100
    all_data[t] = d

vix = yf.download('^VIX', start='2010-01-01', end='2025-01-01', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix[('Close', '^VIX')]
else:
    vix = vix['Close']

vix3m = yf.download('^VIX3M', start='2010-01-01', end='2025-01-01', progress=False)
if isinstance(vix3m.columns, pd.MultiIndex):
    vix3m = vix3m[('Close', '^VIX3M')]
else:
    vix3m = vix3m['Close']

print('Data loaded')

def create_features(df, vix, vix3m):
    """Create comprehensive 22-day features"""
    df = df.copy()
    
    df['VIX'] = vix
    df['VIX3M'] = vix3m
    df['VIX_ma5'] = df['VIX'].rolling(5).mean()
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['VIX_ma66'] = df['VIX'].rolling(66).mean()
    df['VIX_term'] = df['VIX3M'] / (df['VIX'] + 0.01)
    
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_ma22'] = df['CAVB'].rolling(22).mean()
    df['CAVB_ma66'] = df['CAVB'].rolling(66).mean()
    
    vix_long_mean = df['VIX'].rolling(252).mean()
    df['VIX_deviation'] = (df['VIX'] - vix_long_mean) / (vix_long_mean + 0.01)
    
    rv_long_mean = df['RV_22d'].rolling(252).mean()
    df['RV_deviation'] = (df['RV_22d'] - rv_long_mean) / (rv_long_mean + 0.01)
    
    df['RV_5d_lag'] = df['RV_5d'].shift(1)
    df['RV_22d_lag'] = df['RV_22d'].shift(1)
    df['RV_66d_lag'] = df['RV_66d'].shift(1)
    df['RV_ratio_5_22'] = df['RV_5d'] / (df['RV_22d'] + 0.01)
    
    vrp_hist = df['CAVB'].rolling(252).mean()
    df['VRP_adjusted_VIX'] = df['VIX'] - vrp_hist
    
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-22)
    
    return df

# Prepare data for all assets
print('\n[2/5] Preparing datasets...')
datasets = {}

for ticker in ASSETS:
    df = create_features(all_data[ticker], vix, vix3m)
    
    exclude = ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ret']
    fcols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    fcols = [c for c in fcols if df[c].notna().sum() > len(df) * 0.8]
    
    df = df.dropna(subset=fcols + ['Target'])
    
    X, y = df[fcols].values, df['Target'].values
    n = len(y)
    tr_end, val_end = int(n * 0.6), int(n * 0.8)
    
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[:tr_end])
    Xval = scaler.transform(X[tr_end:val_end])
    Xte = scaler.transform(X[val_end:])
    
    ytr, yval, yte = y[:tr_end], y[tr_end:val_end], y[val_end:]
    
    datasets[ticker] = {
        'Xtr': Xtr, 'Xval': Xval, 'Xte': Xte,
        'ytr': ytr, 'yval': yval, 'yte': yte,
        'fcols': fcols, 'scaler': scaler
    }
    
    print(f'  {ticker}: {len(Xtr)} train, {len(Xval)} val, {len(Xte)} test')

# ==========================================
# 3. Optuna Hyperparameter Optimization
# ==========================================
print('\n[3/5] Running Optuna optimization (50 trials)...')

def create_objective(datasets):
    """Create objective function that optimizes average R² across all assets"""
    
    def objective(trial):
        # Hyperparameters to tune
        model_type = trial.suggest_categorical('model_type', ['ridge', 'huber', 'elasticnet', 'svr', 'gb'])
        
        if model_type == 'ridge':
            alpha = trial.suggest_float('ridge_alpha', 0.1, 1000.0, log=True)
            model_class = lambda: Ridge(alpha=alpha)
        
        elif model_type == 'huber':
            epsilon = trial.suggest_float('huber_epsilon', 1.0, 2.5)
            alpha = trial.suggest_float('huber_alpha', 1e-6, 0.1, log=True)
            model_class = lambda: HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)
        
        elif model_type == 'elasticnet':
            alpha = trial.suggest_float('enet_alpha', 0.001, 1.0, log=True)
            l1_ratio = trial.suggest_float('enet_l1', 0.1, 0.9)
            model_class = lambda: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        
        elif model_type == 'svr':
            C = trial.suggest_float('svr_C', 0.1, 100.0, log=True)
            epsilon = trial.suggest_float('svr_epsilon', 0.01, 1.0, log=True)
            kernel = trial.suggest_categorical('svr_kernel', ['linear', 'rbf'])
            model_class = lambda: SVR(kernel=kernel, C=C, epsilon=epsilon)
        
        else:  # gb
            n_est = trial.suggest_int('gb_n_estimators', 30, 150)
            max_depth = trial.suggest_int('gb_max_depth', 2, 5)
            lr = trial.suggest_float('gb_lr', 0.01, 0.2, log=True)
            min_samples = trial.suggest_int('gb_min_samples', 20, 50)
            model_class = lambda: GradientBoostingRegressor(
                n_estimators=n_est, max_depth=max_depth, 
                learning_rate=lr, min_samples_leaf=min_samples, random_state=42
            )
        
        # Evaluate on all assets (validation set)
        scores = []
        for ticker in ASSETS:
            d = datasets[ticker]
            Xtrval = np.vstack([d['Xtr'], d['Xval']])
            ytrval = np.concatenate([d['ytr'], d['yval']])
            
            try:
                model = model_class()
                model.fit(d['Xtr'], d['ytr'])
                pred = model.predict(d['Xval'])
                r2 = r2_score(d['yval'], pred)
                scores.append(r2)
            except:
                scores.append(-1.0)
        
        avg_r2 = np.mean(scores)
        return avg_r2
    
    return objective

# Run Optuna
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(create_objective(datasets), n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_val_r2 = study.best_value

print(f'\nBest Validation R²: {best_val_r2:.4f}')
print(f'Best Parameters: {best_params}')

# ==========================================
# 4. Evaluate Best Model on Test Set
# ==========================================
print('\n[4/5] Evaluating best model on test set...')

def create_best_model(params):
    model_type = params['model_type']
    
    if model_type == 'ridge':
        return Ridge(alpha=params['ridge_alpha'])
    elif model_type == 'huber':
        return HuberRegressor(epsilon=params['huber_epsilon'], 
                             alpha=params['huber_alpha'], max_iter=1000)
    elif model_type == 'elasticnet':
        return ElasticNet(alpha=params['enet_alpha'], 
                         l1_ratio=params['enet_l1'], max_iter=5000)
    elif model_type == 'svr':
        return SVR(kernel=params['svr_kernel'], C=params['svr_C'], 
                  epsilon=params['svr_epsilon'])
    else:
        return GradientBoostingRegressor(
            n_estimators=params['gb_n_estimators'],
            max_depth=params['gb_max_depth'],
            learning_rate=params['gb_lr'],
            min_samples_leaf=params['gb_min_samples'],
            random_state=42
        )

test_results = {}

for ticker in ASSETS:
    d = datasets[ticker]
    
    # Train on train+val
    Xtrval = np.vstack([d['Xtr'], d['Xval']])
    ytrval = np.concatenate([d['ytr'], d['yval']])
    
    model = create_best_model(best_params)
    model.fit(Xtrval, ytrval)
    
    pred = model.predict(d['Xte'])
    r2 = r2_score(d['yte'], pred)
    
    test_results[ticker] = r2
    print(f'  {ticker}: Test R² = {r2:.4f}')

avg_test_r2 = np.mean(list(test_results.values()))
print(f'\nAverage Test R²: {avg_test_r2:.4f}')

# Count positive
positive_count = sum(1 for v in test_results.values() if v > 0)
print(f'Positive R²: {positive_count}/5 assets')

# ==========================================
# 5. Compare with Previous Best
# ==========================================
print('\n[5/5] Comparison with Previous Experiments...')

print('\n22-Day Model Evolution:')
print('  Exp 17 (Baseline):            avg R² = -0.135')
print('  Exp 18 (Literature-based):    avg R² =  0.153 (Ensemble)')
print(f'  Exp 19 (Optuna-tuned):        avg R² = {avg_test_r2:.3f}')

improvement = (avg_test_r2 - 0.153) / 0.153 * 100 if avg_test_r2 > 0.153 else ((avg_test_r2 - 0.153) / 0.153 * 100)
print(f'\nImprovement vs Exp 18: {improvement:+.1f}%')

# Save results
output = {
    'experiment': 'Exp19_22Day_Optuna_Tuning',
    'n_trials': 50,
    'best_params': best_params,
    'best_val_r2': float(best_val_r2),
    'test_results': {k: float(v) for k, v in test_results.items()},
    'avg_test_r2': float(avg_test_r2),
    'positive_assets': positive_count,
    'comparison': {
        'exp17_baseline': -0.135,
        'exp18_literature': 0.153,
        'exp19_tuned': float(avg_test_r2)
    }
}

with open('results/22d_optuna_tuned.json', 'w') as f:
    json.dump(output, f, indent=2)

print('\nResults saved to results/22d_optuna_tuned.json')

# Best recommendation
print('\n' + '='*70)
print('FINAL 22-DAY MODEL RECOMMENDATION')
print('='*70)
print(f'\nModel Type: {best_params["model_type"]}')
print(f'Average R²: {avg_test_r2:.4f}')
print(f'\nAsset Performance:')
for t in sorted(test_results.keys(), key=lambda x: test_results[x], reverse=True):
    status = "Recommended" if test_results[t] > 0.2 else ("Usable" if test_results[t] > 0 else "Not Recommended")
    print(f'  {t}: {test_results[t]:.4f} - {status}')
