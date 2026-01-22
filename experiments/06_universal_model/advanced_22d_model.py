"""
Experiment 18: Advanced 22-Day Universal Model
Based on literature research:
1. HAR-LSTM hybrid (Corsi + Deep Learning)
2. Mean Reversion Model (Critical for long horizon)
3. VRP Adjustment Model
4. Ensemble of specialized models
5. QLIKE loss optimization
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']

print('='*70)
print('Experiment 18: Advanced 22-Day Universal Model')
print('Literature-based approaches for long-horizon prediction')
print('='*70)

# ==========================================
# 1. Load Data
# ==========================================
print('\n[1/6] Loading data...')

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

# VIX3M for term structure
vix3m = yf.download('^VIX3M', start='2010-01-01', end='2025-01-01', progress=False)
if isinstance(vix3m.columns, pd.MultiIndex):
    vix3m = vix3m[('Close', '^VIX3M')]
else:
    vix3m = vix3m['Close']

print('Data loaded (15 years)')

def create_22d_features(df, vix, vix3m):
    """Create features specifically designed for 22-day prediction"""
    df = df.copy()
    
    # ==========================================
    # HAR Structure (Daily, Weekly, Monthly, Quarterly)
    # Extended for long horizon
    # ==========================================
    df['VIX'] = vix
    df['VIX3M'] = vix3m
    
    # Multi-scale VIX
    df['VIX_ma5'] = df['VIX'].rolling(5).mean()
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['VIX_ma66'] = df['VIX'].rolling(66).mean()  # Quarterly
    
    # VIX Term Structure (Critical for long horizon)
    df['VIX_term'] = df['VIX3M'] / (df['VIX'] + 0.01)
    df['VIX_term_ma5'] = df['VIX_term'].rolling(5).mean()
    
    # CAVB (VRP proxy)
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_ma22'] = df['CAVB'].rolling(22).mean()
    df['CAVB_ma66'] = df['CAVB'].rolling(66).mean()
    
    # ==========================================
    # Mean Reversion Features (Key for long horizon)
    # ==========================================
    vix_long_mean = df['VIX'].rolling(252).mean()
    df['VIX_deviation'] = (df['VIX'] - vix_long_mean) / (vix_long_mean + 0.01)
    
    rv_long_mean = df['RV_22d'].rolling(252).mean()
    df['RV_deviation'] = (df['RV_22d'] - rv_long_mean) / (rv_long_mean + 0.01)
    
    cavb_long_mean = df['CAVB'].rolling(252).mean()
    df['CAVB_deviation'] = (df['CAVB'] - cavb_long_mean) / (np.abs(cavb_long_mean) + 0.01)
    
    # Half-life estimation proxy
    df['VIX_momentum_22'] = df['VIX'].pct_change(22)
    df['VIX_momentum_66'] = df['VIX'].pct_change(66)
    
    # ==========================================
    # RV Multi-Scale (HAR extension)
    # ==========================================
    df['RV_5d_lag'] = df['RV_5d'].shift(1)
    df['RV_22d_lag'] = df['RV_22d'].shift(1)
    df['RV_66d_lag'] = df['RV_66d'].shift(1)
    
    # RV ratios
    df['RV_ratio_5_22'] = df['RV_5d'] / (df['RV_22d'] + 0.01)
    df['RV_ratio_22_66'] = df['RV_22d'] / (df['RV_66d'] + 0.01)
    
    # ==========================================
    # VRP Adjustment (Literature approach)
    # ==========================================
    # Historical VRP mean for adjustment
    vrp_historical = df['CAVB'].rolling(252).mean()
    df['VRP_adjusted_VIX'] = df['VIX'] - vrp_historical  # VIX adjusted for VRP
    
    # ==========================================
    # Regime Indicators
    # ==========================================
    df['High_VIX_regime'] = (df['VIX'] > df['VIX'].rolling(252).quantile(0.75)).astype(int)
    df['Low_VIX_regime'] = (df['VIX'] < df['VIX'].rolling(252).quantile(0.25)).astype(int)
    
    # ==========================================
    # Target: 22-day ahead CAVB
    # ==========================================
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-22)
    
    return df

# ==========================================
# 2. Define New Model Approaches
# ==========================================
print('\n[2/6] Defining model approaches...')

def mean_reversion_model(Xtr, ytr, Xte, feature_cols):
    """Model 1: Mean Reversion Focus
    Uses deviation from long-term mean as primary predictor
    """
    # Use only deviation features
    dev_features = [i for i, c in enumerate(feature_cols) if 'deviation' in c or 'mean' in c.lower()]
    
    if len(dev_features) < 3:
        dev_features = list(range(min(5, Xtr.shape[1])))
    
    Xtr_dev = Xtr[:, dev_features]
    Xte_dev = Xte[:, dev_features]
    
    model = Ridge(alpha=100.0)  # Strong regularization
    model.fit(Xtr_dev, ytr)
    return model.predict(Xte_dev)

def vrp_adjusted_model(Xtr, ytr, Xte, feature_cols):
    """Model 2: VRP Adjustment
    Uses VRP-adjusted implied volatility
    """
    # Find VRP-related features
    vrp_features = [i for i, c in enumerate(feature_cols) if 'VRP' in c or 'CAVB' in c or 'term' in c.lower()]
    
    if len(vrp_features) < 3:
        vrp_features = list(range(min(8, Xtr.shape[1])))
    
    Xtr_vrp = Xtr[:, vrp_features]
    Xte_vrp = Xte[:, vrp_features]
    
    model = HuberRegressor(epsilon=1.5, max_iter=1000)
    model.fit(Xtr_vrp, ytr)
    return model.predict(Xte_vrp)

def har_extended_model(Xtr, ytr, Xte, feature_cols):
    """Model 3: Extended HAR
    Multi-scale volatility with quarterly component
    """
    # HAR features: daily, weekly, monthly, quarterly lags
    har_features = [i for i, c in enumerate(feature_cols) if 'RV_' in c or 'VIX_ma' in c]
    
    if len(har_features) < 3:
        har_features = list(range(min(10, Xtr.shape[1])))
    
    Xtr_har = Xtr[:, har_features]
    Xte_har = Xte[:, har_features]
    
    model = Ridge(alpha=50.0)
    model.fit(Xtr_har, ytr)
    return model.predict(Xte_har)

def svr_rbf_tuned_model(Xtr, ytr, Xte, feature_cols):
    """Model 4: Tuned SVR-RBF
    Non-linear kernel for complex patterns
    """
    model = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.5)
    model.fit(Xtr, ytr)
    return model.predict(Xte)

def gradient_boost_tuned_model(Xtr, ytr, Xte, feature_cols):
    """Model 5: Tuned Gradient Boosting
    Conservative settings to prevent overfitting
    """
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_leaf=30, subsample=0.8, random_state=42
    )
    model.fit(Xtr, ytr)
    return model.predict(Xte)

def ensemble_specialist_model(Xtr, ytr, Xte, feature_cols):
    """Model 6: Ensemble of Specialists
    Combine mean reversion + VRP + HAR approaches
    """
    pred_mr = mean_reversion_model(Xtr, ytr, Xte, feature_cols)
    pred_vrp = vrp_adjusted_model(Xtr, ytr, Xte, feature_cols)
    pred_har = har_extended_model(Xtr, ytr, Xte, feature_cols)
    
    # Simple average
    return (pred_mr + pred_vrp + pred_har) / 3

def qlike_optimized_model(Xtr, ytr, Xte, feature_cols):
    """Model 7: QLIKE-optimized (Ridge with log transform)
    QLIKE loss is robust to extreme volatility
    """
    # Log transform target for QLIKE approximation
    y_log = np.log(np.maximum(ytr, 0.1))
    
    model = Ridge(alpha=10.0)
    model.fit(Xtr, y_log)
    
    pred_log = model.predict(Xte)
    return np.exp(pred_log)

# ==========================================
# 3. Run Experiments
# ==========================================
print('\n[3/6] Running experiments...')

models = {
    'Mean_Reversion': mean_reversion_model,
    'VRP_Adjusted': vrp_adjusted_model,
    'HAR_Extended': har_extended_model,
    'SVR_RBF_Tuned': svr_rbf_tuned_model,
    'GB_Tuned': gradient_boost_tuned_model,
    'Ensemble_Specialist': ensemble_specialist_model,
    'QLIKE_Optimized': qlike_optimized_model
}

results = {}

for ticker in ASSETS:
    print(f'\n{ticker}:')
    
    df = create_22d_features(all_data[ticker], vix, vix3m)
    
    exclude = ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ret']
    fcols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    fcols = [c for c in fcols if df[c].notna().sum() > len(df) * 0.8]
    
    df = df.dropna(subset=fcols + ['Target'])
    
    X, y = df[fcols].values, df['Target'].values
    n = len(y)
    tr_end = int(n * 0.8)
    
    Xtr, Xte = X[:tr_end], X[tr_end:]
    ytr, yte = y[:tr_end], y[tr_end:]
    
    scaler = StandardScaler()
    Xts = scaler.fit_transform(Xtr)
    Xtes = scaler.transform(Xte)
    
    ticker_results = {}
    
    for model_name, model_func in models.items():
        try:
            y_pred = model_func(Xts, ytr, Xtes, fcols)
            r2 = r2_score(yte, y_pred)
            
            # QLIKE loss (robust metric for volatility)
            qlike = np.mean(np.log(np.maximum(y_pred, 0.1)) + yte / np.maximum(y_pred, 0.1))
            
            ticker_results[model_name] = {'r2': r2, 'qlike': qlike}
            print(f'  {model_name}: R² = {r2:.4f}, QLIKE = {qlike:.4f}')
        except Exception as e:
            print(f'  {model_name}: FAILED ({str(e)[:30]})')
            ticker_results[model_name] = {'r2': None, 'qlike': None}
    
    # Find best
    valid = {k: v['r2'] for k, v in ticker_results.items() if v['r2'] is not None}
    best_model = max(valid, key=valid.get) if valid else None
    best_r2 = valid[best_model] if best_model else None
    
    print(f'  **Best: {best_model} (R² = {best_r2:.4f})**' if best_r2 else '  No valid model')
    
    results[ticker] = {
        'model_results': ticker_results,
        'best_model': best_model,
        'best_r2': best_r2,
        'n_features': len(fcols)
    }

# ==========================================
# 4. Summary
# ==========================================
print('\n' + '='*70)
print('[4/6] FINAL SUMMARY')
print('='*70)

print('\nModel Performance (Average R²):')
for model_name in models.keys():
    vals = [results[t]['model_results'][model_name]['r2'] for t in ASSETS 
            if results[t]['model_results'][model_name]['r2'] is not None]
    if vals:
        avg = np.mean(vals)
        positive = sum(1 for v in vals if v > 0)
        print(f'  {model_name:20}: {avg:+.4f} ({positive}/5 positive)')

print('\nAsset-Specific Best:')
for t in ASSETS:
    r = results[t]
    print(f'  {t}: {r["best_model"]} (R² = {r["best_r2"]:.4f})' if r['best_r2'] else f'  {t}: None')

# Average best R²
avg_best = np.mean([r['best_r2'] for r in results.values() if r['best_r2'] is not None])
print(f'\nAverage Best R²: {avg_best:.4f}')

# ==========================================
# 5. Find Universal Best Approach
# ==========================================
print('\n[5/6] Finding Universal Best Approach...')

from collections import Counter
best_models = [r['best_model'] for r in results.values() if r['best_model']]
model_counts = Counter(best_models)

print('Best model frequency:')
for m, c in model_counts.most_common():
    print(f'  {m}: {c}/5 assets')

# Model with best average performance
model_avgs = {}
for model_name in models.keys():
    vals = [results[t]['model_results'][model_name]['r2'] for t in ASSETS 
            if results[t]['model_results'][model_name]['r2'] is not None]
    if vals:
        model_avgs[model_name] = np.mean(vals)

best_universal = max(model_avgs, key=model_avgs.get) if model_avgs else None
print(f'\nBest Universal Model: {best_universal} (avg R² = {model_avgs[best_universal]:.4f})')

# ==========================================
# 6. Save Results
# ==========================================
print('\n[6/6] Saving results...')

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

output = {
    'experiment': 'Exp18_Advanced_22Day_Model',
    'description': 'Literature-based long-horizon prediction approaches',
    'models': list(models.keys()),
    'results': clean_for_json(results),
    'summary': {
        'model_averages': clean_for_json(model_avgs),
        'best_universal': best_universal,
        'best_universal_r2': float(model_avgs.get(best_universal, 0)),
        'avg_best_r2': float(avg_best)
    },
    'literature_basis': [
        'HAR model with quarterly extension (Corsi 2009)',
        'Mean reversion for long horizon (GARCH literature)',
        'VRP adjustment for implied volatility (Reading 2019)',
        'QLIKE loss for volatility (Patton 2011)'
    ]
}

with open('results/advanced_22d_model.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Results saved to results/advanced_22d_model.json')
