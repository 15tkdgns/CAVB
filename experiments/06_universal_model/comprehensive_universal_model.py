"""
Experiment 17: Comprehensive Universal Model
- Feature Selection for 5-day and 22-day horizons
- Multiple models: Huber, ElasticNet, Ridge, SVR, RF, GB
- New Universal Model Creation
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']
HORIZONS = [5, 22]

print('='*70)
print('Experiment 17: Comprehensive Universal Model')
print('Feature Selection + Multiple Models + 5-day/22-day Horizons')
print('='*70)

# ==========================================
# 1. Load All Data
# ==========================================
print('\n[1/5] Loading data...')

all_data = {}
for t in ASSETS:
    d = yf.download(t, start='2015-01-01', end='2025-01-01', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d['Ret'] = d['Close'].pct_change()
    d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
    d['RV_22d'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
    all_data[t] = d

vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix[('Close', '^VIX')]
else:
    vix = vix['Close']

# Extended data
extended_tickers = {
    'VIX3M': '^VIX3M', 'SKEW': '^SKEW', 'US10Y': '^TNX',
    'DXY': 'DX-Y.NYB', 'HYG': 'HYG', 'LQD': 'LQD', 'OVX': '^OVX'
}

extended_data = {}
for name, ticker in extended_tickers.items():
    try:
        d = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            extended_data[name] = d[('Close', ticker)]
        else:
            extended_data[name] = d['Close']
    except:
        extended_data[name] = None

print('Data loaded')

def create_features(df, vix, ext_data, horizon):
    """Create features for specified horizon"""
    df = df.copy()
    
    # VIX Features
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['VIX_change'] = df['VIX'].pct_change()
    
    # CAVB Features
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    # RV Features
    df['RV_5d_lag'] = df['RV_5d'].shift(1)
    df['RV_22d_lag'] = df['RV_22d'].shift(1)
    
    # Extended Features
    if ext_data.get('VIX3M') is not None:
        df['VIX3M'] = ext_data['VIX3M']
        df['VIX_term'] = df['VIX3M'] / (df['VIX'] + 0.01)
    
    if ext_data.get('SKEW') is not None:
        df['SKEW'] = ext_data['SKEW']
        df['SKEW_zscore'] = (df['SKEW'] - df['SKEW'].rolling(22).mean()) / (df['SKEW'].rolling(22).std() + 0.01)
    
    if ext_data.get('US10Y') is not None:
        df['US10Y'] = ext_data['US10Y']
        df['US10Y_change'] = df['US10Y'].pct_change()
    
    if ext_data.get('DXY') is not None:
        df['DXY'] = ext_data['DXY']
        df['DXY_momentum'] = df['DXY'].pct_change(22)
    
    if ext_data.get('HYG') is not None and ext_data.get('LQD') is not None:
        df['Credit_spread'] = (ext_data['LQD'] / ext_data['HYG']) - 1
    
    if ext_data.get('OVX') is not None:
        df['OVX'] = ext_data['OVX']
        df['VIX_OVX_ratio'] = df['VIX'] / (df['OVX'] + 0.01)
    
    # Market Features
    df['Ret_lag1'] = df['Ret'].shift(1)
    df['Ret_22d'] = df['Ret'].rolling(22).sum()
    
    # Technical
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 0.001)))
    
    # Target based on horizon
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-horizon)
    
    return df.dropna()

def run_models(Xtr, ytr, Xte, yte):
    """Run all models and return results"""
    results = {}
    
    # 1. Huber
    try:
        m = HuberRegressor(epsilon=1.35, max_iter=1000)
        m.fit(Xtr, ytr)
        results['Huber'] = r2_score(yte, m.predict(Xte))
    except:
        results['Huber'] = None
    
    # 2. ElasticNet
    try:
        m = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=5000)
        m.fit(Xtr, ytr)
        results['ElasticNet'] = r2_score(yte, m.predict(Xte))
    except:
        results['ElasticNet'] = None
    
    # 3. Ridge
    try:
        m = Ridge(alpha=10.0)
        m.fit(Xtr, ytr)
        results['Ridge'] = r2_score(yte, m.predict(Xte))
    except:
        results['Ridge'] = None
    
    # 4. SVR-Linear
    try:
        m = SVR(kernel='linear', C=1.0, epsilon=0.1)
        m.fit(Xtr, ytr)
        results['SVR_Linear'] = r2_score(yte, m.predict(Xte))
    except:
        results['SVR_Linear'] = None
    
    # 5. SVR-RBF
    try:
        m = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        m.fit(Xtr, ytr)
        results['SVR_RBF'] = r2_score(yte, m.predict(Xte))
    except:
        results['SVR_RBF'] = None
    
    # 6. Random Forest (lightweight)
    try:
        m = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=20, random_state=42)
        m.fit(Xtr, ytr)
        results['RF'] = r2_score(yte, m.predict(Xte))
    except:
        results['RF'] = None
    
    # 7. Gradient Boosting (lightweight)
    try:
        m = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        m.fit(Xtr, ytr)
        results['GB'] = r2_score(yte, m.predict(Xte))
    except:
        results['GB'] = None
    
    return results

# ==========================================
# 2. Run Experiments
# ==========================================
print('\n[2/5] Running experiments for each horizon...')

all_results = {}

for horizon in HORIZONS:
    print(f'\n{"="*50}')
    print(f'HORIZON: {horizon}-day')
    print('='*50)
    
    horizon_results = {}
    
    for ticker in ASSETS:
        print(f'\nProcessing {ticker}...')
        
        df = create_features(all_data[ticker], vix, extended_data, horizon)
        
        exclude = ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ret']
        fcols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
        fcols = [c for c in fcols if df[c].notna().sum() > len(df) * 0.9]
        
        X, y = df[fcols], df['Target']
        n = len(y)
        tr_end, val_end = int(n * 0.6), int(n * 0.8)
        
        Xtr, Xval, Xte = X.iloc[:tr_end], X.iloc[tr_end:val_end], X.iloc[val_end:]
        ytr, yval, yte = y.iloc[:tr_end], y.iloc[tr_end:val_end], y.iloc[val_end:]
        
        scaler = StandardScaler()
        Xts = scaler.fit_transform(Xtr)
        Xvs = scaler.transform(Xval)
        Xtes = scaler.transform(Xte)
        
        # Feature Selection using ElasticNet
        selector = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=5000)
        selector.fit(Xts, ytr)
        
        selected_mask = np.abs(selector.coef_) > 0.001
        selected_features = [f for f, m in zip(fcols, selected_mask) if m]
        
        if len(selected_features) < 5:
            top_indices = np.argsort(np.abs(selector.coef_))[-10:]
            selected_features = [fcols[i] for i in top_indices]
        
        print(f'  Selected: {len(selected_features)} features')
        
        # Prepare selected features
        X_sel = df[selected_features]
        
        scaler_sel = StandardScaler()
        Xstr = scaler_sel.fit_transform(X_sel.iloc[:val_end])
        Xste = scaler_sel.transform(X_sel.iloc[val_end:])
        ytrval = pd.concat([ytr, yval])
        
        # Run all models
        model_results = run_models(Xstr, ytrval, Xste, yte)
        
        # Find best model
        valid_results = {k: v for k, v in model_results.items() if v is not None}
        best_model = max(valid_results, key=valid_results.get) if valid_results else None
        best_r2 = valid_results[best_model] if best_model else None
        
        print(f'  Best: {best_model} (R² = {best_r2:.4f})' if best_r2 else '  No valid model')
        
        horizon_results[ticker] = {
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'model_results': model_results,
            'best_model': best_model,
            'best_r2': best_r2
        }
    
    all_results[f'{horizon}d'] = horizon_results

# ==========================================
# 3. Create Universal Model
# ==========================================
print('\n[3/5] Creating Universal Model...')

# Find best overall configuration
best_5d_models = {t: all_results['5d'][t]['best_model'] for t in ASSETS}
best_22d_models = {t: all_results['22d'][t]['best_model'] for t in ASSETS}

# Count model occurrences for 5-day
from collections import Counter
model_counts_5d = Counter(best_5d_models.values())
model_counts_22d = Counter(best_22d_models.values())

# Find common selected features
common_features_5d = set(all_results['5d'][ASSETS[0]]['selected_features'])
for t in ASSETS[1:]:
    common_features_5d &= set(all_results['5d'][t]['selected_features'])

common_features_22d = set(all_results['22d'][ASSETS[0]]['selected_features'])
for t in ASSETS[1:]:
    common_features_22d &= set(all_results['22d'][t]['selected_features'])

print(f'Common features (5-day): {list(common_features_5d)}')
print(f'Common features (22-day): {list(common_features_22d)}')

# ==========================================
# 4. Summary
# ==========================================
print('\n' + '='*70)
print('[4/5] FINAL SUMMARY')
print('='*70)

for horizon in HORIZONS:
    key = f'{horizon}d'
    print(f'\n--- {horizon}-Day Horizon ---')
    
    for t in ASSETS:
        r = all_results[key][t]
        print(f'{t}: {r["best_model"]} = {r["best_r2"]:.4f} ({r["n_features"]} features)')
    
    # Average by model
    print('\nModel Averages:')
    models = ['Huber', 'ElasticNet', 'Ridge', 'SVR_Linear', 'SVR_RBF', 'RF', 'GB']
    for m in models:
        vals = [all_results[key][t]['model_results'].get(m) for t in ASSETS 
                if all_results[key][t]['model_results'].get(m) is not None]
        if vals:
            print(f'  {m}: {np.mean(vals):.4f}')

# ==========================================
# 5. Save Results
# ==========================================
print('\n[5/5] Saving results...')

# Convert numpy types for JSON
def clean_results(results):
    cleaned = {}
    for k, v in results.items():
        if isinstance(v, dict):
            cleaned[k] = clean_results(v)
        elif isinstance(v, (list, np.ndarray)):
            cleaned[k] = [str(x) if isinstance(x, np.floating) else x for x in v]
        elif isinstance(v, np.floating):
            cleaned[k] = float(v)
        elif isinstance(v, np.integer):
            cleaned[k] = int(v)
        else:
            cleaned[k] = v
    return cleaned

output = {
    'experiment': 'Exp17_Comprehensive_Universal_Model',
    'horizons': [5, 22],
    'models_tested': ['Huber', 'ElasticNet', 'Ridge', 'SVR_Linear', 'SVR_RBF', 'RF', 'GB'],
    'results': clean_results(all_results),
    'universal_model': {
        '5d': {
            'recommended_model': model_counts_5d.most_common(1)[0][0] if model_counts_5d else None,
            'common_features': list(common_features_5d)
        },
        '22d': {
            'recommended_model': model_counts_22d.most_common(1)[0][0] if model_counts_22d else None,
            'common_features': list(common_features_22d)
        }
    }
}

with open('results/comprehensive_universal_model.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Results saved to results/comprehensive_universal_model.json')

# Final recommendation
print('\n' + '='*70)
print('UNIVERSAL MODEL RECOMMENDATION')
print('='*70)

print(f'\n5-Day Horizon:')
print(f'  Best Model: {model_counts_5d.most_common(1)[0][0] if model_counts_5d else "N/A"}')
print(f'  Common Features: {len(common_features_5d)} features')
avg_5d = np.mean([all_results['5d'][t]['best_r2'] for t in ASSETS if all_results['5d'][t]['best_r2']])
print(f'  Average Best R²: {avg_5d:.4f}')

print(f'\n22-Day Horizon:')
print(f'  Best Model: {model_counts_22d.most_common(1)[0][0] if model_counts_22d else "N/A"}')
print(f'  Common Features: {len(common_features_22d)} features')
avg_22d = np.mean([all_results['22d'][t]['best_r2'] for t in ASSETS if all_results['22d'][t]['best_r2']])
print(f'  Average Best R²: {avg_22d:.4f}')
