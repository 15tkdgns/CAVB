"""
Experiment 16: Extended Features with Feature Selection
Using ElasticNet L1 regularization to select important features
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']

print('='*70)
print('Experiment 16: Extended Features + Feature Selection')
print('ElasticNet L1 Regularization for Sparse Selection')
print('='*70)

# Load data (same as Exp 15)
print('\n[1/4] Loading data...')

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

def create_extended_features(df, vix, ext_data):
    """Create comprehensive feature set"""
    df = df.copy()
    
    # Basic VIX/CAVB Features
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['VIX_change'] = df['VIX'].pct_change()
    
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    df['RV_5d_lag'] = df['RV_5d'].shift(1)
    df['RV_22d_lag'] = df['RV_22d'].shift(1)
    
    # Options Features
    if ext_data.get('VIX3M') is not None:
        df['VIX3M'] = ext_data['VIX3M']
        df['VIX_term'] = df['VIX3M'] / (df['VIX'] + 0.01)
    
    if ext_data.get('SKEW') is not None:
        df['SKEW'] = ext_data['SKEW']
        df['SKEW_zscore'] = (df['SKEW'] - df['SKEW'].rolling(22).mean()) / (df['SKEW'].rolling(22).std() + 0.01)
    
    # Macro Features
    if ext_data.get('US10Y') is not None:
        df['US10Y'] = ext_data['US10Y']
        df['US10Y_change'] = df['US10Y'].pct_change()
    
    if ext_data.get('DXY') is not None:
        df['DXY'] = ext_data['DXY']
        df['DXY_momentum'] = df['DXY'].pct_change(22)
    
    # Credit Spread
    if ext_data.get('HYG') is not None and ext_data.get('LQD') is not None:
        df['Credit_spread'] = (ext_data['LQD'] / ext_data['HYG']) - 1
    
    # Oil Volatility
    if ext_data.get('OVX') is not None:
        df['OVX'] = ext_data['OVX']
        df['VIX_OVX_ratio'] = df['VIX'] / (df['OVX'] + 0.01)
    
    # Market Return
    df['Ret_lag1'] = df['Ret'].shift(1)
    df['Ret_22d'] = df['Ret'].rolling(22).sum()
    
    # Technical
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 0.001)))
    
    # Target
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    
    return df.dropna()

print('[2/4] Running experiments with feature selection...')

results = {}

for ticker in ASSETS:
    print(f'\nProcessing {ticker}...')
    
    df = create_extended_features(all_data[ticker], vix, extended_data)
    
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
    
    # Step 1: Feature selection using ElasticNet on train set
    selector_model = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=5000)
    selector_model.fit(Xts, ytr)
    
    # Select features with non-zero coefficients
    selected_mask = np.abs(selector_model.coef_) > 0.001
    selected_features = [f for f, m in zip(fcols, selected_mask) if m]
    n_selected = len(selected_features)
    
    print(f'  Selected features: {n_selected}/{len(fcols)}')
    
    if n_selected < 3:
        # If too few selected, use top 10 by coefficient magnitude
        top_indices = np.argsort(np.abs(selector_model.coef_))[-10:]
        selected_features = [fcols[i] for i in top_indices]
        n_selected = len(selected_features)
        print(f'  (Adjusted to {n_selected} features)')
    
    # Step 2: Train final model on selected features
    X_sel = df[selected_features]
    Xstr = scaler.fit_transform(X_sel.iloc[:tr_end])
    Xsval = scaler.transform(X_sel.iloc[tr_end:val_end])
    Xste = scaler.transform(X_sel.iloc[val_end:])
    
    # Combine train+val for final model
    Xstrval = np.vstack([Xstr, Xsval])
    ytrval = pd.concat([ytr, yval])
    
    scaler_full = StandardScaler()
    Xstrval_s = scaler_full.fit_transform(X_sel.iloc[:val_end])
    Xste_s = scaler_full.transform(X_sel.iloc[val_end:])
    
    model = HuberRegressor(epsilon=1.35, max_iter=1000)
    model.fit(Xstrval_s, ytrval)
    r2_selected = r2_score(yte, model.predict(Xste_s))
    
    # Baseline (original 11 core features)
    core_features = ['VIX', 'VIX_lag1', 'VIX_lag5', 'VIX_ma22',
                     'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
                     'RV_5d', 'RV_22d']
    core_features = [c for c in core_features if c in fcols]
    
    X_core = df[core_features]
    scaler_core = StandardScaler()
    Xctr = scaler_core.fit_transform(X_core.iloc[:val_end])
    Xcte = scaler_core.transform(X_core.iloc[val_end:])
    
    model_core = HuberRegressor(epsilon=1.35, max_iter=1000)
    model_core.fit(Xctr, ytrval)
    r2_core = r2_score(yte, model_core.predict(Xcte))
    
    improvement = (r2_selected - r2_core) / abs(r2_core) * 100
    
    print(f'  Core ({len(core_features)} features): R² = {r2_core:.4f}')
    print(f'  Selected ({n_selected} features): R² = {r2_selected:.4f} ({improvement:+.2f}%)')
    print(f'  Selected: {selected_features[:5]}...')
    
    results[ticker] = {
        'n_core': len(core_features),
        'n_selected': n_selected,
        'r2_core': r2_core,
        'r2_selected': r2_selected,
        'improvement': improvement,
        'selected_features': selected_features
    }

# Summary
print('\n' + '='*70)
print('[3/4] FINAL SUMMARY - Feature Selection')
print('='*70)

avg_core = np.mean([results[t]['r2_core'] for t in ASSETS])
avg_sel = np.mean([results[t]['r2_selected'] for t in ASSETS])
avg_imp = (avg_sel - avg_core) / abs(avg_core) * 100

print(f'\nAverage Core R²: {avg_core:.4f}')
print(f'Average Selected R²: {avg_sel:.4f}')
print(f'Average Improvement: {avg_imp:+.2f}%')

print('\nAsset-Specific:')
for t in ASSETS:
    r = results[t]
    print(f'  {t}: {r["r2_core"]:.4f} → {r["r2_selected"]:.4f} ({r["improvement"]:+.2f}%), {r["n_selected"]} features')

# Find consistently selected new features
from collections import Counter
all_selected = []
for t in ASSETS:
    all_selected.extend(results[t]['selected_features'])

feature_counts = Counter(all_selected)
core_set = {'VIX', 'VIX_lag1', 'VIX_lag5', 'VIX_ma22', 'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5', 'RV_5d', 'RV_22d'}

print('\nNew Features Selected (not in core):')
new_features = {f: c for f, c in feature_counts.items() if f not in core_set}
for feat, count in sorted(new_features.items(), key=lambda x: -x[1])[:10]:
    print(f'  {feat}: {count}/5 assets')

# Save
print('\n[4/4] Saving results...')

output = {
    'experiment': 'Exp16_Feature_Selection',
    'method': 'ElasticNet L1 regularization',
    'results': results,
    'summary': {
        'avg_core': avg_core,
        'avg_selected': avg_sel,
        'improvement_pct': avg_imp,
        'new_important_features': dict(sorted(new_features.items(), key=lambda x: -x[1])[:10])
    }
}

with open('results/feature_selection.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Results saved to results/feature_selection.json')
