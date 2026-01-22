"""
Experiment 20: Additional Asset Search for 22-Day Model
Test the Ensemble_Specialist model on new assets to find predictable ones
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

# New assets to test (beyond SPY, GLD, TLT, EFA, EEM)
NEW_ASSETS = {
    # Commodities (GLD-like)
    'SLV': 'Silver',
    'USO': 'Oil',
    'DBA': 'Agriculture',
    'UNG': 'NaturalGas',
    
    # Bonds (TLT-like)
    'IEF': '7-10Y Treasury',
    'HYG': 'High Yield',
    'LQD': 'Investment Grade',
    'TIP': 'TIPS',
    
    # Sectors
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLK': 'Technology',
    'XLU': 'Utilities',
    
    # Regional
    'VWO': 'Emerging All',
    'FXI': 'China',
    'EWJ': 'Japan',
    'EWZ': 'Brazil',
    
    # Alternative
    'VNQ': 'Real Estate',
    'BND': 'Total Bond',
}

print('='*70)
print('Experiment 20: Additional Asset Search for 22-Day Model')
print(f'Testing {len(NEW_ASSETS)} new assets')
print('='*70)

# Load VIX
print('\n[1/4] Loading VIX data...')
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

print('VIX data loaded')

def create_22d_features(df, vix, vix3m):
    """Create 22-day prediction features"""
    df = df.copy()
    
    df['VIX'] = vix
    df['VIX3M'] = vix3m
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
    
    vrp_hist = df['CAVB'].rolling(252).mean()
    df['VRP_adjusted_VIX'] = df['VIX'] - vrp_hist
    
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-22)
    
    return df

def ensemble_specialist_predict(Xtr, ytr, Xte, fcols):
    """Ensemble of Mean Reversion + VRP + HAR"""
    
    # 1. Mean Reversion
    dev_idx = [i for i, c in enumerate(fcols) if 'deviation' in c]
    if len(dev_idx) >= 2:
        m1 = Ridge(alpha=100.0)
        m1.fit(Xtr[:, dev_idx], ytr)
        p1 = m1.predict(Xte[:, dev_idx])
    else:
        m1 = Ridge(alpha=100.0)
        m1.fit(Xtr, ytr)
        p1 = m1.predict(Xte)
    
    # 2. VRP Adjusted
    vrp_idx = [i for i, c in enumerate(fcols) if 'VRP' in c or 'CAVB' in c or 'term' in c.lower()]
    if len(vrp_idx) >= 2:
        m2 = HuberRegressor(epsilon=1.5, max_iter=1000)
        m2.fit(Xtr[:, vrp_idx], ytr)
        p2 = m2.predict(Xte[:, vrp_idx])
    else:
        m2 = HuberRegressor(epsilon=1.5, max_iter=1000)
        m2.fit(Xtr, ytr)
        p2 = m2.predict(Xte)
    
    # 3. HAR Extended
    har_idx = [i for i, c in enumerate(fcols) if 'RV_' in c or 'VIX_ma' in c]
    if len(har_idx) >= 2:
        m3 = Ridge(alpha=50.0)
        m3.fit(Xtr[:, har_idx], ytr)
        p3 = m3.predict(Xte[:, har_idx])
    else:
        m3 = Ridge(alpha=50.0)
        m3.fit(Xtr, ytr)
        p3 = m3.predict(Xte)
    
    return (p1 + p2 + p3) / 3

# ==========================================
# 2. Test Each Asset
# ==========================================
print('\n[2/4] Testing assets...')

results = {}

for ticker, name in NEW_ASSETS.items():
    try:
        # Download data
        d = yf.download(ticker, start='2010-01-01', end='2025-01-01', progress=False)
        
        if len(d) < 1000:
            print(f'  {ticker} ({name}): SKIP - insufficient data ({len(d)} rows)')
            continue
        
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        
        d['Ret'] = d['Close'].pct_change()
        d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
        d['RV_22d'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
        d['RV_66d'] = d['Ret'].rolling(66).std() * np.sqrt(252) * 100
        
        # Create features
        df = create_22d_features(d, vix, vix3m)
        
        exclude = ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ret']
        fcols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
        fcols = [c for c in fcols if df[c].notna().sum() > len(df) * 0.7]
        
        df = df.dropna(subset=fcols + ['Target'])
        
        if len(df) < 500:
            print(f'  {ticker} ({name}): SKIP - insufficient valid data')
            continue
        
        X, y = df[fcols].values, df['Target'].values
        n = len(y)
        tr_end = int(n * 0.8)
        
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[:tr_end])
        Xte = scaler.transform(X[tr_end:])
        ytr, yte = y[:tr_end], y[tr_end:]
        
        # Ensemble prediction
        y_pred = ensemble_specialist_predict(Xtr, ytr, Xte, fcols)
        r2 = r2_score(yte, y_pred)
        
        # Also test Ridge baseline
        ridge = Ridge(alpha=935.5)  # From Optuna
        ridge.fit(Xtr, ytr)
        r2_ridge = r2_score(yte, ridge.predict(Xte))
        
        results[ticker] = {
            'name': name,
            'ensemble_r2': r2,
            'ridge_r2': r2_ridge,
            'best_r2': max(r2, r2_ridge),
            'n_samples': len(df),
            'predictable': r2 > 0.1 or r2_ridge > 0.1
        }
        
        status = 'GOOD' if results[ticker]['best_r2'] > 0.2 else ('OK' if results[ticker]['best_r2'] > 0 else 'NO')
        print(f'  {ticker} ({name}): Ensemble={r2:.3f}, Ridge={r2_ridge:.3f} [{status}]')
        
    except Exception as e:
        print(f'  {ticker} ({name}): ERROR - {str(e)[:40]}')

# ==========================================
# 3. Summary
# ==========================================
print('\n' + '='*70)
print('[3/4] RESULTS SUMMARY')
print('='*70)

# Sort by best R²
sorted_results = sorted(results.items(), key=lambda x: x[1]['best_r2'], reverse=True)

print('\nAsset Rankings (22-Day Prediction):')
print(f'{"Rank":<5} {"Ticker":<6} {"Name":<20} {"Ensemble":<10} {"Ridge":<10} {"Status"}')
print('-'*70)

predictable = []
for i, (ticker, r) in enumerate(sorted_results, 1):
    status = 'Recommended' if r['best_r2'] > 0.2 else ('Usable' if r['best_r2'] > 0 else 'Not Recommended')
    print(f'{i:<5} {ticker:<6} {r["name"]:<20} {r["ensemble_r2"]:>8.3f}  {r["ridge_r2"]:>8.3f}  {status}')
    
    if r['best_r2'] > 0.1:
        predictable.append((ticker, r['name'], r['best_r2']))

print(f'\n\nPredictable Assets (R² > 0.1):')
for t, n, r2 in predictable:
    print(f'  {t} ({n}): R² = {r2:.3f}')

print(f'\nTotal Predictable: {len(predictable)}/{len(results)} tested')

# ==========================================
# 4. Save Results
# ==========================================
print('\n[4/4] Saving results...')

output = {
    'experiment': 'Exp20_Asset_Search',
    'model': 'Ensemble_Specialist + Ridge',
    'assets_tested': len(results),
    'predictable_count': len(predictable),
    'results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                   for kk, vv in v.items()} for k, v in results.items()},
    'predictable_assets': [{'ticker': t, 'name': n, 'r2': float(r2)} for t, n, r2 in predictable],
    'ranking': [{'rank': i+1, 'ticker': t, 'best_r2': float(r['best_r2'])} 
                for i, (t, r) in enumerate(sorted_results)]
}

with open('results/22d_asset_search.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Results saved to results/22d_asset_search.json')

# Final recommendation
print('\n' + '='*70)
print('FINAL RECOMMENDATION: 22-Day Predictable Assets')
print('='*70)

# Combine with original assets
print('\nOriginal Assets:')
print('  GLD (Gold): R² ≈ 0.64-0.66 - RECOMMENDED')
print('  TLT (20Y Treasury): R² ≈ 0.36-0.38 - RECOMMENDED')
print('  EFA (EAFE): R² ≈ 0.10 - USABLE')

print('\nNewly Discovered:')
for t, n, r2 in predictable[:5]:  # Top 5 new
    print(f'  {t} ({n}): R² = {r2:.3f}')
