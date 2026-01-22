"""
Experiment 15: Extended Feature Set
Adding macro, options, cross-asset, and technical features
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']

print('='*70)
print('Experiment 15: Extended Feature Set')
print('Macro + Options + Cross-Asset + Technical Features')
print('='*70)

# ==========================================
# 1. Load Base Data
# ==========================================
print('\n[1/5] Loading base data...')

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

# ==========================================
# 2. Load Extended Data
# ==========================================
print('[2/5] Loading extended data (macro, options, cross-asset)...')

extended_tickers = {
    'VIX3M': '^VIX3M',      # VIX 3-month
    'SKEW': '^SKEW',        # CBOE SKEW Index
    'US10Y': '^TNX',        # 10-year Treasury yield
    'DXY': 'DX-Y.NYB',      # Dollar Index
    'HYG': 'HYG',           # High Yield Bond ETF
    'LQD': 'LQD',           # Investment Grade Bond ETF
    'OVX': '^OVX',          # Oil Volatility
}

extended_data = {}
for name, ticker in extended_tickers.items():
    try:
        d = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            extended_data[name] = d[('Close', ticker)]
        else:
            extended_data[name] = d['Close']
        print(f'  {name}: {len(d)} rows')
    except Exception as e:
        print(f'  {name}: FAILED ({e})')
        extended_data[name] = None

print('Data loading complete')

def create_extended_features(df, vix, ext_data, asset_ticker):
    """Create comprehensive feature set"""
    df = df.copy()
    
    # ==========================================
    # Basic VIX/CAVB Features (13 - baseline)
    # ==========================================
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['VIX_change'] = df['VIX'].pct_change()
    
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    df['RV_1d'] = df['RV_5d'].shift(1)
    df['RV_5d_lag'] = df['RV_5d'].shift(1)
    df['RV_22d_lag'] = df['RV_22d'].shift(1)
    
    # ==========================================
    # Options Features (VIX Term Structure, SKEW)
    # ==========================================
    if ext_data.get('VIX3M') is not None:
        df['VIX3M'] = ext_data['VIX3M']
        df['VIX_term'] = df['VIX3M'] / (df['VIX'] + 0.01)  # Contango/Backwardation
        df['VIX_term_lag1'] = df['VIX_term'].shift(1)
    
    if ext_data.get('SKEW') is not None:
        df['SKEW'] = ext_data['SKEW']
        df['SKEW_lag1'] = df['SKEW'].shift(1)
        df['SKEW_ma5'] = df['SKEW'].rolling(5).mean()
        df['SKEW_zscore'] = (df['SKEW'] - df['SKEW'].rolling(22).mean()) / (df['SKEW'].rolling(22).std() + 0.01)
    
    # ==========================================
    # Macro Features (Yields, Dollar)
    # ==========================================
    if ext_data.get('US10Y') is not None:
        df['US10Y'] = ext_data['US10Y']
        df['US10Y_change'] = df['US10Y'].pct_change()
        df['US10Y_ma22'] = df['US10Y'].rolling(22).mean()
    
    if ext_data.get('DXY') is not None:
        df['DXY'] = ext_data['DXY']
        df['DXY_change'] = df['DXY'].pct_change()
        df['DXY_momentum'] = df['DXY'].pct_change(22)
    
    # ==========================================
    # Credit Spread Features
    # ==========================================
    if ext_data.get('HYG') is not None and ext_data.get('LQD') is not None:
        df['HYG'] = ext_data['HYG']
        df['LQD'] = ext_data['LQD']
        df['Credit_spread'] = (df['LQD'] / df['HYG']) - 1  # Proxy for credit risk
        df['Credit_spread_change'] = df['Credit_spread'].pct_change()
    
    # ==========================================
    # Oil Volatility
    # ==========================================
    if ext_data.get('OVX') is not None:
        df['OVX'] = ext_data['OVX']
        df['OVX_lag1'] = df['OVX'].shift(1)
        df['VIX_OVX_ratio'] = df['VIX'] / (df['OVX'] + 0.01)
    
    # ==========================================
    # Market Return Features
    # ==========================================
    df['Ret_lag1'] = df['Ret'].shift(1)
    df['Ret_5d'] = df['Ret'].rolling(5).sum()
    df['Ret_22d'] = df['Ret'].rolling(22).sum()
    df['Ret_momentum'] = df['Close'].pct_change(22)
    
    # Leverage effect (negative returns -> high volatility)
    df['Leverage_effect'] = df['Ret_lag1'] * df['RV_5d_lag']
    
    # ==========================================
    # Technical Indicators
    # ==========================================
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 0.001)))
    
    # Bollinger Band %B
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_pctB'] = (df['Close'] - (ma20 - 2*std20)) / (4*std20 + 0.001)
    
    # ATR (14-day)
    high = df['High'] if 'High' in df.columns else df['Close']
    low = df['Low'] if 'Low' in df.columns else df['Close']
    tr = pd.concat([high - low, 
                    abs(high - df['Close'].shift(1)),
                    abs(low - df['Close'].shift(1))], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_pct'] = df['ATR'] / df['Close'] * 100
    
    # ==========================================
    # Target
    # ==========================================
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    
    return df.dropna()

# ==========================================
# 3. Run Experiments
# ==========================================
print('\n[3/5] Running experiments...')

results = {}

for ticker in ASSETS:
    print(f'\nProcessing {ticker}...')
    
    df = create_extended_features(all_data[ticker], vix, extended_data, ticker)
    
    # Get all numeric feature columns
    exclude = ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ret']
    fcols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    fcols = [c for c in fcols if df[c].notna().sum() > len(df) * 0.9]
    
    # Baseline features (original 13)
    baseline_cols = ['VIX', 'VIX_lag1', 'VIX_lag5', 'VIX_ma22', 'VIX_change',
                     'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
                     'RV_5d', 'RV_5d_lag', 'RV_22d', 'RV_22d_lag']
    baseline_cols = [c for c in baseline_cols if c in fcols]
    
    # Extended features (all available)
    extended_cols = fcols
    
    X_base = df[baseline_cols]
    X_ext = df[extended_cols]
    y = df['Target']
    
    n = len(y)
    tr_end = int(n * 0.8)
    
    # Baseline
    Xb_tr, Xb_te = X_base.iloc[:tr_end], X_base.iloc[tr_end:]
    Xe_tr, Xe_te = X_ext.iloc[:tr_end], X_ext.iloc[tr_end:]
    ytr, yte = y.iloc[:tr_end], y.iloc[tr_end:]
    
    # Scale
    scaler_b = StandardScaler()
    Xbs = scaler_b.fit_transform(Xb_tr)
    Xbtes = scaler_b.transform(Xb_te)
    
    scaler_e = StandardScaler()
    Xes = scaler_e.fit_transform(Xe_tr)
    Xetes = scaler_e.transform(Xe_te)
    
    # Baseline model
    model_b = HuberRegressor(epsilon=1.35, max_iter=1000)
    model_b.fit(Xbs, ytr)
    r2_base = r2_score(yte, model_b.predict(Xbtes))
    
    # Extended model
    model_e = HuberRegressor(epsilon=1.35, max_iter=1000)
    model_e.fit(Xes, ytr)
    r2_ext = r2_score(yte, model_e.predict(Xetes))
    
    improvement = (r2_ext - r2_base) / abs(r2_base) * 100
    
    print(f'  Baseline ({len(baseline_cols)} features): R² = {r2_base:.4f}')
    print(f'  Extended ({len(extended_cols)} features): R² = {r2_ext:.4f} ({improvement:+.2f}%)')
    
    # Feature importance for extended model
    coef_df = pd.DataFrame({
        'feature': extended_cols,
        'coef': model_e.coef_
    })
    coef_df['abs_coef'] = np.abs(coef_df['coef'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    top_features = coef_df.head(10)['feature'].tolist()
    
    results[ticker] = {
        'n_baseline': len(baseline_cols),
        'n_extended': len(extended_cols),
        'r2_baseline': r2_base,
        'r2_extended': r2_ext,
        'improvement': improvement,
        'top_features': top_features
    }

# ==========================================
# 4. Summary
# ==========================================
print('\n' + '='*70)
print('[4/5] FINAL SUMMARY - Extended Features')
print('='*70)

avg_base = np.mean([results[t]['r2_baseline'] for t in ASSETS])
avg_ext = np.mean([results[t]['r2_extended'] for t in ASSETS])
avg_imp = (avg_ext - avg_base) / abs(avg_base) * 100

print(f'\nAverage Baseline R²: {avg_base:.4f}')
print(f'Average Extended R²: {avg_ext:.4f}')
print(f'Average Improvement: {avg_imp:+.2f}%')

print('\nAsset-Specific Results:')
for t in ASSETS:
    r = results[t]
    print(f'  {t}: {r["r2_baseline"]:.4f} → {r["r2_extended"]:.4f} ({r["improvement"]:+.2f}%)')

# Most important new features across assets
all_top = []
for t in ASSETS:
    all_top.extend(results[t]['top_features'])

from collections import Counter
feature_counts = Counter(all_top)
print('\nMost Important New Features (across all assets):')
for feat, count in feature_counts.most_common(10):
    print(f'  {feat}: {count}/5 assets')

# ==========================================
# 5. Save Results
# ==========================================
print('\n[5/5] Saving results...')

output = {
    'experiment': 'Exp15_Extended_Features',
    'description': 'Macro, options, cross-asset, and technical features',
    'results': results,
    'summary': {
        'avg_baseline': avg_base,
        'avg_extended': avg_ext,
        'improvement_pct': avg_imp,
        'most_important_features': dict(feature_counts.most_common(10))
    },
    'extended_data_sources': list(extended_tickers.keys())
}

with open('results/extended_features.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Results saved to results/extended_features.json')
