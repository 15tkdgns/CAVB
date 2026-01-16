"""
Advanced Universal Model Experiments
Experiment 11: Advanced Feature Engineering

Adding 50+ new features:
- Macro indicators (yields, commodities)
- Credit spreads, Put/Call ratio
- VIX term structure
- Cross-asset correlations
- Technical indicators
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, ElasticNet
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']

print('='*65)
print('Experiment 11: Advanced Feature Engineering')
print('Adding 50+ new features for R² improvement')
print('='*65)

# Load all data sources
print('\n[1/5] Loading base data...')
all_data = {}
for t in ASSETS:
    d = yf.download(t, start='2010-01-01', end='2025-01-01', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d['Ret'] = d['Close'].pct_change()
    d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
    d['RV_10d'] = d['Ret'].rolling(10).std() * np.sqrt(252) * 100
    d['RV_22d'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
    d['RV_66d'] = d['Ret'].rolling(66).std() * np.sqrt(252) * 100  # 3-month
    all_data[t] = d

# VIX
vix_data = yf.download('^VIX', start='2010-01-01', end='2025-01-01', progress=False)
if isinstance(vix_data.columns, pd.MultiIndex):
    vix = vix_data[('Close', '^VIX')]
else:
    vix = vix_data['Close']

print('[2/5] Loading macro/credit indicators...')
# Macro indicators
try:
    # US 10Y Treasury Yield
    tlt_yield = yf.download('^TNX', start='2010-01-01', end='2025-01-01', progress=False)
    if isinstance(tlt_yield.columns, pd.MultiIndex):
        us10y = tlt_yield[('Close', '^TNX')]
    else:
        us10y = tlt_yield['Close']
except:
    us10y = None

try:
    # Dollar Index
    dxy = yf.download('DX-Y.NYB', start='2010-01-01', end='2025-01-01', progress=False)
    if isinstance(dxy.columns, pd.MultiIndex):
        dollar = dxy[('Close', 'DX-Y.NYB')]
    else:
        dollar = dxy['Close']
except:
    dollar = None

try:
    # Oil (WTI)
    oil_data = yf.download('CL=F', start='2010-01-01', end='2025-01-01', progress=False)
    if isinstance(oil_data.columns, pd.MultiIndex):
        oil = oil_data[('Close', 'CL=F')]
    else:
        oil = oil_data['Close']
except:
    oil = None

try:
    # High Yield Spread proxy (HYG vs LQD)
    hyg = yf.download('HYG', start='2010-01-01', end='2025-01-01', progress=False)
    lqd = yf.download('LQD', start='2010-01-01', end='2025-01-01', progress=False)
    if isinstance(hyg.columns, pd.MultiIndex):
        hyg_price = hyg[('Close', 'HYG')]
        lqd_price = lqd[('Close', 'LQD')]
    else:
        hyg_price = hyg['Close']
        lqd_price = lqd['Close']
    credit_spread = (lqd_price / hyg_price - 1) * 100  # Spread proxy
except:
    credit_spread = None

try:
    # VIX 3-month (VXV)
    vxv_data = yf.download('^VIX3M', start='2010-01-01', end='2025-01-01', progress=False)
    if isinstance(vxv_data.columns, pd.MultiIndex):
        vix3m = vxv_data[('Close', '^VIX3M')]
    else:
        vix3m = vxv_data['Close']
except:
    vix3m = None

print('[3/5] Creating advanced features...')

def create_advanced_features(df, vix, us10y, dollar, oil, credit_spread, vix3m, all_data, ticker):
    """Create 50+ advanced features"""
    df = df.copy()
    
    # === Base VIX/RV Features (13) ===
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_lag22'] = df['VIX'].shift(22)
    df['VIX_change_1d'] = df['VIX'].pct_change(1)
    df['VIX_change_5d'] = df['VIX'].pct_change(5)
    df['VIX_ma5'] = df['VIX'].rolling(5).mean()
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['VIX_std22'] = df['VIX'].rolling(22).std()
    df['VIX_zscore'] = (df['VIX'] - df['VIX_ma22']) / df['VIX_std22']
    
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    df['CAVB_ma22'] = df['CAVB'].rolling(22).mean()
    
    # === Realized Volatility Features (8) ===
    df['RV_ratio_5_22'] = df['RV_5d'] / df['RV_22d']
    df['RV_ratio_22_66'] = df['RV_22d'] / df['RV_66d']
    df['RV_change_5d'] = df['RV_5d'].pct_change(5)
    df['RV_ma22'] = df['RV_22d'].rolling(22).mean()
    df['RV_std22'] = df['RV_22d'].rolling(22).std()
    df['RV_zscore'] = (df['RV_22d'] - df['RV_ma22']) / df['RV_std22']
    df['RV_high_22d'] = df['RV_22d'].rolling(22).max()
    df['RV_low_22d'] = df['RV_22d'].rolling(22).min()
    
    # === VIX Term Structure (4) ===
    if vix3m is not None and len(vix3m) > 0:
        df['VIX3M'] = vix3m
        df['VIX_term_spread'] = df['VIX3M'] - df['VIX']  # Contango/Backwardation
        df['VIX_term_ratio'] = df['VIX'] / df['VIX3M']
        df['VIX_term_spread_lag5'] = df['VIX_term_spread'].shift(5)
    
    # === Macro Features (12) ===
    if us10y is not None and len(us10y) > 0:
        df['US10Y'] = us10y
        df['US10Y_change'] = df['US10Y'].pct_change(5)
        df['US10Y_ma22'] = df['US10Y'].rolling(22).mean()
        df['VIX_yield_ratio'] = df['VIX'] / (df['US10Y'] + 1)
    
    if dollar is not None and len(dollar) > 0:
        df['DXY'] = dollar
        df['DXY_change'] = df['DXY'].pct_change(5)
        df['DXY_ma22'] = df['DXY'].rolling(22).mean()
    
    if oil is not None and len(oil) > 0:
        df['OIL'] = oil
        df['OIL_change'] = df['OIL'].pct_change(5)
        df['OIL_vol'] = df['OIL'].pct_change().rolling(22).std() * np.sqrt(252) * 100
    
    # === Credit Spread Features (4) ===
    if credit_spread is not None and len(credit_spread) > 0:
        df['CREDIT_SPREAD'] = credit_spread
        df['CREDIT_SPREAD_change'] = df['CREDIT_SPREAD'].pct_change(5)
        df['CREDIT_SPREAD_ma22'] = df['CREDIT_SPREAD'].rolling(22).mean()
        df['VIX_credit_corr'] = df['VIX'].rolling(22).corr(df['CREDIT_SPREAD'])
    
    # === Technical Indicators (8) ===
    price = df['Close']
    df['RSI'] = compute_rsi(price, 14)
    df['MACD'] = compute_macd(price)
    df['BB_position'] = compute_bb_position(price)
    df['ATR'] = compute_atr(df, 14)
    df['Price_ma_ratio'] = price / price.rolling(22).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(22).mean()
    df['Price_momentum'] = price.pct_change(22)
    df['Price_vol_corr'] = price.pct_change().rolling(22).corr(df['RV_22d'])
    
    # === Cross-Asset Features (8) ===
    for other in ['SPY', 'GLD', 'TLT']:
        if other != ticker and other in all_data:
            other_rv = all_data[other]['RV_22d']
            df[f'RV_corr_{other}'] = df['RV_22d'].rolling(22).corr(other_rv)
            df[f'RV_spread_{other}'] = df['RV_22d'] - other_rv
    
    # === Target ===
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    
    return df.dropna()

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices):
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    return ema12 - ema26

def compute_bb_position(prices, period=20):
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    return (prices - ma) / (2 * std)

def compute_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

print('[4/5] Training models with extended features...')

results = {}

for ticker in ASSETS:
    print(f'\n  Processing {ticker}...')
    
    df = create_advanced_features(
        all_data[ticker], vix, us10y, dollar, oil, credit_spread, vix3m, all_data, ticker
    )
    
    # Get all numeric feature columns
    exclude = ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ret']
    fcols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    
    # Remove columns with too many NaNs
    fcols = [c for c in fcols if df[c].notna().sum() > len(df) * 0.8]
    
    print(f'    Features: {len(fcols)}')
    
    X, y = df[fcols], df['Target']
    n = len(X)
    tr_end, val_end = int(n*0.6), int(n*0.8)
    
    Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
    Xval, yval = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    Xte, yte = X.iloc[val_end:], y.iloc[val_end:]
    
    scaler = StandardScaler()
    Xts = scaler.fit_transform(Xtr)
    Xvs = scaler.transform(Xval)
    Xtes = scaler.transform(Xte)
    
    Xtvs = scaler.fit_transform(pd.concat([Xtr, Xval]))
    Xtes_full = scaler.transform(Xte)
    
    r = {'n_features': len(fcols)}
    
    # Baseline (original features only)
    base_cols = [c for c in fcols if 'corr_' not in c and 'spread_' not in c 
                 and 'RSI' not in c and 'MACD' not in c and 'DXY' not in c
                 and 'OIL' not in c and 'CREDIT' not in c and 'US10Y' not in c
                 and 'VIX3M' not in c and 'term' not in c]
    X_base = Xtr[base_cols]
    X_base_te = Xte[base_cols]
    scaler_base = StandardScaler()
    X_base_s = scaler_base.fit_transform(X_base)
    X_base_te_s = scaler_base.transform(X_base_te)
    
    hub_base = HuberRegressor(epsilon=1.35, max_iter=1000)
    hub_base.fit(X_base_s, ytr)
    r['baseline'] = r2_score(yte, hub_base.predict(X_base_te_s))
    print(f'    Baseline ({len(base_cols)} features): {r["baseline"]:.4f}')
    
    # All features
    hub_all = HuberRegressor(epsilon=1.35, max_iter=1000)
    hub_all.fit(Xts, ytr)
    r['all_features'] = r2_score(yte, hub_all.predict(Xtes))
    print(f'    All Features ({len(fcols)}): {r["all_features"]:.4f}')
    
    # Feature importance (mutual info)
    mi = mutual_info_regression(Xts, ytr, random_state=42)
    mi_df = pd.DataFrame({'feature': fcols, 'mi': mi}).sort_values('mi', ascending=False)
    
    # Top 20 features
    top20 = mi_df.head(20)['feature'].tolist()
    scaler_top = StandardScaler()
    X_top_s = scaler_top.fit_transform(Xtr[top20])
    X_top_te_s = scaler_top.transform(Xte[top20])
    
    hub_top = HuberRegressor(epsilon=1.35, max_iter=1000)
    hub_top.fit(X_top_s, ytr)
    r['top20_features'] = r2_score(yte, hub_top.predict(X_top_te_s))
    r['top20_list'] = top20[:10]  # Save top 10
    print(f'    Top 20 Features: {r["top20_features"]:.4f}')
    
    # Top 30 features
    top30 = mi_df.head(30)['feature'].tolist()
    scaler_t30 = StandardScaler()
    X_t30_s = scaler_t30.fit_transform(Xtr[top30])
    X_t30_te_s = scaler_t30.transform(Xte[top30])
    
    hub_t30 = HuberRegressor(epsilon=1.35, max_iter=1000)
    hub_t30.fit(X_t30_s, ytr)
    r['top30_features'] = r2_score(yte, hub_t30.predict(X_t30_te_s))
    print(f'    Top 30 Features: {r["top30_features"]:.4f}')
    
    results[ticker] = r

# Summary
print('\n' + '='*65)
print('FINAL SUMMARY')
print('='*65)

models = ['baseline', 'all_features', 'top20_features', 'top30_features']
baseline_avg = np.mean([results[t]['baseline'] for t in ASSETS])

for m in models:
    vals = [results[t][m] for t in ASSETS]
    avg = np.mean(vals)
    imp = (avg - baseline_avg) / abs(baseline_avg) * 100
    print(f'{m:18}: {avg:.4f} ({imp:+.2f}%)')

# Best features across assets
print('\n' + '='*65)
print('TOP 10 MOST IMPORTANT FEATURES (across all assets)')
print('='*65)
for ticker in ASSETS:
    print(f'\n{ticker}: {results[ticker]["top20_list"]}')

# Save
output = {
    'experiment': 'Exp11_Advanced_Features',
    'results': {k: {m: v[m] if not isinstance(v[m], list) else v[m] 
                   for m in v.keys()} for k, v in results.items()},
    'summary': {
        'baseline_avg': baseline_avg,
        'all_features_avg': np.mean([results[t]['all_features'] for t in ASSETS]),
        'top20_avg': np.mean([results[t]['top20_features'] for t in ASSETS]),
        'top30_avg': np.mean([results[t]['top30_features'] for t in ASSETS])
    }
}

with open('results/advanced_experiment_11.json', 'w') as f:
    json.dump(output, f, indent=2)

print('\nResults saved to results/advanced_experiment_11.json')
