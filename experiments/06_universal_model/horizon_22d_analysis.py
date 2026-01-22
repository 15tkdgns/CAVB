"""
22-Day Volatility Prediction Analysis
Compare with 5-day prediction
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, ElasticNet
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']

print('='*60)
print('22-Day vs 5-Day Prediction Analysis')
print('='*60)

# Load data
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

print('Data loaded\n')

results = {}

for ticker in ASSETS:
    print(f'Processing {ticker}...')
    df = all_data[ticker].copy()
    df['VIX'] = vix
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    # Target: 5-day ahead
    df['Target_5d'] = df['VIX'] - df['RV_22d'].shift(-5)
    
    # Target: 22-day ahead
    df['Target_22d'] = df['VIX'] - df['RV_22d'].shift(-22)
    
    df_clean = df.dropna()
    
    fcols = ['VIX', 'VIX_lag1', 'VIX_lag5', 'CAVB', 'CAVB_lag1', 'CAVB_lag5', 
             'CAVB_ma5', 'RV_5d', 'RV_22d']
    
    X = df_clean[fcols]
    y_5d = df_clean['Target_5d']
    y_22d = df_clean['Target_22d']
    
    n = len(X)
    tr = int(n * 0.8)
    
    Xtr, Xte = X.iloc[:tr], X.iloc[tr:]
    ytr_5d, yte_5d = y_5d.iloc[:tr], y_5d.iloc[tr:]
    ytr_22d, yte_22d = y_22d.iloc[:tr], y_22d.iloc[tr:]
    
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr)
    Xtes = sc.transform(Xte)
    
    # 5-day prediction
    m_5d = HuberRegressor(epsilon=1.35, max_iter=1000)
    m_5d.fit(Xts, ytr_5d)
    r2_5d = r2_score(yte_5d, m_5d.predict(Xtes))
    
    # 22-day prediction
    m_22d = HuberRegressor(epsilon=1.35, max_iter=1000)
    m_22d.fit(Xts, ytr_22d)
    r2_22d = r2_score(yte_22d, m_22d.predict(Xtes))
    
    results[ticker] = {
        '5d_r2': r2_5d,
        '22d_r2': r2_22d,
        'decay': (r2_22d - r2_5d) / abs(r2_5d) * 100
    }
    
    print(f'  5-day:  R² = {r2_5d:.4f}')
    print(f'  22-day: R² = {r2_22d:.4f} ({results[ticker]["decay"]:+.1f}%)')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)

avg_5d = np.mean([results[t]['5d_r2'] for t in ASSETS])
avg_22d = np.mean([results[t]['22d_r2'] for t in ASSETS])
decay = (avg_22d - avg_5d) / abs(avg_5d) * 100

print(f'Average 5-day R²:  {avg_5d:.4f}')
print(f'Average 22-day R²: {avg_22d:.4f} ({decay:+.1f}%)')
print(f'\n22-day prediction decay: {abs(decay):.1f}%')

# Asset ranking for 22-day
print('\n22-Day Prediction Ranking:')
sorted_assets = sorted(ASSETS, key=lambda x: results[x]['22d_r2'], reverse=True)
for i, t in enumerate(sorted_assets, 1):
    print(f'  {i}. {t}: {results[t]["22d_r2"]:.4f}')

with open('results/horizon_comparison_5d_22d.json', 'w') as f:
    json.dump({
        'experiment': 'Horizon_5d_vs_22d',
        'results': results,
        'summary': {
            'avg_5d': avg_5d,
            'avg_22d': avg_22d,
            'decay_pct': decay
        }
    }, f, indent=2)

print('\nSaved to results/horizon_comparison_5d_22d.json')
