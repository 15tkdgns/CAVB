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

print('='*60)
print('Experiment 12: Feature Interaction')
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
    df = all_data[ticker].copy()
    df['VIX'] = vix
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    
    # Base features
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    
    # INTERACTION FEATURES
    df['VIX_x_RV'] = df['VIX'] * df['RV_22d']
    df['VIX_div_RV'] = df['VIX'] / (df['RV_22d'] + 0.01)
    df['CAVB_x_RV5'] = df['CAVB'] * df['RV_5d']
    df['VIX_sq'] = df['VIX'] ** 2
    df['CAVB_sq'] = df['CAVB'] ** 2
    df['RV_ratio'] = df['RV_5d'] / (df['RV_22d'] + 0.01)
    df['VIX_RV_diff_sq'] = (df['VIX'] - df['RV_22d']) ** 2
    
    df = df.dropna()
    
    # Base features only
    base_cols = ['VIX', 'VIX_lag1', 'CAVB', 'CAVB_lag1', 'RV_5d', 'RV_22d']
    
    # With interactions
    inter_cols = base_cols + ['VIX_x_RV', 'VIX_div_RV', 'CAVB_x_RV5', 'VIX_sq', 'CAVB_sq', 'RV_ratio', 'VIX_RV_diff_sq']
    
    X_base, X_inter = df[base_cols], df[inter_cols]
    y = df['Target']
    n = len(y)
    tr = int(n * 0.8)
    
    Xtr_b = X_base.iloc[:tr]
    Xte_b = X_base.iloc[tr:]
    Xtr_i = X_inter.iloc[:tr]
    Xte_i = X_inter.iloc[tr:]
    ytr = y.iloc[:tr]
    yte = y.iloc[tr:]
    
    sc1, sc2 = StandardScaler(), StandardScaler()
    Xtr_bs = sc1.fit_transform(Xtr_b)
    Xte_bs = sc1.transform(Xte_b)
    Xtr_is = sc2.fit_transform(Xtr_i)
    Xte_is = sc2.transform(Xte_i)
    
    # Baseline
    m1 = HuberRegressor(epsilon=1.35, max_iter=1000)
    m1.fit(Xtr_bs, ytr)
    r_base = r2_score(yte, m1.predict(Xte_bs))
    
    # With interactions
    m2 = HuberRegressor(epsilon=1.35, max_iter=1000)
    m2.fit(Xtr_is, ytr)
    r_inter = r2_score(yte, m2.predict(Xte_is))
    
    imp = (r_inter - r_base) / abs(r_base) * 100
    results[ticker] = {'baseline': r_base, 'interaction': r_inter, 'improvement': imp}
    print(f'{ticker}: Base={r_base:.4f} Inter={r_inter:.4f} ({imp:+.2f}%)')

print('\n' + '='*60)
avg_b = np.mean([results[t]['baseline'] for t in ASSETS])
avg_i = np.mean([results[t]['interaction'] for t in ASSETS])
print(f'Average: Base={avg_b:.4f} Inter={avg_i:.4f} ({(avg_i-avg_b)/abs(avg_b)*100:+.2f}%)')

with open('results/advanced_experiment_12.json', 'w') as f:
    json.dump({'exp': 'Exp12_Interaction', 'results': results}, f, indent=2)
print('\nSaved to results/advanced_experiment_12.json')
