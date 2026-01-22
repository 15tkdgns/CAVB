"""
Hybrid HAR-ElasticNet Experiment
Based on: Gong et al. (2025) - Volatility Spillovers in High-Dimensional Financial Systems

Methodology:
Step 1: Own-volatility dynamics (HAR structure) → OLS (preserve persistence ~0.99)
Step 2: Cross-asset spillovers → ElasticNet (sparse identification)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']

print('='*70)
print('Experiment 13: Hybrid HAR-ElasticNet')
print('Two-Stage Approach: OLS (own-vol) + ElasticNet (spillovers)')
print('='*70)

# Load data
all_data = {}
for t in ASSETS:
    d = yf.download(t, start='2015-01-01', end='2025-01-01', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d['Ret'] = d['Close'].pct_change()
    d['RV_1d'] = d['Ret'].abs() * np.sqrt(252) * 100
    d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
    d['RV_22d'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
    all_data[t] = d

vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix[('Close', '^VIX')]
else:
    vix = vix['Close']

print('Data loaded\n')

def create_hybrid_features(df, vix, all_assets_rv):
    """Create features for hybrid model"""
    df = df.copy()
    
    # VIX features
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    
    # CAVB features
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    # Own-asset HAR terms (for Step 1)
    df['own_RV_1d'] = df['RV_1d'].shift(1)
    df['own_RV_5d'] = df['RV_5d'].shift(1)
    df['own_RV_22d'] = df['RV_22d'].shift(1)
    
    # Cross-asset RV terms (for Step 2 spillovers)
    for asset, asset_df in all_assets_rv.items():
        df[f'cross_{asset}_RV_5d'] = asset_df['RV_5d'].shift(1)
        df[f'cross_{asset}_RV_22d'] = asset_df['RV_22d'].shift(1)
    
    # Target: 5-day ahead CAVB
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    
    return df.dropna()

results = {}

for ticker in ASSETS:
    print(f'\nProcessing {ticker}...')
    
    # Prepare all assets' RV for cross-asset features
    all_assets_rv = {t: all_data[t] for t in ASSETS if t != ticker}
    
    df = create_hybrid_features(all_data[ticker], vix, all_assets_rv)
    
    # Feature sets
    own_features = ['own_RV_1d', 'own_RV_5d', 'own_RV_22d']
    vix_cavb_features = ['VIX', 'VIX_lag1', 'VIX_lag5', 'VIX_ma22',
                         'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
    cross_features = [c for c in df.columns if c.startswith('cross_')]
    
    X_own = df[own_features]
    X_vix_cavb = df[vix_cavb_features]
    X_cross = df[cross_features]
    y = df['Target']
    
    n = len(y)
    tr_end = int(n * 0.8)
    
    Xown_tr, Xown_te = X_own.iloc[:tr_end], X_own.iloc[tr_end:]
    Xvix_tr, Xvix_te = X_vix_cavb.iloc[:tr_end], X_vix_cavb.iloc[tr_end:]
    Xcross_tr, Xcross_te = X_cross.iloc[:tr_end], X_cross.iloc[tr_end:]
    ytr, yte = y.iloc[:tr_end], y.iloc[tr_end:]
    
    # ==========================================
    # Baseline: Standard ElasticNet (all features)
    # ==========================================
    X_all = pd.concat([X_own, X_vix_cavb, X_cross], axis=1)
    Xall_tr, Xall_te = X_all.iloc[:tr_end], X_all.iloc[tr_end:]
    
    scaler_all = StandardScaler()
    Xall_ts = scaler_all.fit_transform(Xall_tr)
    Xall_tes = scaler_all.transform(Xall_te)
    
    enet_baseline = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=5000)
    enet_baseline.fit(Xall_ts, ytr)
    r_baseline = r2_score(yte, enet_baseline.predict(Xall_tes))
    
    print(f'  Baseline ElasticNet: {r_baseline:.4f}')
    
    # ==========================================
    # Step 1: OLS for own-volatility (preserve persistence)
    # ==========================================
    ols_step1 = LinearRegression()
    ols_step1.fit(Xown_tr, ytr)
    
    # Get residuals from Step 1
    ytr_resid = ytr - ols_step1.predict(Xown_tr)
    yte_pred_step1 = ols_step1.predict(Xown_te)
    
    # Check persistence (sum of HAR coefficients)
    persistence = np.sum(ols_step1.coef_)
    print(f'  Step 1 OLS - Persistence: {persistence:.4f}')
    
    # ==========================================
    # Step 2: ElasticNet for cross-asset spillovers + VIX/CAVB
    # ==========================================
    X_step2 = pd.concat([X_vix_cavb, X_cross], axis=1)
    X2_tr, X2_te = X_step2.iloc[:tr_end], X_step2.iloc[tr_end:]
    
    scaler_step2 = StandardScaler()
    X2_ts = scaler_step2.fit_transform(X2_tr)
    X2_tes = scaler_step2.transform(X2_te)
    
    enet_step2 = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=5000)
    enet_step2.fit(X2_ts, ytr_resid)
    
    # Count nonzero spillover coefficients
    cross_indices = [i for i, c in enumerate(X_step2.columns) if c.startswith('cross_')]
    cross_coefs = enet_step2.coef_[cross_indices]
    nonzero_spillovers = np.sum(np.abs(cross_coefs) > 0.001)
    total_spillovers = len(cross_coefs)
    sparsity_rate = nonzero_spillovers / total_spillovers * 100
    
    print(f'  Step 2 ElasticNet - Active spillovers: {nonzero_spillovers}/{total_spillovers} ({sparsity_rate:.1f}%)')
    
    # ==========================================
    # Hybrid prediction: Step1 (OLS) + Step2 (ElasticNet)
    # ==========================================
    yte_pred_step2 = enet_step2.predict(X2_tes)
    yte_pred_hybrid = yte_pred_step1 + yte_pred_step2
    r_hybrid = r2_score(yte, yte_pred_hybrid)
    
    print(f'  Hybrid HAR-ElasticNet: {r_hybrid:.4f} ({(r_hybrid-r_baseline)/abs(r_baseline)*100:+.2f}%)')
    
    # ==========================================
    # Step 1 only (for comparison)
    # ==========================================
    r_step1_only = r2_score(yte, yte_pred_step1)
    print(f'  Step 1 Only (OLS HAR): {r_step1_only:.4f}')
    
    results[ticker] = {
        'baseline_elasticnet': r_baseline,
        'step1_ols_only': r_step1_only,
        'hybrid': r_hybrid,
        'persistence': persistence,
        'nonzero_spillovers': int(nonzero_spillovers),
        'total_spillovers': int(total_spillovers),
        'sparsity_rate': sparsity_rate,
        'improvement_vs_baseline': (r_hybrid - r_baseline) / abs(r_baseline) * 100
    }

# Summary
print('\n' + '='*70)
print('FINAL SUMMARY - Hybrid HAR-ElasticNet')
print('='*70)

models = ['baseline_elasticnet', 'step1_ols_only', 'hybrid']

print('\nModel Performance (Average R²):')
for m in models:
    vals = [results[t][m] for t in ASSETS]
    avg = np.mean(vals)
    print(f'  {m:25}: {avg:.4f}')

# Best model
hybrid_avg = np.mean([results[t]['hybrid'] for t in ASSETS])
baseline_avg = np.mean([results[t]['baseline_elasticnet'] for t in ASSETS])
improvement = (hybrid_avg - baseline_avg) / abs(baseline_avg) * 100

print(f'\nHybrid Improvement: {improvement:+.2f}%')

# Spillover statistics
avg_nonzero = np.mean([results[t]['nonzero_spillovers'] for t in ASSETS])
avg_total = np.mean([results[t]['total_spillovers'] for t in ASSETS])
avg_sparsity = np.mean([results[t]['sparsity_rate'] for t in ASSETS])

print(f'\nSpillover Network:')
print(f'  Average active spillovers: {avg_nonzero:.1f} / {avg_total:.0f}')
print(f'  Average sparsity rate: {avg_sparsity:.1f}%')

# Persistence statistics
avg_persistence = np.mean([results[t]['persistence'] for t in ASSETS])
print(f'\nAverage persistence (sum of HAR coefs): {avg_persistence:.4f}')

# Asset-specific results
print('\nAsset-Specific Results:')
for t in ASSETS:
    print(f'  {t}: Baseline {results[t]["baseline_elasticnet"]:.4f} → Hybrid {results[t]["hybrid"]:.4f} ({results[t]["improvement_vs_baseline"]:+.2f}%)')

# Save results
output = {
    'experiment': 'Exp13_Hybrid_HAR_ElasticNet',
    'methodology': 'Two-stage: OLS (own-vol) + ElasticNet (spillovers)',
    'reference': 'Gong et al. (2025)',
    'results': results,
    'summary': {
        'baseline_avg': baseline_avg,
        'hybrid_avg': hybrid_avg,
        'improvement_pct': improvement,
        'avg_persistence': avg_persistence,
        'avg_active_spillovers': avg_nonzero,
        'avg_sparsity_rate': avg_sparsity
    }
}

with open('results/hybrid_har_elasticnet.json', 'w') as f:
    json.dump(output, f, indent=2)

print('\nResults saved to results/hybrid_har_elasticnet.json')
