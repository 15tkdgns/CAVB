"""
Experiment 14: Prediction Uncertainty Quantification
Based on: Allen et al. (2024) - Confident Risk Premiums using ML Uncertainties

Methodology:
- Bootstrap resampling to generate prediction confidence intervals
- Evaluate correlation between uncertainty and ex-post prediction error
- Regime-dependent analysis (High/Low VIX)
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
N_BOOTSTRAP = 100  # Bootstrap samples

print('='*70)
print('Experiment 14: Prediction Uncertainty Quantification')
print('Bootstrap CI + Uncertainty-Error Correlation')
print('='*70)

# Load data
all_data = {}
for t in ASSETS:
    d = yf.download(t, start='2015-01-01', end='2025-01-01', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d['Ret'] = d['Close'].pct_change()
    d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
    d['RV_10d'] = d['Ret'].rolling(10).std() * np.sqrt(252) * 100
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
    print(f'\nProcessing {ticker}...')
    
    df = all_data[ticker].copy()
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    df = df.dropna()
    
    fcols = ['VIX', 'VIX_lag1', 'VIX_lag5', 'VIX_ma22',
             'CAVB', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
             'RV_5d', 'RV_10d', 'RV_22d']
    
    X, y = df[fcols], df['Target']
    n = len(X)
    tr_end = int(n * 0.8)
    
    Xtr, Xte = X.iloc[:tr_end], X.iloc[tr_end:]
    ytr, yte = y.iloc[:tr_end], y.iloc[tr_end:]
    
    scaler = StandardScaler()
    Xts = scaler.fit_transform(Xtr)
    Xtes = scaler.transform(Xte)
    
    # Point prediction (baseline)
    model = HuberRegressor(epsilon=1.35, max_iter=1000)
    model.fit(Xts, ytr)
    ypred = model.predict(Xtes)
    r2_baseline = r2_score(yte, ypred)
    
    print(f'  Baseline R²: {r2_baseline:.4f}')
    
    # ==========================================
    # Bootstrap for Prediction Uncertainty
    # ==========================================
    print(f'  Running {N_BOOTSTRAP} bootstrap samples...')
    
    bootstrap_preds = np.zeros((N_BOOTSTRAP, len(yte)))
    
    for b in range(N_BOOTSTRAP):
        # Resample training data with replacement
        indices = np.random.choice(len(Xtr), size=len(Xtr), replace=True)
        Xb = Xts[indices]
        yb = ytr.iloc[indices].values
        
        # Fit model on bootstrap sample
        model_b = HuberRegressor(epsilon=1.35, max_iter=1000)
        model_b.fit(Xb, yb)
        
        # Predict on test set
        bootstrap_preds[b] = model_b.predict(Xtes)
    
    # Calculate prediction statistics
    pred_mean = np.mean(bootstrap_preds, axis=0)
    pred_std = np.std(bootstrap_preds, axis=0)
    pred_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
    pred_upper = np.percentile(bootstrap_preds, 97.5, axis=0)
    
    # Prediction interval width
    ci_width = pred_upper - pred_lower
    avg_ci_width = np.mean(ci_width)
    
    # ==========================================
    # Uncertainty-Error Correlation Analysis
    # ==========================================
    actual_errors = np.abs(yte.values - pred_mean)
    
    # Correlation between uncertainty (std) and actual error
    correlation = np.corrcoef(pred_std, actual_errors)[0, 1]
    
    print(f'  Average CI width: {avg_ci_width:.3f}')
    print(f'  Uncertainty-Error correlation: {correlation:.3f}')
    
    # Coverage rate (how often true value is within CI)
    coverage = np.mean((yte.values >= pred_lower) & (yte.values <= pred_upper))
    print(f'  95% CI coverage: {coverage*100:.1f}%')
    
    # ==========================================
    # Regime-Dependent Analysis (High/Low VIX)
    # ==========================================
    vix_test = df['VIX'].iloc[tr_end:].values
    vix_median = np.median(vix_test)
    
    high_vix = vix_test > vix_median
    low_vix = vix_test <= vix_median
    
    r2_high = r2_score(yte[high_vix], pred_mean[high_vix])
    r2_low = r2_score(yte[low_vix], pred_mean[low_vix])
    
    ci_high = np.mean(ci_width[high_vix])
    ci_low = np.mean(ci_width[low_vix])
    
    print(f'  High VIX regime: R² = {r2_high:.4f}, CI = {ci_high:.3f}')
    print(f'  Low VIX regime:  R² = {r2_low:.4f}, CI = {ci_low:.3f}')
    
    results[ticker] = {
        'r2_baseline': r2_baseline,
        'avg_ci_width': avg_ci_width,
        'uncertainty_error_corr': correlation,
        'coverage_95': coverage,
        'r2_high_vix': r2_high,
        'r2_low_vix': r2_low,
        'ci_high_vix': ci_high,
        'ci_low_vix': ci_low
    }

# Summary
print('\n' + '='*70)
print('FINAL SUMMARY - Prediction Uncertainty')
print('='*70)

avg_r2 = np.mean([results[t]['r2_baseline'] for t in ASSETS])
avg_ci = np.mean([results[t]['avg_ci_width'] for t in ASSETS])
avg_corr = np.mean([results[t]['uncertainty_error_corr'] for t in ASSETS])
avg_coverage = np.mean([results[t]['coverage_95'] for t in ASSETS])

print(f'\nAverage R²: {avg_r2:.4f}')
print(f'Average CI width: {avg_ci:.3f}')
print(f'Average Uncertainty-Error correlation: {avg_corr:.3f}')
print(f'Average 95% CI coverage: {avg_coverage*100:.1f}%')

# Regime comparison
avg_r2_high = np.mean([results[t]['r2_high_vix'] for t in ASSETS])
avg_r2_low = np.mean([results[t]['r2_low_vix'] for t in ASSETS])
avg_ci_high = np.mean([results[t]['ci_high_vix'] for t in ASSETS])
avg_ci_low = np.mean([results[t]['ci_low_vix'] for t in ASSETS])

print(f'\nRegime Analysis:')
print(f'  High VIX: R² = {avg_r2_high:.4f}, CI = {avg_ci_high:.3f}')
print(f'  Low VIX:  R² = {avg_r2_low:.4f}, CI = {avg_ci_low:.3f}')

# Key insights
print(f'\nKey Insights:')
if avg_corr > 0.5:
    print(f'  ✓ High correlation ({avg_corr:.2f}) = Uncertainty is predictive of error')
else:
    print(f'  ✗ Low correlation ({avg_corr:.2f}) = Uncertainty not reliable')

if avg_coverage > 0.93 and avg_coverage < 0.97:
    print(f'  ✓ Good calibration ({avg_coverage*100:.1f}% coverage for 95% CI)')
else:
    print(f'  ⚠ Mis-calibrated ({avg_coverage*100:.1f}% coverage for 95% CI)')

if avg_ci_high > avg_ci_low * 1.2:
    print(f'  ✓ Wider CI in high VIX (+{(avg_ci_high/avg_ci_low-1)*100:.1f}%) = Regime-aware')
else:
    print(f'  ~ Similar CI across regimes')

# Save results
output = {
    'experiment': 'Exp14_Prediction_Uncertainty',
    'methodology': 'Bootstrap resampling (100 samples)',
    'reference': 'Allen et al. (2024)',
    'n_bootstrap': N_BOOTSTRAP,
    'results': results,
    'summary': {
        'avg_r2': avg_r2,
        'avg_ci_width': avg_ci,
        'avg_uncertainty_error_corr': avg_corr,
        'avg_coverage': avg_coverage,
        'regime_analysis': {
            'high_vix_r2': avg_r2_high,
            'low_vix_r2': avg_r2_low,
            'high_vix_ci': avg_ci_high,
            'low_vix_ci': avg_ci_low
        }
    }
}

with open('results/prediction_uncertainty.json', 'w') as f:
    json.dump(output, f, indent=2)

print('\nResults saved to results/prediction_uncertainty.json')
