"""
Comprehensive 22-Day Universal Volatility Prediction Model
Maximum generalization with 40+ features and 10+ model types
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import HuberRegressor, ElasticNet, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']

print('='*70)
print('Comprehensive 22-Day Universal Model')
print('40+ Features, 10+ Model Types')
print('='*70)

# Load all data
print('\n[1/4] Loading data...')
all_data = {}
for t in ASSETS:
    d = yf.download(t, start='2010-01-01', end='2025-01-01', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d['Ret'] = d['Close'].pct_change()
    d['RV_1d'] = d['Ret'].abs() * np.sqrt(252) * 100
    d['RV_5d'] = d['Ret'].rolling(5).std() * np.sqrt(252) * 100
    d['RV_10d'] = d['Ret'].rolling(10).std() * np.sqrt(252) * 100
    d['RV_22d'] = d['Ret'].rolling(22).std() * np.sqrt(252) * 100
    d['RV_44d'] = d['Ret'].rolling(44).std() * np.sqrt(252) * 100
    d['RV_66d'] = d['Ret'].rolling(66).std() * np.sqrt(252) * 100
    all_data[t] = d

vix = yf.download('^VIX', start='2010-01-01', end='2025-01-01', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix[('Close', '^VIX')]
else:
    vix = vix['Close']

# Additional macro data
try:
    us10y_data = yf.download('^TNX', start='2010-01-01', end='2025-01-01', progress=False)
    if isinstance(us10y_data.columns, pd.MultiIndex):
        us10y = us10y_data[('Close', '^TNX')]
    else:
        us10y = us10y_data['Close']
except:
    us10y = None

print('Data loaded')

def create_comprehensive_features_22d(df, vix, us10y=None):
    """Create 40+ features optimized for 22-day prediction"""
    df = df.copy()
    
    # ==========================================
    # VIX Features (15)
    # ==========================================
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_lag10'] = df['VIX'].shift(10)
    df['VIX_lag22'] = df['VIX'].shift(22)
    df['VIX_lag44'] = df['VIX'].shift(44)
    df['VIX_change_1d'] = df['VIX'].pct_change(1)
    df['VIX_change_5d'] = df['VIX'].pct_change(5)
    df['VIX_change_22d'] = df['VIX'].pct_change(22)
    df['VIX_ma5'] = df['VIX'].rolling(5).mean()
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['VIX_ma44'] = df['VIX'].rolling(44).mean()
    df['VIX_ma66'] = df['VIX'].rolling(66).mean()
    df['VIX_std22'] = df['VIX'].rolling(22).std()
    df['VIX_zscore'] = (df['VIX'] - df['VIX_ma22']) / (df['VIX_std22'] + 0.01)
    
    # ==========================================
    # CAVB Features (12)
    # ==========================================
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_lag10'] = df['CAVB'].shift(10)
    df['CAVB_lag22'] = df['CAVB'].shift(22)
    df['CAVB_lag44'] = df['CAVB'].shift(44)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    df['CAVB_ma22'] = df['CAVB'].rolling(22).mean()
    df['CAVB_ma44'] = df['CAVB'].rolling(44).mean()
    df['CAVB_ma66'] = df['CAVB'].rolling(66).mean()
    df['CAVB_std22'] = df['CAVB'].rolling(22).std()
    df['CAVB_zscore'] = (df['CAVB'] - df['CAVB_ma22']) / (df['CAVB_std22'] + 0.01)
    
    # ==========================================
    # RV Features (10)
    # ==========================================
    df['RV_ratio_5_22'] = df['RV_5d'] / (df['RV_22d'] + 0.01)
    df['RV_ratio_22_44'] = df['RV_22d'] / (df['RV_44d'] + 0.01)
    df['RV_ratio_22_66'] = df['RV_22d'] / (df['RV_66d'] + 0.01)
    df['RV_momentum_5'] = df['RV_22d'].pct_change(5)
    df['RV_momentum_22'] = df['RV_22d'].pct_change(22)
    df['RV_ma22'] = df['RV_22d'].rolling(22).mean()
    df['RV_ma44'] = df['RV_22d'].rolling(44).mean()
    df['RV_std22'] = df['RV_22d'].rolling(22).std()
    df['RV_high_44'] = df['RV_22d'].rolling(44).max()
    df['RV_low_44'] = df['RV_22d'].rolling(44).min()
    
    # ==========================================
    # VRP Decomposition (4)
    # ==========================================
    df['VRP_persistent'] = df['CAVB'].rolling(66).mean()
    df['VRP_transitory'] = df['CAVB'] - df['VRP_persistent']
    df['VRP_ratio'] = df['VRP_transitory'] / (df['VRP_persistent'].abs() + 0.01)
    df['VRP_momentum'] = df['CAVB'].pct_change(22)
    
    # ==========================================
    # Cross-term Features (5)
    # ==========================================
    df['VIX_RV_ratio'] = df['VIX'] / (df['RV_22d'] + 0.01)
    df['VIX_RV_product'] = df['VIX'] * df['RV_22d'] / 1000
    df['CAVB_VIX_ratio'] = df['CAVB'] / (df['VIX'] + 0.01)
    df['RV_trend'] = df['RV_5d'] - df['RV_66d']
    df['VIX_trend'] = df['VIX'] - df['VIX_ma66']
    
    # ==========================================
    # Macro Features (3) - if available
    # ==========================================
    if us10y is not None and len(us10y) > 0:
        df['US10Y'] = us10y
        df['US10Y_change'] = df['US10Y'].pct_change(22)
        df['VIX_yield_ratio'] = df['VIX'] / (df['US10Y'] + 1)
    
    # ==========================================
    # Target: 22-day ahead
    # ==========================================
    df['Target_22d'] = df['VIX'] - df['RV_22d'].shift(-22)
    
    return df.dropna()

print('[2/4] Creating features...')

results = {}

for ticker in ASSETS:
    print(f'\n[3/4] Processing {ticker}...')
    
    df = create_comprehensive_features_22d(all_data[ticker], vix, us10y)
    
    # Get feature columns (exclude non-features)
    exclude = ['Target_22d', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ret']
    fcols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    fcols = [c for c in fcols if df[c].notna().sum() > len(df) * 0.9]
    
    X, y = df[fcols], df['Target_22d']
    n = len(X)
    tr_end, val_end = int(n*0.6), int(n*0.8)
    
    Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
    Xval, yval = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]
    Xte, yte = X.iloc[val_end:], y.iloc[val_end:]
    
    Xtrval = pd.concat([Xtr, Xval])
    ytrval = pd.concat([ytr, yval])
    
    scaler = RobustScaler()
    Xts = scaler.fit_transform(Xtr)
    Xvs = scaler.transform(Xval)
    Xtes = scaler.transform(Xte)
    
    scaler_full = RobustScaler()
    Xtvs = scaler_full.fit_transform(Xtrval)
    Xtes_full = scaler_full.transform(Xte)
    
    r = {'n_features': len(fcols)}
    print(f'  Features: {len(fcols)}')
    
    # ==========================================
    # 1. Linear Models
    # ==========================================
    print('  Linear models...')
    
    # Huber
    hub = HuberRegressor(epsilon=1.35, max_iter=1000)
    hub.fit(Xtvs, ytrval)
    r['huber'] = r2_score(yte, hub.predict(Xtes_full))
    
    # Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(Xtvs, ytrval)
    r['ridge'] = r2_score(yte, ridge.predict(Xtes_full))
    
    # Lasso
    lasso = Lasso(alpha=0.1, max_iter=5000)
    lasso.fit(Xtvs, ytrval)
    r['lasso'] = r2_score(yte, lasso.predict(Xtes_full))
    
    # ElasticNet
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
    enet.fit(Xtvs, ytrval)
    r['elasticnet'] = r2_score(yte, enet.predict(Xtes_full))
    
    # Bayesian Ridge
    bayesian = BayesianRidge()
    bayesian.fit(Xtvs, ytrval)
    r['bayesian_ridge'] = r2_score(yte, bayesian.predict(Xtes_full))
    
    print(f'    Huber: {r["huber"]:.4f}, Ridge: {r["ridge"]:.4f}, Lasso: {r["lasso"]:.4f}')
    
    # ==========================================
    # 2. SVR Models
    # ==========================================
    print('  SVR models...')
    
    svr_lin = SVR(kernel='linear', C=1.0, epsilon=0.1)
    svr_lin.fit(Xtvs, ytrval)
    r['svr_linear'] = r2_score(yte, svr_lin.predict(Xtes_full))
    
    svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    svr_rbf.fit(Xtvs, ytrval)
    r['svr_rbf'] = r2_score(yte, svr_rbf.predict(Xtes_full))
    
    print(f'    SVR-Linear: {r["svr_linear"]:.4f}, SVR-RBF: {r["svr_rbf"]:.4f}')
    
    # ==========================================
    # 3. Tree-based Models (lightweight)
    # ==========================================
    print('  Tree models...')
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=20, random_state=42)
    rf.fit(Xtr, ytr)
    r['random_forest'] = r2_score(yte, rf.predict(Xte))
    
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(Xtr, ytr)
    r['gradient_boosting'] = r2_score(yte, gb.predict(Xte))
    
    et = ExtraTreesRegressor(n_estimators=50, max_depth=5, min_samples_leaf=20, random_state=42)
    et.fit(Xtr, ytr)
    r['extra_trees'] = r2_score(yte, et.predict(Xte))
    
    print(f'    RF: {r["random_forest"]:.4f}, GB: {r["gradient_boosting"]:.4f}, ET: {r["extra_trees"]:.4f}')
    
    # ==========================================
    # 4. KNN
    # ==========================================
    knn = KNeighborsRegressor(n_neighbors=20, weights='distance')
    knn.fit(Xtvs, ytrval)
    r['knn'] = r2_score(yte, knn.predict(Xtes_full))
    print(f'    KNN: {r["knn"]:.4f}')
    
    # ==========================================
    # 5. MLP
    # ==========================================
    print('  MLP models...')
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, alpha=0.1, random_state=42)
    mlp.fit(Xtvs, ytrval)
    r['mlp'] = r2_score(yte, mlp.predict(Xtes_full))
    print(f'    MLP: {r["mlp"]:.4f}')
    
    # ==========================================
    # 6. Optimized Huber (Optuna)
    # ==========================================
    print('  Optuna Huber (30 trials)...')
    def obj_huber(trial):
        epsilon = trial.suggest_float('epsilon', 1.0, 3.0)
        alpha = trial.suggest_float('alpha', 1e-6, 10.0, log=True)
        m = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)
        m.fit(Xts, ytr)
        return r2_score(yval, m.predict(Xvs))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(obj_huber, n_trials=30, show_progress_bar=False)
    
    hub_opt = HuberRegressor(**study.best_params, max_iter=1000)
    hub_opt.fit(Xtvs, ytrval)
    r['huber_optuna'] = r2_score(yte, hub_opt.predict(Xtes_full))
    r['huber_optuna_params'] = study.best_params
    print(f'    Huber-Optuna: {r["huber_optuna"]:.4f}')
    
    # ==========================================
    # 7. Voting Ensemble
    # ==========================================
    print('  Ensembles...')
    voting = VotingRegressor([
        ('huber', HuberRegressor(epsilon=1.35, max_iter=1000)),
        ('ridge', Ridge(alpha=10.0)),
        ('svr', SVR(kernel='linear', C=1.0))
    ])
    voting.fit(Xtvs, ytrval)
    r['voting_ensemble'] = r2_score(yte, voting.predict(Xtes_full))
    
    # ==========================================
    # 8. Stacking Ensemble
    # ==========================================
    stacking = StackingRegressor(
        estimators=[
            ('huber', HuberRegressor(epsilon=1.35, max_iter=1000)),
            ('ridge', Ridge(alpha=10.0)),
            ('rf', RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42))
        ],
        final_estimator=Ridge(alpha=1.0)
    )
    stacking.fit(Xtr, ytr)
    r['stacking_ensemble'] = r2_score(yte, stacking.predict(Xte))
    
    print(f'    Voting: {r["voting_ensemble"]:.4f}, Stacking: {r["stacking_ensemble"]:.4f}')
    
    # Find best
    model_scores = {k: v for k, v in r.items() if isinstance(v, (int, float)) and k != 'n_features'}
    r['best_model'] = max(model_scores, key=model_scores.get)
    r['best_r2'] = model_scores[r['best_model']]
    
    print(f'  **Best**: {r["best_model"]} (R² = {r["best_r2"]:.4f})')
    
    results[ticker] = r

# Summary
print('\n' + '='*70)
print('[4/4] FINAL SUMMARY - Comprehensive 22-Day Universal Model')
print('='*70)

models = ['huber', 'ridge', 'lasso', 'elasticnet', 'bayesian_ridge',
          'svr_linear', 'svr_rbf', 'random_forest', 'gradient_boosting', 
          'extra_trees', 'knn', 'mlp', 'huber_optuna', 
          'voting_ensemble', 'stacking_ensemble']

print('\nModel Performance (Average R²):')
model_avgs = {}
for m in models:
    vals = [results[t].get(m) for t in ASSETS if results[t].get(m) is not None]
    if vals:
        avg = np.mean(vals)
        model_avgs[m] = avg
        print(f'  {m:20}: {avg:.4f}')

# Rank models
sorted_models = sorted(model_avgs.items(), key=lambda x: x[1], reverse=True)
print('\nTop 5 Models:')
for i, (m, v) in enumerate(sorted_models[:5], 1):
    print(f'  {i}. {m}: {v:.4f}')

best_model = sorted_models[0][0]
best_avg = sorted_models[0][1]

print(f'\n**Best Universal 22-Day Model**: {best_model} (avg R² = {best_avg:.4f})')

print('\nAsset-Specific Best:')
for t in ASSETS:
    print(f'  {t}: {results[t]["best_model"]:20} (R² = {results[t]["best_r2"]:.4f})')

# Save
def clean_for_json(r):
    cleaned = {}
    for k, v in r.items():
        if isinstance(v, dict):
            cleaned[k] = {str(kk): float(vv) if isinstance(vv, (int, float, np.floating)) else str(vv) 
                         for kk, vv in v.items()}
        elif isinstance(v, (list, np.ndarray)):
            cleaned[k] = [float(x) if isinstance(x, (int, float, np.floating)) else str(x) for x in v]
        elif isinstance(v, str):
            cleaned[k] = v
        elif v is not None:
            cleaned[k] = float(v) if isinstance(v, (int, float, np.floating)) else str(v)
        else:
            cleaned[k] = None
    return cleaned

output = {
    'experiment': 'Comprehensive_22Day_Universal_Model',
    'horizon': '22 days',
    'n_features': len(fcols),
    'n_models': len(models),
    'features': fcols,
    'results': {k: clean_for_json(v) for k, v in results.items()},
    'summary': {
        'best_model': best_model,
        'best_avg_r2': best_avg,
        'model_ranking': {m: float(v) for m, v in sorted_models}
    }
}

with open('results/comprehensive_22d_model.json', 'w') as f:
    json.dump(output, f, indent=2)

print('\nResults saved to results/comprehensive_22d_model.json')
