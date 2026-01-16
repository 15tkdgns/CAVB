"""
Advanced Universal Model Experiments
Experiment 3: Asset-Specific Ensemble + Strong Regularization Boosting

Combines best approaches from previous experiments
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['SPY', 'GLD', 'TLT', 'EFA', 'EEM']
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'

def load_data():
    all_data = {}
    for ticker in ASSETS:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data['Return'] = data['Close'].pct_change()
        data['RV_1d'] = data['Return'].abs() * np.sqrt(252) * 100
        data['RV_5d'] = data['Return'].rolling(5).std() * np.sqrt(252) * 100
        data['RV_22d'] = data['Return'].rolling(22).std() * np.sqrt(252) * 100
        all_data[ticker] = data
    
    vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix = vix_data[('Close', '^VIX')]
    else:
        vix = vix_data['Close']
    return all_data, vix

def create_features(df, vix):
    df = df.copy()
    df['VIX'] = vix
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['VIX_ma22'] = df['VIX'].rolling(22).mean()
    df['CAVB'] = df['VIX'] - df['RV_22d']
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    df['RV_10d'] = df['Return'].rolling(10).std() * np.sqrt(252) * 100
    df['Target'] = df['VIX'] - df['RV_22d'].shift(-5)
    return df.dropna()

def train_ensemble(X_train, y_train, X_test, y_test, asset_type='default'):
    """Asset-specific ensemble based on asset characteristics"""
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: ElasticNet (always included)
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    elastic.fit(X_train_scaled, y_train)
    pred_elastic = elastic.predict(X_test_scaled)
    
    # Model 2: Ridge (stable)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    pred_ridge = ridge.predict(X_test_scaled)
    
    # Model 3: GradientBoosting with strong regularization
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=10,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gbr.fit(X_train, y_train)
    pred_gbr = gbr.predict(X_test)
    
    # Asset-specific weighting
    if asset_type == 'emerging':  # EFA, EEM - cross-asset helps
        weights = [0.3, 0.3, 0.4]  # More boosting
    elif asset_type == 'commodity':  # GLD - stable
        weights = [0.5, 0.4, 0.1]  # More linear
    else:  # SPY, TLT - balanced
        weights = [0.4, 0.4, 0.2]
    
    # Weighted ensemble
    pred_ensemble = (weights[0] * pred_elastic + 
                    weights[1] * pred_ridge + 
                    weights[2] * pred_gbr)
    
    return {
        'elastic': r2_score(y_test, pred_elastic),
        'ridge': r2_score(y_test, pred_ridge),
        'gbr': r2_score(y_test, pred_gbr),
        'ensemble': r2_score(y_test, pred_ensemble),
        'weights': weights
    }

def train_strong_reg_lgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """LightGBM with very strong regularization"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Reduced from 31
        'learning_rate': 0.02,  # Slower
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'reg_alpha': 2.0,  # Stronger L1
        'reg_lambda': 2.0,  # Stronger L2
        'min_child_samples': 30,  # More samples per leaf
        'verbose': -1
    }
    
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )
    
    pred = model.predict(X_test)
    return r2_score(y_test, pred), model.best_iteration

def run_experiment():
    print("=" * 60)
    print("Advanced Universal Model Experiment 3 & 4")
    print("Asset-Specific Ensemble + Strong Regularization Boosting")
    print("=" * 60)
    
    print("\n[1/3] Loading data...")
    all_data, vix = load_data()
    
    asset_types = {
        'SPY': 'equity', 'GLD': 'commodity', 'TLT': 'bond',
        'EFA': 'emerging', 'EEM': 'emerging'
    }
    
    results = {}
    
    for ticker in ASSETS:
        print(f"\n[2/3] Processing {ticker}...")
        df = create_features(all_data[ticker], vix)
        
        feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'RV_10d', 'VIX', 'VIX_lag1', 
                       'VIX_lag5', 'VIX_change', 'VIX_ma22', 'CAVB', 'CAVB_lag1', 
                       'CAVB_lag5', 'CAVB_ma5']
        X = df[feature_cols]
        y = df['Target']
        
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        
        # Experiment 3: Asset-Specific Ensemble
        print(f"    Training ensemble ({asset_types[ticker]})...")
        ensemble_results = train_ensemble(X_train, y_train, X_test, y_test, asset_types[ticker])
        
        # Experiment 4: Strong Regularization LightGBM
        print(f"    Training strong-reg LightGBM...")
        lgbm_r2, lgbm_iter = train_strong_reg_lgbm(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        results[ticker] = {
            'asset_type': asset_types[ticker],
            'elastic_r2': ensemble_results['elastic'],
            'ridge_r2': ensemble_results['ridge'],
            'gbr_r2': ensemble_results['gbr'],
            'ensemble_r2': ensemble_results['ensemble'],
            'ensemble_weights': ensemble_results['weights'],
            'strong_reg_lgbm_r2': lgbm_r2,
            'lgbm_best_iter': lgbm_iter
        }
        
        print(f"    ElasticNet: {ensemble_results['elastic']:.4f}")
        print(f"    Ensemble:   {ensemble_results['ensemble']:.4f}")
        print(f"    StrongLGBM: {lgbm_r2:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    avg_elastic = np.mean([r['elastic_r2'] for r in results.values()])
    avg_ensemble = np.mean([r['ensemble_r2'] for r in results.values()])
    avg_lgbm = np.mean([r['strong_reg_lgbm_r2'] for r in results.values()])
    
    print(f"\n{'Asset':<8} {'Elastic':>10} {'Ensemble':>10} {'Ens Imp':>10} {'StrongLGBM':>12} {'LGBM Imp':>10}")
    print("-" * 62)
    
    for ticker in ASSETS:
        r = results[ticker]
        ens_imp = (r['ensemble_r2'] - r['elastic_r2']) / abs(r['elastic_r2']) * 100
        lgbm_imp = (r['strong_reg_lgbm_r2'] - r['elastic_r2']) / abs(r['elastic_r2']) * 100
        print(f"{ticker:<8} {r['elastic_r2']:>10.4f} {r['ensemble_r2']:>10.4f} {ens_imp:>+9.1f}% {r['strong_reg_lgbm_r2']:>12.4f} {lgbm_imp:>+9.1f}%")
    
    print("-" * 62)
    ens_imp = (avg_ensemble - avg_elastic) / abs(avg_elastic) * 100
    lgbm_imp = (avg_lgbm - avg_elastic) / abs(avg_elastic) * 100
    print(f"{'Average':<8} {avg_elastic:>10.4f} {avg_ensemble:>10.4f} {ens_imp:>+9.1f}% {avg_lgbm:>12.4f} {lgbm_imp:>+9.1f}%")
    
    # Save results
    output = {
        'experiment': 'Exp3_Exp4_Ensemble_StrongLGBM',
        'results': results,
        'summary': {
            'avg_elastic_r2': avg_elastic,
            'avg_ensemble_r2': avg_ensemble,
            'avg_strong_lgbm_r2': avg_lgbm,
            'ensemble_improvement_pct': ens_imp,
            'lgbm_improvement_pct': lgbm_imp
        }
    }
    
    with open('results/advanced_experiment_3_4.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/advanced_experiment_3_4.json")
    return output

if __name__ == '__main__':
    results = run_experiment()
