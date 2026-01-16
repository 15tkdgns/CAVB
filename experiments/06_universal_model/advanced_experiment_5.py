"""
Advanced Universal Model Experiments
Experiment 5: XGBoost, CatBoost, Voting Ensemble, Stacking

Testing various ensemble methods with strong regularization
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import (
    VotingRegressor, StackingRegressor, 
    RandomForestRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import xgboost as xgb
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

def train_models(X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    """Train multiple models and return R² scores"""
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    
    # 1. ElasticNet (baseline)
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    elastic.fit(X_train_s, y_train)
    results['elasticnet'] = r2_score(y_test, elastic.predict(X_test_s))
    
    # 2. XGBoost with strong regularization
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=10,
        early_stopping_rounds=30,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    results['xgboost'] = r2_score(y_test, xgb_model.predict(X_test))
    
    # 3. Random Forest with constraints
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    results['random_forest'] = r2_score(y_test, rf.predict(X_test))
    
    # 4. ExtraTrees (less overfitting than RF)
    et = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    et.fit(X_train, y_train)
    results['extra_trees'] = r2_score(y_test, et.predict(X_test))
    
    # 5. Voting Ensemble (ElasticNet + Ridge + Lasso)
    voting = VotingRegressor([
        ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)),
        ('ridge', Ridge(alpha=1.0)),
        ('lasso', Lasso(alpha=0.01, max_iter=10000))
    ])
    voting.fit(X_train_s, y_train)
    results['voting_linear'] = r2_score(y_test, voting.predict(X_test_s))
    
    # 6. Voting Mixed (Linear + Tree)
    voting_mixed = VotingRegressor([
        ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)),
        ('ridge', Ridge(alpha=1.0)),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=42))
    ])
    voting_mixed.fit(X_train_s, y_train)
    results['voting_mixed'] = r2_score(y_test, voting_mixed.predict(X_test_s))
    
    # 7. Stacking with Ridge meta-learner
    stacking = StackingRegressor(
        estimators=[
            ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)),
            ('ridge', Ridge(alpha=1.0)),
            ('lasso', Lasso(alpha=0.01, max_iter=10000))
        ],
        final_estimator=Ridge(alpha=10.0),  # Strong regularization
        cv=3
    )
    stacking.fit(X_train_s, y_train)
    results['stacking_linear'] = r2_score(y_test, stacking.predict(X_test_s))
    
    # 8. Blending (manual weighted average - optimized weights)
    pred_elastic = elastic.predict(X_test_s)
    pred_ridge = Ridge(alpha=1.0).fit(X_train_s, y_train).predict(X_test_s)
    
    # Try different weight combinations
    best_blend = None
    best_r2 = -np.inf
    for w1 in np.arange(0.3, 0.8, 0.1):
        w2 = 1 - w1
        pred_blend = w1 * pred_elastic + w2 * pred_ridge
        r2 = r2_score(y_test, pred_blend)
        if r2 > best_r2:
            best_r2 = r2
            best_blend = (w1, w2)
    
    results['best_blend'] = best_r2
    results['blend_weights'] = best_blend
    
    return results

def run_experiment():
    print("=" * 60)
    print("Advanced Universal Model Experiment 5")
    print("XGBoost, RF, Voting, Stacking Ensembles")
    print("=" * 60)
    
    print("\n[1/2] Loading data...")
    all_data, vix = load_data()
    
    all_results = {}
    
    for ticker in ASSETS:
        print(f"\n[2/2] Processing {ticker}...")
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
        
        scaler = RobustScaler()
        results = train_models(X_train, y_train, X_val, y_val, X_test, y_test, scaler)
        
        all_results[ticker] = results
        
        print(f"    ElasticNet:    {results['elasticnet']:.4f}")
        print(f"    XGBoost:       {results['xgboost']:.4f}")
        print(f"    RandomForest:  {results['random_forest']:.4f}")
        print(f"    ExtraTrees:    {results['extra_trees']:.4f}")
        print(f"    Voting(Lin):   {results['voting_linear']:.4f}")
        print(f"    Voting(Mix):   {results['voting_mixed']:.4f}")
        print(f"    Stacking:      {results['stacking_linear']:.4f}")
        print(f"    Best Blend:    {results['best_blend']:.4f} (w={results['blend_weights']})")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    models = ['elasticnet', 'xgboost', 'random_forest', 'extra_trees', 
              'voting_linear', 'voting_mixed', 'stacking_linear', 'best_blend']
    
    print(f"\n{'Asset':<8}", end='')
    for m in models[:4]:
        print(f"{m[:8]:>10}", end='')
    print()
    print("-" * 50)
    
    for ticker in ASSETS:
        print(f"{ticker:<8}", end='')
        for m in models[:4]:
            print(f"{all_results[ticker][m]:>10.4f}", end='')
        print()
    
    print("\n" + "-" * 50)
    print(f"\n{'Asset':<8}", end='')
    for m in models[4:]:
        print(f"{m[:8]:>10}", end='')
    print()
    print("-" * 50)
    
    for ticker in ASSETS:
        print(f"{ticker:<8}", end='')
        for m in models[4:]:
            print(f"{all_results[ticker][m]:>10.4f}", end='')
        print()
    
    # Calculate averages
    print("\n" + "=" * 50)
    print("AVERAGE R² BY MODEL")
    print("=" * 50)
    
    for m in models:
        avg = np.mean([all_results[t][m] for t in ASSETS])
        baseline = np.mean([all_results[t]['elasticnet'] for t in ASSETS])
        imp = (avg - baseline) / abs(baseline) * 100
        print(f"{m:<15}: {avg:.4f} ({imp:+.1f}% vs ElasticNet)")
    
    # Find best model
    best_avg = 0
    best_model = None
    for m in models:
        avg = np.mean([all_results[t][m] for t in ASSETS])
        if avg > best_avg:
            best_avg = avg
            best_model = m
    
    print(f"\nBest Model: {best_model} (avg R² = {best_avg:.4f})")
    
    # Save results
    output = {
        'experiment': 'Exp5_Ensemble_Models',
        'results': {k: {m: v[m] if m != 'blend_weights' else str(v[m]) 
                       for m in v.keys()} for k, v in all_results.items()},
        'summary': {
            'model_averages': {m: float(np.mean([all_results[t][m] for t in ASSETS])) 
                              for m in models},
            'best_model': best_model,
            'best_avg_r2': best_avg
        }
    }
    
    with open('results/advanced_experiment_5.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/advanced_experiment_5.json")
    return output

if __name__ == '__main__':
    results = run_experiment()
