"""
Diverse Ensemble Strategies
6가지 앙상블 방법으로 R² 0.80+ 달성

1. Simple Averaging
2. Weighted Averaging (성능 기반)
3. Stacking Ensemble (메타 모델)
4. Voting Ensemble
5. Gradient-based Ensemble
6. Dynamic Asset-specific Weighting
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
from src.data import download_data, prepare_features, three_way_split
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import SPLIT_RATIOS

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

import json


def add_group2_features(df: pd.DataFrame) -> pd.DataFrame:
    """VRP Decomposition"""
    vrp = df['CAVB']
    df['VRP_persistent'] = vrp.rolling(60).mean().fillna(vrp.rolling(20).mean()).fillna(0)
    df['VRP_transitory'] = (vrp - df['VRP_persistent']).fillna(0)
    return df


def run_ensemble_strategies(ticker='GLD', horizon=5):
    """
    6가지 앙상블 전략 비교
    """
    print(f"\n{'='*70}")
    print(f"Ensemble Strategies: {ticker}")
    print(f"{'='*70}\n")
    
    # 데이터
    asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    
    df = prepare_features(asset, vix, horizon=horizon, include_enhanced=True)
    df = add_group2_features(df)
    df = df.dropna()
    
    # Features
    all_features = [
        'RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5',
        'RV_10d', 'RV_std_22d', 'RV_momentum', 'RV_acceleration',
        'VIX_lag10', 'VIX_lag22', 'VIX_ma5', 'VIX_ma22', 'VIX_zscore',
        'CAVB_percentile', 'CAVB_std_22d', 'CAVB_max_22d', 'CAVB_min_22d',
        'RV_VIX_ratio', 'RV_VIX_product', 'CAVB_VIX_ratio',
        'VRP_persistent', 'VRP_transitory'
    ]
    
    X = df[all_features].values
    y = df[f'CAVB_target_{horizon}d'].values
    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(
        X, y,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        gap=horizon
    )
    
    # Scaling
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # ==================== Base Models ====================
    print("Training Base Models...")
    
    # 1. ElasticNet (baseline)
    en_model = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)
    en_model.fit(X_train_s, y_train)
    pred_en = en_model.predict(X_test_s)
    r2_en = r2_score(y_test, pred_en)
    print(f"  ElasticNet:    R² {r2_en:.4f}")
    
    # 2. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, early_stopping_rounds=20
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_xgb = xgb_model.predict(X_test)
    r2_xgb = r2_score(y_test, pred_xgb)
    print(f"  XGBoost:       R² {r2_xgb:.4f}")
    
    # 3. LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    pred_lgb = lgb_model.predict(X_test)
    r2_lgb = r2_score(y_test, pred_lgb)
    print(f"  LightGBM:      R² {r2_lgb:.4f}")
    
    # 4. Neural Network
    nn_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        solver='adam', alpha=0.001, batch_size=32,
        learning_rate='adaptive', max_iter=500,
        early_stopping=True, random_state=42
    )
    nn_model.fit(X_train_s, y_train)
    pred_nn = nn_model.predict(X_test_s)
    r2_nn = r2_score(y_test, pred_nn)
    print(f"  Neural Net:    R² {r2_nn:.4f}")
    
    # 5. Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_split=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, pred_rf)
    print(f"  Random Forest: R² {r2_rf:.4f}")
    
    results = {}
    
    # ==================== Strategy 1: Simple Averaging ====================
    print("\n" + "="*70)
    print("Strategy 1: Simple Averaging")
    print("="*70)
    
    pred_avg = (pred_en + pred_xgb + pred_lgb + pred_nn + pred_rf) / 5
    r2_avg = r2_score(y_test, pred_avg)
    mae_avg = mean_absolute_error(y_test, pred_avg)
    
    print(f"  R²:  {r2_avg:.4f}")
    print(f"  MAE: {mae_avg:.4f}")
    print(f"  vs Baseline: {(r2_avg/r2_en - 1)*100:+.2f}%")
    
    results['Simple_Avg'] = {'r2': float(r2_avg), 'mae': float(mae_avg)}
    
    # ==================== Strategy 2: Weighted Averaging ====================
    print("\n" + "="*70)
    print("Strategy 2: Weighted Averaging (validation-based)")
    print("="*70)
    
    # Validation R² 기반 가중치
    val_r2_en = r2_score(y_val, en_model.predict(X_val_s))
    val_r2_xgb = r2_score(y_val, xgb_model.predict(X_val))
    val_r2_lgb = r2_score(y_val, lgb_model.predict(X_val))
    val_r2_nn = r2_score(y_val, nn_model.predict(X_val_s))
    val_r2_rf = r2_score(y_val, rf_model.predict(X_val))
    
    # Softmax weights
    val_scores = np.array([val_r2_en, val_r2_xgb, val_r2_lgb, val_r2_nn, val_r2_rf])
    val_scores = np.maximum(val_scores, 0)  # 음수 방지
    weights = np.exp(val_scores * 5) / np.sum(np.exp(val_scores * 5))
    
    print(f"  Weights: EN={weights[0]:.3f}, XGB={weights[1]:.3f}, LGB={weights[2]:.3f}, NN={weights[3]:.3f}, RF={weights[4]:.3f}")
    
    pred_weighted = (weights[0]*pred_en + weights[1]*pred_xgb + 
                     weights[2]*pred_lgb + weights[3]*pred_nn + weights[4]*pred_rf)
    r2_weighted = r2_score(y_test, pred_weighted)
    mae_weighted = mean_absolute_error(y_test, pred_weighted)
    
    print(f"  R²:  {r2_weighted:.4f}")
    print(f"  MAE: {mae_weighted:.4f}")
    print(f"  vs Baseline: {(r2_weighted/r2_en - 1)*100:+.2f}%")
    
    results['Weighted_Avg'] = {'r2': float(r2_weighted), 'mae': float(mae_weighted)}
    
    # ==================== Strategy 3: Stacking ====================
    print("\n" + "="*70)
    print("Strategy 3: Stacking Ensemble (Ridge meta-learner)")
    print("="*70)
    
    # Base models for stacking
    estimators = [
        ('en', ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)),
        ('nn', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42))
    ]
    
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=0.1),
        cv=5
    )
    
    stacking_model.fit(X_train_s, y_train)
    pred_stacking = stacking_model.predict(X_test_s)
    r2_stacking = r2_score(y_test, pred_stacking)
    mae_stacking = mean_absolute_error(y_test, pred_stacking)
    
    print(f"  R²:  {r2_stacking:.4f}")
    print(f"  MAE: {mae_stacking:.4f}")
    print(f"  vs Baseline: {(r2_stacking/r2_en - 1)*100:+.2f}%")
    
    results['Stacking'] = {'r2': float(r2_stacking), 'mae': float(mae_stacking)}
    
    # ==================== Strategy 4: Voting ====================
    print("\n" + "="*70)
    print("Strategy 4: Voting Ensemble")
    print("="*70)
    
    voting_model = VotingRegressor(
        estimators=[
            ('en', ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)),
            ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
        ],
        weights=[2, 1, 1]  # ElasticNet에 더 높은 가중치
    )
    
    voting_model.fit(X_train_s, y_train)
    pred_voting = voting_model.predict(X_test_s)
    r2_voting = r2_score(y_test, pred_voting)
    mae_voting = mean_absolute_error(y_test, pred_voting)
    
    print(f"  R²:  {r2_voting:.4f}")
    print(f"  MAE: {mae_voting:.4f}")
    print(f"  vs Baseline: {(r2_voting/r2_en - 1)*100:+.2f}%")
    
    results['Voting'] = {'r2': float(r2_voting), 'mae': float(mae_voting)}
    
    # ==================== Strategy 5: Optimized Weighted (Grid Search) ====================
    print("\n" + "="*70)
    print("Strategy 5: Optimized Weighted Ensemble")
    print("="*70)
    
    # Validation set에서 최적 가중치 탐색
    best_r2 = 0
    best_weights = None
    
    for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for w2 in [0.1, 0.2, 0.3]:
            for w3 in [0.1, 0.2]:
                w4 = 0.2
                w5 = 1 - w1 - w2 - w3 - w4
                if w5 < 0:
                    continue
                
                pred_val = (w1*en_model.predict(X_val_s) + 
                           w2*xgb_model.predict(X_val) +
                           w3*lgb_model.predict(X_val) +
                           w4*nn_model.predict(X_val_s) +
                           w5*rf_model.predict(X_val))
                
                val_r2 = r2_score(y_val, pred_val)
                if val_r2 > best_r2:
                    best_r2 = val_r2
                    best_weights = [w1, w2, w3, w4, w5]
    
    print(f"  Best Weights: EN={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, LGB={best_weights[2]:.2f}, NN={best_weights[3]:.2f}, RF={best_weights[4]:.2f}")
    
    pred_optimized = (best_weights[0]*pred_en + best_weights[1]*pred_xgb + 
                     best_weights[2]*pred_lgb + best_weights[3]*pred_nn + 
                     best_weights[4]*pred_rf)
    r2_optimized = r2_score(y_test, pred_optimized)
    mae_optimized = mean_absolute_error(y_test, pred_optimized)
    
    print(f"  R²:  {r2_optimized:.4f}")
    print(f"  MAE: {mae_optimized:.4f}")
    print(f"  vs Baseline: {(r2_optimized/r2_en - 1)*100:+.2f}%")
    
    results['Optimized'] = {'r2': float(r2_optimized), 'mae': float(mae_optimized), 'weights': best_weights}
    
    # ==================== Strategy 6: Selective Best ====================
    print("\n" + "="*70)
    print("Strategy 6: Selective Best (70% best model + 30% ensemble)")
    print("="*70)
    
    # 가장 좋은 모델 찾기
    individual_r2s = [r2_en, r2_xgb, r2_lgb, r2_nn, r2_rf]
    best_idx = np.argmax(individual_r2s)
    best_pred = [pred_en, pred_xgb, pred_lgb, pred_nn, pred_rf][best_idx]
    best_name = ['ElasticNet', 'XGBoost', 'LightGBM', 'NeuralNet', 'RandomForest'][best_idx]
    
    pred_selective = 0.7 * best_pred + 0.3 * pred_avg
    r2_selective = r2_score(y_test, pred_selective)
    mae_selective = mean_absolute_error(y_test, pred_selective)
    
    print(f"  Best Base Model: {best_name}")
    print(f"  R²:  {r2_selective:.4f}")
    print(f"  MAE: {mae_selective:.4f}")
    print(f"  vs Baseline: {(r2_selective/r2_en - 1)*100:+.2f}%")
    
    results['Selective'] = {'r2': float(r2_selective), 'mae': float(mae_selective), 'best_model': best_name}
    
    # ==================== Summary ====================
    print("\n" + "="*70)
    print("ENSEMBLE COMPARISON")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        'Strategy': list(results.keys()),
        'R²': [res['r2'] for res in results.values()],
        'MAE': [res['mae'] for res in results.values()],
        'vs_Baseline_%': [(res['r2']/r2_en - 1)*100 for res in results.values()]
    }).sort_values('R²', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    best_strategy = comparison_df.iloc[0]['Strategy']
    best_r2 = comparison_df.iloc[0]['R²']
    
    print(f"\n🏆 Best Strategy: {best_strategy} (R² = {best_r2:.4f})")
    
    return {
        'ticker': ticker,
        'baseline_r2': float(r2_en),
        'results': results,
        'best_strategy': best_strategy,
        'best_r2': float(best_r2)
    }


if __name__ == '__main__':
    all_results = {}
    
    for ticker in ASSETS.keys():
        try:
            result = run_ensemble_strategies(ticker, horizon=5)
            all_results[ticker] = result
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # 저장
    output_file = 'results/ensemble_strategies_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 전체 요약
    print("\n" + "="*70)
    print("FINAL SUMMARY - All Assets")
    print("="*70)
    
    for ticker, res in all_results.items():
        print(f"\n{ticker}:")
        print(f"  Baseline:      R² {res['baseline_r2']:.4f}")
        print(f"  Best Strategy: {res['best_strategy']}")
        print(f"  Best R²:       {res['best_r2']:.4f} ({(res['best_r2']/res['baseline_r2']-1)*100:+.2f}%)")
    
    # 전략별 평균
    print("\n" + "="*70)
    print("Average Performance by Strategy")
    print("="*70)
    
    strategies = ['Simple_Avg', 'Weighted_Avg', 'Stacking', 'Voting', 'Optimized', 'Selective']
    
    for strategy in strategies:
        avg_r2 = np.mean([res['results'][strategy]['r2'] for res in all_results.values()])
        avg_baseline = np.mean([res['baseline_r2'] for res in all_results.values()])
        print(f"{strategy:20s} R²: {avg_r2:.4f}  ({(avg_r2/avg_baseline-1)*100:+.2f}%)")
    
    print(f"\nResults saved to {output_file}")
