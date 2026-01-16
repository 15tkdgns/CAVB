"""
Advanced ML Models for Volatility Prediction
ElasticNet 대체: XGBoost, LightGBM, Random Forest, Neural Network

Target: R² 0.80+
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
from src.data import download_data, prepare_features, three_way_split
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import SPLIT_RATIOS

# ML Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import json
import time


def add_group2_features(df: pd.DataFrame) -> pd.DataFrame:
    """VRP Decomposition 추가"""
    vrp = df['CAVB']
    df['VRP_persistent'] = vrp.rolling(60).mean().fillna(vrp.rolling(20).mean()).fillna(0)
    df['VRP_transitory'] = (vrp - df['VRP_persistent']).fillna(0)
    return df


def run_advanced_models(ticker='GLD', horizon=5):
    """
    여러 ML 모델 비교
    """
    print(f"\n{'='*70}")
    print(f"Advanced ML Models Comparison: {ticker}")
    print(f"{'='*70}\n")
    
    # 데이터
    asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    
    df = prepare_features(asset, vix, horizon=horizon, include_enhanced=True)
    df = add_group2_features(df)
    df = df.dropna()
    
    # 29 features
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
    
    # Scaling (Neural Network용)
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    
    # ==================== Model 1: XGBoost ====================
    print("Model 1: XGBoost")
    start = time.time()
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred_xgb = xgb_model.predict(X_test)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    train_time_xgb = time.time() - start
    
    print(f"  R²:   {r2_xgb:.4f}")
    print(f"  MAE:  {mae_xgb:.4f}")
    print(f"  RMSE: {rmse_xgb:.4f}")
    print(f"  Time: {train_time_xgb:.2f}s")
    
    results['XGBoost'] = {
        'r2': float(r2_xgb),
        'mae': float(mae_xgb),
        'rmse': float(rmse_xgb),
        'train_time': float(train_time_xgb)
    }
    
    # ==================== Model 2: LightGBM ====================
    print("\nModel 2: LightGBM")
    start = time.time()
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    
    y_pred_lgb = lgb_model.predict(X_test)
    r2_lgb = r2_score(y_test, y_pred_lgb)
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    train_time_lgb = time.time() - start
    
    print(f"  R²:   {r2_lgb:.4f}")
    print(f"  MAE:  {mae_lgb:.4f}")
    print(f"  RMSE: {rmse_lgb:.4f}")
    print(f"  Time: {train_time_lgb:.2f}s")
    
    results['LightGBM'] = {
        'r2': float(r2_lgb),
        'mae': float(mae_lgb),
        'rmse': float(rmse_lgb),
        'train_time': float(train_time_lgb)
    }
    
    # ==================== Model 3: Random Forest ====================
    print("\nModel 3: Random Forest")
    start = time.time()
    
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    train_time_rf = time.time() - start
    
    print(f"  R²:   {r2_rf:.4f}")
    print(f"  MAE:  {mae_rf:.4f}")
    print(f"  RMSE: {rmse_rf:.4f}")
    print(f"  Time: {train_time_rf:.2f}s")
    
    results['RandomForest'] = {
        'r2': float(r2_rf),
        'mae': float(mae_rf),
        'rmse': float(rmse_rf),
        'train_time': float(train_time_rf)
    }
    
    # ==================== Model 4: Neural Network ====================
    print("\nModel 4: Neural Network (MLP)")
    start = time.time()
    
    nn_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    
    nn_model.fit(X_train_s, y_train)
    
    y_pred_nn = nn_model.predict(X_test_s)
    r2_nn = r2_score(y_test, y_pred_nn)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    train_time_nn = time.time() - start
    
    print(f"  R²:   {r2_nn:.4f}")
    print(f"  MAE:  {mae_nn:.4f}")
    print(f"  RMSE: {rmse_nn:.4f}")
    print(f"  Time: {train_time_nn:.2f}s")
    
    results['NeuralNetwork'] = {
        'r2': float(r2_nn),
        'mae': float(mae_nn),
        'rmse': float(rmse_nn),
        'train_time': float(train_time_nn)
    }
    
    # ==================== Model 5: Gradient Boosting (sklearn) ====================
    print("\nModel 5: Gradient Boosting (sklearn)")
    start = time.time()
    
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    
    y_pred_gb = gb_model.predict(X_test)
    r2_gb = r2_score(y_test, y_pred_gb)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    train_time_gb = time.time() - start
    
    print(f"  R²:   {r2_gb:.4f}")
    print(f"  MAE:  {mae_gb:.4f}")
    print(f"  RMSE: {rmse_gb:.4f}")
    print(f"  Time: {train_time_gb:.2f}s")
    
    results['GradientBoosting'] = {
        'r2': float(r2_gb),
        'mae': float(mae_gb),
        'rmse': float(rmse_gb),
        'train_time': float(train_time_gb)
    }
    
    # ==================== 비교 ====================
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('r2', ascending=False)
    
    print(comparison_df.to_string())
    
    best_model = comparison_df.index[0]
    best_r2 = comparison_df.iloc[0]['r2']
    
    print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.4f})")
    
    # Feature Importance (XGBoost)
    if ticker == 'GLD':  # 한 번만 출력
        print(f"\n{'='*70}")
        print("Feature Importance (XGBoost Top 10)")
        print(f"{'='*70}")
        
        importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(importance_df.head(10).to_string(index=False))
    
    return {
        'ticker': ticker,
        'results': results,
        'best_model': best_model,
        'best_r2': float(best_r2)
    }


if __name__ == '__main__':
    all_results = {}
    
    for ticker in ASSETS.keys():
        try:
            result = run_advanced_models(ticker, horizon=5)
            all_results[ticker] = result
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # 저장
    output_file = 'results/advanced_models_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 전체 요약
    print("\n" + "="*70)
    print("FINAL SUMMARY - All Assets")
    print("="*70)
    
    for ticker, res in all_results.items():
        print(f"\n{ticker}:")
        print(f"  Best Model: {res['best_model']}")
        print(f"  Best R²:    {res['best_r2']:.4f}")
    
    # 모델별 평균
    print("\n" + "="*70)
    print("Average Performance by Model")
    print("="*70)
    
    model_names = ['XGBoost', 'LightGBM', 'RandomForest', 'NeuralNetwork', 'GradientBoosting']
    
    for model in model_names:
        avg_r2 = np.mean([res['results'][model]['r2'] for res in all_results.values()])
        avg_time = np.mean([res['results'][model]['train_time'] for res in all_results.values()])
        print(f"{model:20s} R²: {avg_r2:.4f}  Time: {avg_time:.2f}s")
    
    print(f"\nResults saved to {output_file}")
