#!/usr/bin/env python3
"""
Persistence Model 비교 분석
===========================

LSTM/MLP가 단순히 "어제 값 복사"보다 우수한지 검증
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def download_data(ticker, start='2015-01-01', end='2025-01-01'):
    """데이터 다운로드"""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None


def prepare_data(ticker='GLD'):
    """데이터 준비"""
    vix = download_data('^VIX')
    asset = download_data(ticker)
    
    if vix is None or asset is None:
        return None, None
    
    df = asset[['Close']].copy()
    df['VIX'] = vix['Close'].reindex(df.index).ffill().bfill()
    df['returns'] = df['Close'].pct_change()
    
    # 실현변동성
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VIX-RV Spread (이전 명칭: VRP)
    df['Spread'] = df['VIX'] - df['RV_22d']
    
    # 타겟: 22일 후 실현변동성
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['Spread_true'] = df['VIX'] - df['RV_future']
    
    # 특성
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['Spread_lag1'] = df['Spread'].shift(1)
    df['Spread_lag5'] = df['Spread'].shift(5)
    
    df = df.dropna()
    
    feature_cols = ['RV_22d', 'VIX_lag1', 'Spread_lag1', 'Spread_lag5']
    
    return df, feature_cols


def persistence_model_comparison():
    """Persistence Model vs ML 모델 비교"""
    print("\n" + "=" * 70)
    print("Persistence Model 비교 분석")
    print("=" * 70)
    print("\n목적: LSTM이 단순히 '어제 값 복사'보다 우수한지 검증\n")
    
    df, feature_cols = prepare_data('GLD')
    
    if df is None:
        return {'error': '데이터 다운로드 실패'}
    
    # VIX-RV Spread (Target)
    target = df['Spread_true'].values
    
    # Train/Test 분할 (22일 Gap 포함)
    split = int(len(df) * 0.8)
    gap = 22
    
    train_idx = slice(0, split)
    test_idx = slice(split + gap, len(df))
    
    y_train = target[train_idx]
    y_test = target[test_idx]
    
    print(f"데이터: Train {len(y_train)}, Test {len(y_test)}")
    
    results = {}
    
    # ========================================
    # 1. Persistence Model (Naive: y_t = y_{t-1})
    # ========================================
    spread_lag1 = df['Spread_lag1'].values
    y_pred_naive = spread_lag1[test_idx]
    
    r2_naive = r2_score(y_test, y_pred_naive)
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    
    results['Naive (y_{t-1})'] = {
        'r2': r2_naive,
        'mae': mae_naive,
        'rmse': rmse_naive
    }
    
    print(f"\n1. Naive Model (어제 값 복사):")
    print(f"   R² = {r2_naive:.4f}")
    print(f"   MAE = {mae_naive:.2f}%")
    print(f"   RMSE = {rmse_naive:.2f}%")
    
    # ========================================
    # 2. ElasticNet
    # ========================================
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    vix_arr = df['VIX'].values
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_rv_train = y_rv[train_idx]
    vix_test = vix_arr[test_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    en = ElasticNet(alpha=0.01, random_state=SEED)
    en.fit(X_train_s, y_rv_train)
    spread_pred_en = vix_test - en.predict(X_test_s)
    
    r2_en = r2_score(y_test, spread_pred_en)
    mae_en = mean_absolute_error(y_test, spread_pred_en)
    rmse_en = np.sqrt(mean_squared_error(y_test, spread_pred_en))
    
    results['ElasticNet'] = {
        'r2': r2_en,
        'mae': mae_en,
        'rmse': rmse_en
    }
    
    print(f"\n2. ElasticNet:")
    print(f"   R² = {r2_en:.4f}")
    print(f"   MAE = {mae_en:.2f}%")
    print(f"   RMSE = {rmse_en:.2f}%")
    
    # ========================================
    # 3. MLP
    # ========================================
    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, 
                       random_state=SEED, early_stopping=True)
    mlp.fit(X_train_s, y_rv_train)
    spread_pred_mlp = vix_test - mlp.predict(X_test_s)
    
    r2_mlp = r2_score(y_test, spread_pred_mlp)
    mae_mlp = mean_absolute_error(y_test, spread_pred_mlp)
    rmse_mlp = np.sqrt(mean_squared_error(y_test, spread_pred_mlp))
    
    results['MLP (64)'] = {
        'r2': r2_mlp,
        'mae': mae_mlp,
        'rmse': rmse_mlp
    }
    
    print(f"\n3. MLP (64):")
    print(f"   R² = {r2_mlp:.4f}")
    print(f"   MAE = {mae_mlp:.2f}%")
    print(f"   RMSE = {rmse_mlp:.2f}%")
    
    # ========================================
    # 4. LSTM (TensorFlow)
    # ========================================
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(SEED)
        
        # 시퀀스 데이터 생성
        lookback = 22
        X_seq_train, y_seq_train = [], []
        X_seq_test, y_seq_test = [], []
        vix_seq_test = []
        
        X_all = df[feature_cols].values
        y_rv_all = df['RV_future'].values
        vix_all = df['VIX'].values
        target_all = df['Spread_true'].values
        
        for i in range(lookback, split):
            X_seq_train.append(X_all[i-lookback:i])
            y_seq_train.append(y_rv_all[i])
        
        for i in range(split + gap + lookback, len(df)):
            X_seq_test.append(X_all[i-lookback:i])
            y_seq_test.append(target_all[i])
            vix_seq_test.append(vix_all[i])
        
        X_seq_train = np.array(X_seq_train)
        y_seq_train = np.array(y_seq_train)
        X_seq_test = np.array(X_seq_test)
        y_seq_test = np.array(y_seq_test)
        vix_seq_test = np.array(vix_seq_test)
        
        # 스케일링
        X_flat_train = X_seq_train.reshape(-1, X_seq_train.shape[-1])
        X_flat_test = X_seq_test.reshape(-1, X_seq_test.shape[-1])
        scaler_lstm = StandardScaler()
        X_flat_train_s = scaler_lstm.fit_transform(X_flat_train)
        X_flat_test_s = scaler_lstm.transform(X_flat_test)
        X_seq_train_s = X_flat_train_s.reshape(X_seq_train.shape)
        X_seq_test_s = X_flat_test_s.reshape(X_seq_test.shape)
        
        # LSTM 모델
        model = Sequential([
            LSTM(64, input_shape=(lookback, len(feature_cols))),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(X_seq_train_s, y_seq_train, epochs=100, batch_size=32,
                  validation_split=0.1, callbacks=[early_stop], verbose=0)
        
        rv_pred_lstm = model.predict(X_seq_test_s, verbose=0).flatten()
        spread_pred_lstm = vix_seq_test - rv_pred_lstm
        
        r2_lstm = r2_score(y_seq_test, spread_pred_lstm)
        mae_lstm = mean_absolute_error(y_seq_test, spread_pred_lstm)
        rmse_lstm = np.sqrt(mean_squared_error(y_seq_test, spread_pred_lstm))
        
        results['LSTM (64)'] = {
            'r2': r2_lstm,
            'mae': mae_lstm,
            'rmse': rmse_lstm
        }
        
        print(f"\n4. LSTM (64):")
        print(f"   R² = {r2_lstm:.4f}")
        print(f"   MAE = {mae_lstm:.2f}%")
        print(f"   RMSE = {rmse_lstm:.2f}%")
        
    except ImportError:
        print("\n4. LSTM: TensorFlow 미설치")
        r2_lstm = None
    
    # ========================================
    # 비교 요약
    # ========================================
    print("\n" + "=" * 70)
    print("비교 요약")
    print("=" * 70)
    
    print(f"\n{'모델':<20} | {'R²':>10} | {'vs Naive':>15}")
    print("-" * 50)
    
    for model_name, metrics in results.items():
        improvement = ((metrics['r2'] - r2_naive) / abs(r2_naive) * 100) if r2_naive != 0 else 0
        sign = "+" if improvement > 0 else ""
        print(f"{model_name:<20} | {metrics['r2']:>10.4f} | {sign}{improvement:>13.1f}%")
    
    # 핵심 결론
    print("\n" + "=" * 70)
    print("핵심 결론")
    print("=" * 70)
    
    if r2_naive > 0:
        if r2_en > r2_naive or r2_mlp > r2_naive or (r2_lstm and r2_lstm > r2_naive):
            print("\n✓ ML 모델이 Naive Model보다 우수함 → 예측력 유효")
        else:
            print("\n✗ ML 모델이 Naive Model보다 열등함 → 단순 복사와 다를 바 없음")
    else:
        print("\n⚠ Naive Model R² < 0 → VIX-RV Spread에 자기상관 없음")
    
    # 저장
    Path('data/results').mkdir(parents=True, exist_ok=True)
    output = {
        'models': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'baseline': 'Naive (y_{t-1})',
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/results/persistence_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 결과 저장: data/results/persistence_comparison.json")
    
    return results


if __name__ == '__main__':
    persistence_model_comparison()
