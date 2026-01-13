#!/usr/bin/env python3
"""
LSTM 성능 검증 (자산별)
======================

각 자산에 대해 LSTM 성능을 검증하고 명확히 보고
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
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
        return data if len(data) > 500 else None
    except:
        return None


def lstm_verification(asset_ticker, asset_name):
    """자산별 LSTM 검증"""
    print(f"\n{'='*60}")
    print(f"자산: {asset_name} ({asset_ticker})")
    print(f"{'='*60}")
    
    # 데이터 로드
    asset = download_data(asset_ticker)
    vix = download_data('^VIX')
    
    if asset is None or vix is None:
        print(f"  ✗ 데이터 다운로드 실패")
        return None
    
    # 데이터 준비
    df = asset[['Close']].copy()
    df['VIX'] = vix['Close'].reindex(df.index).ffill().bfill()
    df['returns'] = df['Close'].pct_change()
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    df['Spread'] = df['VIX'] - df['RV_22d']
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['Spread_true'] = df['VIX'] - df['RV_future']
    
    # 특성
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['Spread_lag1'] = df['Spread'].shift(1)
    df['Spread_lag5'] = df['Spread'].shift(5)
    df['Spread_ma5'] = df['Spread'].rolling(5).mean()
    
    df = df.dropna()
    print(f"  데이터: {len(df)} 행")
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'Spread_lag1', 'Spread_lag5', 'Spread_ma5']
    
    # Train/Test 분할 (22일 Gap)
    split = int(len(df) * 0.8)
    gap = 22
    
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_spread = df['Spread_true'].values
    vix_arr = df['VIX'].values
    
    X_train, X_test = X[:split], X[split+gap:]
    y_train = y_rv[:split]
    y_test_spread = y_spread[split+gap:]
    vix_test = vix_arr[split+gap:]
    
    if len(X_test) < 50:
        print(f"  ✗ 테스트 데이터 부족")
        return None
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 스케일링
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    
    # ===============================
    # 1. Naive Model (Persistence)
    # ===============================
    spread_lag = df['Spread_lag1'].values[split+gap:]
    r2_naive = r2_score(y_test_spread, spread_lag)
    results['Naive'] = {'r2': r2_naive}
    print(f"\n  [Naive] R² = {r2_naive:.4f}")
    
    # ===============================
    # 2. ElasticNet
    # ===============================
    en = ElasticNet(alpha=0.01, random_state=SEED)
    en.fit(X_train_s, y_train)
    spread_pred_en = vix_test - en.predict(X_test_s)
    r2_en = r2_score(y_test_spread, spread_pred_en)
    results['ElasticNet'] = {'r2': r2_en}
    print(f"  [ElasticNet] R² = {r2_en:.4f}")
    
    # ===============================
    # 3. MLP
    # ===============================
    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, 
                       random_state=SEED, early_stopping=True)
    mlp.fit(X_train_s, y_train)
    spread_pred_mlp = vix_test - mlp.predict(X_test_s)
    r2_mlp = r2_score(y_test_spread, spread_pred_mlp)
    results['MLP'] = {'r2': r2_mlp}
    print(f"  [MLP] R² = {r2_mlp:.4f}")
    
    # ===============================
    # 4. LSTM
    # ===============================
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(SEED)
        
        # 시퀀스 데이터
        lookback = 22
        X_seq_train, y_seq_train = [], []
        X_seq_test, y_seq_test = [], []
        vix_seq_test = []
        
        for i in range(lookback, len(X_train)):
            X_seq_train.append(X_train_s[i-lookback:i])
            y_seq_train.append(y_train[i])
        
        test_start = 0
        for i in range(lookback, len(X_test)):
            X_seq_test.append(X_test_s[i-lookback:i])
            y_seq_test.append(y_test_spread[i])
            vix_seq_test.append(vix_test[i])
        
        X_seq_train = np.array(X_seq_train)
        y_seq_train = np.array(y_seq_train)
        X_seq_test = np.array(X_seq_test)
        y_seq_test = np.array(y_seq_test)
        vix_seq_test = np.array(vix_seq_test)
        
        # LSTM 모델
        model = Sequential([
            LSTM(64, input_shape=(lookback, len(feature_cols))),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(X_seq_train, y_seq_train, epochs=100, batch_size=32,
                  validation_split=0.1, callbacks=[early_stop], verbose=0)
        
        rv_pred_lstm = model.predict(X_seq_test, verbose=0).flatten()
        spread_pred_lstm = vix_seq_test - rv_pred_lstm
        
        r2_lstm = r2_score(y_seq_test, spread_pred_lstm)
        results['LSTM'] = {'r2': r2_lstm}
        print(f"  [LSTM] R² = {r2_lstm:.4f}")
        
        # 방향 정확도
        spread_mean = y_seq_test.mean()
        dir_acc = ((y_seq_test > spread_mean) == (spread_pred_lstm > spread_mean)).mean()
        results['LSTM']['direction_acc'] = dir_acc
        print(f"  [LSTM] 방향정확도 = {dir_acc*100:.1f}%")
        
    except ImportError:
        print(f"  [LSTM] TensorFlow 미설치")
        r2_lstm = None
    
    # 요약
    print(f"\n  {'='*40}")
    print(f"  요약: {asset_name}")
    print(f"  {'='*40}")
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"  최고 성능: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
    
    if 'LSTM' in results and 'Naive' in results:
        improvement = results['LSTM']['r2'] - results['Naive']['r2']
        print(f"  LSTM vs Naive: {improvement:+.4f}")
    
    return {
        'asset': asset_ticker,
        'asset_name': asset_name,
        'n_samples': len(df),
        'n_test': len(X_test),
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'best_model': best_model[0],
        'best_r2': float(best_model[1]['r2'])
    }


def main():
    print("\n" + "🔬" * 30)
    print("LSTM 성능 검증 (자산별)")
    print("🔬" * 30)
    
    assets = [
        ('GLD', 'Gold (금)'),
        ('SPY', 'S&P 500'),
        ('EFA', 'EAFE (선진국)'),
        ('EEM', 'Emerging (신흥국)'),
        ('TLT', '20Y Treasury (국채)'),
        ('IWM', 'Russell 2000 (소형주)'),
    ]
    
    all_results = []
    
    for ticker, name in assets:
        result = lstm_verification(ticker, name)
        if result:
            all_results.append(result)
    
    # 전체 요약
    print("\n" + "=" * 70)
    print("전체 요약")
    print("=" * 70)
    
    print(f"\n{'자산':<20} | {'Naive':>8} | {'ElasticNet':>10} | {'MLP':>8} | {'LSTM':>8} | {'최고':>10}")
    print("-" * 75)
    
    for r in all_results:
        naive = r['results'].get('Naive', {}).get('r2', 0)
        en = r['results'].get('ElasticNet', {}).get('r2', 0)
        mlp = r['results'].get('MLP', {}).get('r2', 0)
        lstm = r['results'].get('LSTM', {}).get('r2', 0)
        best = r['best_model']
        print(f"{r['asset_name']:<20} | {naive:>8.4f} | {en:>10.4f} | {mlp:>8.4f} | {lstm:>8.4f} | {best:>10}")
    
    # 저장
    output = {
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/lstm_verification.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/lstm_verification.json")


if __name__ == '__main__':
    main()
