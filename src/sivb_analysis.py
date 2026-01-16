#!/usr/bin/env python3
"""
Systemic-Idiosyncratic Volatility Basis (SIVB) 분석
====================================================

SIVB = VIX (시장 위험, Systemic Risk) - RV (개별 자산 변동성, Idiosyncratic Vol)

이론적 프레임워크:
- VIX: S&P 500 옵션에서 추출한 시장 전체 내재 위험도
- RV: 개별 자산의 실현 변동성 (고유 위험)
- SIVB: 시장 위험과 개별 자산 위험 간의 괴리 (Basis)

예측 가설:
- SIVB는 시장 위험과 자산 고유 위험의 불일치를 측정
- 이 불일치가 예측 가능한 패턴을 가진다면, 투자 전략에 활용 가능
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
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


def calculate_sivb(asset_ticker, asset_name):
    """
    SIVB (Systemic-Idiosyncratic Volatility Basis) 계산 및 분석
    
    Returns:
        dict: SIVB 분석 결과
    """
    print(f"\n{'='*70}")
    print(f"자산: {asset_name} ({asset_ticker})")
    print(f"{'='*70}")
    
    # 데이터 로드
    asset = download_data(asset_ticker)
    vix = download_data('^VIX')
    spy = download_data('SPY')  # 시장 벤치마크
    
    if asset is None or vix is None or spy is None:
        print("  ✗ 데이터 다운로드 실패")
        return None
    
    # 데이터 준비
    df = asset[['Close']].copy()
    df.columns = ['Price']
    df['VIX'] = vix['Close'].reindex(df.index).ffill().bfill()
    df['SPY'] = spy['Close'].reindex(df.index).ffill().bfill()
    
    # 수익률
    df['returns'] = df['Price'].pct_change()
    df['spy_returns'] = df['SPY'].pct_change()
    
    # 실현변동성 (Idiosyncratic Vol)
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    
    # SPY와의 상관 (Beta 근사)
    df['rolling_corr'] = df['returns'].rolling(60).corr(df['spy_returns'])
    
    # ============================
    # SIVB 정의
    # ============================
    # SIVB = Systemic (VIX) - Idiosyncratic (Asset RV)
    df['SIVB'] = df['VIX'] - df['RV_22d']
    
    # 타겟: 22일 후 SIVB
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['SIVB_true'] = df['VIX'].shift(-22) - df['RV_future']  # Future SIVB
    df['SIVB_target'] = df['VIX'] - df['RV_future']  # 예측 대상
    
    # 래그 특성
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['SIVB_lag1'] = df['SIVB'].shift(1)
    df['SIVB_lag5'] = df['SIVB'].shift(5)
    df['SIVB_ma5'] = df['SIVB'].rolling(5).mean()
    df['SIVB_std5'] = df['SIVB'].rolling(5).std()
    
    df = df.dropna()
    
    print(f"  데이터: {len(df)} 행")
    
    # ============================
    # 기초 통계
    # ============================
    print(f"\n  [SIVB 기초 통계]")
    print(f"    평균: {df['SIVB'].mean():.2f}%")
    print(f"    표준편차: {df['SIVB'].std():.2f}%")
    print(f"    최소/최대: [{df['SIVB'].min():.2f}, {df['SIVB'].max():.2f}]")
    print(f"    VIX-RV 상관: {df['VIX'].corr(df['RV_22d']):.3f}")
    print(f"    SPY와의 상관: {df['rolling_corr'].mean():.3f}")
    
    # ============================
    # SIVB 예측 모델
    # ============================
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'SIVB_lag1', 'SIVB_lag5', 'SIVB_ma5', 
                   'SIVB_std5', 'rolling_corr']
    
    # Train/Test 분할 (22일 Gap)
    split = int(len(df) * 0.8)
    gap = 22
    
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_sivb = df['SIVB_target'].values
    vix_arr = df['VIX'].values
    
    X_train, X_test = X[:split], X[split+gap:]
    y_train = y_rv[:split]
    y_test_sivb = y_sivb[split+gap:]
    vix_test = vix_arr[split+gap:]
    
    if len(X_test) < 50:
        print("  ✗ 테스트 데이터 부족")
        return None
    
    print(f"\n  [모델 학습]")
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 스케일링
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    
    # 1. Naive (Persistence)
    sivb_lag = df['SIVB_lag1'].values[split+gap:]
    r2_naive = r2_score(y_test_sivb, sivb_lag)
    results['Naive'] = {'r2': r2_naive}
    print(f"    Naive: R² = {r2_naive:.4f}")
    
    # 2. ElasticNet
    en = ElasticNet(alpha=0.01, random_state=SEED)
    en.fit(X_train_s, y_train)
    sivb_pred_en = vix_test - en.predict(X_test_s)
    r2_en = r2_score(y_test_sivb, sivb_pred_en)
    dir_en = ((y_test_sivb > y_test_sivb.mean()) == (sivb_pred_en > y_test_sivb.mean())).mean()
    results['ElasticNet'] = {'r2': r2_en, 'direction_acc': dir_en}
    print(f"    ElasticNet: R² = {r2_en:.4f}, Dir = {dir_en*100:.1f}%")
    
    # 3. MLP
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, 
                       random_state=SEED, early_stopping=True)
    mlp.fit(X_train_s, y_train)
    sivb_pred_mlp = vix_test - mlp.predict(X_test_s)
    r2_mlp = r2_score(y_test_sivb, sivb_pred_mlp)
    dir_mlp = ((y_test_sivb > y_test_sivb.mean()) == (sivb_pred_mlp > y_test_sivb.mean())).mean()
    results['MLP'] = {'r2': r2_mlp, 'direction_acc': dir_mlp}
    print(f"    MLP: R² = {r2_mlp:.4f}, Dir = {dir_mlp*100:.1f}%")
    
    # 4. LSTM
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(SEED)
        
        lookback = 22
        X_seq_train, y_seq_train = [], []
        X_seq_test, y_seq_test = [], []
        vix_seq_test = []
        
        for i in range(lookback, len(X_train_s)):
            X_seq_train.append(X_train_s[i-lookback:i])
            y_seq_train.append(y_train[i])
        
        for i in range(lookback, len(X_test_s)):
            X_seq_test.append(X_test_s[i-lookback:i])
            y_seq_test.append(y_test_sivb[i])
            vix_seq_test.append(vix_test[i])
        
        X_seq_train = np.array(X_seq_train)
        y_seq_train = np.array(y_seq_train)
        X_seq_test = np.array(X_seq_test)
        y_seq_test = np.array(y_seq_test)
        vix_seq_test = np.array(vix_seq_test)
        
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
        sivb_pred_lstm = vix_seq_test - rv_pred_lstm
        
        r2_lstm = r2_score(y_seq_test, sivb_pred_lstm)
        dir_lstm = ((y_seq_test > y_seq_test.mean()) == (sivb_pred_lstm > y_seq_test.mean())).mean()
        results['LSTM'] = {'r2': r2_lstm, 'direction_acc': dir_lstm}
        print(f"    LSTM: R² = {r2_lstm:.4f}, Dir = {dir_lstm*100:.1f}%")
        
    except ImportError:
        print("    LSTM: TensorFlow 미설치")
        r2_lstm = None
    
    # ============================
    # 결과 요약
    # ============================
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    
    print(f"\n  [결과 요약]")
    print(f"    최고 모델: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
    
    if 'LSTM' in results:
        print(f"    LSTM vs Naive: {results['LSTM']['r2'] - results['Naive']['r2']:+.4f}")
    
    return {
        'asset': asset_ticker,
        'asset_name': asset_name,
        'n_samples': len(df),
        'sivb_mean': float(df['SIVB'].mean()),
        'sivb_std': float(df['SIVB'].std()),
        'vix_rv_corr': float(df['VIX'].corr(df['RV_22d'])),
        'spy_corr': float(df['rolling_corr'].mean()),
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'best_model': best_model[0],
        'best_r2': float(best_model[1]['r2'])
    }


def main():
    print("\n" + "🔬" * 35)
    print("Systemic-Idiosyncratic Volatility Basis (SIVB) 분석")
    print("🔬" * 35)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  SIVB = VIX (Systemic Risk) - RV (Idiosyncratic Volatility)    │
    │                                                                 │
    │  • VIX: 시장 전체의 내재 위험도 (공포 지수)                       │
    │  • RV:  개별 자산의 실현 변동성 (고유 위험)                       │
    │  • SIVB: 시장 위험과 자산 고유 위험의 괴리                        │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    assets = [
        ('GLD', 'Gold (금)'),
        ('SPY', 'S&P 500'),
        ('EFA', 'EAFE (선진국)'),
        ('EEM', 'Emerging (신흥국)'),
        ('TLT', '20Y Treasury (국채)'),
        ('IWM', 'Russell 2000 (소형주)'),
        ('USO', 'Oil (원유)'),
        ('FXI', 'China (중국)'),
    ]
    
    all_results = []
    
    for ticker, name in assets:
        result = calculate_sivb(ticker, name)
        if result:
            all_results.append(result)
    
    # ==============================
    # 전체 요약
    # ==============================
    print("\n" + "=" * 80)
    print("전체 요약: SIVB 예측력")
    print("=" * 80)
    
    print(f"\n{'자산':<18} | {'SPY상관':>8} | {'Naive':>8} | {'EN':>8} | {'MLP':>8} | {'LSTM':>8} | {'최고':>10}")
    print("-" * 90)
    
    for r in all_results:
        spy_corr = r['spy_corr']
        naive = r['results'].get('Naive', {}).get('r2', 0)
        en = r['results'].get('ElasticNet', {}).get('r2', 0)
        mlp = r['results'].get('MLP', {}).get('r2', 0)
        lstm = r['results'].get('LSTM', {}).get('r2', 0)
        best = r['best_model']
        print(f"{r['asset_name']:<18} | {spy_corr:>8.3f} | {naive:>8.4f} | {en:>8.4f} | {mlp:>8.4f} | {lstm:>8.4f} | {best:>10}")
    
    # ==============================
    # SIVB 예측력 vs SPY 상관 분석
    # ==============================
    if len(all_results) >= 4:
        print("\n" + "=" * 80)
        print("SIVB 예측력과 SPY 상관관계 분석")
        print("=" * 80)
        
        spy_corrs = [r['spy_corr'] for r in all_results]
        best_r2s = [r['best_r2'] for r in all_results]
        
        corr, p_value = stats.pearsonr(spy_corrs, best_r2s)
        print(f"\n  SPY 상관 vs SIVB 예측력:")
        print(f"    Pearson r = {corr:.4f}")
        print(f"    p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"    ✓ 통계적으로 유의 (p < 0.05)")
        else:
            print(f"    ✗ 통계적으로 유의하지 않음 (p > 0.05)")
    
    # ==============================
    # 핵심 결론
    # ==============================
    print("\n" + "=" * 80)
    print("핵심 결론")
    print("=" * 80)
    
    positive_r2 = [r for r in all_results if r['best_r2'] > 0]
    print(f"\n  예측 가능 자산 (R² > 0): {len(positive_r2)}/{len(all_results)}")
    for r in sorted(positive_r2, key=lambda x: -x['best_r2']):
        print(f"    - {r['asset_name']}: R² = {r['best_r2']:.4f} ({r['best_model']})")
    
    negative_r2 = [r for r in all_results if r['best_r2'] <= 0]
    print(f"\n  예측 불가 자산 (R² ≤ 0): {len(negative_r2)}/{len(all_results)}")
    for r in negative_r2:
        print(f"    - {r['asset_name']}: R² = {r['best_r2']:.4f}")
    
    # 저장
    output = {
        'framework': 'SIVB (Systemic-Idiosyncratic Volatility Basis)',
        'definition': 'SIVB = VIX (Systemic Risk) - RV (Idiosyncratic Vol)',
        'results': all_results,
        'summary': {
            'n_assets': len(all_results),
            'n_predictable': len(positive_r2),
            'avg_best_r2': np.mean([r['best_r2'] for r in all_results])
        },
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/sivb_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/sivb_analysis.json")


if __name__ == '__main__':
    main()
