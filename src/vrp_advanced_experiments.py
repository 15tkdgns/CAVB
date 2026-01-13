#!/usr/bin/env python3
"""
VRP 연구 고도화 실험
====================

1. Out-of-Sample 테스트 (2024년)
2. 자산 확대 (8개)
3. VIX-Beta 통계 검정
4. LSTM 모델
5. 거시경제 변수
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def download_data(ticker, start='2015-01-01', end='2025-01-01'):
    """데이터 다운로드 헬퍼"""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        print(f"  ⚠ {ticker} 다운로드 실패: {e}")
        return None


def prepare_features(df, vix_data):
    """특성 및 타겟 생성"""
    df = df.copy()
    df['VIX'] = vix_data['Close'].reindex(df.index).ffill().bfill()
    df['returns'] = df['Close'].pct_change()
    
    # 실현변동성
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VRP
    df['VRP'] = df['VIX'] - df['RV_22d']
    
    # 타겟
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['VRP_true'] = df['VIX'] - df['RV_future']
    
    # 래그 특성
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['VRP_lag1'] = df['VRP'].shift(1)
    df['VRP_lag5'] = df['VRP'].shift(5)
    df['VRP_ma5'] = df['VRP'].rolling(5).mean()
    df['regime_high'] = (df['VIX'] >= 25).astype(int)
    df['return_5d'] = df['returns'].rolling(5).sum()
    df['return_22d'] = df['returns'].rolling(22).sum()
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df


# =============================================================================
# 실험 1: Out-of-Sample 테스트
# =============================================================================
def experiment_1_oos_test():
    """Out-of-Sample 테스트 (2024년 데이터)"""
    print("\n" + "=" * 70)
    print("실험 1: Out-of-Sample 테스트 (2024년)")
    print("=" * 70)
    
    results = {}
    
    # VIX 다운로드
    vix = download_data('^VIX', '2015-01-01', '2025-01-01')
    if vix is None:
        return {'error': 'VIX 데이터 다운로드 실패'}
    
    assets = ['GLD', 'SPY', 'EFA', 'EEM']
    
    print(f"\n  {'자산':8s} | {'Train R²':10s} | {'OOS R²':10s} | {'Train Dir':10s} | {'OOS Dir':10s}")
    print("  " + "-" * 60)
    
    for ticker in assets:
        try:
            data = download_data(ticker, '2015-01-01', '2025-01-01')
            if data is None or len(data) < 500:
                continue
            
            df = prepare_features(data, vix)
            
            feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                           'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                           'regime_high', 'return_5d', 'return_22d']
            
            # 학습: ~2023, 테스트: 2024
            train_mask = df.index.year <= 2023
            test_mask = df.index.year >= 2024
            
            if train_mask.sum() < 100 or test_mask.sum() < 20:
                continue
            
            X_train = df.loc[train_mask, feature_cols].values
            y_train = df.loc[train_mask, 'RV_future'].values
            X_test = df.loc[test_mask, feature_cols].values
            y_vrp_train = df.loc[train_mask, 'VRP_true'].values
            y_vrp_test = df.loc[test_mask, 'VRP_true'].values
            vix_train = df.loc[train_mask, 'VIX'].values
            vix_test = df.loc[test_mask, 'VIX'].values
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # ElasticNet
            en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
            en.fit(X_train_s, y_train)
            
            # Train 예측
            rv_pred_train = en.predict(X_train_s)
            vrp_pred_train = vix_train - rv_pred_train
            r2_train = r2_score(y_vrp_train, vrp_pred_train)
            dir_train = ((y_vrp_train > y_vrp_train.mean()) == (vrp_pred_train > y_vrp_train.mean())).mean()
            
            # OOS 예측
            rv_pred_test = en.predict(X_test_s)
            vrp_pred_test = vix_test - rv_pred_test
            r2_oos = r2_score(y_vrp_test, vrp_pred_test)
            dir_oos = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred_test > y_vrp_test.mean())).mean()
            
            print(f"  {ticker:8s} | {r2_train:10.4f} | {r2_oos:10.4f} | {dir_train*100:8.1f}% | {dir_oos*100:8.1f}%")
            
            results[ticker] = {
                'train_r2': float(r2_train),
                'oos_r2': float(r2_oos),
                'train_dir_acc': float(dir_train),
                'oos_dir_acc': float(dir_oos),
                'train_samples': int(train_mask.sum()),
                'oos_samples': int(test_mask.sum())
            }
            
        except Exception as e:
            print(f"  {ticker:8s} | 오류: {str(e)[:40]}")
    
    return results


# =============================================================================
# 실험 2: 자산 확대 (8개)
# =============================================================================
def experiment_2_asset_expansion():
    """자산 확대 (8개)"""
    print("\n" + "=" * 70)
    print("실험 2: 자산 확대 (8개 자산)")
    print("=" * 70)
    
    vix = download_data('^VIX', '2015-01-01', '2025-01-01')
    if vix is None:
        return {'error': 'VIX 데이터 다운로드 실패'}
    
    assets = [
        ('GLD', 'Gold', 'Commodity'),
        ('SPY', 'S&P 500', 'US Large'),
        ('EFA', 'EAFE', 'Developed'),
        ('EEM', 'Emerging', 'Emerging'),
        ('IWM', 'Russell 2000', 'US Small'),
        ('TLT', '20Y Treasury', 'Bond'),
        ('FXI', 'China', 'Emerging'),
        ('EWJ', 'Japan', 'Developed')
    ]
    
    results = {}
    
    print(f"\n  {'자산':15s} | {'분류':10s} | {'VIX-RV상관':10s} | {'R²':8s} | {'방향':8s}")
    print("  " + "-" * 65)
    
    for ticker, name, category in assets:
        try:
            data = download_data(ticker, '2015-01-01', '2025-01-01')
            if data is None or len(data) < 500:
                print(f"  {name:15s} | 데이터 부족")
                continue
            
            df = prepare_features(data, vix)
            
            # VIX-RV 상관
            vix_rv_corr = df['VIX'].corr(df['RV_22d'])
            
            feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                           'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                           'regime_high', 'return_5d', 'return_22d']
            
            X = df[feature_cols].values
            y_rv = df['RV_future'].values
            y_vrp = df['VRP_true'].values
            vix_arr = df['VIX'].values
            
            split = int(len(X) * 0.8)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X[:split])
            X_test_s = scaler.transform(X[split:])
            
            en = ElasticNet(alpha=0.01, random_state=SEED)
            en.fit(X_train_s, y_rv[:split])
            vrp_pred = vix_arr[split:] - en.predict(X_test_s)
            
            r2 = r2_score(y_vrp[split:], vrp_pred)
            vrp_mean = y_vrp[split:].mean()
            dir_acc = ((y_vrp[split:] > vrp_mean) == (vrp_pred > vrp_mean)).mean()
            
            print(f"  {name:15s} | {category:10s} | {vix_rv_corr:10.3f} | {r2:8.4f} | {dir_acc*100:6.1f}%")
            
            results[ticker] = {
                'name': name,
                'category': category,
                'vix_rv_corr': float(vix_rv_corr),
                'r2': float(r2),
                'direction_acc': float(dir_acc)
            }
            
        except Exception as e:
            print(f"  {name:15s} | 오류: {str(e)[:30]}")
    
    return results


# =============================================================================
# 실험 3: VIX-Beta 통계 검정
# =============================================================================
def experiment_3_vix_beta_stats(asset_results):
    """VIX-Beta 통계 검정"""
    print("\n" + "=" * 70)
    print("실험 3: VIX-Beta 이론 통계 검정")
    print("=" * 70)
    
    if not asset_results or 'error' in asset_results:
        return {'error': '자산 데이터 없음'}
    
    # 데이터 추출
    corrs = []
    r2s = []
    names = []
    
    for ticker, data in asset_results.items():
        if isinstance(data, dict) and 'vix_rv_corr' in data:
            corrs.append(data['vix_rv_corr'])
            r2s.append(data['r2'])
            names.append(data.get('name', ticker))
    
    if len(corrs) < 3:
        return {'error': '데이터 부족'}
    
    corrs = np.array(corrs)
    r2s = np.array(r2s)
    
    # Pearson 상관계수 + p-value
    pearson_r, pearson_p = stats.pearsonr(corrs, r2s)
    
    # Spearman 상관계수 (비모수)
    spearman_r, spearman_p = stats.spearmanr(corrs, r2s)
    
    # 회귀분석: R² = α + β × VIX-RV상관
    slope, intercept, r_value, p_value, std_err = stats.linregress(corrs, r2s)
    
    # Bootstrap 신뢰구간
    n_bootstrap = 1000
    boot_corrs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(corrs), size=len(corrs), replace=True)
        boot_r, _ = stats.pearsonr(corrs[idx], r2s[idx])
        boot_corrs.append(boot_r)
    
    ci_lower = np.percentile(boot_corrs, 2.5)
    ci_upper = np.percentile(boot_corrs, 97.5)
    
    print(f"\n  📊 VIX-Beta 이론 검정 결과:")
    print(f"\n  상관분석:")
    print(f"     Pearson r:  {pearson_r:.4f} (p = {pearson_p:.4f})")
    print(f"     Spearman ρ: {spearman_r:.4f} (p = {spearman_p:.4f})")
    print(f"     95% CI:     [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\n  회귀분석: R² = {intercept:.4f} + ({slope:.4f}) × VIX-RV상관")
    print(f"     기울기 p-value: {p_value:.4f}")
    print(f"     R²: {r_value**2:.4f}")
    
    # 해석
    if pearson_p < 0.05 and pearson_r < -0.5:
        interpretation = "✓ VIX-Beta 이론 통계적으로 유의 (p < 0.05, r < -0.5)"
    elif pearson_p < 0.10:
        interpretation = "△ 약한 유의성 (0.05 < p < 0.10)"
    else:
        interpretation = "✗ 통계적으로 유의하지 않음"
    
    print(f"\n  결론: {interpretation}")
    
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'regression_slope': float(slope),
        'regression_intercept': float(intercept),
        'regression_p': float(p_value),
        'n_assets': len(corrs),
        'interpretation': interpretation
    }


# =============================================================================
# 실험 4: LSTM 모델
# =============================================================================
def experiment_4_lstm():
    """LSTM 모델"""
    print("\n" + "=" * 70)
    print("실험 4: LSTM 모델")
    print("=" * 70)
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(SEED)
    except ImportError:
        print("  ⚠ TensorFlow 미설치. MLP로 대체...")
        return experiment_4_lstm_alternative()
    
    vix = download_data('^VIX', '2015-01-01', '2025-01-01')
    gld = download_data('GLD', '2015-01-01', '2025-01-01')
    
    if vix is None or gld is None:
        return {'error': '데이터 다운로드 실패'}
    
    df = prepare_features(gld, vix)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    # 시퀀스 데이터 생성
    lookback = 22
    X_seq = []
    y_seq = []
    vix_seq = []
    vrp_seq = []
    
    X_raw = df[feature_cols].values
    y_raw = df['RV_future'].values
    vix_raw = df['VIX'].values
    vrp_raw = df['VRP_true'].values
    
    for i in range(lookback, len(X_raw)):
        X_seq.append(X_raw[i-lookback:i])
        y_seq.append(y_raw[i])
        vix_seq.append(vix_raw[i])
        vrp_seq.append(vrp_raw[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    vix_seq = np.array(vix_seq)
    vrp_seq = np.array(vrp_seq)
    
    # 분할
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    vix_test = vix_seq[split:]
    vrp_test = vrp_seq[split:]
    
    # 스케일링
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    scaler = StandardScaler()
    X_train_flat_s = scaler.fit_transform(X_train_flat)
    X_test_flat_s = scaler.transform(X_test_flat)
    X_train_s = X_train_flat_s.reshape(X_train.shape)
    X_test_s = X_test_flat_s.reshape(X_test.shape)
    
    # LSTM 모델
    model = Sequential([
        LSTM(64, input_shape=(lookback, len(feature_cols)), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    
    print("  LSTM 학습 중...")
    history = model.fit(
        X_train_s, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )
    
    # 예측
    rv_pred_lstm = model.predict(X_test_s, verbose=0).flatten()
    vrp_pred_lstm = vix_test - rv_pred_lstm
    
    r2_lstm = r2_score(vrp_test, vrp_pred_lstm)
    vrp_mean = vrp_test.mean()
    dir_lstm = ((vrp_test > vrp_mean) == (vrp_pred_lstm > vrp_mean)).mean()
    
    # MLP 비교
    X_train_2d = X_train_s.reshape(X_train_s.shape[0], -1)
    X_test_2d = X_test_s.reshape(X_test_s.shape[0], -1)
    
    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=SEED, early_stopping=True)
    mlp.fit(X_train_2d, y_train)
    rv_pred_mlp = mlp.predict(X_test_2d)
    vrp_pred_mlp = vix_test - rv_pred_mlp
    r2_mlp = r2_score(vrp_test, vrp_pred_mlp)
    dir_mlp = ((vrp_test > vrp_mean) == (vrp_pred_mlp > vrp_mean)).mean()
    
    print(f"\n  📊 모델 비교 (GLD):")
    print(f"     {'모델':15s} | {'R²':10s} | {'방향정확도':10s}")
    print("     " + "-" * 40)
    print(f"     {'LSTM':15s} | {r2_lstm:10.4f} | {dir_lstm*100:8.1f}%")
    print(f"     {'MLP':15s} | {r2_mlp:10.4f} | {dir_mlp*100:8.1f}%")
    
    improvement = (r2_lstm - r2_mlp) / abs(r2_mlp) * 100 if r2_mlp != 0 else 0
    print(f"\n  LSTM vs MLP: {improvement:+.1f}%")
    
    return {
        'lstm_r2': float(r2_lstm),
        'lstm_dir_acc': float(dir_lstm),
        'mlp_r2': float(r2_mlp),
        'mlp_dir_acc': float(dir_mlp),
        'improvement': float(improvement),
        'epochs_trained': len(history.history['loss'])
    }


def experiment_4_lstm_alternative():
    """LSTM 대체 (MLP 심층 비교)"""
    print("  MLP 심층 비교로 대체...")
    
    vix = download_data('^VIX', '2015-01-01', '2025-01-01')
    gld = download_data('GLD', '2015-01-01', '2025-01-01')
    
    if vix is None or gld is None:
        return {'error': '데이터 다운로드 실패'}
    
    df = prepare_features(gld, vix)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_vrp = df['VRP_true'].values
    vix_arr = df['VIX'].values
    
    split = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split])
    X_test_s = scaler.transform(X[split:])
    
    results = {}
    architectures = [
        ('MLP(64)', (64,)),
        ('MLP(128,64)', (128, 64)),
        ('MLP(256,128,64)', (256, 128, 64))
    ]
    
    print(f"\n  {'모델':20s} | {'R²':10s} | {'방향정확도':10s}")
    print("  " + "-" * 45)
    
    for name, layers in architectures:
        mlp = MLPRegressor(hidden_layer_sizes=layers, max_iter=500, random_state=SEED, early_stopping=True)
        mlp.fit(X_train_s, y_rv[:split])
        rv_pred = mlp.predict(X_test_s)
        vrp_pred = vix_arr[split:] - rv_pred
        
        r2 = r2_score(y_vrp[split:], vrp_pred)
        vrp_mean = y_vrp[split:].mean()
        dir_acc = ((y_vrp[split:] > vrp_mean) == (vrp_pred > vrp_mean)).mean()
        
        print(f"  {name:20s} | {r2:10.4f} | {dir_acc*100:8.1f}%")
        results[name] = {'r2': float(r2), 'dir_acc': float(dir_acc)}
    
    return {'mlp_comparison': results, 'note': 'TensorFlow 미설치로 MLP 비교만 수행'}


# =============================================================================
# 실험 5: 거시경제 변수
# =============================================================================
def experiment_5_macro_variables():
    """거시경제 변수 추가"""
    print("\n" + "=" * 70)
    print("실험 5: 거시경제 변수 추가")
    print("=" * 70)
    
    vix = download_data('^VIX', '2015-01-01', '2025-01-01')
    gld = download_data('GLD', '2015-01-01', '2025-01-01')
    
    # 추가 지표
    tlt = download_data('TLT', '2015-01-01', '2025-01-01')  # 장기 국채
    ief = download_data('IEF', '2015-01-01', '2025-01-01')  # 중기 국채
    hyg = download_data('HYG', '2015-01-01', '2025-01-01')  # 하이일드 채권
    
    if vix is None or gld is None:
        return {'error': '데이터 다운로드 실패'}
    
    df = prepare_features(gld, vix)
    
    # 거시 변수 추가
    if tlt is not None and ief is not None:
        df['term_spread'] = (tlt['Close'].reindex(df.index).ffill() / 
                            ief['Close'].reindex(df.index).ffill() - 1) * 100
    
    if hyg is not None:
        df['credit_proxy'] = hyg['Close'].reindex(df.index).ffill().pct_change(22) * 100
    
    df = df.dropna()
    
    # 기본 특성
    base_features = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                    'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                    'regime_high', 'return_5d', 'return_22d']
    
    # 확장 특성
    extended_features = base_features.copy()
    if 'term_spread' in df.columns:
        extended_features.append('term_spread')
    if 'credit_proxy' in df.columns:
        extended_features.append('credit_proxy')
    
    X_base = df[base_features].values
    X_ext = df[extended_features].values
    y_rv = df['RV_future'].values
    y_vrp = df['VRP_true'].values
    vix_arr = df['VIX'].values
    
    split = int(len(X_base) * 0.8)
    
    results = {}
    
    for name, X in [('기본 (12개)', X_base), ('확장 (+거시)', X_ext)]:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[:split])
        X_test_s = scaler.transform(X[split:])
        
        en = ElasticNet(alpha=0.01, random_state=SEED)
        en.fit(X_train_s, y_rv[:split])
        vrp_pred = vix_arr[split:] - en.predict(X_test_s)
        
        r2 = r2_score(y_vrp[split:], vrp_pred)
        vrp_mean = y_vrp[split:].mean()
        dir_acc = ((y_vrp[split:] > vrp_mean) == (vrp_pred > vrp_mean)).mean()
        
        results[name] = {'r2': float(r2), 'dir_acc': float(dir_acc), 'n_features': X.shape[1]}
    
    print(f"\n  {'특성 세트':15s} | {'N':5s} | {'R²':10s} | {'방향정확도':10s}")
    print("  " + "-" * 50)
    for name, data in results.items():
        print(f"  {name:15s} | {data['n_features']:5d} | {data['r2']:10.4f} | {data['dir_acc']*100:8.1f}%")
    
    if len(results) >= 2:
        base_r2 = list(results.values())[0]['r2']
        ext_r2 = list(results.values())[1]['r2']
        improvement = (ext_r2 - base_r2) / abs(base_r2) * 100 if base_r2 != 0 else 0
        print(f"\n  거시변수 효과: {improvement:+.1f}%")
        results['improvement'] = float(improvement)
    
    return results


# =============================================================================
# 메인 실행
# =============================================================================
def main():
    print("\n" + "🔬" * 35)
    print("VRP 연구 고도화 실험")
    print("🔬" * 35)
    
    all_results = {}
    
    # 실험 1
    all_results['oos_test'] = experiment_1_oos_test()
    
    # 실험 2
    all_results['asset_expansion'] = experiment_2_asset_expansion()
    
    # 실험 3 (실험 2 결과 필요)
    all_results['vix_beta_stats'] = experiment_3_vix_beta_stats(all_results['asset_expansion'])
    
    # 실험 4
    all_results['lstm'] = experiment_4_lstm()
    
    # 실험 5
    all_results['macro_variables'] = experiment_5_macro_variables()
    
    # 결과 저장
    all_results['timestamp'] = datetime.now().isoformat()
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/advanced_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 최종 요약
    print("\n" + "=" * 70)
    print("📊 최종 요약")
    print("=" * 70)
    
    print("""
    1. Out-of-Sample (2024년): 완료
    2. 자산 확대 (8개): 완료
    3. VIX-Beta 통계 검정: 완료
    4. LSTM 모델: 완료
    5. 거시경제 변수: 완료
    """)
    
    print(f"💾 결과 저장: data/results/advanced_experiments.json")
    
    return all_results


if __name__ == '__main__':
    main()
