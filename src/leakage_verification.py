#!/usr/bin/env python3
"""
데이터 누출(Data Leakage) 검증
==============================

의심 사항:
- R² = 0.40 은 금융 시계열에서 매우 높은 수치
- Look-ahead bias 가능성 체크 필요

검증 방법:
1. 무작위 타겟 테스트 (Shuffled Target)
2. 미래 데이터 제거 테스트 (Strict Temporal)
3. 22일 Gap 강화 테스트
4. Scaler 누출 테스트
5. Autocorrelation 분석
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
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


def prepare_data():
    """EFA 데이터 준비 (최고 성능 자산)"""
    print("데이터 준비...")
    
    asset = download_data('EFA')
    vix = download_data('^VIX')
    spy = download_data('SPY')
    
    if asset is None or vix is None:
        return None
    
    df = asset[['Close']].copy()
    df.columns = ['Price']
    df['VIX'] = vix['Close'].reindex(df.index).ffill().bfill()
    df['SPY'] = spy['Close'].reindex(df.index).ffill().bfill()
    df['returns'] = df['Price'].pct_change()
    df['spy_returns'] = df['SPY'].pct_change()
    
    # 실현변동성
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    
    # SIVB
    df['SIVB'] = df['VIX'] - df['RV_22d']
    
    # 타겟: 22일 후 RV
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['SIVB_target'] = df['VIX'] - df['RV_future']
    
    # 특성 (과거 데이터만)
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['SIVB_lag1'] = df['SIVB'].shift(1)
    df['SIVB_lag5'] = df['SIVB'].shift(5)
    df['SIVB_ma5'] = df['SIVB'].rolling(5).mean()
    df['rolling_corr'] = df['returns'].rolling(60).corr(df['spy_returns'])
    
    df = df.dropna()
    
    print(f"  데이터 행 수: {len(df)}")
    return df


def test_1_shuffled_target(df):
    """
    테스트 1: 무작위 타겟 (Shuffled Target)
    
    기대 결과: R² ≈ 0 (예측 불가)
    만약 R² > 0.1이면: 심각한 데이터 누출 의심
    """
    print("\n" + "=" * 60)
    print("테스트 1: 무작위 타겟 (Shuffled Target)")
    print("=" * 60)
    print("기대 결과: R² ≈ 0 (무작위면 예측 불가)")
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'SIVB_lag1', 'SIVB_lag5', 'SIVB_ma5', 'rolling_corr']
    
    X = df[feature_cols].values
    y_true = df['SIVB_target'].values
    
    # 타겟 셔플
    y_shuffled = np.random.permutation(y_true)
    
    split = int(len(X) * 0.8)
    gap = 22
    
    X_train, X_test = X[:split], X[split+gap:]
    y_train_shuffled = y_shuffled[:split]
    y_test_shuffled = y_shuffled[split+gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    en = ElasticNet(alpha=0.01, random_state=SEED)
    en.fit(X_train_s, y_train_shuffled)
    y_pred = en.predict(X_test_s)
    
    r2 = r2_score(y_test_shuffled, y_pred)
    
    print(f"\n  R² (Shuffled Target): {r2:.4f}")
    
    if abs(r2) < 0.05:
        print("  ✓ 통과: 무작위 타겟이 예측 불가 (정상)")
        result = "PASS"
    else:
        print("  ✗ 실패: 무작위 타겟도 예측됨 (누출 의심)")
        result = "FAIL"
    
    return {'test': 'shuffled_target', 'r2': float(r2), 'result': result}


def test_2_strict_temporal(df):
    """
    테스트 2: 엄격한 시간 기반 분할
    
    - Train: 2015-2022
    - Gap: 2023
    - Test: 2024
    
    기대 결과: R² < 원래 결과 (하지만 여전히 > 0)
    """
    print("\n" + "=" * 60)
    print("테스트 2: 엄격한 시간 기반 분할")
    print("=" * 60)
    print("Train: ~2022, Gap: 2023, Test: 2024")
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'SIVB_lag1', 'SIVB_lag5', 'SIVB_ma5', 'rolling_corr']
    
    # 시간 기반 분할
    train_mask = df.index.year <= 2022
    test_mask = df.index.year >= 2024
    
    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, 'RV_future'].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, 'SIVB_target'].values
    vix_test = df.loc[test_mask, 'VIX'].values
    
    print(f"  Train: {train_mask.sum()}, Test: {test_mask.sum()}")
    
    if len(X_test) < 30:
        print("  ⚠ 테스트 데이터 부족")
        return {'test': 'strict_temporal', 'r2': None, 'result': 'SKIP'}
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    en = ElasticNet(alpha=0.01, random_state=SEED)
    en.fit(X_train_s, y_train)
    sivb_pred = vix_test - en.predict(X_test_s)
    
    r2 = r2_score(y_test, sivb_pred)
    
    print(f"\n  R² (Strict Temporal): {r2:.4f}")
    
    if r2 > 0:
        print("  ✓ 통과: OOS에서도 예측력 유지")
        result = "PASS"
    else:
        print("  △ 주의: OOS 성능 저하")
        result = "WARNING"
    
    return {'test': 'strict_temporal', 'r2': float(r2), 'result': result}


def test_3_extended_gap(df):
    """
    테스트 3: 확장된 Gap (44일)
    
    기대 결과: 22일 Gap과 유사하거나 약간 낮은 R²
    """
    print("\n" + "=" * 60)
    print("테스트 3: 확장된 Gap (44일 vs 22일)")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'SIVB_lag1', 'SIVB_lag5', 'SIVB_ma5', 'rolling_corr']
    
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_sivb = df['SIVB_target'].values
    vix_arr = df['VIX'].values
    
    split = int(len(X) * 0.8)
    
    results = {}
    
    for gap in [22, 44, 66]:
        X_train, X_test = X[:split], X[split+gap:]
        y_train = y_rv[:split]
        y_test = y_sivb[split+gap:]
        vix_test = vix_arr[split+gap:]
        
        if len(X_test) < 30:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=0.01, random_state=SEED)
        en.fit(X_train_s, y_train)
        sivb_pred = vix_test - en.predict(X_test_s)
        
        r2 = r2_score(y_test, sivb_pred)
        results[f'gap_{gap}'] = r2
        print(f"  Gap {gap}일: R² = {r2:.4f}")
    
    if len(results) >= 2:
        r2_22 = results.get('gap_22', 0)
        r2_44 = results.get('gap_44', 0)
        diff = r2_22 - r2_44
        print(f"\n  Gap 22 vs 44 차이: {diff:.4f}")
        
        if abs(diff) < 0.2:
            print("  ✓ 통과: Gap 확장해도 유사한 성능 (정상)")
            result = "PASS"
        else:
            print("  △ 주의: Gap에 따라 성능이 크게 변함")
            result = "WARNING"
    else:
        result = "SKIP"
    
    return {'test': 'extended_gap', 'results': {k: float(v) for k, v in results.items()}, 'result': result}


def test_4_scaler_leak(df):
    """
    테스트 4: Scaler 누출 테스트
    
    비교:
    - A) 정상: Scaler를 Train에만 fit
    - B) 누출: Scaler를 전체 데이터에 fit
    
    기대 결과: A ≈ B (차이 작아야 함, 하지만 B > A면 누출)
    """
    print("\n" + "=" * 60)
    print("테스트 4: Scaler 누출 테스트")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'SIVB_lag1', 'SIVB_lag5', 'SIVB_ma5', 'rolling_corr']
    
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_sivb = df['SIVB_target'].values
    vix_arr = df['VIX'].values
    
    split = int(len(X) * 0.8)
    gap = 22
    
    X_train, X_test = X[:split], X[split+gap:]
    y_train = y_rv[:split]
    y_test = y_sivb[split+gap:]
    vix_test = vix_arr[split+gap:]
    
    # A) 정상: Train에만 fit
    scaler_a = StandardScaler()
    X_train_a = scaler_a.fit_transform(X_train)
    X_test_a = scaler_a.transform(X_test)
    
    en_a = ElasticNet(alpha=0.01, random_state=SEED)
    en_a.fit(X_train_a, y_train)
    sivb_pred_a = vix_test - en_a.predict(X_test_a)
    r2_a = r2_score(y_test, sivb_pred_a)
    
    # B) 누출: 전체 데이터에 fit
    scaler_b = StandardScaler()
    X_all_b = scaler_b.fit_transform(X)  # 전체 데이터로 fit
    X_train_b = X_all_b[:split]
    X_test_b = X_all_b[split+gap:]
    
    en_b = ElasticNet(alpha=0.01, random_state=SEED)
    en_b.fit(X_train_b, y_train)
    sivb_pred_b = vix_test - en_b.predict(X_test_b)
    r2_b = r2_score(y_test, sivb_pred_b)
    
    print(f"\n  A) Train-only Scaler: R² = {r2_a:.4f}")
    print(f"  B) Full-data Scaler:  R² = {r2_b:.4f}")
    print(f"  차이 (B - A): {r2_b - r2_a:.4f}")
    
    if abs(r2_b - r2_a) < 0.02:
        print("  ✓ 통과: Scaler 누출 영향 미미")
        result = "PASS"
    else:
        print("  △ 주의: Scaler 누출이 결과에 영향")
        result = "WARNING"
    
    return {
        'test': 'scaler_leak', 
        'r2_correct': float(r2_a),
        'r2_leaked': float(r2_b),
        'difference': float(r2_b - r2_a),
        'result': result
    }


def test_5_autocorrelation(df):
    """
    테스트 5: 자기상관 분석
    
    SIVB가 강한 자기상관을 가지면 Naive와 유사해야 함
    """
    print("\n" + "=" * 60)
    print("테스트 5: 자기상관 분석")
    print("=" * 60)
    
    sivb = df['SIVB_target'].values
    
    # Lag 1 자기상관
    autocorr_1 = np.corrcoef(sivb[:-1], sivb[1:])[0, 1]
    autocorr_22 = np.corrcoef(sivb[:-22], sivb[22:])[0, 1]
    
    print(f"\n  SIVB 자기상관:")
    print(f"    Lag 1:  {autocorr_1:.4f}")
    print(f"    Lag 22: {autocorr_22:.4f}")
    
    if autocorr_22 > 0.7:
        print("  ⚠ 주의: 높은 자기상관 → Naive가 강할 수 있음")
        result = "WARNING"
    else:
        print("  ✓ 통과: 자기상관 낮음 → ML 예측력 유효")
        result = "PASS"
    
    return {
        'test': 'autocorrelation',
        'lag_1': float(autocorr_1),
        'lag_22': float(autocorr_22),
        'result': result
    }


def test_6_future_feature(df):
    """
    테스트 6: 미래 특성 의도적 추가 테스트
    
    만약 미래 RV를 특성으로 추가하면: R² ≈ 1.0 (완벽 예측)
    현재 특성만 사용 시: R² < 1.0
    """
    print("\n" + "=" * 60)
    print("테스트 6: 미래 특성 추가 테스트 (의도적 누출)")
    print("=" * 60)
    
    # 현재 특성
    current_features = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                       'VIX_change', 'SIVB_lag1', 'SIVB_lag5', 'SIVB_ma5', 'rolling_corr']
    
    X_current = df[current_features].values
    y_rv = df['RV_future'].values
    y_sivb = df['SIVB_target'].values
    vix_arr = df['VIX'].values
    
    # 미래 특성 추가 (의도적 누출)
    X_leaked = np.column_stack([X_current, df['RV_future'].values])
    
    split = int(len(X_current) * 0.8)
    gap = 22
    
    # 현재 특성만
    scaler_c = StandardScaler()
    X_train_c = scaler_c.fit_transform(X_current[:split])
    X_test_c = scaler_c.transform(X_current[split+gap:])
    
    en_c = ElasticNet(alpha=0.01, random_state=SEED)
    en_c.fit(X_train_c, y_rv[:split])
    sivb_pred_c = vix_arr[split+gap:] - en_c.predict(X_test_c)
    r2_current = r2_score(y_sivb[split+gap:], sivb_pred_c)
    
    # 미래 특성 포함 (누출)
    scaler_l = StandardScaler()
    X_train_l = scaler_l.fit_transform(X_leaked[:split])
    X_test_l = scaler_l.transform(X_leaked[split+gap:])
    
    en_l = ElasticNet(alpha=0.01, random_state=SEED)
    en_l.fit(X_train_l, y_rv[:split])
    sivb_pred_l = vix_arr[split+gap:] - en_l.predict(X_test_l)
    r2_leaked = r2_score(y_sivb[split+gap:], sivb_pred_l)
    
    print(f"\n  현재 특성만: R² = {r2_current:.4f}")
    print(f"  미래 특성 포함: R² = {r2_leaked:.4f}")
    
    if r2_leaked > 0.9:
        print("  ✓ 대조군 확인: 미래 데이터 사용 시 완벽 예측 (정상 반응)")
    
    if r2_current < r2_leaked * 0.8:
        print("  ✓ 통과: 현재 특성이 미래 특성보다 낮음 (누출 없음)")
        result = "PASS"
    else:
        print("  ✗ 실패: 현재 특성이 미래 정보 포함 가능성")
        result = "FAIL"
    
    return {
        'test': 'future_feature',
        'r2_current': float(r2_current),
        'r2_with_future': float(r2_leaked),
        'result': result
    }


def main():
    print("\n" + "🔍" * 30)
    print("데이터 누출(Data Leakage) 검증")
    print("🔍" * 30)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  의심 사항: R² = 0.40은 금융 시계열에서 매우 높음               │
    │  검증 대상: EFA (EAFE) - 최고 성능 자산                         │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    df = prepare_data()
    if df is None:
        print("데이터 준비 실패")
        return
    
    all_results = []
    
    # 테스트 실행
    all_results.append(test_1_shuffled_target(df))
    all_results.append(test_2_strict_temporal(df))
    all_results.append(test_3_extended_gap(df))
    all_results.append(test_4_scaler_leak(df))
    all_results.append(test_5_autocorrelation(df))
    all_results.append(test_6_future_feature(df))
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("📊 최종 요약")
    print("=" * 60)
    
    pass_count = sum(1 for r in all_results if r['result'] == 'PASS')
    warn_count = sum(1 for r in all_results if r['result'] == 'WARNING')
    fail_count = sum(1 for r in all_results if r['result'] == 'FAIL')
    
    print(f"\n  ✓ PASS: {pass_count}")
    print(f"  △ WARNING: {warn_count}")
    print(f"  ✗ FAIL: {fail_count}")
    
    for r in all_results:
        status = {'PASS': '✓', 'WARNING': '△', 'FAIL': '✗', 'SKIP': '-'}.get(r['result'], '?')
        print(f"\n  {status} {r['test']}: {r['result']}")
    
    if fail_count == 0 and warn_count <= 1:
        print("\n  ✓ 결론: 데이터 누출 증거 없음")
    elif fail_count > 0:
        print("\n  ✗ 결론: 데이터 누출 의심")
    else:
        print("\n  △ 결론: 추가 검증 필요")
    
    # 저장
    output = {
        'asset': 'EFA',
        'tests': all_results,
        'summary': {
            'pass': pass_count,
            'warning': warn_count,
            'fail': fail_count
        },
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/leakage_verification.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: data/results/leakage_verification.json")


if __name__ == '__main__':
    main()
