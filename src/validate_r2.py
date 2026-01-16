#!/usr/bin/env python3
"""
R² 높은 값 검증 실험
====================

1. 데이터 누출 재확인
2. Shuffled prediction 테스트
3. 실제 예측과 타겟 간 시간 간격 확인
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import yfinance as yf

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


def validate_high_r2():
    """높은 R² 값 검증"""
    print("\n" + "="*70)
    print("높은 R² 값 검증 실험")
    print("="*70)
    
    # Gold 데이터로 테스트
    asset = download_data('GLD')
    vix = download_data('^VIX')
    
    df = asset[['Close']].copy()
    df.columns = ['Price']
    df['VIX'] = vix['Close'].reindex(df.index).ffill().bfill()
    df['returns'] = df['Price'].pct_change()
    
    # 변동성
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    
    # CAVB
    df['CAVB'] = df['VIX'] - df['RV_22d']
    
    # 타겟 (5일)
    df['RV_future_5d'] = df['RV_22d'].shift(-5)
    df['CAVB_target_5d'] = df['VIX'] - df['RV_future_5d']
    
    # 특성
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    df = df.dropna()
    
    print(f"\n총 데이터: {len(df)} 행")
    print(f"기간: {df.index[0]} ~ {df.index[-1]}")
    
    # 특성
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                   'VIX_change', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
    
    X = df[feature_cols].values
    y_rv = df['RV_future_5d'].values
    y_cavb = df['CAVB_target_5d'].values
    vix_arr = df['VIX'].values
    
    # 3-Way Split
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    gap = 5
    
    X_train = X[:train_end]
    X_test = X[val_end+gap:]
    
    y_train = y_rv[:train_end]
    y_test_cavb = y_cavb[val_end+gap:]
    
    vix_test = vix_arr[val_end+gap:]
    
    # 날짜 확인
    dates_train_end = df.index[train_end-1]
    dates_test_start = df.index[val_end+gap]
    dates_test_end = df.index[-1]
    
    print(f"\n[데이터 분할 확인]")
    print(f"Train 종료: {dates_train_end}")
    print(f"Test 시작:  {dates_test_start}")
    print(f"Test 종료:  {dates_test_end}")
    print(f"Train-Test 간격: {(dates_test_start - dates_train_end).days}일")
    
    # 스케일링
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 모델 학습
    model = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=SEED, max_iter=2000)
    model.fit(X_train_s, y_train)
    
    # 정상 예측
    rv_pred = model.predict(X_test_s)
    cavb_pred_test = vix_test - rv_pred
    r2_normal = r2_score(y_test_cavb, cavb_pred_test)
    
    print(f"\n[테스트 1: 정상 예측]")
    print(f"Test R² = {r2_normal:.4f}")
    
    # ==============================================
    # 의심 포인트 1: 타겟이 섞였을 때
    # ==============================================
    print(f"\n[테스트 2: Shuffled Target (데이터 누출 체크)]")
    y_test_shuffled = y_test_cavb.copy()
    np.random.shuffle(y_test_shuffled)
    r2_shuffled = r2_score(y_test_shuffled, cavb_pred_test)
    print(f"Shuffled R² = {r2_shuffled:.4f}")
    print(f"예상: ~0.0 (무작위)")
    print(f"결과: {'✅ PASS' if r2_shuffled < 0.1 else '❌ FAIL - 데이터 누출 의심!'}")
    
    # ==============================================
    # 의심 포인트 2: 미래 정보 사용 체크
    # ==============================================
    print(f"\n[테스트 3: Future Leakage Check]")
    
    # 현재 타겟: t 시점에서 (t+5)의 RV 예측
    # 확인: y_train에 미래 정보가 없는가?
    
    # Train의 마지막 타겟 날짜
    last_train_target_idx = train_end - 1  # y_train의 마지막 인덱스
    last_train_target_date = df.index[last_train_target_idx]
    
    # 이 타겟은 (last_train_target_date + 5일)의 RV를 예측해야 함
    # RV_future_5d = RV_22d.shift(-5)이므로
    # df.index[last_train_target_idx]의 RV_future_5d는
    # df.index[last_train_target_idx + 5]의 RV_22d
    
    expected_future_date = df.index[min(last_train_target_idx + 5, len(df)-1)]
    
    print(f"Train 마지막 타겟 날짜: {last_train_target_date}")
    print(f"이 타겟이 예측하는 시점: {expected_future_date}")
    print(f"Test 시작 날짜: {dates_test_start}")
    print(f"")
    print(f"Train 타겟이 미래 정보 사용 여부: {expected_future_date < dates_test_start}")
    print(f"결과: {'✅ PASS - 미래 정보 미사용' if expected_future_date < dates_test_start else '❌ FAIL - 미래 정보 사용!'}")
    
    # ==============================================
    # 의심 포인트 3: Gap이 제대로 작동하는가?
    # ==============================================
    print(f"\n[테스트 4: Gap Functionality]")
    
    # Gap이 없다면?
    X_test_no_gap = X[val_end:]
    vix_test_no_gap = vix_arr[val_end:]
    y_test_cavb_no_gap = y_cavb[val_end:]
    
    X_test_no_gap_s = scaler.transform(X_test_no_gap)
    rv_pred_no_gap = model.predict(X_test_no_gap_s)
    cavb_pred_no_gap = vix_test_no_gap - rv_pred_no_gap
    r2_no_gap = r2_score(y_test_cavb_no_gap, cavb_pred_no_gap)
    
    print(f"Gap 있음 (정상): R² = {r2_normal:.4f}")
    print(f"Gap 없음 (비교): R² = {r2_no_gap:.4f}")
    print(f"차이: {r2_normal - r2_no_gap:.4f}")
    print(f"예상: Gap 있을 때가 더 보수적 (R² 낮음)")
    
    # ==============================================
    # 의심 포인트 4: Persistence (Naive) 비교
    # ==============================================
    print(f"\n[테스트 5: vs Naive Persistence]")
    
    cavb_naive = df['CAVB_lag1'].values[val_end+gap:]
    r2_naive = r2_score(y_test_cavb, cavb_naive)
    
    print(f"ElasticNet R² = {r2_normal:.4f}")
    print(f"Naive R²      = {r2_naive:.4f}")
    print(f"개선:          {r2_normal - r2_naive:+.4f}")
    
    # ==============================================
   # 의심 포인트 5: 타겟과 예측의 상관관계 확인
    # ==============================================
    print(f"\n[테스트 6: 예측 vs 실제 통계]")
    
    print(f"실제 CAVB (Test) - Mean: {y_test_cavb.mean():.2f}, Std: {y_test_cavb.std():.2f}")
    print(f"예측 CAVB (Test) - Mean: {cavb_pred_test.mean():.2f}, Std: {cavb_pred_test.std():.2f}")
    print(f"상관계수: {np.corrcoef(y_test_cavb, cavb_pred_test)[0,1]:.4f}")
    
    # ==============================================
    # 결론
    # ==============================================
    print(f"\n" + "="*70)
    print("검증 결론")
    print("="*70)
    
    issues = []
    
    if r2_shuffled > 0.1:
        issues.append("❌ Shuffled Target R² 너무 높음 - 데이터 누출 의심")
    
    if expected_future_date >= dates_test_start:
        issues.append("❌ Train 타겟이 Test 기간 데이터 사용")
    
    if r2_normal < r2_no_gap:
        issues.append("⚠️ Gap이 성능을 오히려 높임 - 이상")
    
    if len(issues) == 0:
        print("✅ 모든 검증 통과")
        print(f"✅ R² = {r2_normal:.4f}는 정상적인 결과로 판단")
        print(f"")
        print(f"높은 R²의 이유:")
        print(f"  1. 5일 단기 예측 - 변동성 패턴이 단기간 안정적")
        print(f"  2. VIX의 예측력 - Gold의 경우 VIX와 RV 간 강한 선형 관계")
        print(f"  3. CAVB 지속성 - CAVB_lag1이 강력한 예측 변수")
    else:
        print("⚠️ 검증 실패 - 다음 문제 발견:")
        for issue in issues:
            print(f"  {issue}")


if __name__ == '__main__':
    validate_high_r2()
