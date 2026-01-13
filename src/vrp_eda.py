#!/usr/bin/env python3
"""
VRP 예측을 위한 데이터 탐색적 분석 (EDA)
=========================================

1. 데이터 특성 분석
2. VRP 분포 및 통계
3. 상관관계 분석
4. 시계열 특성 분석
5. Regime 분석
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
from datetime import datetime
import json

np.random.seed(42)


def load_data():
    """데이터 로드"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill().bfill()
    spy['returns'] = spy['Close'].pct_change()
    
    # 실현변동성 (연율화 %)
    spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VRP
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    
    # 미래 RV
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
    spy = spy.dropna()
    return spy


def analyze_basic_stats(spy):
    """1. 기본 통계량 분석"""
    print("\n" + "=" * 70)
    print("1. 기본 통계량")
    print("=" * 70)
    
    cols = ['VIX', 'RV_22d', 'VRP', 'VRP_true', 'returns']
    
    stats_df = pd.DataFrame()
    for col in cols:
        stats_df[col] = {
            'Mean': spy[col].mean(),
            'Std': spy[col].std(),
            'Min': spy[col].min(),
            'Max': spy[col].max(),
            'Skew': spy[col].skew(),
            'Kurt': spy[col].kurtosis(),
            'Q25': spy[col].quantile(0.25),
            'Q50': spy[col].quantile(0.50),
            'Q75': spy[col].quantile(0.75)
        }
    
    print(stats_df.round(4).to_string())
    
    # VRP 분석
    print(f"\n📊 VRP 주요 특성:")
    print(f"  • 평균 VRP: {spy['VRP'].mean():.2f}% (VIX가 평균적으로 RV보다 높음)")
    print(f"  • VRP 양수 비율: {(spy['VRP'] > 0).mean()*100:.1f}%")
    print(f"  • VRP 범위: [{spy['VRP'].min():.2f}%, {spy['VRP'].max():.2f}%]")
    
    # 분포 정규성 검정
    _, p_value = stats.normaltest(spy['VRP'].dropna())
    print(f"  • 정규성 검정 (D'Agostino): p={p_value:.4f} ({'정규' if p_value > 0.05 else '비정규'})")
    
    return stats_df


def analyze_correlation(spy):
    """2. 상관관계 분석"""
    print("\n" + "=" * 70)
    print("2. 상관관계 분석")
    print("=" * 70)
    
    # 특성 생성
    spy['VIX_lag1'] = spy['VIX'].shift(1)
    spy['VIX_lag5'] = spy['VIX'].shift(5)
    spy['VIX_change'] = spy['VIX'].pct_change()
    spy['VRP_lag1'] = spy['VRP'].shift(1)
    spy['VRP_lag5'] = spy['VRP'].shift(5)
    spy['return_5d'] = spy['returns'].rolling(5).sum()
    
    # VRP_true와의 상관관계
    corr_cols = ['VIX', 'RV_22d', 'VRP', 'VIX_lag1', 'VIX_lag5', 'VIX_change', 
                 'VRP_lag1', 'VRP_lag5', 'return_5d', 'RV_5d']
    
    correlations = {}
    for col in corr_cols:
        if col in spy.columns:
            corr = spy[col].corr(spy['VRP_true'])
            correlations[col] = corr
    
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\n📊 VRP_true (미래 VRP)와의 상관관계:")
    for feat, corr in sorted_corr:
        bar = "█" * int(abs(corr) * 50)
        sign = "+" if corr > 0 else "-"
        print(f"  {feat:15s}: {sign}{abs(corr):.4f} {bar}")
    
    # VIX-RV 상관관계
    vix_rv_corr = spy['VIX'].corr(spy['RV_22d'])
    print(f"\n📊 VIX-RV 상관관계: {vix_rv_corr:.4f}")
    print(f"   → {'높은 상관관계' if vix_rv_corr > 0.7 else '중간 상관관계' if vix_rv_corr > 0.5 else '낮은 상관관계'}")
    
    return correlations


def analyze_autocorrelation(spy):
    """3. 자기상관관계 분석"""
    print("\n" + "=" * 70)
    print("3. 자기상관관계 (Autocorrelation)")
    print("=" * 70)
    
    lags = [1, 2, 3, 5, 10, 22]
    
    print("\n📊 VRP 자기상관:")
    vrp_ac = {}
    for lag in lags:
        ac = spy['VRP'].autocorr(lag=lag)
        vrp_ac[lag] = ac
        bar = "█" * int(abs(ac) * 50)
        print(f"  Lag {lag:2d}: {ac:.4f} {bar}")
    
    print("\n📊 VRP_true 자기상관:")
    vrp_true_ac = {}
    for lag in lags:
        ac = spy['VRP_true'].autocorr(lag=lag)
        vrp_true_ac[lag] = ac
        bar = "█" * int(abs(ac) * 50)
        print(f"  Lag {lag:2d}: {ac:.4f} {bar}")
    
    print("\n📊 VIX 자기상관:")
    for lag in lags:
        ac = spy['VIX'].autocorr(lag=lag)
        bar = "█" * int(abs(ac) * 50)
        print(f"  Lag {lag:2d}: {ac:.4f} {bar}")
    
    # 시사점
    print("\n💡 시사점:")
    print(f"  • VRP는 lag 1에서 상관계수 {vrp_ac[1]:.3f} → 강한 지속성(Persistence)")
    print(f"  • VRP_true는 lag 1에서 상관계수 {vrp_true_ac.get(1, 0):.3f}")
    print(f"  • HAR 스타일 모델(lag 1, 5, 22)이 적합할 수 있음")
    
    return vrp_ac


def analyze_regime(spy):
    """4. Regime 분석"""
    print("\n" + "=" * 70)
    print("4. Regime 분석")
    print("=" * 70)
    
    # VIX 기반 Regime 분류
    spy['regime'] = pd.cut(spy['VIX'], 
                          bins=[0, 15, 20, 25, 35, 100],
                          labels=['Very Low', 'Low', 'Normal', 'High', 'Crisis'])
    
    print("\n📊 Regime 분포:")
    regime_counts = spy['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(spy) * 100
        bar = "█" * int(pct)
        print(f"  {regime:10s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("\n📊 Regime별 VRP 통계:")
    regime_vrp = spy.groupby('regime')['VRP'].agg(['mean', 'std', 'count'])
    print(regime_vrp.round(2).to_string())
    
    print("\n📊 Regime별 VRP_true 통계:")
    regime_vrp_true = spy.groupby('regime')['VRP_true'].agg(['mean', 'std', 'count'])
    print(regime_vrp_true.round(2).to_string())
    
    # Regime 전환 패턴
    spy['regime_change'] = (spy['regime'] != spy['regime'].shift(1)).astype(int)
    regime_changes = spy['regime_change'].sum()
    print(f"\n📊 Regime 전환 횟수: {regime_changes} (평균 {len(spy)/regime_changes:.1f}일마다)")
    
    return regime_counts


def analyze_seasonality(spy):
    """5. 계절성 분석"""
    print("\n" + "=" * 70)
    print("5. 시간적 패턴 분석")
    print("=" * 70)
    
    spy['month'] = spy.index.month
    spy['weekday'] = spy.index.weekday
    spy['year'] = spy.index.year
    
    print("\n📊 월별 VRP 평균:")
    monthly_vrp = spy.groupby('month')['VRP'].mean()
    for month, vrp in monthly_vrp.items():
        bar = "█" * int(abs(vrp))
        print(f"  {month:2d}월: {vrp:6.2f}% {bar}")
    
    print("\n📊 연도별 VRP 평균:")
    yearly_vrp = spy.groupby('year')['VRP'].agg(['mean', 'std'])
    print(yearly_vrp.round(2).to_string())
    
    print("\n📊 요일별 VRP 평균:")
    weekday_vrp = spy.groupby('weekday')['VRP'].mean()
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    for wd, vrp in weekday_vrp.items():
        print(f"  {weekday_names[wd]}: {vrp:6.2f}%")
    
    return monthly_vrp


def analyze_stationarity(spy):
    """6. 정상성 분석"""
    print("\n" + "=" * 70)
    print("6. 정상성 분석 (Stationarity)")
    print("=" * 70)
    
    from statsmodels.tsa.stattools import adfuller
    
    print("\n📊 Augmented Dickey-Fuller 검정:")
    for col in ['VIX', 'RV_22d', 'VRP', 'VRP_true']:
        result = adfuller(spy[col].dropna())
        stat, pvalue = result[0], result[1]
        stationary = "정상" if pvalue < 0.05 else "비정상"
        print(f"  {col:12s}: ADF={stat:.4f}, p={pvalue:.4f} → {stationary}")
    
    return None


def generate_recommendations(spy):
    """7. 모델링 권장사항"""
    print("\n" + "=" * 70)
    print("7. 모델링 권장사항")
    print("=" * 70)
    
    # 데이터 특성 기반 권장사항
    recommendations = {
        "특성 엔지니어링": [
            "HAR 스타일: RV(1d), RV(5d), RV(22d) - 자기상관이 강함",
            "VRP 래그: VRP(t-1), VRP(t-5) - 높은 지속성",
            "VIX 기간구조: VIX/VIX_MA20 - Contango/Backwardation",
            "Regime 더미: VIX 기준 고변동성/위기 구분"
        ],
        "전처리": [
            "표준화 (StandardScaler) - 선형 모델용",
            "로그 변환 불필요 - VRP는 이미 스프레드",
            "이상치 클리핑: 극단적 VRP 값 (상하위 1%)"
        ],
        "모델 선택": [
            "ElasticNet (현재 최고): 선형성이 강하고 해석가능",
            "HAR-X: HAR-RV + 외생변수 (참고문헌 기반)",
            "GARCH-MIDAS: 고빈도+저빈도 결합 (시간 있으면)",
            "Rolling 재학습: 63일마다 갱신"
        ],
        "주의사항": [
            "VIX-RV 상관관계가 높음 → VIX 자체가 좋은 예측자",
            "비선형 모델(RF, GB)은 과적합 경향",
            "미래 데이터 누출 주의 (shift 사용)"
        ]
    }
    
    for category, items in recommendations.items():
        print(f"\n📌 {category}:")
        for item in items:
            print(f"   • {item}")
    
    return recommendations


def main():
    print("\n" + "🔍" * 35)
    print("VRP 예측을 위한 데이터 탐색적 분석")
    print("🔍" * 35)
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    spy = load_data()
    print(f"  ✓ 데이터: {len(spy)} 행")
    print(f"  ✓ 기간: {spy.index[0].date()} ~ {spy.index[-1].date()}")
    
    # 분석 실행
    stats_df = analyze_basic_stats(spy)
    correlations = analyze_correlation(spy)
    autocorr = analyze_autocorrelation(spy)
    regimes = analyze_regime(spy)
    seasonality = analyze_seasonality(spy)
    
    try:
        analyze_stationarity(spy)
    except:
        print("  (statsmodels 없음 - 정상성 분석 생략)")
    
    recommendations = generate_recommendations(spy)
    
    # 결과 저장
    output = {
        'basic_stats': stats_df.to_dict(),
        'correlations': correlations,
        'autocorrelation': autocorr,
        'regime_distribution': regimes.to_dict(),
        'recommendations': recommendations,
        'data_info': {
            'n_samples': len(spy),
            'start_date': str(spy.index[0].date()),
            'end_date': str(spy.index[-1].date())
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/vrp_eda_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 결과 저장: paper/vrp_eda_results.json")


if __name__ == '__main__':
    main()
