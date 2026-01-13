"""
Novel Features from Academic Literature
기발하고 창의적인 변수들
"""
import pandas as pd
import numpy as np
from scipy import stats


def realized_skewness(returns: pd.Series, window: int = 22) -> pd.Series:
    """
    Realized Skewness (Neuberger 2012, Amaya et al. 2015)
    
    비대칭 위험 측정:
    - Skew < 0: 왼쪽 꼬리 위험 (큰 손실 가능성)
    - Skew > 0: 오른쪽 왼쪽 (큰 수익 가능성)
    
    VRP와 관계: 음의 skew일 때 VRP 높음
    """
    def calc_skew(x):
        if len(x) > 3:
            return stats.skew(x)
        return 0
    
    rs = returns.rolling(window).apply(calc_skew)
    return rs.fillna(0)


def realized_kurtosis(returns: pd.Series, window: int = 22) -> pd.Series:
    """
    Realized Kurtosis (Amaya et al. 2015)
    
    꼬리 두께 측정:
    - Kurtosis > 3: Fat tails (극단 사건 많음)
    - Kurtosis = 3: Normal distribution
    
    VRP 예측력: Kurt ↑ → VRP ↑
    """
    def calc_kurt(x):
        if len(x) > 4:
            return stats.kurtosis(x, fisher=False)  # Pearson kurtosis
        return 3
    
    rk = returns.rolling(window).apply(calc_kurt)
    return rk.fillna(3)


def good_bad_volatility(returns: pd.Series, window: int = 22) -> tuple:
    """
    Good/Bad Volatility (Segal et al. 2015)
    
    개념:
    - Good Vol: 상승 움직임의 변동성
    - Bad Vol: 하락 움직임의 변동성
    
    투자자는 Bad Vol에 더 민감 → VRP에 비대칭 영향
    """
    def calc_good_bad(x):
        positive = x[x > 0]
        negative = x[x < 0]
        
        good_vol = positive.std() * np.sqrt(252) * 100 if len(positive) > 1 else 0
        bad_vol = abs(negative.std()) * np.sqrt(252) * 100 if len(negative) > 1 else 0
        
        return good_vol, bad_vol
    
    results = returns.rolling(window).apply(
        lambda x: calc_good_bad(pd.Series(x))[0], raw=False
    )
    
    good_vol = returns.rolling(window).apply(
        lambda x: x[x > 0].std() * np.sqrt(252) * 100 if len(x[x > 0]) > 1 else 0
    ).fillna(0)
    
    bad_vol = returns.rolling(window).apply(
        lambda x: abs(x[x < 0].std()) * np.sqrt(252) * 100 if len(x[x < 0]) > 1 else 0
    ).fillna(0)
    
    return good_vol, bad_vol


def variance_risk_premium_decomposition(vix: pd.Series, rv_22d: pd.Series) -> dict:
    """
    VRP Decomposition (Bollerslev et al. 2009, Drechsler & Yaron 2011)
    
    VRP = E[RV] - IV
    
    분해:
    1. Persistent component (장기)
    2. Transitory component (단기)
    """
    # VRP
    vrp = vix - rv_22d
    
    # Persistent (60일 이동평균)
    vrp_persistent = vrp.rolling(60).mean().fillna(vrp)
    
    # Transitory (잔차)
    vrp_transitory = vrp - vrp_persistent
    
    # Variance ratio (persistent/transitory)
    vrp_variance_ratio = (vrp_persistent.rolling(22).std() / 
                          (vrp_transitory.rolling(22).std() + 1e-6)).fillna(1)
    
    return {
        'VRP_persistent': vrp_persistent,
        'VRP_transitory': vrp_transitory,
        'VRP_variance_ratio': vrp_variance_ratio
    }


def tail_risk_hedging_demand(vix: pd.Series, rv_22d: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Tail Risk Hedging Demand (Kelly et al. 2016)
    
    개념: 투자자들이 꼬리 위험을 헤지하려는 수요
    
    Proxy:
    - VIX가 높지만 realized vol이 낮을 때
    - 큰 손실 이후 급등하는 VIX
    """
    # VIX spike after negative returns
    negative_shock = (returns < returns.rolling(60).quantile(0.1)).astype(float)
    vix_spike = (vix - vix.rolling(22).mean()) / (vix.rolling(22).std() + 1e-6)
    
    tail_demand = negative_shock.shift(1) * vix_spike
    return tail_demand.fillna(0)


def leverage_effect_proxy(returns: pd.Series, rv: pd.Series) -> pd.Series:
    """
    Leverage Effect (Black 1976, Christie 1982)
    
    개념: 
    - 주가 하락 → 레버리지 증가 → 변동성 증가
    - 비대칭적 관계
    
    Proxy: Correlation(returns_t, RV_t+1)
    """
    # Rolling correlation between returns and future RV
    window = 22
    
    def calc_corr(ret_series):
        if len(ret_series) < 5:
            return 0
        rv_future = rv.loc[ret_series.index].shift(-1)
        return ret_series.corr(rv_future) if len(rv_future.dropna()) > 0 else 0
    
    leverage = returns.rolling(window).apply(
        lambda x: pd.Series(x).corr(rv.shift(-1).loc[pd.Series(x).index]) 
        if len(x) > 5 else 0,
        raw=False
    ).fillna(0)
    
    return leverage


def vix_term_structure_slope(vix: pd.Series) -> pd.Series:
    """
    VIX Term Structure Slope (Mixon 2007)
    
    이론:
    - Contango (slope > 0): 정상, 낮은 현재 위험
    - Backwardation (slope < 0): 스트레스, 즉각적 위험
    
    Proxy: VIX의 1일 vs 5일 vs 22일 변화율 차이
    """
    vix_1d_change = vix.pct_change(1)
    vix_5d_change = vix.pct_change(5)
    vix_22d_change = vix.pct_change(22)
    
    # Short-term slope
    slope_short = vix_5d_change - vix_1d_change
    
    # Long-term slope  
    slope_long = vix_22d_change - vix_5d_change
    
    # Curvature
    curvature = slope_long - slope_short
    
    return slope_short.fillna(0), slope_long.fillna(0), curvature.fillna(0)


def realized_correlation_rv_vix(rv: pd.Series, vix: pd.Series, window: int = 22) -> pd.Series:
    """
    Realized Correlation between RV and VIX (Buss & Vilkov 2012)
    
    개념:
    - RV ↑, VIX ↑: 정상적 동조
    - RV ↓, VIX ↑: 공포 (미래 위험 예상)
    - Correlation < 0: 역설적 상황
    """
    def calc_corr(rv_window, vix_window):
        if len(rv_window) > 3 and len(vix_window) > 3:
            return np.corrcoef(rv_window, vix_window)[0, 1]
        return 0
    
    # Rolling correlation
    corr = pd.Series(index=rv.index, dtype=float)
    for i in range(window, len(rv)):
        rv_win = rv.iloc[i-window:i].values
        vix_win = vix.iloc[i-window:i].values
        corr.iloc[i] = calc_corr(rv_win, vix_win)
    
    return corr.fillna(0)


def information_discreteness(returns: pd.Series, window: int = 22) -> pd.Series:
    """
    Information Discreteness (Jiang & Oomen 2008)
    
    개념: 
    - 뉴스/정보가 불연속적으로 도착
    - 연속 변동 vs 점프 분리
    
    Proxy: Realized Range / RV
    - Range가 크면 jumps 많음
    """
    rv = returns.rolling(window).std() * np.sqrt(252) * 100
    realized_range = (returns.rolling(window).max() - 
                     returns.rolling(window).min()) * np.sqrt(252) * 100
    
    discreteness = (realized_range / (rv + 1e-6)).fillna(1)
    return discreteness


def add_novel_features(df: pd.DataFrame) -> pd.DataFrame:
    """모든 기발한 변수 추가"""
    
    # Higher moments
    df['RS_skew'] = realized_skewness(df['returns'], 22)
    df['RK_kurt'] = realized_kurtosis(df['returns'], 22)
    
    # Good/Bad volatility
    good_vol, bad_vol = good_bad_volatility(df['returns'], 22)
    df['good_volatility'] = good_vol
    df['bad_volatility'] = bad_vol
    df['bad_good_ratio'] = (bad_vol / (good_vol + 1e-6)).fillna(1)
    
    # VRP decomposition
    vrp_decomp = variance_risk_premium_decomposition(df['VIX'], df['RV_22d'])
    df['VRP_persistent'] = vrp_decomp['VRP_persistent']
    df['VRP_transitory'] = vrp_decomp['VRP_transitory']
    df['VRP_variance_ratio'] = vrp_decomp['VRP_variance_ratio']
    
    # Tail risk
    df['tail_risk_demand'] = tail_risk_hedging_demand(df['VIX'], df['RV_22d'], df['returns'])
    
    # VIX term structure
    slope_short, slope_long, curvature = vix_term_structure_slope(df['VIX'])
    df['VIX_slope_short'] = slope_short
    df['VIX_slope_long'] = slope_long
    df['VIX_curvature'] = curvature
    
    # RV-VIX correlation
    df['RV_VIX_correlation'] = realized_correlation_rv_vix(df['RV_22d'], df['VIX'], 22)
    
    # Information discreteness
    df['info_discreteness'] = information_discreteness(df['returns'], 22)
    
    return df
