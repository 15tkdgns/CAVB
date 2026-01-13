"""
Cross-Asset RV Features (즉시 추가 가능)
기존 5개 자산 데이터로 계산
"""
import pandas as pd
import numpy as np


def add_cross_asset_features(all_assets_rv: dict) -> pd.DataFrame:
    """
    Cross-Asset RV 피처 생성
    
    Args:
        all_assets_rv: {'SPY': rv_series, 'GLD': rv_series, ...}
    
    Returns:
        Cross-asset features DataFrame
    """
    # 모든 RV를 DataFrame으로 결합
    rv_df = pd.DataFrame(all_assets_rv)
    
    # === Spreads (자산 간 괴리) ===
    features = {}
    
    # SPY-GLD (Flight-to-Quality)
    features['rv_spy_gld_spread'] = rv_df['SPY'] - rv_df['GLD']
    
    # SPY-TLT (주식-채권)
    features['rv_spy_tlt_spread'] = rv_df['SPY'] - rv_df['TLT']
    
    # EFA-EEM (선진-신흥)
    features['rv_efa_eem_spread'] = rv_df['EFA'] - rv_df['EEM']
    
    # === Aggregates (전체 시장) ===
    
    # Cross-Asset 평균
    features['rv_cross_mean'] = rv_df.mean(axis=1)
    
    # Cross-Asset 표준편차 (dispersion)
    features['rv_cross_std'] = rv_df.std(axis=1)
    
    # Cross-Asset 최대-최소 (range)
    features['rv_cross_range'] = rv_df.max(axis=1) - rv_df.min(axis=1)
    
    # === Correlation proxy ===
    
    # SPY와 다른 자산의 RV 상관 (rolling 22d)
    features['rv_spy_correlation'] = rv_df.corrwith(rv_df['SPY'], axis=1, method='pearson')
    
    return pd.DataFrame(features)


def add_downside_rv(returns: pd.Series, window: int = 22) -> pd.Series:
    """
    Downside RV (Bekaert & Wu 2000)
    """
    def calc_downside(x):
        negative = x[x < 0]
        if len(negative) > 0:
            return negative.std() * np.sqrt(252) * 100
        return 0
    
    downside_rv = returns.rolling(window).apply(calc_downside)
    return downside_rv.fillna(0)
