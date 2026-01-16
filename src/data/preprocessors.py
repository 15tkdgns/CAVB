"""
Enhanced preprocessors with 25 features
9 기본 + 16 추가 = 25개 변수
"""
import pandas as pd
import numpy as np
from typing import Tuple

def calculate_realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """실현 변동성 계산"""
    return returns.rolling(window).std() * np.sqrt(252) * 100


def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Market Regime Indicator 추가"""
    df['regime'] = pd.cut(
        df['VIX'], 
        bins=[0, 15, 25, 100], 
        labels=['low', 'mid', 'high']
    )
    df['regime_low'] = (df['regime'] == 'low').astype(float)
    df['regime_high'] = (df['regime'] == 'high').astype(float)
    df['VIX_slope'] = df['VIX'].pct_change(5).fillna(0)
    
    return df


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    16개 추가 피처 생성
    """
    # === RV 확장 (4개) ===
    df['RV_10d'] = calculate_realized_volatility(df['returns'], 10)
    df['RV_std_22d'] = df['RV_22d'].rolling(22).std().fillna(0)
    df['RV_momentum'] = ((df['RV_5d'] - df['RV_22d']) / (df['RV_22d'] + 1e-6)).fillna(0)
    df['RV_acceleration'] = (df['RV_1d'] - 2*df['RV_5d'] + df['RV_22d']).fillna(0)
    
    # === VIX 확장 (5개) ===
    df['VIX_lag10'] = df['VIX'].shift(10).fillna(method='bfill')
    df['VIX_lag22'] = df['VIX'].shift(22).fillna(method='bfill')
    df['VIX_ma5'] = df['VIX'].rolling(5).mean().fillna(df['VIX'])
    df['VIX_ma22'] = df['VIX'].rolling(22).mean().fillna(df['VIX'])
    df['VIX_zscore'] = ((df['VIX'] - df['VIX'].rolling(60).mean()) / 
                         (df['VIX'].rolling(60).std() + 1e-6)).fillna(0)
    
    # === CAVB 파생 (4개) ===
    df['CAVB_percentile'] = df['CAVB'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    ).fillna(0.5)
    df['CAVB_std_22d'] = df['CAVB'].rolling(22).std().fillna(0)
    df['CAVB_max_22d'] = df['CAVB'].rolling(22).max().fillna(df['CAVB'])
    df['CAVB_min_22d'] = df['CAVB'].rolling(22).min().fillna(df['CAVB'])
    
    # === Cross-term (3개) ===
    df['RV_VIX_ratio'] = (df['RV_22d'] / (df['VIX'] + 1e-6)).fillna(0)
    df['RV_VIX_product'] = (df['RV_22d'] * df['VIX']).fillna(0)
    df['CAVB_VIX_ratio'] = (df['CAVB'] / (df['VIX'] + 1e-6)).fillna(0)
    
    return df


def prepare_features(asset_data: pd.DataFrame, vix_data: pd.DataFrame, 
                    horizon: int = 5, include_regime: bool = False,
                    include_enhanced: bool = False) -> pd.DataFrame:
    """
    예측 변수 생성 (Enhanced 옵션 추가)
    
    Args:
        asset_data: 자산 가격 데이터
        vix_data: VIX 데이터
        horizon: 예측 시계
        include_regime: Regime indicator 포함 여부
        include_enhanced: Enhanced features (16개) 포함 여부
    
    Returns:
        피처가 포함된 DataFrame
    """
    df = asset_data[['Close']].copy()
    df.columns = ['Price']
    
    # VIX 정렬
    df['VIX'] = vix_data['Close'].reindex(df.index).ffill().bfill()
    
    # 수익률
    df['returns'] = df['Price'].pct_change()
    
    # 기본 실현 변동성
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    df['RV_5d'] = calculate_realized_volatility(df['returns'], 5)
    df['RV_22d'] = calculate_realized_volatility(df['returns'], 22)
    
    # CAVB
    df['CAVB'] = df['VIX'] - df['RV_22d']
    
    # 타겟 변수
    df[f'RV_future_{horizon}d'] = df['RV_22d'].shift(-horizon)
    df[f'CAVB_target_{horizon}d'] = df['VIX'] - df[f'RV_future_{horizon}d']
    
    # 기본 예측 변수 (9개)
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['CAVB_lag1'] = df['CAVB'].shift(1)
    df['CAVB_lag5'] = df['CAVB'].shift(5)
    df['CAVB_ma5'] = df['CAVB'].rolling(5).mean()
    
    # Enhanced features (옵션)
    if include_enhanced:
        df = add_enhanced_features(df)
    
    # Regime features (옵션)
    if include_regime:
        df = add_regime_indicators(df)
    
    return df.dropna()


def extract_features_and_target(df: pd.DataFrame, horizon: int = 5, 
                               include_regime: bool = False,
                               include_enhanced: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X, y_rv, y_cavb 추출 (Enhanced 옵션)
    
    Args:
        df: prepare_features 결과
        horizon: 예측 시계
        include_regime: Regime features 포함 여부
        include_enhanced: Enhanced features 포함 여부
    
    Returns:
        (X, y_rv, y_cavb) 튜플
    """
    # 기본 9개
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                    'VIX_change', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
    
    # Enhanced 16개
    if include_enhanced:
        feature_cols += [
            # RV 확장 (4)
            'RV_10d', 'RV_std_22d', 'RV_momentum', 'RV_acceleration',
            # VIX 확장 (5)
            'VIX_lag10', 'VIX_lag22', 'VIX_ma5', 'VIX_ma22', 'VIX_zscore',
            # CAVB 파생 (4)
            'CAVB_percentile', 'CAVB_std_22d', 'CAVB_max_22d', 'CAVB_min_22d',
            # Cross-term (3)
            'RV_VIX_ratio', 'RV_VIX_product', 'CAVB_VIX_ratio'
        ]
    
    # Regime 3개
    if include_regime:
        feature_cols += ['regime_low', 'regime_high', 'VIX_slope']
    
    X = df[feature_cols].values
    y_rv = df[f'RV_future_{horizon}d'].values
    y_cavb = df[f'CAVB_target_{horizon}d'].values
    
    return X, y_rv, y_cavb


def get_feature_names(include_enhanced: bool = False, include_regime: bool = False) -> list:
    """피처 이름 리스트 반환"""
    features = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5',
                'VIX_change', 'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5']
    
    if include_enhanced:
        features += [
            'RV_10d', 'RV_std_22d', 'RV_momentum', 'RV_acceleration',
            'VIX_lag10', 'VIX_lag22', 'VIX_ma5', 'VIX_ma22', 'VIX_zscore',
            'CAVB_percentile', 'CAVB_std_22d', 'CAVB_max_22d', 'CAVB_min_22d',
            'RV_VIX_ratio', 'RV_VIX_product', 'CAVB_VIX_ratio'
        ]
    
    if include_regime:
        features += ['regime_low', 'regime_high', 'VIX_slope']
    
    return features
