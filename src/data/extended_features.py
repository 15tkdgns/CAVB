"""
Extended Feature Engineering for VRP Prediction
추가 피처: 거시경제, 기술적 지표, Cross-Asset
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def add_macro_features(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    거시경제 피처 추가
    - VIX Term Structure (VIX/VIX3M)
    - 금리 proxy (TLT 기반)
    - 달러 strength (UUP)
    """
    # VIX3M (3-month VIX) - VIX term structure
    try:
        vix3m = yf.download('^VIX3M', start=start_date, end=end_date, progress=False)['Close']
        if len(vix3m) > 0:
            vix3m = vix3m.reindex(df.index).ffill()
            df['VIX3M'] = vix3m
            df['VIX_term_structure'] = df['VIX'] / (df['VIX3M'] + 1e-6)
            # Contango/Backwardation indicator
            df['VIX_contango'] = (df['VIX_term_structure'] < 1).astype(int)
    except:
        df['VIX_term_structure'] = 1.0
        df['VIX_contango'] = 0
    
    # Treasury proxy using TLT returns volatility
    try:
        tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)['Close']
        if len(tlt) > 0:
            tlt = tlt.reindex(df.index).ffill()
            tlt_returns = tlt.pct_change()
            df['TLT_vol_5d'] = tlt_returns.rolling(5).std() * np.sqrt(252) * 100
            df['TLT_vol_22d'] = tlt_returns.rolling(22).std() * np.sqrt(252) * 100
            # Flight to quality indicator
            df['flight_to_quality'] = tlt_returns.rolling(5).mean() * 100
    except:
        df['TLT_vol_5d'] = 0
        df['TLT_vol_22d'] = 0
        df['flight_to_quality'] = 0
    
    # Dollar strength (UUP ETF)
    try:
        uup = yf.download('UUP', start=start_date, end=end_date, progress=False)['Close']
        if len(uup) > 0:
            uup = uup.reindex(df.index).ffill()
            df['USD_strength'] = uup.pct_change(22) * 100  # 1-month change
    except:
        df['USD_strength'] = 0
    
    return df


def add_technical_features(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    기술적 지표 추가
    - RSI (14)
    - MACD
    - Bollinger Band Width
    - ATR
    """
    if price_col not in df.columns:
        return df
    
    price = df[price_col]
    
    # RSI (14-day)
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Band Width
    sma_20 = price.rolling(20).mean()
    std_20 = price.rolling(20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    df['BB_width'] = (upper_band - lower_band) / (sma_20 + 1e-10) * 100
    df['BB_position'] = (price - lower_band) / (upper_band - lower_band + 1e-10)
    
    # Momentum
    df['momentum_10d'] = price.pct_change(10) * 100
    df['momentum_22d'] = price.pct_change(22) * 100
    
    return df


def add_cross_asset_features(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Cross-Asset 피처 추가
    - Gold-SPY Correlation (안전자산 수요)
    - Bond-Equity Correlation (Risk-on/Risk-off)
    - Sector Dispersion
    """
    # Gold (GLD)
    try:
        gld = yf.download('GLD', start=start_date, end=end_date, progress=False)['Close']
        gld = gld.reindex(df.index).ffill()
        gld_returns = gld.pct_change()
        
        if 'returns' in df.columns:
            # Rolling correlation with SPY
            df['gold_equity_corr'] = df['returns'].rolling(22).corr(gld_returns)
            # Gold outperformance (safe haven demand)
            df['gold_outperformance'] = (gld_returns.rolling(5).mean() - 
                                          df['returns'].rolling(5).mean()) * 100
    except:
        df['gold_equity_corr'] = 0
        df['gold_outperformance'] = 0
    
    # Bond-Equity correlation
    try:
        tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)['Close']
        tlt = tlt.reindex(df.index).ffill()
        tlt_returns = tlt.pct_change()
        
        if 'returns' in df.columns:
            df['bond_equity_corr'] = df['returns'].rolling(22).corr(tlt_returns)
            # Risk-on/Risk-off indicator
            df['risk_on_off'] = np.where(df['bond_equity_corr'] < -0.2, 1, 
                                         np.where(df['bond_equity_corr'] > 0.2, -1, 0))
    except:
        df['bond_equity_corr'] = 0
        df['risk_on_off'] = 0
    
    # Sector dispersion (using sector ETFs)
    sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI']  # Tech, Finance, Energy, Health, Industrial
    try:
        sector_returns = []
        for etf in sector_etfs:
            data = yf.download(etf, start=start_date, end=end_date, progress=False)['Close']
            data = data.reindex(df.index).ffill()
            sector_returns.append(data.pct_change())
        
        if sector_returns:
            sector_df = pd.concat(sector_returns, axis=1)
            df['sector_dispersion'] = sector_df.std(axis=1) * np.sqrt(252) * 100
    except:
        df['sector_dispersion'] = 0
    
    return df


def add_volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    변동성 레짐 피처
    - VIX percentile rank
    - Volatility regime (low/normal/high/extreme)
    - Regime duration
    """
    # VIX percentile (rolling 252-day)
    df['VIX_percentile'] = df['VIX'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5
    )
    
    # Volatility regime
    vix_20 = df['VIX'].rolling(252).quantile(0.2)
    vix_50 = df['VIX'].rolling(252).quantile(0.5)
    vix_80 = df['VIX'].rolling(252).quantile(0.8)
    
    df['vol_regime'] = np.select(
        [df['VIX'] <= vix_20, 
         (df['VIX'] > vix_20) & (df['VIX'] <= vix_50),
         (df['VIX'] > vix_50) & (df['VIX'] <= vix_80),
         df['VIX'] > vix_80],
        [0, 1, 2, 3],  # Low, Normal, High, Extreme
        default=1
    )
    
    # Days since regime change
    regime_change = df['vol_regime'].diff().ne(0).astype(int)
    df['regime_duration'] = regime_change.groupby(regime_change.cumsum()).cumcount() + 1
    
    return df


def add_all_extended_features(df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    모든 확장 피처 추가
    """
    if start_date is None:
        start_date = df.index.min().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = df.index.max().strftime('%Y-%m-%d')
    
    print("Adding macro features...")
    df = add_macro_features(df, start_date, end_date)
    
    print("Adding technical features...")
    if 'Close' in df.columns:
        df = add_technical_features(df, 'Close')
    
    print("Adding cross-asset features...")
    df = add_cross_asset_features(df, start_date, end_date)
    
    print("Adding volatility regime features...")
    df = add_volatility_regime_features(df)
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    return df


if __name__ == "__main__":
    # Test
    print("Extended features module loaded successfully!")
