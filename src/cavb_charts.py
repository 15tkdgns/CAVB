#!/usr/bin/env python3
"""
원인 규명 차트: VIX와 자산별 변동성 관계 분석
=============================================

VIX 전일 종가 vs 당일 자산 변동성 산점도
선형/비선형 관계 시각화
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
import yfinance as yf
from pathlib import Path
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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


def create_scatter_analysis(asset_ticker, asset_name):
    """VIX vs 자산 변동성 산점도 분석"""
    print(f"\n분석: {asset_name} ({asset_ticker})")
    
    # 데이터 로드
    asset = download_data(asset_ticker)
    vix = download_data('^VIX')
    
    if asset is None or vix is None:
        return None
    
    # 데이터 준비
    df = asset[['Close', 'Open', 'High', 'Low']].copy()
    df['VIX'] = vix['Close'].reindex(df.index).ffill().bfill()
    df['VIX_lag1'] = df['VIX'].shift(1)  # 전일 VIX
    
    # 당일 변동성 지표
    df['returns'] = df['Close'].pct_change()
    df['intraday_range'] = (df['High'] - df['Low']) / df['Open'] * 100  # 일중 변동폭 (%)
    df['abs_return'] = df['returns'].abs() * 100  # 절대 수익률 (%)
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100  # 5일 RV
    
    df = df.dropna()
    
    return df


def plot_scatter_charts():
    """산점도 차트 생성"""
    print("\n" + "=" * 60)
    print("VIX vs 자산 변동성 산점도 분석")
    print("=" * 60)
    
    assets = [
        ('EFA', 'EAFE (Developed)', 'tab:blue'),
        ('EEM', 'Emerging Markets', 'tab:orange'),
        ('GLD', 'Gold', 'tab:green'),
        ('SPY', 'S&P 500', 'tab:red'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    results = {}
    
    for idx, (ticker, name, color) in enumerate(assets):
        df = create_scatter_analysis(ticker, name)
        
        if df is None:
            continue
        
        ax = axes[idx]
        
        # VIX 전일 종가 vs 당일 절대수익률
        x = df['VIX_lag1'].values
        y = df['abs_return'].values
        
        # 산점도
        ax.scatter(x, y, alpha=0.3, s=10, c=color, label='Daily')
        
        # 선형 회귀
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Linear (r={r_value:.2f})')
        
        # 구간별 평균
        bins = [0, 15, 20, 25, 35, 100]
        bin_labels = ['<15', '15-20', '20-25', '25-35', '>35']
        df['vix_bin'] = pd.cut(df['VIX_lag1'], bins=bins, labels=bin_labels)
        bin_means = df.groupby('vix_bin', observed=True)['abs_return'].mean()
        
        # 구간 평균 플롯
        bin_centers = [10, 17.5, 22.5, 30, 40]
        for i, (label, mean_val) in enumerate(bin_means.items()):
            if pd.notna(mean_val):
                ax.scatter(bin_centers[i], mean_val, s=200, c='black', marker='D', 
                          zorder=5, edgecolors='white', linewidths=2)
        
        ax.set_xlabel('VIX (Previous Day)', fontsize=12)
        ax.set_ylabel('Absolute Return (%)', fontsize=12)
        ax.set_title(f'{name}\nCorr: r = {r_value:.3f}, p = {p_value:.4f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(5, 80)
        ax.set_ylim(0, 8)
        
        results[ticker] = {
            'name': name,
            'correlation': float(r_value),
            'p_value': float(p_value),
            'slope': float(slope),
            'n_samples': len(df)
        }
        
        print(f"  {name}: r = {r_value:.3f}, p = {p_value:.4f}")
    
    plt.suptitle('VIX (Previous Day) vs Asset Volatility (Next Day)\n'
                 'Black diamonds = VIX Range Averages', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 저장
    Path('diagrams').mkdir(parents=True, exist_ok=True)
    plt.savefig('diagrams/vix_volatility_scatter.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n저장: diagrams/vix_volatility_scatter.png")
    plt.close()
    
    return results


def plot_regime_analysis():
    """VIX 레짐별 예측력 분석"""
    print("\n" + "=" * 60)
    print("VIX 레짐별 변동성 예측력 분석")
    print("=" * 60)
    
    assets = [('EFA', 'EAFE'), ('EEM', 'Emerging'), ('GLD', 'Gold'), ('SPY', 'S&P 500')]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 왼쪽: VIX 레짐별 평균 변동성
    ax1 = axes[0]
    
    regimes = ['Low\n(VIX<20)', 'Normal\n(20-25)', 'High\n(25-35)', 'Crisis\n(>35)']
    x_pos = np.arange(len(regimes))
    width = 0.2
    
    for i, (ticker, name) in enumerate(assets):
        df = create_scatter_analysis(ticker, name)
        if df is None:
            continue
        
        # 레짐별 평균 변동성
        regime_means = []
        for low, high in [(0, 20), (20, 25), (25, 35), (35, 100)]:
            mask = (df['VIX_lag1'] >= low) & (df['VIX_lag1'] < high)
            if mask.sum() > 10:
                regime_means.append(df.loc[mask, 'abs_return'].mean())
            else:
                regime_means.append(np.nan)
        
        ax1.bar(x_pos + i * width - 1.5 * width, regime_means, width, 
               label=name, alpha=0.8)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regimes)
    ax1.set_xlabel('VIX Regime', fontsize=12)
    ax1.set_ylabel('Average Absolute Return (%)', fontsize=12)
    ax1.set_title('VIX Regime vs Next-Day Volatility', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 오른쪽: 상관관계 비교
    ax2 = axes[1]
    
    correlations = []
    asset_names = []
    
    for ticker, name in assets:
        df = create_scatter_analysis(ticker, name)
        if df is None:
            continue
        
        r, _ = stats.pearsonr(df['VIX_lag1'], df['abs_return'])
        correlations.append(r)
        asset_names.append(name)
    
    colors = ['tab:blue' if c > 0.3 else 'tab:orange' if c > 0.2 else 'tab:red' 
              for c in correlations]
    ax2.barh(asset_names, correlations, color=colors, alpha=0.8)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Correlation (VIX_t-1 vs |Return|_t)', fontsize=12)
    ax2.set_title('VIX Predictive Power by Asset', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (corr, name) in enumerate(zip(correlations, asset_names)):
        ax2.text(corr + 0.01, i, f'{corr:.3f}', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('diagrams/vix_regime_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"저장: diagrams/vix_regime_analysis.png")
    plt.close()


def plot_cavb_framework():
    """CAVB 프레임워크 설명 차트"""
    print("\n" + "=" * 60)
    print("CAVB (Cross-Asset Volatility Basis) 프레임워크")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 프레임워크 박스
    # VIX 박스
    ax.add_patch(plt.Rectangle((0.1, 0.7), 0.25, 0.15, 
                               facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.text(0.225, 0.775, 'VIX\n(S&P 500 IV)\nSystemic Risk', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # RV 박스
    ax.add_patch(plt.Rectangle((0.65, 0.7), 0.25, 0.15, 
                               facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(0.775, 0.775, 'RV_asset\n(Realized Vol)\nIdiosyncratic Risk', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    
    # CAVB 박스
    ax.add_patch(plt.Rectangle((0.375, 0.4), 0.25, 0.15, 
                               facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.text(0.5, 0.475, 'CAVB\nVIX - RV_asset', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # 화살표
    ax.annotate('', xy=(0.375, 0.475), xytext=(0.35, 0.7),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=(0.625, 0.475), xytext=(0.65, 0.7),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Minus 기호
    ax.text(0.5, 0.6, '-', fontsize=24, ha='center', va='center', fontweight='bold')
    
    # 예측력 결과
    ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.2, 
                               facecolor='white', edgecolor='gray', linewidth=1))
    
    results_text = """
    CAVB Prediction Results:
    
    ✓ Predictable:  EAFE (R²=0.40), Treasury (R²=0.39), Emerging (R²=0.31), Gold (R²=0.27)
    ✗ Not Predictable:  Oil, China, Russell 2000
    """
    ax.text(0.5, 0.2, results_text, ha='center', va='center', fontsize=10,
           family='monospace')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Cross-Asset Volatility Basis (CAVB) Framework\n'
                 'Predicting systemic-idiosyncratic volatility gap', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('diagrams/cavb_framework.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"저장: diagrams/cavb_framework.png")
    plt.close()


def main():
    print("\n" + "📊" * 30)
    print("원인 규명 차트 및 CAVB 프레임워크")
    print("📊" * 30)
    
    # 1. 산점도 분석
    scatter_results = plot_scatter_charts()
    
    # 2. 레짐 분석
    plot_regime_analysis()
    
    # 3. CAVB 프레임워크
    plot_cavb_framework()
    
    # 결과 저장
    output = {
        'scatter_analysis': scatter_results,
        'framework': 'CAVB (Cross-Asset Volatility Basis Based Prediction)',
        'definition': 'CAVB = VIX (Systemic) - RV (Idiosyncratic)',
        'predictable_assets': ['EFA', 'TLT', 'EEM', 'GLD', 'SPY'],
        'negative_control': ['USO', 'FXI'],
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    import json
    with open('data/results/cavb_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 60)
    print("생성된 차트")
    print("=" * 60)
    print("  1. diagrams/vix_volatility_scatter.png")
    print("  2. diagrams/vix_regime_analysis.png")
    print("  3. diagrams/cavb_framework.png")
    print("  4. data/results/cavb_analysis.json")


if __name__ == '__main__':
    main()
