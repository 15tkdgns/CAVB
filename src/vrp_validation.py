#!/usr/bin/env python3
"""
VRP 예측 모델 검증 및 추가 분석
================================

1. Bootstrap 신뢰구간
2. Regime별 성능 분석
3. 연도별 안정성
4. 특성 중요도
5. 방향 예측 정밀도/재현율
6. 트레이딩 시뮬레이션
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def load_data_and_train():
    """데이터 로드 및 모델 학습"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill().bfill()
    spy['returns'] = spy['Close'].pct_change()
    
    # 특성 생성
    spy['RV_1d'] = spy['returns'].abs() * np.sqrt(252) * 100
    spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
    spy['VIX_lag1'] = spy['VIX'].shift(1)
    spy['VIX_lag5'] = spy['VIX'].shift(5)
    spy['VIX_change'] = spy['VIX'].pct_change()
    spy['VRP_lag1'] = spy['VRP'].shift(1)
    spy['VRP_lag5'] = spy['VRP'].shift(5)
    spy['VRP_ma5'] = spy['VRP'].rolling(5).mean()
    spy['regime_high'] = (spy['VIX'] >= 25).astype(int)
    spy['return_5d'] = spy['returns'].rolling(5).sum()
    spy['return_22d'] = spy['returns'].rolling(22).sum()
    
    spy = spy.replace([np.inf, -np.inf], np.nan)
    spy = spy.dropna()
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    return spy, feature_cols


def validation_1_bootstrap(spy, feature_cols):
    """검증 1: Bootstrap 신뢰구간"""
    print("\n" + "=" * 60)
    print("[1/6] Bootstrap 신뢰구간")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    y_vrp = spy['VRP_true'].values
    vix = spy['VIX'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_rv_train, y_rv_test = y_rv[:split_idx], y_rv[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    vix_test = vix[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 모델 학습
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_rv_train)
    rv_pred = en.predict(X_test_s)
    vrp_pred = vix_test - rv_pred
    
    # Bootstrap
    n_bootstrap = 1000
    r2_scores = []
    
    for i in range(n_bootstrap):
        idx = np.random.choice(len(y_vrp_test), size=len(y_vrp_test), replace=True)
        r2 = r2_score(y_vrp_test[idx], vrp_pred[idx])
        r2_scores.append(r2)
    
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    ci_lower = np.percentile(r2_scores, 2.5)
    ci_upper = np.percentile(r2_scores, 97.5)
    
    print(f"\n  📊 VRP R² 통계:")
    print(f"     점추정: {r2_score(y_vrp_test, vrp_pred):.4f}")
    print(f"     Bootstrap 평균: {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"     95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'point_estimate': float(r2_score(y_vrp_test, vrp_pred)),
        'bootstrap_mean': float(r2_mean),
        'bootstrap_std': float(r2_std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }, vrp_pred, y_vrp_test, spy.index[split_idx:]


def validation_2_regime(spy, feature_cols, vrp_pred, y_vrp_test, test_dates):
    """검증 2: Regime별 성능"""
    print("\n" + "=" * 60)
    print("[2/6] Regime별 성능")
    print("=" * 60)
    
    split_idx = int(len(spy) * 0.8)
    vix_test = spy['VIX'].values[split_idx:]
    
    regimes = {
        'Low Vol (VIX<20)': vix_test < 20,
        'Normal (20≤VIX<25)': (vix_test >= 20) & (vix_test < 25),
        'High Vol (25≤VIX<35)': (vix_test >= 25) & (vix_test < 35),
        'Crisis (VIX≥35)': vix_test >= 35
    }
    
    results = {}
    print(f"\n  {'Regime':25s} | {'샘플':6s} | {'R²':8s} | {'방향정확도':10s}")
    print("  " + "-" * 60)
    
    for regime, mask in regimes.items():
        if mask.sum() >= 10:
            r2 = r2_score(y_vrp_test[mask], vrp_pred[mask])
            
            vrp_mean = y_vrp_test.mean()
            dir_acc = ((y_vrp_test[mask] > vrp_mean) == (vrp_pred[mask] > vrp_mean)).mean()
            
            print(f"  {regime:25s} | {mask.sum():6d} | {r2:8.4f} | {dir_acc*100:8.1f}%")
            results[regime] = {'r2': float(r2), 'n_samples': int(mask.sum()), 'direction_acc': float(dir_acc)}
        else:
            print(f"  {regime:25s} | {mask.sum():6d} | 샘플 부족")
    
    return results


def validation_3_yearly(spy, feature_cols, vrp_pred, y_vrp_test, test_dates):
    """검증 3: 연도별 안정성"""
    print("\n" + "=" * 60)
    print("[3/6] 연도별 안정성")
    print("=" * 60)
    
    test_df = pd.DataFrame({
        'actual': y_vrp_test,
        'pred': vrp_pred
    }, index=test_dates)
    
    test_df['year'] = test_df.index.year
    
    results = {}
    print(f"\n  {'연도':6s} | {'샘플':6s} | {'R²':8s} | {'방향정확도':10s}")
    print("  " + "-" * 40)
    
    for year in sorted(test_df['year'].unique()):
        mask = test_df['year'] == year
        if mask.sum() >= 10:
            r2 = r2_score(test_df.loc[mask, 'actual'], test_df.loc[mask, 'pred'])
            
            vrp_mean = y_vrp_test.mean()
            dir_acc = ((test_df.loc[mask, 'actual'] > vrp_mean) == (test_df.loc[mask, 'pred'] > vrp_mean)).mean()
            
            print(f"  {year:6d} | {mask.sum():6d} | {r2:8.4f} | {dir_acc*100:8.1f}%")
            results[year] = {'r2': float(r2), 'n_samples': int(mask.sum()), 'direction_acc': float(dir_acc)}
    
    return results


def validation_4_feature_importance(spy, feature_cols):
    """검증 4: 특성 중요도"""
    print("\n" + "=" * 60)
    print("[4/6] 특성 중요도")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train = X[:split_idx]
    y_train = y_rv[:split_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_train)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': np.abs(en.coef_)
    }).sort_values('coefficient', ascending=False)
    
    print(f"\n  📊 특성 중요도 (절대 계수):")
    for _, row in importance.iterrows():
        bar = "█" * int(row['coefficient'] * 10)
        print(f"     {row['feature']:15s}: {row['coefficient']:.4f} {bar}")
    
    return importance.to_dict('records')


def validation_5_direction_metrics(vrp_pred, y_vrp_test):
    """검증 5: 방향 예측 정밀도/재현율"""
    print("\n" + "=" * 60)
    print("[5/6] 방향 예측 메트릭")
    print("=" * 60)
    
    vrp_mean = y_vrp_test.mean()
    
    # 이진 분류: VRP > 평균
    y_actual = (y_vrp_test > vrp_mean).astype(int)
    y_pred = (vrp_pred > vrp_mean).astype(int)
    
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    
    print(f"\n  📊 이진 분류 (VRP > {vrp_mean:.2f}%):")
    print(f"     정확도:    {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"     정밀도:    {precision:.4f}")
    print(f"     재현율:    {recall:.4f}")
    print(f"     F1 Score:  {f1:.4f}")
    
    # 3분위 분류
    q33 = np.percentile(y_vrp_test, 33)
    q67 = np.percentile(y_vrp_test, 67)
    
    y_actual_3 = np.where(y_vrp_test < q33, 0, np.where(y_vrp_test < q67, 1, 2))
    y_pred_3 = np.where(vrp_pred < q33, 0, np.where(vrp_pred < q67, 1, 2))
    
    acc_3 = accuracy_score(y_actual_3, y_pred_3)
    print(f"\n  📊 3분위 분류:")
    print(f"     정확도:    {acc_3:.4f} ({acc_3*100:.1f}%)")
    
    return {
        'binary': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'tertile': {
            'accuracy': float(acc_3)
        }
    }


def validation_mlp_comparison(spy, feature_cols):
    """MLP vs ElasticNet 비교 검증"""
    print("\n" + "=" * 60)
    print("[MLP] 모델 성능 비교 (MLP vs ElasticNet)")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    y_vrp = spy['VRP_true'].values
    vix = spy['VIX'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_rv_train, y_rv_test = y_rv[:split_idx], y_rv[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    vix_test = vix[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    
    # ElasticNet
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_rv_train)
    rv_pred_en = en.predict(X_test_s)
    vrp_pred_en = vix_test - rv_pred_en
    r2_en = r2_score(y_vrp_test, vrp_pred_en)
    results['ElasticNet'] = r2_en
    print(f"\n  ElasticNet:    R² = {r2_en:.4f}")
    
    # MLP (64)
    mlp_64 = MLPRegressor(
        hidden_layer_sizes=(64,),
        max_iter=500,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_64.fit(X_train_s, y_rv_train)
    rv_pred_mlp64 = mlp_64.predict(X_test_s)
    vrp_pred_mlp64 = vix_test - rv_pred_mlp64
    r2_mlp64 = r2_score(y_vrp_test, vrp_pred_mlp64)
    results['MLP(64)'] = r2_mlp64
    print(f"  MLP(64):       R² = {r2_mlp64:.4f}")
    
    # MLP (128, 64)
    mlp_128_64 = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_128_64.fit(X_train_s, y_rv_train)
    rv_pred_mlp128 = mlp_128_64.predict(X_test_s)
    vrp_pred_mlp128 = vix_test - rv_pred_mlp128
    r2_mlp128 = r2_score(y_vrp_test, vrp_pred_mlp128)
    results['MLP(128,64)'] = r2_mlp128
    print(f"  MLP(128,64):   R² = {r2_mlp128:.4f}")
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\n  ✓ 최고 성능: {best_model[0]} (R² = {best_model[1]:.4f})")
    
    # 방향 정확도 비교
    vrp_mean = y_vrp_test.mean()
    dir_en = ((y_vrp_test > vrp_mean) == (vrp_pred_en > vrp_mean)).mean()
    dir_mlp64 = ((y_vrp_test > vrp_mean) == (vrp_pred_mlp64 > vrp_mean)).mean()
    dir_mlp128 = ((y_vrp_test > vrp_mean) == (vrp_pred_mlp128 > vrp_mean)).mean()
    
    print(f"\n  📊 방향 정확도:")
    print(f"     ElasticNet:  {dir_en*100:.1f}%")
    print(f"     MLP(64):     {dir_mlp64*100:.1f}%")
    print(f"     MLP(128,64): {dir_mlp128*100:.1f}%")
    
    return {
        'r2_scores': {k: float(v) for k, v in results.items()},
        'direction_accuracy': {
            'ElasticNet': float(dir_en),
            'MLP(64)': float(dir_mlp64),
            'MLP(128,64)': float(dir_mlp128)
        },
        'best_model': best_model[0],
        'best_r2': float(best_model[1])
    }

def validation_6_trading_simulation(vrp_pred, y_vrp_test):
    """검증 6: 트레이딩 시뮬레이션 (거래비용 및 Sharpe Ratio 포함)"""
    print("\n" + "=" * 60)
    print("[6/6] 트레이딩 시뮬레이션")
    print("=" * 60)
    
    vrp_mean = y_vrp_test.mean()
    
    # 전략: 예측 VRP > 평균이면 변동성 매도 (VRP 수취)
    positions = (vrp_pred > vrp_mean).astype(int)
    
    # 일별 수익률 (VRP를 일별 수익률로 변환: 22일 평균)
    daily_returns = y_vrp_test / 22 / 100  # 퍼센트를 소수로, 22일로 나눔
    strategy_returns = positions * daily_returns
    
    # 거래비용 반영 (0.1% per trade)
    transaction_cost = 0.001
    position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
    costs = position_changes * transaction_cost
    net_returns = strategy_returns - costs
    
    # 통계
    n_trades = positions.sum()
    total_return = net_returns.sum() * 100  # 퍼센트로 변환
    avg_return = net_returns[positions == 1].mean() * 100 if n_trades > 0 else 0
    win_rate = (net_returns[positions == 1] > 0).mean() if n_trades > 0 else 0
    
    # 올바른 Sharpe Ratio 계산 (연환산)
    annualized_return = net_returns.mean() * 252
    annualized_std = net_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_std if annualized_std > 0 else 0
    
    # Buy & Hold (항상 매도)
    bh_returns = daily_returns
    bh_total = bh_returns.sum() * 100
    bh_sharpe = (bh_returns.mean() * 252) / (bh_returns.std() * np.sqrt(252)) if bh_returns.std() > 0 else 0
    
    print(f"\n  📊 VRP 매도 전략 (예측 VRP > 평균시):")
    print(f"     거래 횟수:     {n_trades}/{len(positions)} ({n_trades/len(positions)*100:.1f}%)")
    print(f"     총 수익:       {total_return:.2f}%")
    print(f"     평균 일별수익: {avg_return:.4f}%")
    print(f"     승률:          {win_rate*100:.1f}%")
    print(f"     Sharpe Ratio:  {sharpe_ratio:.2f}")
    
    print(f"\n  📊 Buy & Hold (항상 매도):")
    print(f"     총 수익:       {bh_total:.2f}%")
    print(f"     Sharpe Ratio:  {bh_sharpe:.2f}")
    
    outperformance = sharpe_ratio - bh_sharpe
    print(f"\n  📊 전략 vs Buy & Hold:")
    print(f"     Sharpe 차이: {outperformance:+.2f}")
    
    return {
        'strategy': {
            'n_trades': int(n_trades),
            'total_return': float(total_return),
            'avg_return': float(avg_return),
            'win_rate': float(win_rate),
            'sharpe_ratio': float(sharpe_ratio)
        },
        'buy_hold': {
            'total_return': float(bh_total),
            'sharpe_ratio': float(bh_sharpe)
        },
        'sharpe_improvement': float(outperformance)
    }


def validation_7_deflated_sharpe(net_returns, n_trials=10):
    """
    검증 7: Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
    Multiple testing 보정을 통한 과적합 위험 평가
    """
    print("\n" + "=" * 60)
    print("[7] Deflated Sharpe Ratio (Multiple Testing 보정)")
    print("=" * 60)
    
    from scipy import stats
    
    # 기본 Sharpe Ratio
    annualized_return = net_returns.mean() * 252
    annualized_std = net_returns.std() * np.sqrt(252)
    observed_sharpe = annualized_return / annualized_std if annualized_std > 0 else 0
    
    # 수익률 분포 통계
    T = len(net_returns)  # 관측 수
    skewness = stats.skew(net_returns)
    kurtosis = stats.kurtosis(net_returns)
    
    # Expected Maximum Sharpe under null (E[SR*])
    # Approximation: E[SR*] ≈ sqrt(2 * log(n_trials)) * (1 - gamma) / sqrt(T)
    # where gamma ≈ 0.5772 (Euler-Mascheroni constant)
    gamma = 0.5772
    expected_max_sharpe = np.sqrt(2 * np.log(n_trials)) * (1 - gamma) / np.sqrt(T) * np.sqrt(252)
    
    # Standard deviation of SR estimate
    sr_std = np.sqrt((1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe 
                      + (kurtosis / 4) * observed_sharpe**2) / T) * np.sqrt(252)
    
    # Deflated Sharpe Ratio (DSR)
    # DSR = Prob(SR > SR*) where SR* is expected max under null
    z_score = (observed_sharpe - expected_max_sharpe) / sr_std if sr_std > 0 else 0
    dsr_pvalue = 1 - stats.norm.cdf(z_score)
    
    # Hair-cut Sharpe (conservative estimate)
    haircut_sharpe = observed_sharpe - expected_max_sharpe
    
    print(f"\n  📊 Sharpe Ratio 분석:")
    print(f"     관측 Sharpe:      {observed_sharpe:.4f}")
    print(f"     기대 최대 Sharpe: {expected_max_sharpe:.4f} (n={n_trials} 테스트)")
    print(f"     Hair-cut Sharpe:  {haircut_sharpe:.4f}")
    print(f"     DSR p-value:      {dsr_pvalue:.4f}")
    
    print(f"\n  📊 수익률 분포:")
    print(f"     왜도 (Skewness):  {skewness:.4f}")
    print(f"     첨도 (Kurtosis):  {kurtosis:.4f}")
    
    # 해석
    if dsr_pvalue < 0.05:
        interpretation = "✓ 통계적으로 유의 (과적합 위험 낮음)"
    else:
        interpretation = "⚠ 과적합 위험 있음 (더 많은 검증 필요)"
    print(f"\n  {interpretation}")
    
    return {
        'observed_sharpe': float(observed_sharpe),
        'expected_max_sharpe': float(expected_max_sharpe),
        'haircut_sharpe': float(haircut_sharpe),
        'dsr_pvalue': float(dsr_pvalue),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'n_trials': n_trials,
        'n_observations': T
    }


def validation_8_walk_forward(spy, feature_cols, n_splits=10):
    """
    검증 8: Walk-Forward Validation
    시간 순서를 유지하면서 여러 테스트 기간에서 검증
    """
    print("\n" + "=" * 60)
    print(f"[8] Walk-Forward Validation ({n_splits}개 기간)")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    y_vrp = spy['VRP_true'].values
    vix = spy['VIX'].values
    
    n = len(spy)
    fold_size = n // (n_splits + 1)
    train_size = fold_size * 2  # 최소 학습 크기
    
    results = []
    
    print(f"\n  {'기간':8s} | {'학습':8s} | {'테스트':8s} | {'R²':8s} | {'방향':8s}")
    print("  " + "-" * 50)
    
    for i in range(n_splits):
        train_end = train_size + i * fold_size
        test_start = train_end + 22  # 22일 Gap
        test_end = min(test_start + fold_size, n)
        
        if test_end <= test_start:
            break
        
        X_train = X[:train_end]
        y_train = y_rv[:train_end]
        X_test = X[test_start:test_end]
        y_vrp_test = y_vrp[test_start:test_end]
        vix_test = vix[test_start:test_end]
        
        if len(X_train) < 50 or len(X_test) < 10:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        rv_pred = en.predict(X_test_s)
        vrp_pred = vix_test - rv_pred
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        vrp_mean = y_vrp_test.mean()
        dir_acc = ((y_vrp_test > vrp_mean) == (vrp_pred > vrp_mean)).mean()
        
        results.append({
            'fold': i + 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'r2': r2,
            'direction_acc': dir_acc
        })
        
        print(f"  Fold {i+1:2d} | {len(X_train):8d} | {len(X_test):8d} | {r2:8.4f} | {dir_acc*100:6.1f}%")
    
    # 종합 통계
    r2_values = [r['r2'] for r in results]
    dir_values = [r['direction_acc'] for r in results]
    
    print(f"\n  📊 Walk-Forward 종합:")
    print(f"     평균 R²:        {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
    print(f"     평균 방향정확도: {np.mean(dir_values)*100:.1f}% ± {np.std(dir_values)*100:.1f}%")
    print(f"     최소/최대 R²:   [{min(r2_values):.4f}, {max(r2_values):.4f}]")
    
    return {
        'folds': results,
        'mean_r2': float(np.mean(r2_values)),
        'std_r2': float(np.std(r2_values)),
        'mean_dir_acc': float(np.mean(dir_values)),
        'std_dir_acc': float(np.std(dir_values))
    }


def validation_9_asset_iv_indices(spy_data=None):
    """
    검증 9: 자산별 IV 지수 활용 (GVZ, OVX)
    GLD → GVZ (Gold Volatility Index)
    Oil → OVX (Oil Volatility Index)
    """
    print("\n" + "=" * 60)
    print("[9] 자산별 IV 지수 활용 (GVZ, OVX)")
    print("=" * 60)
    
    results = {}
    
    assets = [
        ('GLD', 'GVZ', 'Gold'),
        ('USO', 'OVX', 'Oil'),
        ('SPY', '^VIX', 'S&P 500')
    ]
    
    print(f"\n  {'자산':12s} | {'IV지수':8s} | {'R²':8s} | {'방향':8s} | {'개선':8s}")
    print("  " + "-" * 60)
    
    for ticker, iv_ticker, name in assets:
        try:
            # 데이터 다운로드
            asset = yf.download(ticker, start='2015-01-01', end='2024-12-01', progress=False)
            iv = yf.download(iv_ticker if not iv_ticker.startswith('^') else iv_ticker, 
                           start='2015-01-01', end='2024-12-01', progress=False)
            
            if isinstance(asset.columns, pd.MultiIndex):
                asset.columns = asset.columns.get_level_values(0)
            if isinstance(iv.columns, pd.MultiIndex):
                iv.columns = iv.columns.get_level_values(0)
            
            if len(asset) < 500 or len(iv) < 500:
                print(f"  {name:12s} | 데이터 부족")
                continue
            
            # 데이터 준비
            df = asset[['Close']].copy()
            df['IV'] = iv['Close'].reindex(df.index).ffill().bfill()
            df['returns'] = df['Close'].pct_change()
            df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
            df['VRP'] = df['IV'] - df['RV_22d']
            df['RV_future'] = df['RV_22d'].shift(-22)
            df['VRP_true'] = df['IV'] - df['RV_future']
            
            # 특성
            df['IV_lag1'] = df['IV'].shift(1)
            df['VRP_lag1'] = df['VRP'].shift(1)
            df = df.dropna()
            
            if len(df) < 500:
                print(f"  {name:12s} | 데이터 부족 (처리 후)")
                continue
            
            # 학습/테스트
            X = df[['RV_22d', 'IV_lag1', 'VRP_lag1']].values
            y_rv = df['RV_future'].values
            y_vrp = df['VRP_true'].values
            iv_arr = df['IV'].values
            
            split = int(len(X) * 0.8)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X[:split])
            X_test_s = scaler.transform(X[split:])
            
            en = ElasticNet(alpha=0.01, random_state=SEED)
            en.fit(X_train_s, y_rv[:split])
            vrp_pred = iv_arr[split:] - en.predict(X_test_s)
            
            r2 = r2_score(y_vrp[split:], vrp_pred)
            vrp_mean = y_vrp[split:].mean()
            dir_acc = ((y_vrp[split:] > vrp_mean) == (vrp_pred > vrp_mean)).mean()
            
            # VIX 기반과 비교
            improvement = "N/A"
            if iv_ticker != '^VIX':
                # VIX로 같은 자산 예측
                vix = yf.download('^VIX', start='2015-01-01', end='2024-12-01', progress=False)
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                df_vix = asset[['Close']].copy()
                df_vix['IV'] = vix['Close'].reindex(df_vix.index).ffill().bfill()
                df_vix['returns'] = df_vix['Close'].pct_change()
                df_vix['RV_22d'] = df_vix['returns'].rolling(22).std() * np.sqrt(252) * 100
                df_vix['VRP'] = df_vix['IV'] - df_vix['RV_22d']
                df_vix['RV_future'] = df_vix['RV_22d'].shift(-22)
                df_vix['VRP_true'] = df_vix['IV'] - df_vix['RV_future']
                df_vix['IV_lag1'] = df_vix['IV'].shift(1)
                df_vix['VRP_lag1'] = df_vix['VRP'].shift(1)
                df_vix = df_vix.dropna()
                
                if len(df_vix) >= 500:
                    X_vix = df_vix[['RV_22d', 'IV_lag1', 'VRP_lag1']].values
                    y_rv_vix = df_vix['RV_future'].values
                    y_vrp_vix = df_vix['VRP_true'].values
                    iv_vix = df_vix['IV'].values
                    
                    split_vix = int(len(X_vix) * 0.8)
                    scaler_vix = StandardScaler()
                    X_train_vix = scaler_vix.fit_transform(X_vix[:split_vix])
                    X_test_vix = scaler_vix.transform(X_vix[split_vix:])
                    
                    en_vix = ElasticNet(alpha=0.01, random_state=SEED)
                    en_vix.fit(X_train_vix, y_rv_vix[:split_vix])
                    vrp_pred_vix = iv_vix[split_vix:] - en_vix.predict(X_test_vix)
                    r2_vix = r2_score(y_vrp_vix[split_vix:], vrp_pred_vix)
                    
                    improvement = f"+{(r2 - r2_vix)*100:.1f}%p"
            
            print(f"  {name:12s} | {iv_ticker:8s} | {r2:8.4f} | {dir_acc*100:6.1f}% | {improvement:8s}")
            
            results[name] = {
                'ticker': ticker,
                'iv_index': iv_ticker,
                'r2': float(r2),
                'direction_acc': float(dir_acc)
            }
            
        except Exception as e:
            print(f"  {name:12s} | 오류: {str(e)[:30]}")
    
    return results


def main():
    print("\n" + "🔬" * 30)
    print("VRP 예측 모델 검증 및 추가 분석")
    print("🔬" * 30)
    
    # 데이터 준비 및 예측
    print("\n데이터 준비...")
    spy, feature_cols = load_data_and_train()
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    # 검증 1: Bootstrap
    bootstrap_results, vrp_pred, y_vrp_test, test_dates = validation_1_bootstrap(spy, feature_cols)
    
    # 검증 2: Regime
    regime_results = validation_2_regime(spy, feature_cols, vrp_pred, y_vrp_test, test_dates)
    
    # 검증 3: 연도별
    yearly_results = validation_3_yearly(spy, feature_cols, vrp_pred, y_vrp_test, test_dates)
    
    # 검증 4: 특성 중요도
    importance_results = validation_4_feature_importance(spy, feature_cols)
    
    # 검증 5: 방향 메트릭
    direction_results = validation_5_direction_metrics(vrp_pred, y_vrp_test)
    
    # 검증 6: 트레이딩 (거래비용 반영)
    trading_results = validation_6_trading_simulation(vrp_pred, y_vrp_test)
    
    # 검증 7: Deflated Sharpe Ratio
    vrp_mean = y_vrp_test.mean()
    positions = (vrp_pred > vrp_mean).astype(int)
    daily_returns = y_vrp_test / 22 / 100
    strategy_returns = positions * daily_returns
    transaction_cost = 0.001
    position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
    costs = position_changes * transaction_cost
    net_returns = strategy_returns - costs
    deflated_sharpe_results = validation_7_deflated_sharpe(net_returns, n_trials=10)
    
    # 검증 8: Walk-Forward Validation
    walk_forward_results = validation_8_walk_forward(spy, feature_cols, n_splits=10)
    
    # 검증 9: 자산별 IV 지수 (선택적 - 네트워크 필요)
    try:
        iv_index_results = validation_9_asset_iv_indices()
    except Exception as e:
        print(f"\n  ⚠ IV 지수 검증 스킵: {e}")
        iv_index_results = {}
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 최종 요약")
    print("=" * 60)
    
    print(f"""
    🎯 VRP 예측 성능:
       R² = {bootstrap_results['point_estimate']:.4f}
       95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]
    
    📊 방향 예측:
       정확도 = {direction_results['binary']['accuracy']*100:.1f}%
       F1 Score = {direction_results['binary']['f1']:.4f}
    
    💰 트레이딩 성과:
       Sharpe Ratio: {trading_results['strategy']['sharpe_ratio']:.2f}
       Hair-cut Sharpe: {deflated_sharpe_results['haircut_sharpe']:.2f}
       승률: {trading_results['strategy']['win_rate']*100:.1f}%
    
    🔄 Walk-Forward:
       평균 R²: {walk_forward_results['mean_r2']:.4f} ± {walk_forward_results['std_r2']:.4f}
    """)
    
    # 저장
    output = {
        'bootstrap': bootstrap_results,
        'regime': regime_results,
        'yearly': {str(k): v for k, v in yearly_results.items()},
        'feature_importance': importance_results,
        'direction_metrics': direction_results,
        'trading': trading_results,
        'deflated_sharpe': deflated_sharpe_results,
        'walk_forward': walk_forward_results,
        'iv_indices': iv_index_results,
        'timestamp': datetime.now().isoformat()
    }
    
    Path('data/results').mkdir(parents=True, exist_ok=True)
    with open('data/results/vrp_validation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 결과 저장: data/results/vrp_validation_results.json")


if __name__ == '__main__':
    main()

