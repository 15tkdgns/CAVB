"""
Week 2: Regime-Adaptive Model
VIX 구간별로 별도 모델 학습 (변수 추가 X, 모델 분리 O)
"""
import sys
sys.path.insert(0, '../../..')

import numpy as np
import pandas as pd
from src.data import download_data, prepare_features, extract_features_and_target, three_way_split
from config.data_config import ASSETS, DATA_START_DATE, DATA_END_DATE
from config.model_config import ELASTIC_NET_PARAMS, SPLIT_RATIOS
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import json


class RegimeAdaptiveModel:
    """VIX 구간별 모델 분리"""
    
    def __init__(self, vix_low=15, vix_high=25):
        self.vix_low = vix_low
        self.vix_high = vix_high
        
        # 3개 모델
        self.model_low = ElasticNet(**ELASTIC_NET_PARAMS)
        self.model_mid = ElasticNet(**ELASTIC_NET_PARAMS)
        self.model_high = ElasticNet(**ELASTIC_NET_PARAMS)
        
        # Scaler
        self.scaler_low = RobustScaler()
        self.scaler_mid = RobustScaler()
        self.scaler_high = RobustScaler()
    
    def _detect_regime(self, vix_values):
        """VIX 값으로 regime 분류"""
        regimes = np.zeros(len(vix_values), dtype=int)
        regimes[vix_values < self.vix_low] = 0  # low
        regimes[(vix_values >= self.vix_low) & (vix_values < self.vix_high)] = 1  # mid
        regimes[vix_values >= self.vix_high] = 2  # high
        return regimes
    
    def fit(self, X, y, vix_values):
        """Regime별 모델 학습"""
        regimes = self._detect_regime(vix_values)
        
        # Low regime
        mask_low = (regimes == 0)
        if mask_low.sum() > 50:  # 최소 샘플 수
            X_low_s = self.scaler_low.fit_transform(X[mask_low])
            self.model_low.fit(X_low_s, y[mask_low])
            print(f"  Low Vol:  {mask_low.sum()} samples trained")
        
        # Mid regime
        mask_mid = (regimes == 1)
        if mask_mid.sum() > 50:
            X_mid_s = self.scaler_mid.fit_transform(X[mask_mid])
            self.model_mid.fit(X_mid_s, y[mask_mid])
            print(f"  Mid Vol:  {mask_mid.sum()} samples trained")
        
        # High regime
        mask_high = (regimes == 2)
        if mask_high.sum() > 50:
            X_high_s = self.scaler_high.fit_transform(X[mask_high])
            self.model_high.fit(X_high_s, y[mask_high])
            print(f"  High Vol: {mask_high.sum()} samples trained")
        
        return self
    
    def predict(self, X, vix_values):
        """Regime별 예측"""
        regimes = self._detect_regime(vix_values)
        predictions = np.zeros(len(X))
        
        # Low
        mask_low = (regimes == 0)
        if mask_low.sum() > 0:
            X_low_s = self.scaler_low.transform(X[mask_low])
            predictions[mask_low] = self.model_low.predict(X_low_s)
        
        # Mid
        mask_mid = (regimes == 1)
        if mask_mid.sum() > 0:
            X_mid_s = self.scaler_mid.transform(X[mask_mid])
            predictions[mask_mid] = self.model_mid.predict(X_mid_s)
        
        # High
        mask_high = (regimes == 2)
        if mask_high.sum() > 0:
            X_high_s = self.scaler_high.transform(X[mask_high])
            predictions[mask_high] = self.model_high.predict(X_high_s)
        
        return predictions


def run_regime_adaptive(ticker='GLD', horizon=5):
    """
    Regime-Adaptive Model 실험
    """
    print(f"\n{'='*60}")
    print(f"Regime-Adaptive Model: {ticker}")
    print(f"{'='*60}\n")
    
    # 데이터 로드
    asset = download_data(ticker, DATA_START_DATE, DATA_END_DATE)
    vix = download_data('^VIX', DATA_START_DATE, DATA_END_DATE)
    
    # 피처 준비 (Regime 변수 없이 기본만)
    df = prepare_features(asset, vix, horizon=horizon, include_regime=False)
    X, _, y_cavb = extract_features_and_target(df, horizon=horizon, include_regime=False)
    
    # VIX 값 추출 (Regime 판단용)
    vix_series = df['VIX'].values
    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(
        X, y_cavb,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['val'],
        gap=horizon
    )
    
    # VIX도 split
    vix_train = vix_series[:len(X_train)]
    vix_test = vix_series[len(X_train) + horizon + len(X_val) + horizon:len(X_train) + horizon + len(X_val) + horizon + len(X_test)]
    
    # Baseline (단일 모델)
    print("\nBaseline (Single Model):")
    scaler_base = RobustScaler()
    X_train_s = scaler_base.fit_transform(X_train)
    X_test_s = scaler_base.transform(X_test)
    
    model_base = ElasticNet(**ELASTIC_NET_PARAMS)
    model_base.fit(X_train_s, y_train)
    
    y_pred_base = model_base.predict(X_test_s)
    r2_base = r2_score(y_test, y_pred_base)
    mae_base = mean_absolute_error(y_test, y_pred_base)
    
    print(f"  Test R²:  {r2_base:.4f}")
    print(f"  Test MAE: {mae_base:.4f}")
    
    # Regime-Adaptive Model
    print("\nRegime-Adaptive Model:")
    model_adaptive = RegimeAdaptiveModel(vix_low=15, vix_high=25)
    model_adaptive.fit(X_train, y_train, vix_train)
    
    y_pred_adaptive = model_adaptive.predict(X_test, vix_test)
    r2_adaptive = r2_score(y_test, y_pred_adaptive)
    mae_adaptive = mean_absolute_error(y_test, y_pred_adaptive)
    
    print(f"\n  Test R²:  {r2_adaptive:.4f}")
    print(f"  Test MAE: {mae_adaptive:.4f}")
    
    # 개선도
    print("\n" + "="*60)
    print("IMPROVEMENT")
    print("="*60)
    print(f"ΔR²:  {r2_adaptive - r2_base:+.4f} ({(r2_adaptive/r2_base - 1)*100:+.2f}%)")
    print(f"ΔMAE: {mae_adaptive - mae_base:+.4f}")
    
    # Regime별 성능
    regimes_test = model_adaptive._detect_regime(vix_test)
    print("\n" + "="*60)
    print("Performance by Regime")
    print("="*60)
    
    for regime_id, regime_name in [(0, 'LOW'), (1, 'MID'), (2, 'HIGH')]:
        mask = (regimes_test == regime_id)
        if mask.sum() > 0:
            r2_regime = r2_score(y_test[mask], y_pred_adaptive[mask])
            mae_regime = mean_absolute_error(y_test[mask], y_pred_adaptive[mask])
            print(f"{regime_name} Vol ({mask.sum()} samples):")
            print(f"  R²:  {r2_regime:.4f}")
            print(f"  MAE: {mae_regime:.4f}")
    
    # 결과 저장
    results = {
        'ticker': ticker,
        'baseline': {'r2': float(r2_base), 'mae': float(mae_base)},
        'regime_adaptive': {'r2': float(r2_adaptive), 'mae': float(mae_adaptive)},
        'improvement': {
            'r2_delta': float(r2_adaptive - r2_base),
            'r2_pct': float((r2_adaptive/r2_base - 1)*100),
            'mae_delta': float(mae_adaptive - mae_base)
        }
    }
    
    return results


if __name__ == '__main__':
    # 모든 자산 분석
    all_results = {}
    
    for ticker in ASSETS.keys():
        try:
            results = run_regime_adaptive(ticker, horizon=5)
            all_results[ticker] = results
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 전체 결과 저장
    output_file = 'results/regime_adaptive_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 요약
    print("\n" + "="*60)
    print("SUMMARY - All Assets")
    print("="*60)
    
    summary_df = pd.DataFrame([
        {
            'Asset': ticker,
            'Baseline R²': res['baseline']['r2'],
            'Adaptive R²': res['regime_adaptive']['r2'],
            'ΔR²': res['improvement']['r2_delta'],
            'Improvement %': res['improvement']['r2_pct']
        }
        for ticker, res in all_results.items()
    ])
    
    print(summary_df.to_string(index=False))
    print(f"\nAverage Improvement: {summary_df['ΔR²'].mean():+.4f} ({summary_df['Improvement %'].mean():+.2f}%)")
    print(f"\nResults saved to {output_file}")
