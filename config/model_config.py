"""
프로젝트 설정 - 모델 하이퍼파라미터
"""

# ElasticNet 파라미터
ELASTIC_NET_PARAMS = {
    'alpha': 0.01,
    'l1_ratio': 0.7,
    'max_iter': 10000,
    'random_state': 42,
    'selection': 'random'
}

# RobustScaler 파라미터
SCALER_PARAMS = {
    'quantile_range': (25.0, 75.0),
    'with_centering': True,
    'with_scaling': True
}

# Train/Val/Test Split 비율
SPLIT_RATIOS = {
    'train': 0.6,
    'val': 0.2,
    'test': 0.2
}

# Gap (전방 편향 방지)
FORECAST_GAPS = {
    '1d': 1,
    '5d': 5,
    '22d': 22
}

# 피처 이름
FEATURE_COLS = [
    'RV_1d', 'RV_5d', 'RV_22d',
    'VIX_lag1', 'VIX_lag5', 'VIX_change',
    'CAVB_lag1', 'CAVB_lag5', 'CAVB_ma5'
]
