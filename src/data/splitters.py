"""
데이터 분할 - Train/Val/Test Split
"""
import numpy as np
from typing import Tuple

def three_way_split(X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.6, val_ratio: float = 0.2,
                   gap: int = 5) -> Tuple:
    """
    3-Way Split with Gap
    
    Args:
        X: 피처 배열
        y: 타겟 배열
        train_ratio: 학습셋 비율
        val_ratio: 검증셋 비율
        gap: 분할 간 간격 (전방 편향 방지)
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    
    # 분할 지점 계산
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split
    X_train = X[:train_end]
    X_val = X[train_end + gap:val_end]
    X_test = X[val_end + gap:]
    
    y_train = y[:train_end]
    y_val = y[train_end + gap:val_end]
    y_test = y[val_end + gap:]
    
    print(f"Split completed:")
    print(f"  Train: {len(X_train)} samples (0 - {train_end})")
    print(f"  Val:   {len(X_val)} samples ({train_end + gap} - {val_end})")
    print(f"  Test:  {len(X_test)} samples ({val_end + gap} - {n})")
    print(f"  Gap:   {gap} days")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_non_overlapping_test(X: np.ndarray, y: np.ndarray, interval: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-overlapping 테스트셋 생성
    
    Args:
        X: 피처 배열
        y: 타겟 배열
        interval: 샘플링 간격
    
    Returns:
        (X_non_overlap, y_non_overlap)
    """
    indices = np.arange(0, len(X), interval)
    return X[indices], y[indices]
