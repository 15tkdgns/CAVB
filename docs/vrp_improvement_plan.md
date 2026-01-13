# VRP 예측 연구 개선 계획

# ========================

## 1. 데이터 특성 기반 핵심 발견

### 1.1 긍정적 특성

- VRP는 정상(Stationary) → 직접 모델링 가능
- VRP 자기상관 0.96 → HAR 스타일 모델 적합
- VRP 양수 비율 83% → 체계적 프리미엄 존재

### 1.2 문제점

- VRP_true(미래)와 현재 변수 상관관계 낮음 (최고 0.22)
- VIX-RV 상관 0.75 → VIX만으로 RV 대부분 설명됨
- 비정규 분포, 극단값 존재 (-46% ~ +23%)

---

## 2. 전처리 계획

### 2.1 타겟 재정의

```
기존: VRP_true = VIX - RV_future
문제: 예측하기 어려움 (상관관계 낮음)

대안 1: VRP 변화 예측
  target = VRP(t+1) - VRP(t)
  
대안 2: RV 직접 예측 후 VRP 계산
  target = RV_future
  VRP_pred = VIX - RV_pred
  
대안 3: VRP Regime 분류
  target = {high, normal, low} VRP
```

### 2.2 특성 엔지니어링

```python
# HAR 스타일 (높은 자기상관 활용)
RV_1d, RV_5d, RV_22d

# VRP 래그 (지속성 활용)
VRP_lag1, VRP_lag5, VRP_lag22

# VIX 기간구조
VIX_term = VIX / VIX_MA20  # Contango/Backwardation

# Regime 특성
regime_crisis = (VIX >= 35)
regime_high = (VIX >= 25)

# 차분 (변화량)
VRP_change = VRP.diff()
VIX_change = VIX.pct_change()
```

### 2.3 이상치 처리

```python
# 상하위 1% 윈저라이징
VRP_clipped = VRP.clip(lower=VRP.quantile(0.01), upper=VRP.quantile(0.99))
```

---

## 3. 모델 선택 (참고문헌 기반)

### 3.1 벤치마크: HAR-RV-X (Corsi, 2009)

```
RV_future = β0 + β1*RV_1d + β2*RV_5d + β3*RV_22d + β4*VIX + ε
```

- 장점: 해석 가능, 안정적
- 예상 R²: 0.10-0.15

### 3.2 주력: ElasticNet + 최적화

```
현재 R² = 0.15
목표 R² = 0.20+
방법:
  1. alpha/l1_ratio 세밀 튜닝
  2. 특성 선택 최적화
  3. 이상치 제거
```

### 3.3 고급: ARIMA-GARCH 하이브리드

```
VRP의 높은 자기상관 → AR 모델이 효과적일 수 있음
ARIMA(1,0,0) 또는 ARIMA(5,0,0)
```

### 3.4 딥러닝 (조건부)

```
LSTM: 충분한 데이터(10,000+ 샘플) 필요
현재 1,380 샘플 → 과적합 위험 높음
→ 현재는 권장하지 않음
```

---

## 4. 평가 지표

### 4.1 주요 지표

- R² (회귀 성능)
- 방향 정확도 (VRP > mean vs <= mean)
- RMSE, MAE

### 4.2 실무 지표

- 고VRP 예측 Precision (트레이딩용)
- Sharpe Ratio (백테스트)

---

## 5. 실험 순서

### Phase 1: 타겟 재정의 (10분)

1. VRP 변화 예측 테스트
2. RV 직접 예측 → VRP 계산

### Phase 2: 특성 최적화 (15분)

1. HAR-X 모델 구현
2. 이상치 제거 효과 검증
3. 특성 선택 (상위 10개)

### Phase 3: 모델 튜닝 (20분)

1. ElasticNet 세밀 튜닝
2. ARIMA 모델 테스트
3. 앙상블 구성

### Phase 4: 검증 (10분)

1. Bootstrap 신뢰구간
2. Rolling Window 검증

---

## 6. 예상 결과

| 접근법 | 예상 R² | 방향 정확도 |
|--------|---------|------------|
| 현재 (ElasticNet) | 0.15 | 71% |
| HAR-X | 0.18 | 72% |
| 이상치 제거 + 튜닝 | 0.20 | 75% |
| ARIMA | 0.20-0.25 | 75%+ |
| 최종 앙상블 | 0.22-0.25 | 76%+ |

---

## 7. 참고문헌

1. Corsi (2009) - HAR-RV 모델
2. Bollerslev et al. (2009) - VRP와 주식 수익률
3. Bekaert & Hoerova (2014) - VIX 분해
4. Ait-Sahalia et al. (2015) - 고빈도 변동성 예측
