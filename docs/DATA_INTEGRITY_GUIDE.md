# 데이터 무결성 가이드

> 금융 ML에서 반드시 준수해야 할 무결성 규칙

---

## 3대 금기사항

### 1. 데이터 하드코딩 금지

**잘못된 예**:
```python
# ❌ 절대 금지
returns = [0.01, 0.02, -0.01, 0.03, ...]
prices = pd.DataFrame({'Close': [400, 401, 402, ...]})
```

**올바른 예**:
```python
# ✅ 올바른 방법
import yfinance as yf
spy = yf.download('SPY', start='2015-01-01', end='2024-12-31')
```

---

### 2. Random 데이터 생성 금지

**잘못된 예**:
```python
# ❌ 절대 금지
np.random.seed(42)
fake_returns = np.random.randn(1000) * 0.02
```

**올바른 예**:
```python
# ✅ 실제 시장 데이터만 사용
real_returns = spy['Close'].pct_change().dropna()
```

---

### 3. 데이터 누출 방지

**잘못된 예**:
```python
# ❌ 미래 정보 사용
df['feature'] = df['target'].shift(-5)  # 미래 타겟으로 특성 생성
scaler.fit(X_all)  # 테스트 데이터 포함하여 스케일링
```

**올바른 예**:
```python
# ✅ 시간적 분리
df['feature'] = df['price'].shift(1)  # t-1 정보만 사용
scaler.fit(X_train)  # 학습 데이터만으로 fit
X_test_scaled = scaler.transform(X_test)  # 테스트는 transform만
```

---

## 시간적 분리 원칙

### 기본 규칙
```
시점      t-N ... t-1   t    t+1 ... t+22
특성      [══════════]      
타겟                        [═══════════]
                    ↑
              예측 시점 (t)
```

### 타겟 설계
```python
# 5일 후 변동성 예측 예시
df['target_vol_5d'] = df['returns'].shift(-1).rolling(5).std().shift(-4)
#                                    ↑                            ↑
#                        t+1부터 시작                    t+5까지
```

---

## 검증 방법

### Purged K-Fold CV
금융 시계열에서 표준 K-Fold는 데이터 누출 위험이 있음.

```
일반 K-Fold (위험):
Train  [...............] | Test [........]
                        ↑
              훈련 마지막의 타겟이 테스트와 겹칠 수 있음

Purged K-Fold (안전):
Train  [...........] | PURGE | Test [........]
                        ↑
              최소 22일 간격 (타겟 계산 기간)
```

### 파라미터 설정
```python
PurgedKFold(
    n_splits=5,       # 5개 폴드
    purge_length=22,  # 22거래일 제거 (월간 변동성)
    embargo_length=0  # 추가 금지 기간
)
```

---

## 성능 현실성 검사

### 의심 기준
| R² 범위 | 해석 |
|---------|------|
| **> 0.95** | ⚠️ 데이터 누출 의심 |
| 0.50 ~ 0.95 | 검토 필요 |
| 0.15 ~ 0.50 | ✅ 현실적 |
| 0.05 ~ 0.15 | ✅ 변동성 예측 적정 |
| < 0.05 | 수익률 예측 (거의 불가능) |

### VRP 예측 현실적 범위
- **R²**: 0.13 ~ 0.20
- **방향 정확도**: 70% ~ 75%
- **이 이상이면 누출 의심**

---

## 코드 검사 체크리스트

```bash
# 1. Random 데이터 검색
grep -r "np.random" src/
grep -r "random.random" src/

# 2. 하드코딩 검색
grep -r "= \[" src/  # 리스트 리터럴
grep -r "pd.DataFrame({" src/  # 직접 생성

# 3. 미래 정보 검색
grep -r "shift(-" src/  # 음수 shift (미래 데이터)
```

---

## 무결성 검증 리포트 예시

```json
{
  "temporal_separation": true,
  "no_random_data": true,
  "no_hardcoded_data": true,
  "reasonable_correlation": true,
  "realistic_cv_variance": true,
  "model_r2": 0.19,
  "r2_below_95": true,
  "validation_method": "Purged K-Fold CV",
  "validation": "PASSED"
}
```

---

**작성일**: 2025-12-20
