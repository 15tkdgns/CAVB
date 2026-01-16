# VIX-RV Basis를 활용한 자산 간 변동성 예측

> Cross-Asset Volatility Forecasting Using VIX-RV Basis

---

## 🎯 핵심 성과

**평균 R²: 0.746** (모든 자산 예측 가능)

| 자산 | 5일 R² | 22일 R² | 개선 | 상태 |
|------|--------|---------|------|------|
| **Gold (금)** | **0.857** | 0.32 | +169% | ✅ 예측 가능 |
| **Treasury (국채)** | **0.783** | 0.08 | +878% | ✅ 예측 가능 |
| **EAFE (선진국)** | **0.732** | 0.18 | +307% | ✅ 예측 가능 |
| **S&P 500** | **0.706** | -0.04 | +1865% | ✅ 예측 가능 |
| **Emerging (신흥국)** | **0.654** | -0.36 | +282% | ✅ 예측 가능 |

**주요 발견**:
- ✅ **5일 예측**이 22일 대비 **평균 +717% 우수**
- ✅ **VIX 추가**로 HAR-RV 대비 **90% 성능** 달성
- ✅ **단순 ElasticNet**이 복잡한 Stacking보다 우수
- ✅ **데이터 누출 6-fold** 검증 통과

---

## 📖 CAVB 정의

```
CAVB = VIX (Systemic Risk) - RV (Idiosyncratic Volatility)
```

**VIX를 공통 변수로 사용하는 이유:**
- VIX는 시장 전체의 systemic risk 측정
- 개별 자산의 RV는 고유 위험 반영  
- 이 둘의 **괴리(Basis)**가 예측 가능한 패턴 형성
- HAR-RV 대비 VIX 추가로 90% 성능 달성 (실증)

---

## 🚀 빠른 시작

### 1. 기본 환경

```bash
pip install streamlit pandas numpy scikit-learn yfinance plotly
```

### 2. 고급 실험 (가상환경)

```bash
python3 -m venv venv_ml
source venv_ml/bin/activate  # Windows: venv_ml\Scripts\activate
pip install numpy==1.26.4 pandas scikit-learn yfinance
```

### 3. 대시보드 실행

```bash
streamlit run app.py
# http://localhost:8501
```

---

## 📂 프로젝트 구조

```
vrp-prediction/
├── app.py                       # Streamlit 대시보드 (5일 예측)
├── src/
│   ├── horizon_comparison.py    # ⭐ 5일 vs 22일 비교
│   ├── har_rv_benchmark.py      # ⭐ HAR-RV 벤치마크
│   ├── advanced_pipeline.py     # Feature Eng (과적합 사례)
│   ├── statistical_validation.py
│   ├── subperiod_analysis.py
│   ├── rolling_window_validation.py
│   └── leakage_verification.py  # 데이터 누출 검증
├── data/results/
│   ├── horizon_comparison.json  # 5일 vs 22일 결과
│   └── har_rv_benchmark.json    # HAR-RV 비교
└── venv_ml/                     # 가상환경 (NumPy 1.26.4)
```

---

## 🔬 핵심 실험 결과

### 1. 예측 기간 최적화

**실험**: 5일 vs 22일 예측 비교

**결론**: 5일 예측이 월등히 우수 (+717%)

### 2. HAR-RV 벤치마크

**비교 모델**:
- HAR-RV: Linear(RV_1d, RV_5d, RV_22d)
- HAR-RV+VIX: HAR + VIX 변수
- CAVB (Full): ElasticNet (9 features)

**결론**: 
- VIX 추가만으로 90% 성능 달성
- CAVB 변수는 S&P 500에서만 통계적 유의
- 대부분 자산에서 HAR-RV+VIX로 충분

### 3. 과적합 실험

**실험**: Feature Engineering (21 features) + Stacking (XGB+RF+GBM)

**결론**: **-30.7% 성능 악화** (심각한 과적합)
- 단순 ElasticNet이 최적
- Occam's Razor 확인

### 4. 데이터 누출 검증

**6가지 테스트 모두 통과**:
1. ✅ Shuffled Target
2. ✅ Strict Temporal Split
3. ✅ Extended Gap (22/44/66일)
4. ✅ Scaler Leakage
5. ✅ Autocorrelation
6. ✅ Future Feature

---

## 💡 실무 시사점

### 단기 변동성 예측 가능
- 5일 예측 시 **모든 주요 자산 예측 가능**
- Gold, Treasury 등 안전자산 특히 우수 (R² > 0.78)
- S&P 500도 VIX 기반으로 예측 가능 (R² = 0.71)

### VIX의 핵심 역할
- HAR-RV에 VIX 추가로 90% 성능 달성
- 개별 IV (GVZ, MOVE) 대비 우수
- Systemic risk spillover 효과 입증

### 단순 모델 우월성
- ElasticNet (9 features) 최적
- 복잡한 모델은 과적합 (-30%)
- Occam's Razor 적용

---

## 🛠 기술 스택

- **Python**: 3.12
- **ML**: scikit-learn (ElasticNet, RobustScaler)
- **Dashboard**: Streamlit, Plotly
- **Data**: pandas, numpy, yfinance
- **Validation**: HAR-RV Benchmark, 6-fold Leakage Tests

---

## 📊 SCI 저널 제출 준비

### 제목
"Short-Term Volatility Forecasting via VIX Spillover Effects: Evidence from Cross-Asset Analysis"

### 주요 기여
1. 5일 단기 예측의 우월성 입증 (+717% vs 22일)
2. VIX spillover effects 실증 (HAR-RV+VIX로 90% 성능)
3. 단순 모델 효과성 입증 (과적합 방지)
4. 엄격한 검증 (6-fold leakage + HAR-RV benchmark)

### 검증 수준
- ✅ 예측 기간 최적화 (5일 vs 22일)
- ✅ HAR-RV 벤치마크 비교
- ✅ 과적합 테스트 (Stacking 실패)
- ✅ 데이터 누출 6-fold 검증
- ✅ 3-Way Split (60/20/20)

---

## 📝 문서

- **CAVB_Summary.md**: 연구 전체 요약
- **walkthrough.md**: 실험 과정 및 결과
- **task.md**: 작업 내역

---

## 📄 라이선스

MIT License

---

**최종 업데이트**: 2026-01-09  
**프로젝트 상태**: SCI 저널 제출 준비 완료  
**평균 R²**: 0.746 (모든 자산 예측 가능)
