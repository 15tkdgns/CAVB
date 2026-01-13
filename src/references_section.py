# References section content
REFERENCES_CONTENT = """
# ========================================
# 섹션 5: 참고문헌
# ========================================
st.markdown('<div class="section-header">5. 주요 참고문헌</div>', unsafe_allow_html=True)

st.markdown(\"""
본 연구의 이론적 기반과 방법론적 선택의 근거가 된 핵심 문헌을 소개합니다.
\""")

# 섹션 5.1: 선형 모델 vs 머신러닝
st.markdown("### 5.1 선형 모델 vs 머신러닝")

with st.expander("📄 Branco et al. (2023) - Does Anything Beat Linear Models?"):
    st.markdown(\"""
    **제목**: Forecasting Realized Volatility: Does Anything Beat Linear Models?
    
    **연구 배경**: 딥러닝과 머신러닝(ML) 기법의 발전에도 불구하고, 금융 시계열 예측, 특히 실현 변동성(RV) 예측에 있어 
    이러한 복잡한 모델들이 전통적인 계량경제 모델보다 실질적인 우위를 가지는지에 대한 논쟁이 지속되고 있습니다.
    
    **방법론**:
    - 벤치마크: HAR-RV (이질적 자기회귀) 모델
    - 비교군: Random Forest, GBM, 인공신경망 등
    - 예측 시계: 단기(1일) 및 주간(5일)
    - 대상: 전 세계 주요 10개 주가 지수
    
    **연구 결과**:
    - 거시경제 변수나 기술적 지표 추가는 예측 오차 감소에 기여
    - 그러나 비선형 ML 모델이 선형 HAR 모델을 **통계적으로 유의미하게 능가한다는 증거 발견되지 않음**
    
    **결론**:
    - 변동성 데이터의 높은 노이즈와 제한적 샘플 크기로 인해 복잡한 모델은 과적합되기 쉬움
    - 실무적으로는 변수 선택에 신중을 기한 **단순 선형 모델이 최적 전략**
    - "오컴의 면도날" 원칙 지지
    \""")

with st.expander("📄 Working Paper (2024) - HARd to Beat"):
    st.markdown(\"""
    **제목**: HARd to Beat: The Overlooked Impact of Rolling Windows in the Era of Machine Learning
    
    **연구 배경**: 
    기존 연구들이 머신러닝 모델의 우월성을 보고했으나, 이는 종종 고정된 학습 구간이나 부적절한 검증 방식을 사용한 결과일 수 있습니다.
    
    **핵심 발견**:
    - 모델 파라미터를 매일/주기적으로 재추정하는 **롤링 윈도우 방식**을 엄격하게 적용할 경우,
      머신러닝 모델의 겉보기 우위는 **사라짐**
    - 적절히 튜닝된 HAR 모델은 RV와 VIX 정보를 효율적으로 처리
    - 복잡한 비선형 모델보다 **예측 안정성** 면에서 우수
    
    **결론**:
    - 변동성 예측 문헌에서 ML 모델의 성과가 **과대포장**되었을 가능성 제기
    - 구조적 변화를 반영하기 위해 모델을 지속적으로 업데이트하는 환경에서는
      단순하고 견고한 **HAR 모델이 여전히 가장 강력한 벤치마크**
    \""")

# 섹션 5.2: RV와 VIX의 결합 및 예측 시계
st.markdown("### 5.2 RV와 VIX의 결합 및 예측 시계")

with st.expander("📄 Martin (2021) - Informational Content of RV and VIX"):
    st.markdown(\"""
    **제목**: The Informational Content of RV and VIX for Forecasting
    
    **연구 질문**: 
    과거의 실현 변동성(RV)과 내재 변동성(VIX) 중 무엇이 더 우수한 예측 변수인가? 
    두 변수를 결합했을 때 시너지 효과가 발생하는가?
    
    **분석 결과**:
    - RV와 VIX는 **상호 보완적(Complementary)** 정보 포함
    - RV: 과거 가격 움직임의 **지속성(Persistence)** 정보
    - VIX: 미래 불확실성에 대한 시장의 **프리미엄** 정보
    - 두 변수 동시 투입 시 1일 및 5일 예측에서 **R² 비약적 상승**
    
    **결론**:
    - 단기 예측에서 RV와 VIX의 결합은 **가용 정보의 대부분을 포괄**
    - 복잡한 파생 변수 없이 **이 두 핵심 변수만으로 최적 성과** 달성 가능
    \""")

with st.expander("📄 Degiannakis et al. (2018) - Multiple Horizons and Decay"):
    st.markdown(\"""
    **제목**: Multiple Days Ahead Realized Volatility Forecasting: Horizons and Decay
    
    **연구 목적**: 
    1일, 5일, 10일, 22일 등 다중 시계에 걸쳐 예측 모델 성과가 어떻게 변화하는지 추적
    
    **실증 분석**:
    - HAR-RV, ARFIMA 등 주요 모델 적용
    - 예측 시계가 길어질수록 예측력은 **지수함수적으로 감쇠**
    - **1일-5일**: 높은 설명력 유지
    - **10일 기점**: 예측 오차 급증
    - **22일(월간)**: 모델 간 성능 차이 무의미, 설명력 급락
    
    **시사점**:
    - 변동성 정보의 '기억(Memory)'은 **단기에 집중**
    - **5일(주간) 예측**: 정보 유효성과 실무 활용도가 균형을 이루는 **최적 구간**
    - 월간 이상 예측: 변동성 지속성보다 **거시경제 레짐 변화**가 더 중요
    \""")

with st.expander("📄 Yfanti (2022) - Option-Implied Information"):
    st.markdown(\"""
    **제목**: Financial Volatility Modeling with Option-Implied Information
    
    **연구 개요**: 
    역사적 수익률 데이터만 의존하는 전통적 시계열 모델의 한계를 극복하기 위해
    옵션 시장의 내재 변동성 정보를 통합하는 방안 연구
    
    **결과**:
    - HAR 모델에 **VIX 추가 확장** 시 통계적으로 유의미한 예측력 개선
    - 개선 효과는 **모든 예측 시계**(1일, 5일, 10일, 22일)에서 일관되게 나타남
    - 특히 시장 불확실성 증가 구간에서 옵션 정보가 변동성 급등을 **더 빠르게 포착**
    
    **결론**:
    - 옵션 내재 정보는 과거 데이터가 설명하지 못하는 **미래 위험 프리미엄** 포함
    - **'HAR-RV + VIX' 모델링**이 복잡한 구조 변경 없이 정확도를 높이는
      **가장 효율적인 베이스라인**
    \""")

st.markdown(\"""
---
**본 연구와의 연관성**:

위 문헌들은 본 연구의 핵심 발견을 뒷받침합니다:
1. **단순 선형 모델의 우월성**: Branco et al.과 본 연구 모두 ElasticNet이 복잡한 ML보다 우수함을 확인
2. **RV-VIX 결합의 효과**: Martin, Yfanti의 발견과 일치하여 HAR-RV+VIX가 90% 성능 달성
3. **5일 예측 최적성**: Degiannakis의 horizon decay 분석이 본 연구의 5일/22일 비교 결과를 이론적으로 지지
4. **롤링 윈도우의 중요성**: 본 연구는 3-way split + 5일 gap으로 엄격한 시간 검증 적용
\""")
"""
