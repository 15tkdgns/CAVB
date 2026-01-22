"""
방법론 탭 - 종합 방법론 설명
Updated: 2026-01-16
"""
import streamlit as st
import pandas as pd

def render_methodology():
    """방법론 렌더링 - 확장된 버전"""
    
    st.markdown('<div class="section-header">2. 방법론</div>', unsafe_allow_html=True)
    
    # ==========================================
    # 2.1 예측 시계 비교 (5일 vs 22일)
    # ==========================================
    st.markdown("### 2.1 예측 시계 비교")
    
    st.markdown("""
    **타겟 변수**: `CAVB_t+h = VIX_t - RV_t+h` (h = 예측 기간)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 5일 예측 (권장)")
        st.metric("평균 R²", "0.789", "Huber-Tuned")
        
        perf_5d = {
            '자산': ['GLD', 'TLT', 'EFA', 'SPY', 'EEM'],
            'R²': ['0.871', '0.828', '0.775', '0.753', '0.717']
        }
        st.dataframe(pd.DataFrame(perf_5d), use_container_width=True, hide_index=True)
        
        st.success("**장점**: 노이즈 감소, 정보 유지, 전 자산 예측 가능")
    
    with col2:
        st.markdown("#### 22일 예측 (선택적)")
        st.metric("평균 R²", "0.153", "Ensemble (채권 위주)")
        
        perf_22d = {
            '자산': ['TIP', 'IEF', 'GLD', 'TLT', 'SPY'],
            'R²': ['0.82', '0.79', '0.64', '0.38', '<0']
        }
        st.dataframe(pd.DataFrame(perf_22d), use_container_width=True, hide_index=True)
        
        st.warning("**한계**: 채권/상품만 가능, 주식 예측 불가")
    
    st.info("""
    **핵심 결론**:
    - **5일**: 전 자산 예측 가능 (R² 0.72~0.87)
    - **22일**: 채권(TIP, IEF) > 상품(GLD) > 주식(불가)
    - 권장: **5일 예측** 사용, 22일은 채권/상품 한정
    """)

    
    # ==========================================
    # 2.2 Baseline 변수 (9개)
    # ==========================================
    st.markdown("### 2.2 Baseline 예측 변수 (9개)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **실현변동성 (RV)**:
        - `RV_1d`: 1일 변동성
        - `RV_5d`: 5일 변동성
        - `RV_22d`: 22일 변동성
        """)
    
    with col2:
        st.markdown("""
        **내재변동성 (VIX)**:
        - `VIX_lag1`: VIX (t-1)
        - `VIX_lag5`: VIX (t-5)
        - `VIX_change`: VIX 변화율
        """)
    
    with col3:
        st.markdown("""
        **괴리 지속성 (CAVB)**:
        - `CAVB_lag1`: CAVB (t-1)
        - `CAVB_lag5`: CAVB (t-5)
        - `CAVB_ma5`: CAVB 5일 MA
        """)
    
    # ==========================================
    # 2.3 Feature Groups (신규)
    # ==========================================
    st.markdown("### 2.3 Feature Groups (29개 변수)")
    
    with st.expander("Feature Group별 상세", expanded=True):
        group_data = {
            'Group': ['Baseline', 'Enhanced', 'Group 2', 'Group 3', 'Group 4'],
            '변수 수': [9, 16, 4, 7, 10],
            '누적': [9, 25, 29, 32, 35],
            '효과': ['-', '+2.97%', '+1.05%', '+0.66%', '+0.26%'],
            '권장': ['필수', '필수', '필수', 'EEM만', '선택적'],
            '출처': ['HAR-RV', '신규 설계', 'Bollerslev (2009)', 'Segal (2015)', 'Term Structure']
        }
        
        df_group = pd.DataFrame(group_data)
        st.dataframe(df_group, use_container_width=True, hide_index=True)
        
        st.markdown("#### Baseline (9개)")
        st.code("""
RV_1d, RV_5d, RV_22d,           # 실현변동성
VIX_lag1, VIX_lag5, VIX_change,  # VIX 관련
CAVB_lag1, CAVB_lag5, CAVB_ma5   # CAVB 지속성
        """)
        
        st.markdown("#### Enhanced (16개 추가 → 누적 25개)")
        st.code("""
RV_10d, RV_std_22d, RV_momentum, RV_acceleration,  # RV 확장
VIX_lag10, VIX_lag22, VIX_ma5, VIX_ma22, VIX_zscore,  # VIX 확장
CAVB_percentile, CAVB_std_22d, CAVB_max_22d, CAVB_min_22d,  # CAVB 파생
RV_VIX_ratio, RV_VIX_product, CAVB_VIX_ratio  # Cross-term
        """)
        
        st.markdown("#### Group 2: VRP Decomposition (4개 추가 → 누적 29개)")
        st.code("""
VRP_persistent,      # CAVB 60일 이동평균 (장기 성분)
VRP_transitory,      # CAVB - Persistent (단기 성분)
VRP_variance_ratio,  # Transitory / Persistent
VRP_momentum         # VRP 변화율
        """)
        
        st.info("""
        **최적 구성**: 29개 (Baseline 25 + Group 2)
        - Group 2 (VRP Decomposition)이 가장 효과적 (+1.05%)
        - Group 3는 신흥시장(EEM)에서만 유효
        - Group 4는 과적합 위험
        """)
    
    # ==========================================
    # 2.4 Enhanced 변수 (16개) 상세
    # ==========================================
    st.markdown("### 2.4 Enhanced 변수 (16개)")
    
    with st.expander("추가 변수 상세"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **RV 확장 (4개)**:
            - `RV_10d`: 10일 변동성
            - `RV_std_22d`: RV 표준편차
            - `RV_momentum`: RV 모멘텀
            - `RV_acceleration`: RV 가속도
            
            **VIX 확장 (5개)**:
            - `VIX_lag10`: VIX (t-10)
            - `VIX_lag22`: VIX (t-22)
            - `VIX_ma5`: VIX 5일 MA
            - `VIX_ma22`: VIX 22일 MA
            - `VIX_zscore`: VIX z-score
            """)
        
        with col2:
            st.markdown("""
            **CAVB 파생 (4개)**:
            - `CAVB_percentile`: CAVB 백분위
            - `CAVB_std_22d`: CAVB 표준편차
            - `CAVB_max_22d`: CAVB 22일 최대
            - `CAVB_min_22d`: CAVB 22일 최소
            
            **Cross-term (3개)**:
            - `RV_VIX_ratio`: RV/VIX 비율
            - `RV_VIX_product`: RV*VIX 곱
            - `CAVB_VIX_ratio`: CAVB/VIX 비율
            """)
    
    # ==========================================
    # 2.5 Group 2: VRP Decomposition (핵심)
    # ==========================================
    st.markdown("### 2.5 Group 2: VRP Decomposition (핵심)")
    
    with st.expander("VRP 분해 상세"):
        st.markdown("""
        **이론적 근거**: Bollerslev et al. (2009)
        
        **변수**:
        - `VRP_persistent`: CAVB의 60일 이동평균 (장기 성분)
        - `VRP_transitory`: CAVB - Persistent (단기 성분)
        - `VRP_variance_ratio`: Transitory / Persistent
        
        **효과**:
        | 자산 | Baseline | +Group2 | 효과 |
        |------|----------|---------|------|
        | GLD | 0.870 | 0.875 | +0.57% |
        | EFA | 0.728 | 0.743 | +2.06% |
        | EEM | 0.677 | 0.694 | +2.51% |
        """)
        
        st.success("**평균 효과**: +1.05% (모든 자산에서 안정적)")
    
    # ==========================================
    # 2.6 Feature Selection 전략
    # ==========================================
    st.markdown("### 2.6 Feature Selection 전략")
    
    with st.expander("3가지 전략 비교"):
        fs_data = {
            '전략': ['RFE 15', 'Full 29', 'MI Top 20'],
            '변수 수': [15, 29, 20],
            '평균 R²': [0.769, 0.769, 0.757],
            '효율성': ['99.8%', '100%', '98.4%'],
            '권장': ['실전 배포', '학술 연구', '신흥시장 특화']
        }
        
        df_fs = pd.DataFrame(fs_data)
        st.dataframe(df_fs, use_container_width=True, hide_index=True)
        
        st.info("""
        **권장**: 
        - 학술 연구: Full 29 (완전한 분석)
        - 실전 배포: RFE 15 (48% 변수로 99.8% 성능)
        """)
    
    # ==========================================
    # 2.7 모델 사양 (12개 실험 기반)
    # ==========================================
    st.markdown("### 2.7 모델 사양 (12개 실험 검증)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **최적 모델: Huber Regressor**
        ```python
        from sklearn.linear_model import HuberRegressor
        
        model = HuberRegressor(
            epsilon=1.35,    # Outlier threshold
            alpha=1e-6,      # Regularization
            max_iter=1000
        )
        # Average R² = 0.789
        ```
        """)
    
    with col2:
        st.markdown("""
        **12개 실험 모델 비교 (40+ 모델)**
        
        | 순위 | 모델 | R² |
        |------|------|-----|
        | 1 | **Huber-Tuned** | **0.789** |
        | 2 | SVR-Tuned | 0.787 |
        | 3 | SVR-Linear | 0.785 |
        | 4 | ElasticNet | 0.768 |
        """)
    
    st.warning("""
    **실험 결론**: 
    - 비선형 모델(XGBoost, LSTM, RF)은 모두 과적합으로 성능 저하 (-10~30%)
    - 선형 모델이 VRP 예측에 가장 적합 (본질적 선형 관계)
    - Huber Regressor가 이상치에 강건하여 가장 안정적
    """)
    
    # ==========================================
    # 2.8 검증 프로토콜
    # ==========================================
    st.markdown("### 2.8 검증 프로토콜")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **3-Way Split**:
        - Train: 60%
        - Validation: 20%
        - Test: 20%
        - **Gap**: 5일 (전방 편향 방지)
        """)
    
    with col2:
        st.markdown("""
        **벤치마크 모델**:
        - HAR-RV (기본): 0.733
        - HAR-RV + VIX: 0.744
        - ElasticNet (29개): 0.769
        - **Huber-Tuned (13개): 0.789**
        """)
    
    # ==========================================
    # 2.9 핵심 Feature (13개)
    # ==========================================
    st.markdown("### 2.9 핵심 Feature (13개)")
    
    with st.expander("최적 Feature Set", expanded=True):
        feature_cols = st.columns(3)
        
        with feature_cols[0]:
            st.markdown("""
            **VIX 관련 (5개)**:
            - `VIX`
            - `VIX_lag1`
            - `VIX_lag5`
            - `VIX_change`
            - `VIX_ma22`
            """)
        
        with feature_cols[1]:
            st.markdown("""
            **CAVB 관련 (4개)**:
            - `CAVB`
            - `CAVB_lag1`
            - `CAVB_lag5`
            - `CAVB_ma5`
            """)
        
        with feature_cols[2]:
            st.markdown("""
            **RV 관련 (4개)**:
            - `RV_1d`
            - `RV_5d`
            - `RV_10d`
            - `RV_22d`
            """)
        
        st.success("**CAVB가 가장 중요한 변수** (약 79% 분산 설명)")
    
    # ==========================================
    # 2.10 확장 피처 (Exp 15-17)
    # ==========================================
    st.markdown("### 2.10 확장 피처 (7개 데이터 소스)")
    
    with st.expander("확장 피처 상세", expanded=True):
        st.markdown("""
        **Exp 15-17에서 테스트된 추가 피처** (yfinance 활용)
        """)
        
        extended_data = {
            '소스': ['VIX3M', 'SKEW', 'US10Y', 'DXY', 'HYG', 'LQD', 'OVX'],
            'Ticker': ['^VIX3M', '^SKEW', '^TNX', 'DX-Y.NYB', 'HYG', 'LQD', '^OVX'],
            '파생 변수': ['VIX_term (VIX3M/VIX)', 'SKEW_zscore', 'US10Y, US10Y_change', 
                        'DXY_momentum', 'Credit_spread', 'Credit_spread', 'VIX_OVX_ratio'],
            '예측력': ['높음 (장기)', '중간', '높음', '중간', '중간', '중간', '낮음'],
            '활용': ['22일 예측', '전체', 'Mean Reversion', '22일 예측', 'VRP 조정', 'VRP 조정', '옵션']
        }
        df_ext = pd.DataFrame(extended_data)
        st.dataframe(df_ext, use_container_width=True, hide_index=True)
        
        st.warning("""
        **주의: Feature Selection 필수**
        - 43개 확장 피처 전체 사용: **-34% 성능 저하** (과적합)
        - ElasticNet L1 선택 후 12-15개 사용: **-2% 성능 저하** (수용 가능)
        - 핵심 가치 피처: **SKEW_zscore, US10Y, DXY_momentum, VIX_term**
        """)
    
    # ==========================================
    # 2.11 22일 예측 방법론 (Exp 18-19)
    # ==========================================
    st.markdown("### 2.11 22일 예측 방법론 (Ensemble)")
    
    with st.expander("22일 Ensemble 접근법", expanded=True):
        st.markdown("""
        **문헌 기반 접근법** (Exp 18):
        1. **Mean Reversion**: 장기 평균 회귀 특성 활용
        2. **VRP Adjusted**: VIX - 역사적 VRP로 조정
        3. **HAR Extended**: 분기(66일) 수준까지 multi-scale 확장
        
        **Ensemble 공식**:
        ```
        Prediction = (Mean_Reversion + VRP_Adjusted + HAR_Extended) / 3
        ```
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **22일 예측 성능 (Exp 18)**:
            | 모델 | 평균 R² | 양수 자산 |
            |------|---------|----------|
            | **Ensemble** | **0.153** | 3/5 |
            | VRP_Adjusted | 0.148 | 3/5 |
            | HAR_Extended | 0.079 | 2/5 |
            """)
        
        with col2:
            st.markdown("""
            **Optuna 튜닝 결과 (Exp 19)**:
            - 최적 모델: Ridge (alpha=935)
            - 테스트 R²: 0.114
            - 강한 정규화가 과적합 방지에 필수
            """)
        
        st.info("""
        **22일 예측 핵심 인사이트**:
        - 채권(TIP, IEF)에서 **R² > 0.7** 달성 가능
        - 주식(SPY)에서는 **예측 불가** (R² < 0)
        - 강한 정규화 (alpha > 500)가 필수
        """)
    
    # ==========================================
    # 2.12 다자산 전략
    # ==========================================
    st.markdown("### 2.12 다자산 예측 전략 (23개 자산)")
    
    with st.expander("자산별 최적 모델", expanded=True):
        st.markdown("""
        **Exp 20: 18개 신규 자산 테스트 결과**
        """)
        
        asset_strategy = {
            '자산군': ['채권', '채권', '채권', '상품', '상품', '주식'],
            '자산': ['TIP, IEF, BND', 'TLT, HYG, LQD', '-', 'GLD, DBA, SLV', '-', 'SPY, XLK, XLF'],
            '5일 모델': ['Huber', 'Huber', '-', 'Huber', '-', 'Huber'],
            '5일 R²': ['0.79+', '0.70+', '-', '0.65+', '-', '0.76'],
            '22일 모델': ['Ensemble', 'Ensemble', '-', 'Ridge', '-', '예측불가'],
            '22일 R²': ['**0.70-0.82**', '0.38-0.56', '-', '0.35-0.65', '-', '**<0**'],
            '권장': ['5일+22일', '5일+22일', '-', '5일+22일', '-', '5일만']
        }
        df_strategy = pd.DataFrame(asset_strategy)
        st.dataframe(df_strategy, use_container_width=True, hide_index=True)
        
        st.success("""
        **자산별 권장 사항**:
        - **채권 (TIP, IEF)**: 5일 + 22일 모두 예측 → **가장 효과적**
        - **상품 (GLD, DBA)**: 5일 + 22일 모두 예측 → 효과적
        - **주식 (SPY)**: **5일만 예측** → 22일은 예측 불가
        - **신흥국 (EEM)**: 5일만 약간 가능 → 권장하지 않음
        """)
    
    # ==========================================
    # 2.13 최적 모델 상세 (5일: Huber, 22일: Ensemble)
    # ==========================================
    st.markdown("### 2.13 최적 모델 상세")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 5일 예측: Huber Regressor")
        st.metric("평균 R²", "0.789", "vs ElasticNet +2.6%")
        
        st.code("""
from sklearn.linear_model import HuberRegressor

model = HuberRegressor(
    epsilon=1.35,   # 이상치 임계값
    alpha=1e-6,     # 정규화 (최소)
    max_iter=1000
)
        """, language='python')
        
        st.markdown("""
        **왜 Huber인가?**
        - **이상치에 강건**: COVID(2020), VIX 급등에도 안정적
        - **선형성 유지**: VRP는 본질적 선형 관계
        - **정규화 최소**: 데이터 충분, 수축 불필요
        
        **vs 다른 모델**:
        | 모델 | R² | 차이 |
        |------|-----|------|
        | Huber | **0.789** | - |
        | ElasticNet | 0.769 | +2.6% |
        | XGBoost | 0.680 | +16% |
        | RF | 0.608 | +30% |
        """)
    
    with col2:
        st.markdown("#### 22일 예측: Ensemble Specialist")
        st.metric("평균 R²", "0.153", "TIP 0.82, IEF 0.79")
        
        st.code("""
# Ensemble = 3개 모델 평균
prediction = (
    mean_reversion_pred +  # 평균 회귀
    vrp_adjusted_pred +    # VRP 조정
    har_extended_pred      # HAR 확장
) / 3
        """, language='python')
        
        st.markdown("""
        **3개 구성 모델**:
        | 모델 | 역할 | 핵심 피처 |
        |------|------|----------|
        | Mean Reversion | 장기 평균 회귀 | VIX_deviation |
        | VRP Adjusted | VRP 패턴 | VIX_term |
        | HAR Extended | Multi-scale | RV_66d |
        
        **정규화**: Ridge alpha=50~100 (강함)
        
        **vs 단일 모델**:
        | 모델 | R² |
        |------|-----|
        | **Ensemble** | **0.153** |
        | VRP_Adjusted | 0.148 |
        | Ridge | 0.114 |
        """)
    
    st.info("""
    **모델 선택 요약**:
    - **5일**: Huber (단일, 정규화 약함) → 전 자산 적용
    - **22일**: Ensemble (3개 앙상블, 정규화 강함) → 채권/상품만 적용
    - **핵심 차이**: 예측 기간이 길수록 강한 정규화 필요
    """)

