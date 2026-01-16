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
    # 2.1 타겟 변수 정의
    # ==========================================
    st.markdown("### 2.1 타겟 변수 정의")
    
    st.markdown("""
    **예측 대상**: 5일 선행 CAVB (VIX-RV 괴리)
    ```
    CAVB_t+5 = VIX_t - RV_t+5
    ```
    
    **근거**: 
    - 미래 괴리를 예측함으로써 기대되는 RV를 추론
    - VIX-RV 수렴 패턴 활용
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
    # 2.7 모델 사양
    # ==========================================
    st.markdown("### 2.7 모델 사양")
    
    st.markdown("""
    **ElasticNet 회귀**:
    ```
    CAVB_t+5 = α + Σ β_i X_i,t + ε
    ```
    
    **하이퍼파라미터**:
    - α = 0.01 (L1/L2 혼합 강도)
    - l1_ratio = 0.7 (L1 비중)
    - 표준화: RobustScaler (이상치 강건)
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
        - HAR-RV (기본)
        - HAR-RV + VIX (개선)
        - CAVB Full (9개 변수)
        - **29 Features (최종)**
        """)
