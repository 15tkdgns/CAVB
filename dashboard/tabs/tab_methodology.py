"""
방법론 탭
"""
import streamlit as st

def render_methodology():
    """방법론 렌더링"""
    
    st.markdown('<div class="section-header">2. 방법론</div>', unsafe_allow_html=True)
    
    # 변수 정의
    st.markdown("### 2.1 변수 정의")
    
    st.markdown("""
    **타겟 변수**: 5일 선행 CAVB (VIX-RV 괴리)
    ```
    CAVB_t+5 = VIX_t - RV_t+5
    ```
    """)
    
    st.markdown("**예측 변수** (9개):")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **실현변동성 (RV)**:
        - RV_1d: 1일 변동성
        - RV_5d: 5일 변동성
        - RV_22d: 22일 변동성
        """)
    
    with col2:
        st.markdown("""
        **내재변동성 (VIX)**:
        - VIX_lag1: VIX (t-1)
        - VIX_lag5: VIX (t-5)
        - VIX_change: VIX 변화율
        """)
    
    with col3:
        st.markdown("""
        **괴리 지속성 (CAVB)**:
        - CAVB_lag1: CAVB (t-1)
        - CAVB_lag5: CAVB (t-5)
        - CAVB_ma5: CAVB 5일 이동평균
        """)
    
    # 모델
    st.markdown("### 2.2 모델")
    
    st.markdown("""
    **ElasticNet 회귀**:
    ```
    CAVB_t+5 = α + Σ β_i X_i,t + ε
    ```
    
    **하이퍼파라미터**:
    - α = 0.01 (L1/L2 혼합 강도)
    - l1_ratio = 0.7 (L1 비중)
    - 표준화: RobustScaler
    """)
    
    # 검증 프로토콜
    st.markdown("### 2.3 검증 프로토콜")
    
    st.markdown("""
    **3-Way Split**:
    - Train: 60%
    - Validation: 20%
    - Test: 20%
    - **Gap**: 5일 (전방 편향 방지)
    
    **벤치마크 모델**:
    - HAR-RV (기본)
    - HAR-RV + VIX (개선)
    - CAVB Full (9개 변수)
    """)
