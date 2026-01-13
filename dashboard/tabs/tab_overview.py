"""
연구 개요 탭
"""
import streamlit as st

def render_overview():
    """연구 개요 렌더링"""
    
    st.markdown('<div class="section-header">1. 연구 개요</div>', unsafe_allow_html=True)
    
    # 핵심 결과 요약
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("평균 Test R²", "0.746", "+717% vs 22일")
    
    with col2:
        st.metric("최고 성능", "0.857", "Gold (5일)")
    
    with col3:
        st.metric("벤치마크", "HAR-RV+VIX", "90% 달성")
    
    with col4:
        st.metric("검증 통과", "6/6", "Leakage Tests")
    
    st.markdown("---")
    
    # 연구 질문
    st.markdown("### 1.1 연구 질문")
    st.markdown("""
    시장 전체 내재변동성(VIX)과 자산별 실현변동성 간 괴리(basis)가 
    여러 자산군에 걸친 **단기 변동성 예측**을 개선할 수 있는가?
    """)
    
    # 가설
    st.markdown("### 1.2 가설")
    st.markdown("""
    - **H1**: VIX 기반 변수가 전통적인 HAR-RV 모델을 개선한다
    - **H2**: VIX-RV 괴리(CAVB)가 VIX 단독 사용 대비 추가적인 예측력을 제공한다
    - **H3**: 단순 선형 모델이 복잡한 앙상블 방법보다 우수한 성능을 보인다
    """)
    
    # 핵심 발견
    st.markdown("### 1.3 핵심 발견")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ** 검증된 가설**:
        - H1: HAR-RV (0.733) → HAR+VIX (0.744) = +1.5%
        - H3: ElasticNet이 Stacking보다 +30.7% 우수
        """)
    
    with col2:
        st.markdown("""
        **️ 부분 검증**:
        - H2: CAVB는 S&P 500에서만 유의 (p=0.008)
        - 다른 4개 자산: HAR+VIX로 충분
        """)
    
    # Horizon 비교
    st.markdown("### 1.4 예측 시계 최적화")
    
    import pandas as pd
    
    horizon_data = {
        'Horizon': ['1일', '5일 (채택)', '22일'],
        '평균 R²': [0.682, 0.746, 0.097],
        '예측 가능 자산': ['5/5', '5/5', '2/5'],
        'vs 5일': ['-9%', '100%', '-87%']
    }
    
    df = pd.DataFrame(horizon_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **결론**: 5일 예측이 최적 구간 (Degiannakis et al. 2018 이론 재확인)
    """)
