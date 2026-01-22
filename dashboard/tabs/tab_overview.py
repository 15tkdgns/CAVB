"""
연구 개요 탭 - Executive Summary
Updated: 2026-01-21
"""
import streamlit as st
import pandas as pd

def render_overview():
    """연구 개요 렌더링 - 확장된 버전"""
    
    st.markdown('<div class="section-header">1. 연구 개요</div>', unsafe_allow_html=True)
    
    # ==========================================
    # Executive Summary (핵심 지표)
    # ==========================================
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("5일 Test R²", "0.789", "Huber-Tuned")
    
    with col2:
        st.metric("22일 Test R²", "0.82", "TIP (채권)")
    
    with col3:
        st.metric("테스트 자산", "23개", "5개 기존 + 18개 신규")
    
    with col4:
        st.metric("테스트 모델", "60+", "20개 실험")
    
    with col5:
        st.metric("vs HAR-RV", "+11%", "베이스라인 대비")
    
    st.markdown("---")
    
    # ==========================================
    # 연구 질문
    # ==========================================
    st.markdown("### 1.1 연구 질문")
    st.info("""
    시장 전체 내재변동성(VIX)과 자산별 실현변동성 간 괴리(**CAVB**)가 
    여러 자산군에 걸친 **5일 및 22일 선행 VRP 예측**을 개선할 수 있는가?
    """)
    
    # ==========================================
    # 가설 및 검증 결과
    # ==========================================
    st.markdown("### 1.2 가설 및 검증 결과")
    
    hypo_data = {
        '가설': ['H1', 'H2', 'H3', 'H4'],
        '내용': [
            'CAVB가 HAR-RV 모델 개선',
            '선형 모델 > 복잡한 ML',
            '22일 예측은 채권에서 효과적',
            '확장 피처는 Feature Selection 필요'
        ],
        '결과': ['검증됨', '검증됨', '검증됨', '검증됨'],
        '근거': [
            'HAR 대비 +11% R²',
            'Huber > XGBoost +16%',
            'TIP 0.82, IEF 0.79',
            '43개→15개 선택시 과적합 방지'
        ]
    }
    
    df_hypo = pd.DataFrame(hypo_data)
    st.dataframe(df_hypo, use_container_width=True, hide_index=True)
    
    # ==========================================
    # 주요 실험 결과 요약
    # ==========================================
    st.markdown("### 1.3 주요 실험 결과 (20개 실험)")
    
    with st.expander("핵심 발견", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **5일 예측 (전 자산 가능)**:
            - 최적 모델: **Huber** (R² 0.789)
            - GLD 0.87 > TLT 0.83 > SPY 0.76
            - HAR-RV 대비 **+11%** 개선
            
            **베이스라인 비교**:
            - vs Naive Mean: +182%
            - vs Random Walk: +88%
            - vs GARCH: +76%
            """)
        
        with col2:
            st.markdown("""
            **22일 예측 (채권/상품만)**:
            - 최적 모델: **Ensemble**
            - TIP **0.82** > IEF 0.79 > GLD 0.64
            - 주식(SPY): **예측 불가** (R² < 0)
            
            **자산 발견 (Exp 20)**:
            - 23개 자산 중 10개 예측 가능
            - **채권 > 상품 > 주식**
            """)
    

    # ==========================================
    # 자산별 성능
    # ==========================================
    st.markdown("### 1.4 자산별 성능")
    st.caption("**5일 예측 | ElasticNet (29 Features) | Test Set (2022-2025)**")
    
    asset_data = {
        '자산': ['SPY (S&P 500)', 'GLD (Gold)', 'TLT (Treasury)', 'EFA (EAFE)', 'EEM (Emerging)'],
        '5일 R² (ElasticNet)': [0.699, 0.873, 0.837, 0.742, 0.694],
        'vs HAR-RV': ['+4.3%', '+2.1%', '+6.5%', '+5.2%', '+6.6%'],
        '최적 구성': ['Regime-Adaptive', '25+Group2', '25+Group2', '25+Group2', '25+Group3']
    }
    
    df_asset = pd.DataFrame(asset_data)
    st.dataframe(df_asset, use_container_width=True, hide_index=True)
    
    # ==========================================
    # 연구 기여점
    # ==========================================
    st.markdown("### 1.5 연구 기여점 (Research Contribution)")
    
    st.success("""
    **이론적 기여**:
    1. VIX-RV Basis (CAVB) 개념을 변동성 예측에 도입
    2. 5일 예측의 최적성 실증 (Degiannakis decay 이론 재확인)
    3. 중간 규모 데이터에서 선형 모형 우위 입증
    
    **실무적 기여**:
    1. 29개 Feature로 R² 0.769 달성 (또는 RFE 15개로 동일 성능)
    2. 6중 데이터 누출 테스트로 강건성 확보
    3. 자산별 맞춤 전략 제시
    """)
    
    # ==========================================
    # 대상 저널
    # ==========================================
    st.markdown("### 1.6 SCI 저널 제출 준비")
    
    with st.expander("목표 저널 및 전략"):
        st.markdown("""
        **목표 저널**:
        - Journal of Empirical Finance
        - Finance Research Letters
        - Quantitative Finance
        
        **핵심 기여 포인트**:
        1. 5일 예측이 22일 대비 +717% 우수
        2. ElasticNet이 ML 대비 +30% 우수
        3. 6중 검증으로 데이터 누출 부재 확인
        
        **제안 제목**: "VIX-RV Basis를 활용한 자산 간 변동성 예측"
        """)
