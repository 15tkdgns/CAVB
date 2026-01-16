"""
연구 개요 탭 - Executive Summary
Updated: 2026-01-16
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
        st.metric("평균 Test R²", "0.789", "+2.7% vs ElasticNet")
    
    with col2:
        st.metric("최적 예측 기간", "5일", "+717% vs 22일")
    
    with col3:
        st.metric("최적 모델", "Huber-Tuned", "40+ 모델 비교")
    
    with col4:
        st.metric("Feature 수", "13개", "핵심 변수만")
    
    with col5:
        st.metric("실험 수", "12개", "총 40+ 모델")
    
    st.markdown("---")
    
    # ==========================================
    # 연구 질문
    # ==========================================
    st.markdown("### 1.1 연구 질문")
    st.info("""
    시장 전체 내재변동성(VIX)과 자산별 실현변동성 간 괴리(basis)가 
    여러 자산군에 걸친 **5일 선행 변동성 예측**을 개선할 수 있는가?
    """)
    
    # ==========================================
    # 가설 및 검증 결과
    # ==========================================
    st.markdown("### 1.2 가설 및 검증 결과")
    
    hypo_data = {
        '가설': ['H1', 'H2', 'H3'],
        '내용': [
            'VIX 기반 변수가 HAR-RV 모델 개선',
            'CAVB가 VIX 단독 대비 추가 예측력 제공',
            '단순 선형 모델 > 복잡한 ML'
        ],
        '결과': ['검증됨', '부분 검증', '검증됨'],
        '근거': [
            'HAR+VIX: +1.5% 개선',
            'S&P 500에서만 유의 (p=0.008)',
            'ElasticNet > Stacking (+30.7%)'
        ]
    }
    
    df_hypo = pd.DataFrame(hypo_data)
    st.dataframe(df_hypo, use_container_width=True, hide_index=True)
    
    # ==========================================
    # 주요 실험 결과 요약
    # ==========================================
    st.markdown("### 1.3 주요 실험 결과")
    
    with st.expander("실험별 핵심 발견", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **예측 기간 비교**:
            - 1일: R² 0.682 (노이즈 과다)
            - **5일**: R² 0.746 (최적)
            - 22일: R² 0.097 (정보 감쇠)
            
            **모델 구성**:
            - Baseline (9): R² 0.740
            - Enhanced (25): R² 0.762
            - **29 Features**: R² **0.770**
            """)
        
        with col2:
            st.markdown("""
            **ML vs Linear**:
            - ElasticNet: R² **0.770**
            - Neural Network: R² 0.707
            - Stacking: R² 0.540
            
            **Feature Groups**:
            - Group 2 (VRP): **+1.05%** (필수)
            - Group 3 (Good/Bad): +0.66%
            - RFE 15: 99.8% 효율
            """)
    
    # ==========================================
    # 자산별 성능
    # ==========================================
    st.markdown("### 1.4 자산별 성능 (29 Features)")
    
    asset_data = {
        '자산': ['S&P 500', 'Gold', 'Treasury', 'EAFE', 'Emerging'],
        'Test R²': [0.699, 0.873, 0.837, 0.742, 0.694],
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
