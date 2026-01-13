"""
결과 탭
"""
import streamlit as st
import pandas as pd

def render_results():
    """결과 렌더링"""
    
    st.markdown('<div class="section-header">3. 결과</div>', unsafe_allow_html=True)
    
    # 모델 성능 비교
    st.markdown("### 3.1 모델 성능 비교 (Test R²)")
    
    results_data = {
        '자산': ['S&P 500', 'Gold', 'Treasury', 'EAFE', 'Emerging', '평균'],
        'HAR-RV': [0.670, 0.855, 0.786, 0.705, 0.651, 0.733],
        'HAR+VIX': [0.683, 0.857, 0.789, 0.732, 0.661, 0.744],
        'CAVB (Full)': [0.706, 0.857, 0.783, 0.732, 0.654, 0.746],
        'HAR+VIX 대비': ['+0.023', '+0.000', '-0.006', '+0.001', '-0.007', '+0.002'],
        '통계적 유의성': [' (p=0.008)', '', '', '', '', '1/5']
    }
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **핵심 발견**:
    - HAR-RV+VIX가 평균 R² 0.744 달성 (CAVB Full 성능의 90%)
    - CAVB 추가 변수는 S&P 500에서만 통계적으로 유의
    - **결론**: HAR+VIX로 충분 (Occam's Razor 원칙)
    """)
    
    # 변수 중요도
    st.markdown("### 3.2 변수 중요도 (S&P 500)")
    
    importance_data = {
        '변수': ['VIX_lag1', 'RV_22d', 'CAVB_lag1', 'VIX_change', '기타'],
        '절댓값 계수': [0.58, 0.22, 0.09, 0.06, 0.05],
        'R² 기여도': ['60%', '20%', '5%', '3%', '12%']
    }
    
    df_imp = pd.DataFrame(importance_data)
    st.dataframe(df_imp, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **해석**:
    - VIX_lag1이 전체 설명력의 60% 담당
    - S&P 500 특수성: VIX가 SPY 옵션에서 직접 도출 → 구조적 연결
    """)
    
    # Horizon 비교
    st.markdown("### 3.3 예측 시계 비교")
    
    horizon_data = {
        '자산': ['Gold', 'Treasury', 'EAFE', 'S&P 500', 'Emerging'],
        '1일 R²': [0.823, 0.751, 0.698, 0.664, 0.573],
        '5일 R²': [0.857, 0.783, 0.732, 0.706, 0.654],
        '22일 R²': [0.317, 0.082, 0.176, -0.045, -0.361],
        '5일 vs 22일': ['+169%', '+855%', '+316%', '+1669%', '+281%']
    }
    
    df_h = pd.DataFrame(horizon_data)
    st.dataframe(df_h, use_container_width=True, hide_index=True)
