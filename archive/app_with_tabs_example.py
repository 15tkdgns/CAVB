"""
간단한 탭 기반 대시보드 - 모든 탭 포함
"""

import streamlit as st
import sys
sys.path.insert(0, '.')

# 페이지 설정
st.set_page_config(
    page_title="CAVB 변동성 예측",
    page_icon="📊",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 타이틀
st.title("📊 VIX-RV Basis 기반 Cross-Asset 변동성 예측")
st.markdown("**5일 선행 예측 | ElasticNet 모델 | HAR-RV 벤치마크**")
st.markdown("---")

# 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 연구 개요", 
    "🔬 방법론", 
    "📈 결과",
    "✓ 검증",
    "📚 참고문헌"
])

with tab1:
    try:
        from dashboard.tabs.tab_overview import render_overview
        render_overview()
    except:
        st.markdown("### 연구 개요")
        st.write("기존 app.py의 섹션 1-2 내용")
    
with tab2:
    try:
        from dashboard.tabs.tab_methodology import render_methodology
        render_methodology()
    except:
        st.markdown("### 방법론")
        st.write("기존 app.py의 변수 설명 등")

with tab3:
    try:
        from dashboard.tabs.tab_results import render_results
        render_results()
    except:
        st.markdown("### 결과")
        st.write("기존 app.py의 성능 비교 등")

with tab4:
    try:
        from dashboard.tabs.tab_validation import render_validation
        render_validation()
    except:
        st.markdown("### 검증")
        st.write("기존 app.py의 검증 섹션")

with tab5:
    from tabs.tab_references import render_references
    render_references()

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#999; font-size:0.9rem;">
    CAVB 예측 연구 (5일 예측 기간) | 데이터: Yahoo Finance (2015-2025)
</p>
""", unsafe_allow_html=True)
