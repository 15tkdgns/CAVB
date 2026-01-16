#!/usr/bin/env python3
"""
변동성 예측 모델 폐기 이유 설명 대시보드
핵심: VIX 상관관계 문제
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="모델 전환 이유",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .insight-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .problem-box {
        background: #fff5f5;
        border-left: 5px solid #e74c3c;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .solution-box {
        background: #f0fff4;
        border-left: 5px solid #38a169;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<div class="main-title">RV 모델 폐기 이유: VIX 상관관계 문제</div>', unsafe_allow_html=True)

# 핵심 인사이트
st.markdown("""
<div class="insight-box">
    <h3>핵심 발견</h3>
    <p style="font-size: 1.3rem; margin: 0;">
        <strong>VIX_lag 변수가 예측력의 대부분을 차지</strong> → 모델이 실질적으로 "VIX 따라가기"에 불과
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# 두 컬럼
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 특성 중요도 (계수)")
    
    # 특성 중요도 차트
    importance_data = pd.DataFrame({
        '특성': ['VIX_lag1', 'VIX_lag5', 'RV_22d', 'VRP_lag5', 'VRP_ma5', 
                 'VIX_change', 'regime_high', 'RV_1d', 'RV_5d', 'VRP_lag1'],
        '계수': [5.77, 5.47, 4.25, 2.36, 1.88, 1.51, 1.22, 1.00, 0.92, 0.65],
        '유형': ['VIX', 'VIX', 'RV', 'VRP', 'VRP', 'VIX', '기타', 'RV', 'RV', 'VRP']
    })
    
    fig = px.bar(importance_data, x='계수', y='특성', orientation='h',
                 color='유형',
                 color_discrete_map={'VIX': '#e74c3c', 'RV': '#3498db', 'VRP': '#2ecc71', '기타': '#95a5a6'},
                 text='계수',
                 title='')
    fig.update_traces(textposition='inside', texttemplate='%{text:.2f}', textfont_size=12)
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="problem-box">
        <strong>문제:</strong> VIX 관련 변수(빨강)가 <strong>상위 1, 2, 6위</strong> 차지<br>
        → 나머지 변수들의 기여도가 상대적으로 낮음
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### VIX-RV 상관관계")
    
    # 자산별 상관관계 vs 예측력
    asset_data = pd.DataFrame({
        '자산': ['GLD (금)', 'EFA (선진국)', 'EEM (신흥국)', 'SPY (S&P 500)'],
        'VIX-RV 상관': [0.51, 0.75, 0.69, 0.83],
        '예측력 R²': [0.37, 0.31, -0.21, 0.02]
    })
    
    fig2 = px.scatter(asset_data, x='VIX-RV 상관', y='예측력 R²', 
                      text='자산', size=[50, 40, 40, 40],
                      title='')
    fig2.update_traces(textposition='top center', marker=dict(color=['#2ecc71', '#3498db', '#e74c3c', '#e74c3c']))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # 추세선
    fig2.add_trace(go.Scatter(
        x=[0.5, 0.85], y=[0.4, -0.1],
        mode='lines', line=dict(dash='dash', color='purple'),
        name='추세 (r=-0.87)'
    ))
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>VIX-Beta 이론:</strong><br>
        VIX-RV 상관 높을수록 → 예측력 낮음<br>
        <strong>SPY:</strong> 상관 0.83 → R²=0.02 (예측 불가)<br>
        <strong>GLD:</strong> 상관 0.51 → R²=0.37 (예측 가능)
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 문제 설명
st.markdown("## 왜 RV 모델이 실패했나?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="problem-box">
        <h4>1. VIX가 이미 좋은 예측자</h4>
        <p>옵션 시장의 집단 지성이 이미 변동성을 잘 예측</p>
        <p><strong>VIX-RV 상관: 0.83</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="problem-box">
        <h4>2. 추가 변수 기여도 낮음</h4>
        <p>VIX_lag1, lag5가 계수 5.77, 5.47</p>
        <p>나머지는 대부분 <strong>2 이하</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="problem-box">
        <h4>3. 실질적 "VIX 따라가기"</h4>
        <p>모델 = VIX 래그의 선형 조합</p>
        <p><strong>새로운 가치 창출 불가</strong></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 해결책
st.markdown("## 해결책: RV → VRP로 전환")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="solution-box">
        <h4>VRP = VIX - RV (차이 예측)</h4>
        <ul>
            <li>VIX: 시장의 변동성 예상</li>
            <li>RV: 실제 변동성</li>
            <li><strong>VRP: "오차"를 예측</strong></li>
        </ul>
        <p>→ VIX가 틀리는 부분을 예측하므로 의미 있음</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="solution-box">
        <h4>VRP 모델 성과</h4>
        <table style="width: 100%;">
            <tr><td>R²</td><td><strong>0.19</strong></td></tr>
            <tr><td>방향 정확도</td><td><strong>73.5%</strong></td></tr>
            <tr><td>트레이딩 승률</td><td><strong>77.7%</strong></td></tr>
            <tr><td>거래당 초과수익</td><td><strong>+3.09%</strong></td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 최종 결론
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 12px; text-align: center;">
    <h2 style="margin: 0;">핵심 결론</h2>
    <p style="font-size: 1.2rem; margin: 1rem 0;">
        RV 예측 모델 = VIX의 선형 조합 → <strong>새로운 정보 없음</strong><br><br>
        VRP 예측 모델 = VIX와 RV의 "차이" 예측 → <strong>실질적 가치 창출</strong>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | VIX 상관관계 분석 기반")

