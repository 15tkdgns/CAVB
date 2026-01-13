#!/usr/bin/env python3
"""
VRP 예측 연구 발표자료
====================

머신러닝을 활용한 변동성 위험 프리미엄 예측 연구
발표 및 보고서용 Streamlit 대시보드
Mermaid 다이어그램 및 연구 그래프 버전
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="VRP 예측 연구 발표",
    page_icon="📊",
    layout="wide"
)

# 이미지 경로 설정
IMAGES_PATH = Path(__file__).parent / "images"

def render_mermaid(code, height=400):
    """Mermaid 다이어그램 렌더링"""
    html = f"""
    <div class="mermaid" style="display: flex; justify-content: center;">
    {code}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true, theme:'neutral'}});</script>
    """
    components.html(html, height=height)

def display_research_image(image_name, caption=None, use_full_width=True):
    """연구 그래프 이미지 표시"""
    image_path = IMAGES_PATH / image_name
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=use_full_width)
    else:
        st.warning(f"이미지를 찾을 수 없습니다: {image_name}")

# CSS 스타일
st.markdown("""
<style>
    .slide-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        padding: 1.5rem;
        background: transparent;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .slide-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
    }
    .key-point {
        background: #f7fafc;
        color: #1a202c;
        border-left: 4px solid #4a5568;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
    }
    .explanation {
        background: #f0f9ff;
        border-left: 4px solid #3498db;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .mermaid {
        background: white;
        padding: 1rem;
        border-radius: 8px;
    }
    .research-figure {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 제목 슬라이드
# ============================================================================
st.markdown("""
<div class="slide-title">
    <h1 style="margin: 0; font-size: 1.8rem; color: #1a202c;">머신러닝을 활용한 변동성 위험 프리미엄 예측</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; color: #4a5568;">자산별 예측력 차이에 관한 연구</p>
    <hr style="border: 1px solid #e2e8f0; margin: 0.8rem 0;">
    <p style="margin: 0; font-size: 0.9rem; color: #718096;">2024년 12월</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 1. VRP 개념 다이어그램
# ============================================================================
st.markdown('<h2 class="section-header">1. VRP 개념</h2>', unsafe_allow_html=True)

# 연구 그래프 표시
col1, col2 = st.columns(2)
with col1:
    render_mermaid("""
flowchart LR
    subgraph 내재변동성["📈 내재변동성 (IV)"]
        VIX["VIX 지수<br/>옵션 시장 기대"]
    end
    
    subgraph 실현변동성["📉 실현변동성 (RV)"]
        RV["실제 변동성<br/>과거 데이터 기반"]
    end
    
    subgraph VRP결과["💰 VRP"]
        VRP["VRP = VIX - RV<br/>변동성 위험 프리미엄"]
    end
    
    VIX --> VRP
    RV --> VRP
    
    VRP -->|VRP > 0| PREMIUM["프리미엄 존재<br/>옵션 매도 수익 가능"]
    VRP -->|VRP < 0| DISCOUNT["디스카운트<br/>옵션 매수 유리"]
""", height=350)

with col2:
    st.markdown('<div class="research-figure">', unsafe_allow_html=True)
    display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_1.jpg", "VRP 개념 및 VIX-RV 관계")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
<strong>VRP(Volatility Risk Premium)</strong>: 옵션 시장이 예상하는 변동성(VIX)과 실제 실현된 변동성(RV)의 차이입니다.
VRP > 0이면 시장이 변동성을 과대평가하고 있어 변동성 매도 전략이 유리합니다.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 2. 연구 갭 다이어그램
# ============================================================================
st.markdown('<h2 class="section-header">2. 연구 갭</h2>', unsafe_allow_html=True)

render_mermaid("""
flowchart TB
    subgraph 기존연구["기존 연구"]
        A1["VIX만 사용"]
        A2["단일 자산 분석"]
        A3["정적 모델"]
    end
    
    subgraph 한계["한계점"]
        B1["예측력 낮음"]
        B2["일반화 어려움"]
        B3["시장 변화 미반영"]
    end
    
    subgraph 본연구["본 연구 기여"]
        C1["다중 특성 활용<br/>(VIX + RV + VRP lag)"]
        C2["다중 자산 비교<br/>(SPY, GLD, EFA, EEM)"]
        C3["ML 모델 도입<br/>(ElasticNet, MLP)"]
    end
    
    A1 --> B1 --> C1
    A2 --> B2 --> C2
    A3 --> B3 --> C3
""", height=400)

# ============================================================================
# 3. VRP 시계열 및 분포
# ============================================================================
st.markdown('<h2 class="section-header">3. VRP 시계열 및 분포 분석</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="research-figure">', unsafe_allow_html=True)
    display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_2.jpg", "Market Fear Premium 시계열")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="research-figure">', unsafe_allow_html=True)
    display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_3.jpg", "VRP 분포 히스토그램")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="key-point">
<strong>핵심 관찰:</strong> VRP는 평균적으로 양의 값을 가지며, 이는 옵션 시장에서 변동성 위험에 대한 프리미엄이 존재함을 의미합니다.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 4. 가설
# ============================================================================
st.markdown('<h2 class="section-header">4. 연구 가설</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="slide-card">
        <h4>H1: VRP 예측 가능성</h4>
        <p>과거 VIX, RV, VRP 정보를 활용하여 미래 VRP를 예측할 수 있다.</p>
        <div class="key-point">
            <strong>결과:</strong> R² = 0.19, 방향정확도 73.5%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="slide-card">
        <h4>H2: 자산별 차이</h4>
        <p>VIX-RV 상관관계가 낮은 자산일수록 VRP 예측력이 높다.</p>
        <div class="key-point">
            <strong>결과:</strong> GLD(R²=0.37) > SPY(R²=0.02)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="slide-card">
        <h4>H3: 경제적 가치</h4>
        <p>VRP 예측을 활용한 트레이딩 전략이 Buy & Hold를 초과한다.</p>
        <div class="key-point">
            <strong>결과:</strong> 77.7% 승률, +3.09%/거래
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 5. VIX-RV 상관관계 분석
# ============================================================================
st.markdown('<h2 class="section-header">5. VIX-RV 상관관계 분석</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="research-figure">', unsafe_allow_html=True)
    display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_4.jpg", "VIX-RV 상관관계 산점도")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # 산점도 차트
    asset_data = pd.DataFrame({
        '자산': ['GLD (금)', 'EFA (선진국)', 'EEM (신흥국)', 'SPY (S&P 500)'],
        'VIX-RV 상관': [0.51, 0.75, 0.69, 0.83],
        '예측력 R²': [0.37, 0.31, -0.21, 0.02]
    })

    fig2 = px.scatter(asset_data, x='VIX-RV 상관', y='예측력 R²', 
                      text='자산', size=[50, 40, 40, 40],
                      title='VIX-RV 상관관계 vs VRP 예측력')
    fig2.update_traces(textposition='top center', marker=dict(color=['#2ecc71', '#3498db', '#e74c3c', '#e74c3c']))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.add_trace(go.Scatter(
        x=[0.5, 0.85], y=[0.4, -0.1],
        mode='lines', line=dict(dash='dash', color='purple'),
        name='추세선 (r=-0.87)'
    ))
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div class="key-point">
<strong>VIX-Beta 이론:</strong> VIX-RV 상관관계가 높을수록 → VRP 예측력이 낮음<br/>
• <strong>SPY:</strong> 상관 0.83 → R²=0.02 (VIX가 이미 잘 설명)<br/>
• <strong>GLD:</strong> 상관 0.51 → R²=0.37 (VIX가 설명 못하는 부분 예측 가능)
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 6. 자산별 비교 분석
# ============================================================================
st.markdown('<h2 class="section-header">6. 자산별 VRP 비교 분석</h2>', unsafe_allow_html=True)

st.markdown('<div class="research-figure">', unsafe_allow_html=True)
display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_5.jpg", "SPY, GLD, EFA, EEM 자산별 VRP 비교")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# 7. 분석 파이프라인
# ============================================================================
st.markdown('<h2 class="section-header">7. 분석 파이프라인</h2>', unsafe_allow_html=True)

render_mermaid("""
flowchart LR
    subgraph 데이터["1️⃣ 데이터"]
        D1["yfinance<br/>SPY, VIX, GLD..."]
    end
    
    subgraph 전처리["2️⃣ 전처리"]
        P1["RV 계산<br/>(5d, 22d)"]
        P2["VRP 계산<br/>(VIX - RV)"]
        P3["래그 변수<br/>생성"]
    end
    
    subgraph 모델링["3️⃣ 모델링"]
        M1["ElasticNet"]
        M2["MLP"]
        M3["XGBoost"]
    end
    
    subgraph 검증["4️⃣ 검증"]
        V1["Purged K-Fold"]
        V2["Bootstrap CI"]
        V3["백테스트"]
    end
    
    D1 --> P1 --> P2 --> P3 --> M1 & M2 & M3 --> V1 --> V2 --> V3
""", height=300)

# ============================================================================
# 8. 모델 성능 비교
# ============================================================================
st.markdown('<h2 class="section-header">8. 모델 성능 비교</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="research-figure">', unsafe_allow_html=True)
    display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_6.jpg", "ElasticNet, MLP, XGBoost 모델 성능 비교")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    features_df = pd.DataFrame({
        '특성': ['VIX_lag1', 'VIX_lag5', 'RV_22d', 'VRP_lag5', 'VRP_ma5', 
                 'VIX_change', 'regime_high', 'RV_1d', 'RV_5d', 'VRP_lag1'],
        '계수': [5.77, 5.47, 4.25, 2.36, 1.88, 1.51, 1.22, 1.00, 0.92, 0.65],
        '유형': ['VIX', 'VIX', 'RV', 'VRP', 'VRP', 'VIX', '기타', 'RV', 'RV', 'VRP']
    })

    fig = px.bar(features_df, x='계수', y='특성', orientation='h',
                 color='유형',
                 color_discrete_map={'VIX': '#e74c3c', 'RV': '#3498db', 'VRP': '#2ecc71', '기타': '#95a5a6'},
                 text='계수',
                 title='ElasticNet 모델 특성 중요도')
    fig.update_traces(textposition='inside', texttemplate='%{text:.2f}', textfont_size=12)
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="key-point">
<strong>핵심 발견:</strong> VIX 관련 변수(lag1, lag5)가 예측력의 대부분을 차지 → 모델이 실질적으로 "VIX 따라가기"에 가까움
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 9. 특성 중요도 상세
# ============================================================================
st.markdown('<h2 class="section-header">9. 특성 중요도 상세 분석</h2>', unsafe_allow_html=True)

st.markdown('<div class="research-figure">', unsafe_allow_html=True)
display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_7.jpg", "Feature Importance 상세 분석")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# 10. MLP 구조 다이어그램
# ============================================================================
st.markdown('<h2 class="section-header">10. MLP 모델 구조</h2>', unsafe_allow_html=True)

render_mermaid("""
flowchart LR
    subgraph Input["입력층<br/>(12개 특성)"]
        I1["VIX_lag1"]
        I2["VIX_lag5"]
        I3["RV_22d"]
        I4["..."]
    end
    
    subgraph Hidden1["은닉층 1<br/>(64 뉴런)"]
        H1["ReLU<br/>Dropout 0.3"]
    end
    
    subgraph Hidden2["은닉층 2<br/>(32 뉴런)"]
        H2["ReLU<br/>Dropout 0.3"]
    end
    
    subgraph Output["출력층"]
        O1["VRP 예측값"]
    end
    
    I1 & I2 & I3 & I4 --> H1 --> H2 --> O1
""", height=300)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Test R²", "0.44", delta="+0.25 vs ElasticNet")
with col2:
    st.metric("아키텍처", "12-64-32-1", delta="2 hidden layers")
with col3:
    st.metric("정규화", "Dropout 0.3", delta="과적합 방지")

# ============================================================================
# 11. 데이터 분할 다이어그램
# ============================================================================
st.markdown('<h2 class="section-header">11. 데이터 분할 (Purged K-Fold)</h2>', unsafe_allow_html=True)

render_mermaid("""
gantt
    title Purged K-Fold Cross-Validation
    dateFormat X
    axisFormat %s
    
    section Fold 1
    Train     :done, 0, 60
    Purge     :crit, 60, 62
    Test      :active, 62, 80
    
    section Fold 2
    Train     :done, 0, 40
    Train     :done, 52, 80
    Purge     :crit, 40, 42
    Purge     :crit, 50, 52
    Test      :active, 42, 50
    
    section Fold 3
    Train     :done, 20, 80
    Purge     :crit, 18, 20
    Test      :active, 0, 18
""", height=250)

st.markdown("""
<div class="explanation">
<strong>Purged K-Fold</strong>: 금융 시계열에서 데이터 누출을 방지하기 위해 학습/테스트 세트 사이에 22일(타겟 계산 기간)의 간격(Purge)을 둠
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 12. 백테스트 결과
# ============================================================================
st.markdown('<h2 class="section-header">12. 백테스트 결과</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="research-figure">', unsafe_allow_html=True)
    display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_8.jpg", "누적 수익률 백테스트")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="research-figure">', unsafe_allow_html=True)
    display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_9.jpg", "트레이딩 전략 성과 비교")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# 13. Regime 분석
# ============================================================================
st.markdown('<h2 class="section-header">13. Regime별 VRP 분석</h2>', unsafe_allow_html=True)

st.markdown('<div class="research-figure">', unsafe_allow_html=True)
display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_10.jpg", "Regime별 (High/Low Volatility) VRP 분석")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# 14. 성과 요약
# ============================================================================
st.markdown('<h2 class="section-header">14. 모델 성과 요약</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.19</div>
        <div class="metric-label">Test R² (ElasticNet)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">73.5%</div>
        <div class="metric-label">방향 예측 정확도</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">77.7%</div>
        <div class="metric-label">트레이딩 승률</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">+3.09%</div>
        <div class="metric-label">거래당 초과수익</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 15. 결론 다이어그램
# ============================================================================
st.markdown('<h2 class="section-header">15. 결론</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    render_mermaid("""
flowchart TB
    subgraph 연구질문["연구 질문"]
        Q1["VRP 예측 가능한가?"]
        Q2["자산별 차이 존재?"]
        Q3["경제적 가치 있는가?"]
    end
    
    subgraph 결과["연구 결과"]
        R1["✓ R²=0.19<br/>방향정확도 73.5%"]
        R2["✓ VIX-Beta 이론<br/>GLD > SPY"]
        R3["✓ 77.7% 승률<br/>+3.09%/거래"]
    end
    
    subgraph 시사점["시사점"]
        I1["VRP 예측의<br/>현실적 상한선 제시"]
        I2["자산별 전략<br/>차별화 근거"]
        I3["리스크 관리<br/>활용 가치"]
    end
    
    Q1 --> R1 --> I1
    Q2 --> R2 --> I2
    Q3 --> R3 --> I3
""", height=350)

with col2:
    st.markdown('<div class="research-figure">', unsafe_allow_html=True)
    display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_11.jpg", "연구 결론 및 시사점")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# 16. 연구 흐름
# ============================================================================
st.markdown('<h2 class="section-header">16. 연구 흐름 요약</h2>', unsafe_allow_html=True)

render_mermaid("""
flowchart LR
    A["문헌 검토"] --> B["데이터 수집<br/>yfinance"]
    B --> C["VRP 계산<br/>VIX - RV"]
    C --> D["특성 공학<br/>래그, MA"]
    D --> E["모델 학습<br/>ElasticNet, MLP"]
    E --> F["검증<br/>Purged K-Fold"]
    F --> G["VIX-Beta<br/>이론 도출"]
    G --> H["경제적 가치<br/>검증"]
    H --> I["결론 및<br/>향후 연구"]
""", height=200)

# ============================================================================
# 17. 향후 연구
# ============================================================================
st.markdown('<h2 class="section-header">17. 한계 및 향후 연구</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 한계점
    - VIX 래그 변수 의존도 높음
    - 단일 시장(미국) 분석
    - 거래비용 미반영
    - 실시간 예측 미검증
    """)

with col2:
    st.markdown('<div class="research-figure">', unsafe_allow_html=True)
    display_research_image("Fear_Premium_Decoding_Prediction_and_Profit_12.jpg", "향후 연구 방향")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    #### 향후 연구 방향
    - 다른 자산군 확장 (암호화폐, 채권)
    - 고빈도 데이터 활용
    - Regime-Switching 모델
    - 실시간 트레이딩 시스템
    """)

# ============================================================================
# 참고문헌
# ============================================================================
st.markdown('<h2 class="section-header">참고문헌</h2>', unsafe_allow_html=True)

with st.expander("References"):
    st.markdown("""
1. **Lopez de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley.
2. **Corsi, F. (2009)**. A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*.
3. **Bollerslev, T., Tauchen, G., & Zhou, H. (2009)**. Expected Stock Returns and Variance Risk Premia. *Review of Financial Studies*.
4. **Christoffersen, P., & Mazzotta, S. (2005)**. The accuracy of density forecasts from foreign exchange options. *Journal of Financial Econometrics*.
5. **Bekaert, G., & Hoerova, M. (2014)**. The VIX, the variance premium and stock market volatility. *Journal of Econometrics*.
""")
