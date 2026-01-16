"""
결과 탭 - 종합 실험 결과 표시
Updated: 2026-01-16
"""
import streamlit as st
import pandas as pd

def render_results():
    """결과 렌더링 - 확장된 버전"""
    
    st.markdown('<div class="section-header">3. 실험 결과</div>', unsafe_allow_html=True)
    
    # ==========================================
    # 3.1 모델 성능 비교 (기존)
    # ==========================================
    st.markdown("### 3.1 벤치마크 성능 비교 (Test R²)")
    
    results_data = {
        '자산': ['S&P 500', 'Gold', 'Treasury', 'EAFE', 'Emerging', '**평균**'],
        'HAR-RV': [0.670, 0.855, 0.786, 0.705, 0.651, 0.733],
        'HAR+VIX': [0.683, 0.857, 0.789, 0.732, 0.661, 0.744],
        'CAVB (9)': [0.706, 0.857, 0.783, 0.732, 0.654, 0.746],
        '**29 Features**': [0.699, 0.873, 0.837, 0.742, 0.694, '**0.769**'],
        'vs HAR-RV': ['+4.3%', '+2.1%', '+6.5%', '+5.2%', '+6.6%', '**+4.9%**']
    }
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.info("""
    **핵심 발견**: 
    - 29개 Feature 구성이 평균 R² **0.769** 달성
    - HAR-RV 대비 **+4.9%** 개선
    - S&P 500에서만 CAVB 추가 변수가 통계적으로 유의 (p=0.008)
    """)
    
    # ==========================================
    # 3.2 모델 구성 비교 (신규)
    # ==========================================
    st.markdown("### 3.2 모델 구성별 성능 비교")
    
    with st.expander("구성별 상세 결과", expanded=True):
        config_data = {
            '구성': ['Baseline (9)', 'Enhanced (25)', 'Regime-Adaptive', '**29 Features**', '32 Features'],
            '변수 수': [9, 25, 9, 29, 32],
            '평균 R²': [0.740, 0.762, 0.756, '**0.770**', 0.768],
            'vs Baseline': ['-', '+2.97%', '+2.16%', '**+4.05%**', '+3.78%'],
            '주요 특징': ['HAR-RV+VIX+CAVB', '+16개 저위험 변수', 'VIX 구간별 모델 분리', 
                        '+VRP Decomposition', '+Good/Bad Vol (과적합 시작)']
        }
        
        df_config = pd.DataFrame(config_data)
        st.dataframe(df_config, use_container_width=True, hide_index=True)
        
        st.success("**최적 구성**: 29 Features (Baseline 25 + Group 2)")
    
    # ==========================================
    # 3.3 ML vs Linear 비교 (신규)
    # ==========================================
    st.markdown("### 3.3 ML vs Linear Model 비교")
    
    with st.expander("모델 유형별 성능", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            ml_data = {
                'Model': ['**ElasticNet**', 'Neural Network', 'XGBoost', 'LightGBM', 'Random Forest', 'Stacking'],
                'Avg R²': ['**0.770**', 0.707, 0.680, 0.672, 0.608, 0.540],
                '훈련 시간': ['0.15s', '0.52s', '0.21s', '0.06s', '0.31s', '2.36s'],
                '순위': ['**1위**', '2위', '3위', '4위', '5위', '6위']
            }
            
            df_ml = pd.DataFrame(ml_data)
            st.dataframe(df_ml, use_container_width=True, hide_index=True)
        
        with col2:
            overfit_data = {
                'Model': ['ElasticNet', 'Neural Network', 'XGBoost', 'Random Forest'],
                'Train R²': [0.782, 0.854, 0.792, 0.833],
                'Test R²': [0.770, 0.707, 0.680, 0.608],
                '과적합': ['1.5%', '17.2%', '14.1%', '27.0%']
            }
            
            df_overfit = pd.DataFrame(overfit_data)
            st.dataframe(df_overfit, use_container_width=True, hide_index=True)
        
        st.warning("""
        **결론**: ElasticNet이 모든 ML 모델을 능가
        - vs Neural Network: **+8.9%**
        - vs Stacking: **+42.6%**
        - 이유: 제한된 표본 크기(N~2,500)에서 정규화된 선형 모형이 과적합 방지에 효과적
        """)
    
    # ==========================================
    # 3.4 자산별 최적 구성 (신규)
    # ==========================================
    st.markdown("### 3.4 자산별 최적 구성")
    
    with st.expander("자산별 맞춤 전략"):
        asset_opt = {
            '자산': ['S&P 500', 'Gold', 'Treasury', 'EAFE', 'Emerging'],
            '최적 구성': ['Regime-Adaptive', '25+Group2', '25+Group2', '25+Group2', '25+Group3'],
            '최고 R²': [0.741, 0.875, 0.837, 0.743, 0.697],
            'vs Baseline': ['+6.8%', '+2.8%', '+5.9%', '+2.8%', '+8.2%'],
            '핵심 변수': ['VIX 구간 분리', 'VRP 분해', 'VRP 분해', 'VRP 분해', 'Good/Bad Vol']
        }
        
        df_asset = pd.DataFrame(asset_opt)
        st.dataframe(df_asset, use_container_width=True, hide_index=True)
    
    # ==========================================
    # 3.5 변수 중요도 Top 10 (확장)
    # ==========================================
    st.markdown("### 3.5 변수 중요도 Top 10 (29 Features)")
    
    with st.expander("핵심 변수 분석"):
        importance_data = {
            '순위': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            '변수': ['VRP_transitory', 'VIX_ma22', 'VRP_persistent', 'CAVB_std_22d', 
                    'RV_22d', 'CAVB_max_22d', 'RV_10d', 'CAVB_lag1', 'VIX_lag10', 'CAVB_min_22d'],
            '평균 계수': [0.821, 0.299, 0.305, 0.160, 0.071, 0.138, 0.088, 0.045, 0.052, 0.095],
            '선택률': ['100%', '100%', '100%', '80%', '100%', '80%', '80%', '80%', '60%', '60%'],
            '출처': ['Group 2', '신규', 'Group 2', '신규', '기본', '신규', '신규', '기본', '신규', '신규']
        }
        
        df_imp = pd.DataFrame(importance_data)
        st.dataframe(df_imp, use_container_width=True, hide_index=True)
        
        st.info("""
        **발견**:
        - Group 2 (VRP Decomposition) 변수 2개가 Top 3 진입
        - 신규 추가 변수 8개가 Top 10에 포함
        - 기본 9개 변수 중 2개만 Top 10 유지
        """)
    
    # ==========================================
    # 3.6 예측 시계 비교 (기존)
    # ==========================================
    st.markdown("### 3.6 예측 시계 비교 (1일 vs 5일 vs 22일)")
    
    horizon_data = {
        '자산': ['Gold', 'Treasury', 'EAFE', 'S&P 500', 'Emerging', '**평균**'],
        '1일 R²': [0.823, 0.751, 0.698, 0.664, 0.573, 0.682],
        '**5일 R²**': ['**0.857**', '**0.783**', '**0.732**', '**0.706**', '**0.654**', '**0.746**'],
        '22일 R²': [0.317, 0.082, 0.176, -0.045, -0.361, 0.097],
        '5일 vs 22일': ['+169%', '+855%', '+316%', '+1669%', '+281%', '**+717%**']
    }
    
    df_h = pd.DataFrame(horizon_data)
    st.dataframe(df_h, use_container_width=True, hide_index=True)
    
    st.success("""
    **결론**: 5일 예측이 최적
    - 1일 대비: **+9.4%** (노이즈 감소)
    - 22일 대비: **+717%** (정보 감쇠 회피)
    - Degiannakis decay 이론 재확인 (정보 감쇠율 8.5%/일)
    """)
    
    # ==========================================
    # 3.7 Universal Model 고급 실험 (신규)
    # ==========================================
    st.markdown("### 3.7 Universal Model 고급 실험 (12개 실험, 40+ 모델)")
    
    with st.expander("고급 모델 비교 실험", expanded=True):
        st.markdown("#### 최적 모델 순위")
        
        best_models = {
            '순위': [1, 2, 3, 4, 5],
            '모델': ['**Huber-Tuned (Optuna)**', 'SVR-Tuned', 'SVR-Linear', 'Stacking', 'ElasticNet'],
            '평균 R²': ['**0.789**', 0.787, 0.785, 0.768, 0.768],
            'vs Baseline': ['**+0.47%**', '+0.30%', '+0.00%', '+0.00%', '-'],
            '특징': ['100 trials HPO', '100 trials HPO', 'Robust Linear', 'Linear Ensemble', 'L1+L2 정규화']
        }
        
        df_best = pd.DataFrame(best_models)
        st.dataframe(df_best, use_container_width=True, hide_index=True)
        
        st.markdown("#### 모델 카테고리별 성능")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**성능 우수 (선형 모델)**")
            good_models = {
                '카테고리': ['Robust Linear', 'Linear', 'Linear Ensemble'],
                '대표 모델': ['Huber, SVR-Linear', 'ElasticNet, Ridge', 'Stacking, Voting'],
                '평균 R²': ['0.785-0.789', '0.768', '0.768']
            }
            st.dataframe(pd.DataFrame(good_models), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**성능 저조 (비선형 모델)**")
            bad_models = {
                '카테고리': ['Boosting', 'Tree', 'Deep Learning', 'Time Series'],
                '대표 모델': ['XGBoost, LightGBM', 'RF, ExtraTrees', 'LSTM, GRU', 'ARIMA, GARCH'],
                '평균 R²': ['0.65-0.70', '0.67-0.68', '0.00-0.75', '음수']
            }
            st.dataframe(pd.DataFrame(bad_models), use_container_width=True, hide_index=True)
        
        st.markdown("#### 자산별 최적 모델 성능")
        
        asset_results = {
            '자산': ['Gold', 'Treasury', 'EFA', 'S&P 500', 'Emerging'],
            'Huber-Tuned R²': ['**0.871**', '**0.828**', 0.775, 0.753, 0.717],
            'vs Baseline': ['+0.3%', '+0.2%', '+0.3%', '+0.7%', '+1.0%']
        }
        
        df_asset = pd.DataFrame(asset_results)
        st.dataframe(df_asset, use_container_width=True, hide_index=True)
    
    with st.expander("12개 실험 요약"):
        exp_summary = {
            '실험': ['Exp1', 'Exp2', 'Exp3-4', 'Exp5', 'Exp6', 'Exp7', 'Exp8', 'Exp9', 'Exp10', 'Exp11', 'Exp12'],
            '접근법': ['Cross-Asset LightGBM', 'Multi-Task Learning', 'Ensemble+StrongReg', 
                     'XGBoost/RF/Stacking', 'Huber/SVR-Linear', 'ARIMA/GARCH/LSTM',
                     'CatBoost/GP/PolyRidge', 'Optuna HPO (20)', 'Optuna HPO (100)',
                     'Advanced Features (57)', 'Feature Interaction'],
            '결과': ['-6.7%', '+1.0%', '-0.6%/-12%', '+0.1%', '**+2.3%**', 
                   '-101%', '+0.2%', '+0.7%', '**+0.47%**', '-21%', '+0.08%'],
            '결론': ['EFA만 개선', 'EEM 개선', '불안정', 'Linear 최고', '**최적 모델**',
                   '모두 실패', '동등', 'HPO 효과', '**미세 개선**', '과적합', '효과 미미']
        }
        
        df_exp = pd.DataFrame(exp_summary)
        st.dataframe(df_exp, use_container_width=True, hide_index=True)
    
    st.info("""
    **핵심 발견**:
    - **R² 0.789가 현재 데이터로 달성 가능한 최대치**
    - VRP 예측은 본질적으로 **선형 문제** (CAVB가 ~79% 설명)
    - 비선형 모델은 **과적합**으로 성능 저하
    - 추가 개선은 **새로운 데이터 소스** (옵션 Greeks, 고빈도 데이터) 필요
    """)
    
    st.code("""
# 최적 모델 설정
from sklearn.linear_model import HuberRegressor

model = HuberRegressor(
    epsilon=1.35,    # Outlier threshold
    alpha=1e-6,      # Regularization (very low)
    max_iter=1000
)
# Average R² = 0.789
    """, language='python')

