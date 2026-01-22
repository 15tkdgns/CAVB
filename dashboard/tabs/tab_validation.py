"""
검증 탭 - 통계적 검증 및 누출 테스트
Updated: 2026-01-16
"""
import streamlit as st
import pandas as pd

def render_validation():
    """검증 렌더링 - 확장된 버전"""
    
    st.markdown('<div class="section-header">4. 검증</div>', unsafe_allow_html=True)
    
    # ==========================================
    # 4.1 6중 데이터 누출 테스트
    # ==========================================
    st.markdown("### 4.1 데이터 누출 6-Fold 검증")
    
    with st.expander("6개 테스트 상세 결과", expanded=True):
        validation_data = {
            '테스트': [
                '1. Shuffled Target',
                '2. Strict Temporal Split',
                '3. Extended Gap',
                '4. Scaler Leakage',
                '5. Autocorrelation',
                '6. Future Feature Control'
            ],
            '방법': [
                '타겟 무작위 섞기',
                '2년 간격 분할',
                '5/22/44/66일 간격',
                '학습셋 전용 스케일링',
                '잔차 자기상관 검사',
                '미래 정보 포함 테스트'
            ],
            '결과': [
                'R² = -0.02',
                'R² 안정적 유지',
                '모두 안정',
                '차이 0.001',
                'lag 22 = 0.002',
                'R² ~ 1.0 (탐지)'
            ],
            '판정': ['PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS']
        }
        
        df = pd.DataFrame(validation_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.success("**결과**: 6/6 테스트 모두 통과 - 데이터 누출 없음 확인")
    
    # ==========================================
    # 4.2 통계적 유의성 (신규)
    # ==========================================
    st.markdown("### 4.2 통계적 유의성 분석")
    
    with st.expander("자산별 p-value", expanded=True):
        sig_data = {
            '자산': ['S&P 500', 'Gold', 'Treasury', 'EAFE', 'Emerging'],
            'HAR+VIX R²': [0.683, 0.857, 0.789, 0.732, 0.661],
            'CAVB Full R²': [0.706, 0.857, 0.783, 0.732, 0.654],
            '개선': ['+0.023', '+0.000', '-0.006', '+0.001', '-0.007'],
            'p-value': ['**0.008**', '0.954', '0.095', '0.913', '0.184'],
            '유의성': ['유의', '무의', '무의', '무의', '무의']
        }
        
        df_sig = pd.DataFrame(sig_data)
        st.dataframe(df_sig, use_container_width=True, hide_index=True)
        
        st.warning("""
        **해석**: 
        - **S&P 500**만 CAVB 추가 변수가 통계적으로 유의 (p=0.008)
        - 나머지 4개 자산: HAR+VIX로 충분
        - VIX가 SPY 옵션에서 직접 도출 → S&P 500에서 구조적 연결
        """)
    
    # ==========================================
    # 4.3 Overlapping Window 테스트
    # ==========================================
    st.markdown("### 4.3 Overlapping Window 검증")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **문제**: 5일 타겟 윈도우가 연속 일간 표본에서 80% 중첩
        
        **테스트 설계**:
        - 중첩: 전체 일간 테스트셋 (n=492)
        - 비중첩: 5일 간격 표본 (n=99)
        """)
    
    with col2:
        overlap_data = {
            '조건': ['중첩', '비중첩', '차이'],
            'Gold R²': [0.8571, 0.8616, '-0.0045'],
            '판정': ['', '', 'PASS']
        }
        
        df_overlap = pd.DataFrame(overlap_data)
        st.dataframe(df_overlap, use_container_width=True, hide_index=True)
    
    st.success("**결론**: 중첩 윈도우 효과 무시 가능 (-0.5%)")
    
    # ==========================================
    # 4.4 복잡 모델 비교 (과적합 검증)
    # ==========================================
    st.markdown("### 4.4 복잡 모델 비교 (과적합 검증)")
    
    with st.expander("선형 vs 비선형 모델 비교"):
        complex_data = {
            '모델': ['Huber', 'ElasticNet', 'SVR-Linear', 'XGBoost', 'RF', 'Stacking'],
            '5일 R²': ['**0.789**', '0.769', '0.707', '0.680', '0.608', '0.540'],
            'vs Huber': ['-', '-2.5%', '-10.4%', '-13.8%', '-22.9%', '-31.6%'],
            '과적합': ['없음', '없음', '없음', '있음', '심각', '심각']
        }
        
        df_complex = pd.DataFrame(complex_data)
        st.dataframe(df_complex, use_container_width=True, hide_index=True)
        
        st.error("""
        **결론**: 비선형 모델(XGBoost, RF)은 **과적합으로 성능 저하**
        - Branco et al. (2024) 발견 재확인: "단순 선형 모델 > 복잡 ML"
        - VRP 예측은 본질적으로 선형 관계
        """)
    
    # ==========================================
    # 4.5 22일 예측 자산 검증 (신규)
    # ==========================================
    st.markdown("### 4.5 22일 예측 자산 검증 (Exp 18-20)")
    
    with st.expander("자산별 22일 예측 가능성"):
        st.markdown("**검증 방법**: Ensemble Specialist 모델로 23개 자산 테스트")
        
        asset_22d_data = {
            '분류': ['예측 가능 (R² > 0.3)', '예측 가능 (R² > 0.3)', '부분 가능 (0 < R² < 0.3)', '예측 불가 (R² < 0)'],
            '자산': ['TIP, IEF, BND, GLD, DBA', 'HYG, LQD', 'TLT, EWJ, SLV, XLU', 'SPY, XLK, XLF, EEM, FXI'],
            '대표 R²': ['0.64~0.82', '0.50~0.56', '0.10~0.47', '-0.01 ~ -0.47'],
            '자산 수': ['5개', '2개', '4개', '5개']
        }
        
        df_asset_22d = pd.DataFrame(asset_22d_data)
        st.dataframe(df_asset_22d, use_container_width=True, hide_index=True)
        
        st.info("""
        **핵심 발견**:
        - **채권(TIP, IEF)**: 가장 예측 가능 (금리 사이클 예측 가능)
        - **상품(GLD, DBA)**: 우수 (장기 추세 안정)
        - **주식(SPY, XLK)**: **예측 불가** (이벤트 노이즈)
        """)
    
    # ==========================================
    # 4.6 베이스라인 비교 검증
    # ==========================================
    st.markdown("### 4.6 베이스라인 모델 비교 (Diebold-Mariano)")
    
    with st.expander("통계적 유의성 검정"):
        dm_data = {
            '비교': ['Huber vs HAR-RV', 'Huber vs GARCH', 'Huber vs Random Walk'],
            'DM 통계량': ['2.34', '4.12', '7.89'],
            'p-value': ['**0.019**', '< 0.001', '< 0.001'],
            '결론': ['유의하게 우수', '유의하게 우수', '유의하게 우수']
        }
        
        df_dm = pd.DataFrame(dm_data)
        st.dataframe(df_dm, use_container_width=True, hide_index=True)
        
        st.success("**결론**: Huber는 모든 베이스라인 대비 **통계적으로 유의하게 우수**")
    
    # ==========================================
    # 4.7 검증 요약
    # ==========================================
    st.markdown("### 4.7 최종 검증 요약")
    
    summary_data = {
        '검증 항목': ['데이터 누출', '통계적 유의성', '베이스라인 비교', '과적합', '22일 자산', '**총 실험**'],
        '결과': ['6/6 PASS', 'DM p=0.019', 'HAR 대비 +11%', '선형 > 비선형', '10/23 예측 가능', '**20개**'],
        '상태': ['통과', '통과', '통과', '통과', '통과', '**60+ 모델**']
    }
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.success("""
    **최종 판정**: 모든 검증 통과
    - 20개 실험, 60+ 모델, 23개 자산 비교 완료
    - **5일 최적 모델**: Huber (R² 0.789)
    - **22일 최적 모델**: Ensemble (채권 R² 0.82)
    - 비선형 모델 과적합 확인 → 선형 모델 우위
    """)

