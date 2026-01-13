"""
검증 탭
"""
import streamlit as st
import pandas as pd

def render_validation():
    """검증 렌더링"""
    
    st.markdown('<div class="section-header">4. 검증</div>', unsafe_allow_html=True)
    
    # 데이터 누출 테스트
    st.markdown("### 4.1 데이터 누출 6-Fold 검증")
    
    validation_data = {
        '테스트': [
            '1. Shuffled Target',
            '2. Strict Temporal Split',
            '3. Extended Gap',
            '4. Scaler Leakage',
            '5. Autocorrelation',
            '6. Future Feature Control'
        ],
        '결과': [
            'R² = -0.02 (무작위 예측 불가)',
            'R² 안정적 유지',
            '5/22/44/66일 모두 안정',
            '차이 0.001 (무시 가능)',
            'lag 22 자기상관 = 0.002',
            'R²≈1.0 (탐지 확인)'
        ],
        '상태': [' PASS', ' PASS', ' PASS', ' PASS', ' PASS', ' PASS']
    }
    
    df = pd.DataFrame(validation_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Overlapping Window 테스트
    st.markdown("### 4.2 Overlapping Window 검증")
    
    st.markdown("""
    **문제**: 5일 타겟 윈도우가 연속 일간 표본에서 80% 중첩 → R² 부풀림 가능
    
    **테스트**: 전체 일간 테스트셋 vs 비중첩 표본 (5일 간격) 비교
    
    **결과** (Gold):
    - 중첩 R²: 0.8571
    - 비중첩 R²: 0.8616
    - 차이: **-0.0045** (-0.5%)
    
    **결론**: 중첩 윈도우 효과 무시 가능 
    """)
    
    # 복잡 모델 비교
    st.markdown("### 4.3 복잡 모델 비교 (과적합 검증)")
    
    complex_data = {
        '자산': ['Gold', 'EAFE', 'Treasury', 'S&P 500', 'Emerging', '평균'],
        'ElasticNet (기준)': [0.859, 0.753, 0.770, 0.725, 0.674, 1.000],
        'Stacking': [0.829, 0.552, 0.556, 0.347, 0.380, 0.693],
        '변화': ['-3.5%', '-26.7%', '-27.8%', '-52.1%', '-43.6%', '**-30.7%**']
    }
    
    df_complex = pd.DataFrame(complex_data)
    st.dataframe(df_complex, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **해석**: 
    - 복잡한 앙상블 모델(XGBoost, RF, GBM)이 평균 -30.7% 악화
    - Branco et al. (2023) 발견 재확인: "단순 선형 모델 > 복잡 ML"
    - 제한된 샘플 크기에서 과적합 발생
    """)
