"""
프로젝트 설정 - 자산 정의
"""

# S&P 500을 메인 자산으로 우선 표시
ASSETS = {
    'SPY': {
        'name': 'S&P 500 (메인)',
        'group': 'Baseline',
        'color': '#4299e1',
        'ticker': 'SPY'
    },
    'GLD': {
        'name': 'Gold (금)',
        'group': 'Safety',
        'color': '#38b2ac',
        'ticker': 'GLD'
    },
    'TLT': {
        'name': 'Treasury (국채)',
        'group': 'Safety',
        'color': '#48bb78',
        'ticker': 'TLT'
    },
    'EFA': {
        'name': 'EAFE (선진국)',
        'group': 'Lag Effect',
        'color': '#667eea',
        'ticker': 'EFA'
    },
    'EEM': {
        'name': 'Emerging (신흥국)',
        'group': 'Lag Effect',
        'color': '#805ad5',
        'ticker': 'EEM'
    },
}

# 데이터 기간
DATA_START_DATE = '2015-01-01'
DATA_END_DATE = '2025-01-01'

# 결과 저장 경로
RESULTS_DIR = 'data/results'
FIGURES_DIR = 'results/figures'
MODELS_DIR = 'results/models'
