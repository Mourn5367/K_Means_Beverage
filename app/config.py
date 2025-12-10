"""
애플리케이션 설정
"""

import logging
import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 디렉토리
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# 정적 파일 디렉토리
STATIC_DIR = PROJECT_ROOT / "static"
VISUALIZATION_DIR = STATIC_DIR / "visualizations"

# 로그 디렉토리
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)

# 기본 K값
DEFAULT_K_VALUES = [2, 3, 4, 5]

# 피쳐 정의
CAFFEINE_FEATURES = [
    '에너지(kcal)', '단백질(g)', '당류(g)',
    '나트륨(mg)', '포화지방산(g)', '카페인(mg)'
]

NONCAFFEINE_FEATURES = [
    '에너지(kcal)', '단백질(g)', '당류(g)',
    '나트륨(mg)', '포화지방산(g)'
]

METADATA_COLUMNS = ['식품명', '업체명', '대표식품명']

# 앱 정보
APP_NAME = "K-Means 음료 군집화 API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "음료 데이터 K-means 군집화 및 분석 시스템"
