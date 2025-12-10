"""
FastAPI 메인 애플리케이션
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from datetime import datetime
import logging

from app.config import APP_NAME, APP_VERSION, APP_DESCRIPTION, DEFAULT_K_VALUES
from app.services.data_preprocessor import DataPreprocessor
from app.services.model_cache import ModelCache
from app.services.clustering_service import ClusteringService
from app.services.insights_service import InsightsService
from app.services.prediction_service import PredictionService
from app.api.routes import clustering, insights, prediction
from app.models.schemas import HealthResponse, MetadataResponse

logger = logging.getLogger(__name__)

# 전역 서비스 인스턴스
data_preprocessor = None
model_cache = None
clustering_service_instance = None
insights_service_instance = None
prediction_service_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 이벤트"""
    # 시작
    logger.info("=" * 60)
    logger.info(f"{APP_NAME} v{APP_VERSION} 시작")
    logger.info("=" * 60)

    global data_preprocessor, model_cache, clustering_service_instance
    global insights_service_instance, prediction_service_instance

    try:
        # 1. 데이터 전처리 서비스 초기화
        logger.info("1. 데이터 전처리 서비스 초기화 중...")
        data_preprocessor = DataPreprocessor()
        caffeine_df, noncaffeine_df = data_preprocessor.load_and_split()
        logger.info(f"   ✓ 카페인 음료: {len(caffeine_df)}개")
        logger.info(f"   ✓ 논카페인 음료: {len(noncaffeine_df)}개")

        # 2. 모델 캐시 초기화
        logger.info("2. 모델 캐시 초기화 중...")
        model_cache = ModelCache()

        # 3. 서비스 초기화
        logger.info("3. 서비스 초기화 중...")
        clustering_service_instance = ClusteringService(data_preprocessor, model_cache)
        insights_service_instance = InsightsService(model_cache)
        prediction_service_instance = PredictionService(data_preprocessor, model_cache)

        # 4. 라우트에 서비스 주입
        logger.info("4. API 라우트 설정 중...")
        clustering.init_clustering_routes(clustering_service_instance)
        insights.init_insights_routes(insights_service_instance)
        prediction.init_prediction_routes(prediction_service_instance)

        # 5. 사전 학습된 모델 로드
        logger.info("5. 사전 학습된 모델 로드 중...")
        model_cache.preload_models(DEFAULT_K_VALUES)

        cache_info = model_cache.get_cache_info()
        logger.info(f"   ✓ 카페인 모델: {cache_info['caffeine']['disk_available']}")
        logger.info(f"   ✓ 논카페인 모델: {cache_info['noncaffeine']['disk_available']}")

        logger.info("=" * 60)
        logger.info("서버 초기화 완료!")
        logger.info(f"API 문서: http://localhost:8000/docs")
        logger.info(f"대체 문서: http://localhost:8000/redoc")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.error(f"서버 초기화 실패: {e}")
        raise

    # 종료
    logger.info("서버 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan
)

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 정적 파일 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(clustering.router, prefix="/api/v1", tags=["Clustering"])
app.include_router(insights.router, prefix="/api/v1", tags=["Insights"])
app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])


@app.get("/", tags=["Root"])
async def root(request: Request):
    """웹 인터페이스"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Utility"])
async def health_check():
    """헬스체크"""
    if model_cache is None:
        return HealthResponse(
            status="initializing",
            timestamp=datetime.now(),
            models_loaded={"caffeine": [], "noncaffeine": []}
        )

    models_loaded = model_cache.list_all()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=models_loaded
    )


@app.get("/api/v1/metadata", response_model=MetadataResponse, tags=["Utility"])
async def get_metadata():
    """데이터셋 메타데이터"""
    if data_preprocessor is None:
        return MetadataResponse(
            total_beverages=0,
            caffeine_beverages=0,
            noncaffeine_beverages=0,
            features={}
        )

    caffeine_df, noncaffeine_df = data_preprocessor.caffeine_df, data_preprocessor.noncaffeine_df

    # 대표식품명 분포
    all_df = data_preprocessor.get_data('caffeine')
    representative_types = all_df['대표식품명'].value_counts().head(10)
    rep_types_list = [
        {"name": name, "count": int(count)}
        for name, count in representative_types.items()
    ]

    return MetadataResponse(
        total_beverages=len(caffeine_df) + len(noncaffeine_df),
        caffeine_beverages=len(caffeine_df),
        noncaffeine_beverages=len(noncaffeine_df),
        features={
            "caffeine": data_preprocessor.caffeine_features,
            "noncaffeine": data_preprocessor.noncaffeine_features
        },
        representative_types=rep_types_list
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)
