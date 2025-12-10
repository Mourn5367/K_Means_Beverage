"""
예측 API 라우트
"""

from fastapi import APIRouter, HTTPException, status
from app.models.schemas import (
    PredictionRequest, PredictionResponse,
    SimilarBeveragesRequest, SimilarBeveragesResponse,
    CompareBeveragesRequest, CompareBeveragesResponse
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

prediction_service = None


def init_prediction_routes(service):
    global prediction_service
    prediction_service = service


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    군집 예측

    - **beverage_type**: 'caffeine' 또는 'noncaffeine'
    - **k**: 군집 수
    - **features**: 피쳐 딕셔너리 (예: {"에너지(kcal)": 65.0, ...})
    """
    try:
        result = prediction_service.predict_cluster(
            request.beverage_type,
            request.k,
            request.features
        )
        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"예측 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/predict/similar", response_model=SimilarBeveragesResponse)
async def find_similar(request: SimilarBeveragesRequest):
    """
    유사 음료 검색

    - **beverage_type**: 'caffeine' 또는 'noncaffeine'
    - **k**: 군집 수
    - **features**: 피쳐 딕셔너리 (빈 값은 0으로 처리)
    - **n_neighbors**: 반환할 이웃 수 (기본값: 10)
    - **search_scope**: 'predicted_cluster' 또는 'all_clusters' (기본값: 'predicted_cluster')
    """
    try:
        # 디버깅: 요청 데이터 로그
        logger.info(f"[유사 음료 검색 요청]")
        logger.info(f"  - beverage_type: {request.beverage_type}")
        logger.info(f"  - k: {request.k}")
        logger.info(f"  - features: {request.features}")

        result = prediction_service.find_similar_beverages(
            request.beverage_type,
            request.k,
            request.features,
            request.n_neighbors,
            request.search_scope
        )
        return SimilarBeveragesResponse(**result)

    except ValueError as e:
        logger.error(f"[유사 음료 검색 실패 - ValueError] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"입력 값 오류: {str(e)}"
        )
    except Exception as e:
        logger.error(f"유사 음료 검색 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/compare", response_model=CompareBeveragesResponse)
async def compare(request: CompareBeveragesRequest):
    """
    두 음료 비교

    - **beverage_type**: 'caffeine' 또는 'noncaffeine'
    - **beverage1**: 첫 번째 음료 이름
    - **beverage2**: 두 번째 음료 이름
    """
    try:
        result = prediction_service.compare_beverages(
            request.beverage1,
            request.beverage2,
            request.beverage_type
        )
        return CompareBeveragesResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"음료 비교 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
