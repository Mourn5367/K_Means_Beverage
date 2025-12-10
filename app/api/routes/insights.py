"""
인사이트 API 라우트
"""

from fastapi import APIRouter, HTTPException, status
from app.models.schemas import InsightsResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

insights_service = None


def init_insights_routes(service):
    global insights_service
    insights_service = service


@router.get("/insights/{beverage_type}/{k}", response_model=InsightsResponse)
async def get_insights(beverage_type: str, k: int):
    """
    군집 인사이트 조회

    - **beverage_type**: 'caffeine' 또는 'noncaffeine'
    - **k**: 군집 수
    """
    try:
        insights = insights_service.get_cluster_insights(beverage_type, k)
        return InsightsResponse(**insights)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"인사이트 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


