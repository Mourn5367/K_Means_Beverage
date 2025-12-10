"""
군집화 API 라우트
"""

from fastapi import APIRouter, HTTPException, status
from app.models.schemas import (
    ClusterRequest, ClusterResponse, ClusterInfo, ClusterMetrics,
    RetrainRequest, RetrainResponse, RetrainResult,
    OptimalKResponse
)
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# 전역 서비스 인스턴스 (main.py에서 주입)
clustering_service = None


def init_clustering_routes(service):
    """서비스 인스턴스 주입"""
    global clustering_service
    clustering_service = service


@router.post("/cluster", response_model=ClusterResponse)
async def perform_clustering(request: ClusterRequest):
    """
    K-means 군집화 수행

    - **beverage_type**: 'caffeine' 또는 'noncaffeine'
    - **k**: 군집 수 (2-20)
    - **force_retrain**: 강제 재학습 여부 (기본값: false)
    """
    try:
        logger.info(f"군집화 요청: {request.beverage_type} k={request.k}")

        model_data = clustering_service.train_kmeans(
            request.beverage_type,
            request.k,
            request.force_retrain
        )

        # 응답 구성
        clusters = []
        for i in range(request.k):
            clusters.append(ClusterInfo(
                cluster_id=i,
                size=model_data['metrics']['cluster_sizes'][i],
                centroid=model_data['centroids'][i].tolist()
            ))

        # 각 데이터 포인트의 상세 정보 포함
        df = model_data['df'].copy()
        df['cluster'] = model_data['labels']
        data_points = df.to_dict('records')

        response = ClusterResponse(
            cluster_id=f"{request.beverage_type}_k{request.k}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            beverage_type=request.beverage_type,
            k=request.k,
            n_samples=model_data['n_samples'],
            features_used=model_data['features'],
            clusters=clusters,
            metrics=ClusterMetrics(**model_data['metrics']),
            trained_at=model_data['trained_at'],
            training_time=model_data['training_time'],
            data_points=data_points
        )

        logger.info(f"군집화 완료: {request.beverage_type} k={request.k}")
        return response

    except Exception as e:
        logger.error(f"군집화 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/clusters/{beverage_type}/{k}")
async def get_cluster(beverage_type: str, k: int):
    """
    캐시된 군집화 결과 조회

    - **beverage_type**: 'caffeine' 또는 'noncaffeine'
    - **k**: 군집 수
    """
    try:
        model_data = clustering_service.get_model(beverage_type, k)

        if model_data is None:
            available = clustering_service.model_cache.list_available(beverage_type)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "message": f"모델을 찾을 수 없습니다: {beverage_type} k={k}",
                    "suggestion": "POST /api/v1/cluster로 모델을 먼저 학습하세요",
                    "available_models": available
                }
            )

        # 응답 구성
        clusters = []
        for i in range(k):
            clusters.append(ClusterInfo(
                cluster_id=i,
                size=model_data['metrics']['cluster_sizes'][i],
                centroid=model_data['centroids'][i].tolist()
            ))

        # 각 데이터 포인트의 상세 정보 포함
        df = model_data['df'].copy()
        df['cluster'] = model_data['labels']
        data_points = df.to_dict('records')

        response = ClusterResponse(
            cluster_id=f"{beverage_type}_k{k}",
            beverage_type=beverage_type,
            k=k,
            n_samples=model_data['n_samples'],
            features_used=model_data['features'],
            clusters=clusters,
            metrics=ClusterMetrics(**model_data['metrics']),
            trained_at=model_data['trained_at'],
            training_time=model_data.get('training_time', 0.0),
            data_points=data_points
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"군집 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_models(request: RetrainRequest):
    """
    모델 재학습

    - **beverage_type**: 'all', 'caffeine', 또는 'noncaffeine'
    - **k_values**: 재학습할 K값 리스트 (기본값: [2,3,4,5])
    """
    try:
        logger.info(f"재학습 요청: {request.beverage_type} k_values={request.k_values}")

        result = clustering_service.retrain_models(
            request.beverage_type,
            request.k_values
        )

        # 응답 구성
        retrain_results = []
        for r in result['results']:
            if r['status'] == 'success':
                retrain_results.append(RetrainResult(
                    beverage_type=r['beverage_type'],
                    k=r['k'],
                    status=r['status'],
                    metrics=ClusterMetrics(**r['metrics'])
                ))
            else:
                retrain_results.append(RetrainResult(
                    beverage_type=r['beverage_type'],
                    k=r['k'],
                    status=r['status'],
                    error=r.get('error')
                ))

        response = RetrainResponse(
            results=retrain_results,
            total_time=result['total_time'],
            success_count=result['success_count'],
            failed_count=result['failed_count']
        )

        logger.info(f"재학습 완료: 성공 {result['success_count']}, 실패 {result['failed_count']}")
        return response

    except Exception as e:
        logger.error(f"재학습 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/optimal_k/{beverage_type}", response_model=OptimalKResponse)
async def find_optimal_k(beverage_type: str, k_min: int = 2, k_max: int = 11):
    """
    최적 K값 탐색 (Elbow method)

    - **beverage_type**: 'caffeine' 또는 'noncaffeine'
    - **k_min**: 최소 K (기본값: 2)
    - **k_max**: 최대 K (기본값: 11, 미포함)
    """
    try:
        if beverage_type not in ['caffeine', 'noncaffeine']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="beverage_type은 'caffeine' 또는 'noncaffeine'이어야 합니다"
            )

        logger.info(f"최적 K 탐색: {beverage_type} range=({k_min},{k_max})")

        result = clustering_service.find_optimal_k(beverage_type, (k_min, k_max))

        response = OptimalKResponse(**result)

        logger.info(f"최적 K 탐색 완료: 추천 K={result['recommended_k_overall']}")
        return response

    except Exception as e:
        logger.error(f"최적 K 탐색 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
