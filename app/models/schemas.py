"""
API 요청/응답 스키마 정의 (Pydantic 모델)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from datetime import datetime


class ClusterRequest(BaseModel):
    """군집화 요청"""
    beverage_type: str = Field(..., pattern="^(caffeine|noncaffeine)$")
    k: int = Field(..., ge=2, le=20)
    force_retrain: bool = False

    @field_validator('k')
    @classmethod
    def validate_k(cls, v, info):
        """K 값 검증"""
        if v < 2:
            raise ValueError("K는 2 이상이어야 합니다")
        if v > 20:
            raise ValueError("K는 20 이하를 권장합니다")
        return v


class ClusterInfo(BaseModel):
    """군집 정보"""
    cluster_id: int
    size: int
    centroid: List[float]


class ClusterMetrics(BaseModel):
    """군집 품질 메트릭"""
    inertia: float
    silhouette_score: float
    silhouette_per_cluster: Dict[int, float]
    calinski_harabasz_score: float
    davies_bouldin_score: float
    cluster_sizes: Dict[int, int]


class ClusterResponse(BaseModel):
    """군집화 응답"""
    cluster_id: str
    beverage_type: str
    k: int
    n_samples: int
    features_used: List[str]
    clusters: List[ClusterInfo]
    metrics: ClusterMetrics
    trained_at: datetime
    training_time: float
    data_points: Optional[List[Dict[str, Any]]] = None


class PredictionRequest(BaseModel):
    """예측 요청"""
    beverage_type: str = Field(..., pattern="^(caffeine|noncaffeine)$")
    k: int = Field(..., ge=2, le=20)
    features: Dict[str, Optional[float]]  # null 값은 0으로 처리


class AlternativeCluster(BaseModel):
    """대안 군집 정보"""
    cluster_id: int
    distance: float


class PredictionResponse(BaseModel):
    """예측 응답"""
    predicted_cluster: int
    confidence: float
    distance_to_centroid: float
    alternative_clusters: List[AlternativeCluster]


class SimilarBeveragesRequest(BaseModel):
    """유사 음료 검색 요청"""
    beverage_type: str = Field(..., pattern="^(caffeine|noncaffeine)$")
    k: int = Field(..., ge=2, le=20)
    features: Dict[str, Optional[float]]  # null 값은 0으로 처리
    n_neighbors: int = Field(default=10, ge=1, le=50)
    search_scope: str = Field(default="predicted_cluster", pattern="^(predicted_cluster|all_clusters)$")


class SimilarBeverage(BaseModel):
    """유사 음료 정보"""
    rank: int
    beverage_name: str
    company: str
    food_type: str
    features: Dict[str, float]
    distance: float
    similarity_score: float


class SimilarBeveragesResponse(BaseModel):
    """유사 음료 검색 응답"""
    predicted_cluster: int
    similar_beverages: List[SimilarBeverage]
    search_summary: Dict[str, Any]


class FeatureStatistics(BaseModel):
    """피쳐 통계"""
    mean: float
    std: float
    variance: float
    min: float
    max: float
    median: float
    q1: float
    q3: float
    iqr: float


class RepresentativeType(BaseModel):
    """대표 식품명 정보"""
    name: str
    count: int
    percentage: float


class ClusterInsight(BaseModel):
    """군집 인사이트"""
    cluster_id: int
    name: str
    size: int
    percentage: float
    representative_types: List[RepresentativeType]
    feature_statistics: Dict[str, FeatureStatistics]
    centroid: Dict[str, float]
    characteristics: List[str]
    sample_beverages: List[Dict[str, Any]]


class InsightsResponse(BaseModel):
    """인사이트 응답"""
    beverage_type: str
    k: int
    clusters: List[ClusterInsight]
    global_statistics: Dict[str, Any]


class RetrainRequest(BaseModel):
    """재학습 요청"""
    beverage_type: str = Field(default="all", pattern="^(all|caffeine|noncaffeine)$")
    k_values: Optional[List[int]] = Field(default=[2, 3, 4, 5])


class RetrainResult(BaseModel):
    """재학습 결과"""
    beverage_type: str
    k: int
    status: str
    metrics: Optional[ClusterMetrics] = None
    error: Optional[str] = None


class RetrainResponse(BaseModel):
    """재학습 응답"""
    results: List[RetrainResult]
    total_time: float
    success_count: int
    failed_count: int


class OptimalKResponse(BaseModel):
    """최적 K 응답"""
    k_values: List[int]
    inertias: List[float]
    silhouettes: List[float]
    calinski_harabasz_scores: List[float]
    davies_bouldin_scores: List[float]
    recommended_k_elbow: int
    recommended_k_silhouette: int
    recommended_k_overall: int


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    timestamp: datetime
    models_loaded: Dict[str, List[int]]


class MetadataResponse(BaseModel):
    """메타데이터 응답"""
    total_beverages: int
    caffeine_beverages: int
    noncaffeine_beverages: int
    features: Dict[str, List[str]]
    representative_types: Optional[List[Dict[str, Any]]] = None


class BeverageInfo(BaseModel):
    """음료 정보"""
    beverage_name: str
    company: str
    food_type: str
    features: Dict[str, float]
    cluster_assignments: Dict[str, int]


class CompareBeveragesRequest(BaseModel):
    """음료 비교 요청"""
    beverage_type: str = Field(..., pattern="^(caffeine|noncaffeine)$")
    beverage1: str
    beverage2: str


class BeverageComparison(BaseModel):
    """음료 정보 (비교용)"""
    beverage_name: str
    company: str
    food_type: str
    features: Dict[str, float]
    cluster_assignment: Dict[str, int]


class CompareBeveragesResponse(BaseModel):
    """음료 비교 응답"""
    beverages: List[BeverageComparison]
    distance: float
    similarity_percentage: float
    feature_differences: Dict[str, Dict[str, float]]


class VisualizationFormat(BaseModel):
    """시각화 형식"""
    format: str = Field(default="png", pattern="^(png|json|html|svg)$")
    width: int = Field(default=800, ge=400, le=2000)
    height: int = Field(default=600, ge=300, le=1500)
