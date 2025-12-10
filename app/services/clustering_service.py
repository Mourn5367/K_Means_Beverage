"""
군집화 서비스
K-means 학습, 메트릭 계산, 최적 K 탐색
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.spatial.distance import pdist, squareform
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ClusteringService:
    """군집화 서비스 클래스"""

    def __init__(self, data_preprocessor, model_cache):
        """
        Args:
            data_preprocessor: DataPreprocessor 인스턴스
            model_cache: ModelCache 인스턴스
        """
        self.data_preprocessor = data_preprocessor
        self.model_cache = model_cache

    def train_kmeans(self, beverage_type, k, force_retrain=False):
        """
        K-means 모델 학습

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k (int): 군집 수
            force_retrain (bool): 강제 재학습 여부

        Returns:
            dict: 모델 데이터
        """
        # 1. 캐시 확인
        if not force_retrain:
            cached_model = self.model_cache.get(beverage_type, k)
            if cached_model is not None:
                logger.info(f"캐시된 모델 사용: {beverage_type} k={k}")
                return cached_model

        logger.info(f"K-means 학습 시작: {beverage_type} k={k}")
        start_time = datetime.now()

        # 2. 데이터 준비
        data = self.data_preprocessor.prepare_clustering_data(beverage_type)
        X_scaled = data['X_scaled']
        scaler = data['scaler']
        features = data['features']
        metadata = data['metadata']
        df = data['df']

        # 3. K-means 학습
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42,
            algorithm='elkan'
        )

        labels = kmeans.fit_predict(X_scaled)

        # 빈 군집 처리
        unique_labels = np.unique(labels)
        if len(unique_labels) < k:
            logger.warning(f"빈 군집 발생! 재학습 시도 (n_init=20)")
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=20,
                max_iter=300,
                random_state=None,
                algorithm='elkan'
            )
            labels = kmeans.fit_predict(X_scaled)

            if len(np.unique(labels)) < k:
                raise ValueError(f"K={k}로 {k}개의 비어있지 않은 군집을 생성할 수 없습니다.")

        # 4. 메트릭 계산
        metrics = self.calculate_metrics(X_scaled, labels, kmeans.cluster_centers_)

        # 5. 모델 데이터 구성
        model_data = {
            'kmeans': kmeans,
            'labels': labels,
            'scaler': scaler,
            'features': features,
            'X_scaled': X_scaled,
            'X_original': data['X'],
            'centroids': kmeans.cluster_centers_,
            'metadata': metadata,
            'df': df,
            'metrics': metrics,
            'beverage_type': beverage_type,
            'k': k,
            'n_samples': len(X_scaled),
            'trained_at': datetime.now(),
            'training_time': (datetime.now() - start_time).total_seconds()
        }

        # 6. 캐시에 저장
        self.model_cache.set(beverage_type, k, model_data)

        logger.info(f"K-means 학습 완료: {beverage_type} k={k}")
        logger.info(f"  - 학습 시간: {model_data['training_time']:.2f}초")
        logger.info(f"  - Inertia: {metrics['inertia']:.2f}")
        logger.info(f"  - Silhouette: {metrics['silhouette_score']:.4f}")

        return model_data

    def calculate_metrics(self, X, labels, centroids):
        """
        군집 품질 메트릭 계산

        Args:
            X (np.ndarray): 스케일링된 데이터
            labels (np.ndarray): 군집 레이블
            centroids (np.ndarray): 군집 중심

        Returns:
            dict: 메트릭 딕셔너리
        """
        metrics = {}

        # 1. Inertia (군집 내 제곱합)
        inertia = sum([
            np.linalg.norm(X[labels == i] - centroids[i]) ** 2
            for i in range(len(centroids))
        ])
        metrics['inertia'] = float(inertia)

        # 2. Silhouette Score (-1~1, 높을수록 좋음)
        silhouette_avg = silhouette_score(X, labels)
        metrics['silhouette_score'] = float(silhouette_avg)

        # 3. 군집별 Silhouette Score
        silhouette_vals = silhouette_samples(X, labels)
        silhouette_per_cluster = {}
        for i in range(len(centroids)):
            cluster_silhouette = silhouette_vals[labels == i].mean()
            silhouette_per_cluster[int(i)] = float(cluster_silhouette)
        metrics['silhouette_per_cluster'] = silhouette_per_cluster

        # 4. Calinski-Harabasz Score (높을수록 좋음)
        ch_score = calinski_harabasz_score(X, labels)
        metrics['calinski_harabasz_score'] = float(ch_score)

        # 5. Davies-Bouldin Score (낮을수록 좋음)
        db_score = davies_bouldin_score(X, labels)
        metrics['davies_bouldin_score'] = float(db_score)

        # 6. 군집 크기 분포
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}
        metrics['cluster_sizes'] = cluster_sizes

        # 7. 군집 간 거리
        centroid_distances = squareform(pdist(centroids))
        metrics['inter_cluster_distances'] = centroid_distances.tolist()

        return metrics

    def find_optimal_k(self, beverage_type, k_range=(2, 11)):
        """
        최적 K 탐색 (Elbow method)

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k_range (tuple): (min_k, max_k)

        Returns:
            dict: 각 K별 메트릭 및 추천 K
        """
        logger.info(f"최적 K 탐색 시작: {beverage_type}, range={k_range}")

        # 데이터 준비
        data = self.data_preprocessor.prepare_clustering_data(beverage_type)
        X_scaled = data['X_scaled']

        results = {
            'k_values': [],
            'inertias': [],
            'silhouettes': [],
            'calinski_harabasz_scores': [],
            'davies_bouldin_scores': []
        }

        for k in range(k_range[0], k_range[1]):
            logger.info(f"  K={k} 테스트 중...")

            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42,
                algorithm='elkan'
            )
            labels = kmeans.fit_predict(X_scaled)

            results['k_values'].append(k)
            results['inertias'].append(kmeans.inertia_)
            results['silhouettes'].append(silhouette_score(X_scaled, labels))
            results['calinski_harabasz_scores'].append(calinski_harabasz_score(X_scaled, labels))
            results['davies_bouldin_scores'].append(davies_bouldin_score(X_scaled, labels))

        # Elbow method로 최적 K 찾기
        inertias_normalized = np.array(results['inertias']) / results['inertias'][0]
        if len(inertias_normalized) > 2:
            second_derivative = np.diff(inertias_normalized, 2)
            elbow_k = np.argmax(second_derivative) + k_range[0] + 1
        else:
            elbow_k = k_range[0] + 1

        # Silhouette score 기준 최적 K
        silhouette_k = results['k_values'][np.argmax(results['silhouettes'])]

        results['recommended_k_elbow'] = int(elbow_k)
        results['recommended_k_silhouette'] = int(silhouette_k)
        results['recommended_k_overall'] = int(silhouette_k)  # Silhouette 우선

        logger.info(f"최적 K 탐색 완료:")
        logger.info(f"  - Elbow method: K={elbow_k}")
        logger.info(f"  - Silhouette: K={silhouette_k}")

        return results

    def get_model(self, beverage_type, k):
        """
        모델 조회 (캐시에서만)

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k (int): 군집 수

        Returns:
            dict or None: 모델 데이터
        """
        return self.model_cache.get(beverage_type, k)

    def retrain_models(self, beverage_type='all', k_values=None):
        """
        모델 재학습

        Args:
            beverage_type (str): 'all', 'caffeine', or 'noncaffeine'
            k_values (list, optional): 재학습할 K값 리스트

        Returns:
            dict: 재학습 결과
        """
        if k_values is None:
            k_values = [2, 3, 4, 5]

        if beverage_type == 'all':
            types = ['caffeine', 'noncaffeine']
        else:
            types = [beverage_type]

        results = []
        start_time = datetime.now()

        for bev_type in types:
            for k in k_values:
                try:
                    # 캐시 삭제
                    self.model_cache.clear(bev_type, k)

                    # 재학습
                    model_data = self.train_kmeans(bev_type, k, force_retrain=True)

                    results.append({
                        'beverage_type': bev_type,
                        'k': k,
                        'status': 'success',
                        'metrics': model_data['metrics']
                    })
                    logger.info(f"재학습 성공: {bev_type} k={k}")

                except Exception as e:
                    results.append({
                        'beverage_type': bev_type,
                        'k': k,
                        'status': 'failed',
                        'error': str(e)
                    })
                    logger.error(f"재학습 실패: {bev_type} k={k}, {e}")

        total_time = (datetime.now() - start_time).total_seconds()

        return {
            'results': results,
            'total_time': total_time,
            'success_count': sum(1 for r in results if r['status'] == 'success'),
            'failed_count': sum(1 for r in results if r['status'] == 'failed')
        }
