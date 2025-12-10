"""
예측 서비스
군집 예측 및 유사 음료 검색
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """예측 서비스 클래스"""

    def __init__(self, data_preprocessor, model_cache):
        self.data_preprocessor = data_preprocessor
        self.model_cache = model_cache

    def predict_cluster(self, beverage_type, k, features_dict):
        """
        군집 예측

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k (int): 군집 수
            features_dict (dict): 피쳐 딕셔너리 (빈 값은 0으로 처리)

        Returns:
            dict: 예측 결과
        """
        # 1. 모델 로드
        model_data = self.model_cache.get(beverage_type, k)
        if model_data is None:
            raise ValueError(f"모델을 찾을 수 없습니다: {beverage_type} k={k}")

        # 2. 빈 값을 0으로 채우기
        features = model_data['features']
        filled_dict = {}
        for feature in features:
            value = features_dict.get(feature)
            if value is None or value == '':
                filled_dict[feature] = 0.0
            else:
                filled_dict[feature] = float(value)

        # 3. 피쳐 배열 변환
        X = self.data_preprocessor.features_dict_to_array(beverage_type, filled_dict)

        # 4. 스케일링
        scaler = model_data['scaler']
        X_scaled, _ = self.data_preprocessor.scale_features(X, scaler)

        # 5. 예측
        kmeans = model_data['kmeans']
        predicted_cluster = int(kmeans.predict(X_scaled)[0])

        # 6. 중심까지의 거리
        centroid = model_data['centroids'][predicted_cluster]
        distance = float(np.linalg.norm(X_scaled[0] - centroid))

        # 7. 신뢰도 계산 (다른 군집과의 거리 비율)
        distances_to_all = np.linalg.norm(
            X_scaled[0] - model_data['centroids'], axis=1
        )
        sorted_distances = np.sort(distances_to_all)
        if len(sorted_distances) > 1:
            confidence = 1.0 - (sorted_distances[0] / sorted_distances[1])
        else:
            confidence = 1.0

        # 8. 대안 군집
        alternative_clusters = []
        for i, dist in enumerate(distances_to_all):
            if i != predicted_cluster:
                alternative_clusters.append({
                    'cluster_id': int(i),
                    'distance': float(dist)
                })
        alternative_clusters.sort(key=lambda x: x['distance'])

        result = {
            'predicted_cluster': predicted_cluster,
            'confidence': float(confidence),
            'distance_to_centroid': distance,
            'alternative_clusters': alternative_clusters
        }

        logger.info(f"예측 완료: {beverage_type} k={k} -> cluster {predicted_cluster}")
        return result

    def find_similar_beverages(self, beverage_type, k, features_dict, n_neighbors=10, scope='predicted_cluster'):
        """
        유사 음료 검색

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k (int): 군집 수
            features_dict (dict): 피쳐 딕셔너리 (빈 값은 0으로 처리)
            n_neighbors (int): 반환할 이웃 수
            scope (str): 'predicted_cluster' or 'all_clusters'

        Returns:
            dict: 유사 음료 리스트
        """
        # 1. 군집 예측
        prediction = self.predict_cluster(beverage_type, k, features_dict)
        predicted_cluster = prediction['predicted_cluster']

        # 2. 모델 데이터 로드
        model_data = self.model_cache.get(beverage_type, k)
        df = model_data['df']
        labels = model_data['labels']
        features = model_data['features']
        X_scaled = model_data['X_scaled']
        scaler = model_data['scaler']

        # 3. 입력 피쳐 스케일링
        X = self.data_preprocessor.features_dict_to_array(beverage_type, features_dict)
        X_input_scaled, _ = self.data_preprocessor.scale_features(X, scaler)

        # 4. 검색 범위 결정
        if scope == 'predicted_cluster':
            search_mask = labels == predicted_cluster
            search_df = df[search_mask]
            search_X = X_scaled[search_mask]
        else:
            search_mask = np.ones(len(df), dtype=bool)
            search_df = df
            search_X = X_scaled

        # 5. 거리 계산
        distances = np.linalg.norm(search_X - X_input_scaled, axis=1)

        # 6. 상위 N개 선택
        n_neighbors = min(n_neighbors, len(distances))
        closest_indices = distances.argsort()[:n_neighbors]

        # 7. 결과 구성
        similar_beverages = []
        for rank, idx in enumerate(closest_indices, 1):
            import math

            actual_idx = search_df.index[idx]
            beverage = search_df.loc[actual_idx]
            distance = float(distances[idx])

            # distance가 유효한지 확인
            if math.isnan(distance) or math.isinf(distance):
                distance = 999999.0

            similarity_score = 1.0 / (1.0 + distance)

            # NaN, inf 값을 안전하게 처리
            safe_features = {}
            for f in features:
                value = beverage[f]
                # NaN, inf 체크 (numpy, pandas 값 모두 처리)
                try:
                    float_value = float(value)
                    if math.isnan(float_value) or math.isinf(float_value):
                        safe_features[f] = 0.0
                    else:
                        safe_features[f] = float_value
                except (ValueError, TypeError):
                    safe_features[f] = 0.0

            similar_beverages.append({
                'rank': rank,
                'beverage_name': beverage['식품명'],
                'company': beverage['업체명'],
                'food_type': beverage['대표식품명'],
                'features': safe_features,
                'distance': distance,
                'similarity_score': similarity_score
            })

        result = {
            'predicted_cluster': predicted_cluster,
            'similar_beverages': similar_beverages,
            'search_summary': {
                'total_searched': int(search_mask.sum()),
                'cluster_id': predicted_cluster if scope == 'predicted_cluster' else None,
                'n_returned': len(similar_beverages),
                'scope': scope
            }
        }

        logger.info(f"유사 음료 검색 완료: {len(similar_beverages)}개 발견")
        return result

    def compare_beverages(self, beverage1_name, beverage2_name, beverage_type):
        """
        두 음료 비교

        Args:
            beverage1_name (str): 첫 번째 음료 이름
            beverage2_name (str): 두 번째 음료 이름
            beverage_type (str): 'caffeine' or 'noncaffeine'

        Returns:
            dict: 비교 결과
        """
        # 아무 모델이나 하나 가져오기 (데이터 조회용)
        available_k = self.model_cache.list_available(beverage_type)
        if not available_k:
            raise ValueError(f"{beverage_type}에 사용 가능한 모델이 없습니다")

        model_data = self.model_cache.get(beverage_type, available_k[0])
        df = model_data['df']
        features = model_data['features']

        # 음료 찾기
        beverage1 = df[df['식품명'].str.contains(beverage1_name, case=False, na=False)]
        beverage2 = df[df['식품명'].str.contains(beverage2_name, case=False, na=False)]

        if len(beverage1) == 0:
            raise ValueError(f"음료를 찾을 수 없습니다: {beverage1_name}")
        if len(beverage2) == 0:
            raise ValueError(f"음료를 찾을 수 없습니다: {beverage2_name}")

        bev1 = beverage1.iloc[0]
        bev2 = beverage2.iloc[0]

        # 군집 배정 조회
        cluster_assignments1 = {}
        cluster_assignments2 = {}
        for k in available_k:
            model_data = self.model_cache.get(beverage_type, k)
            if model_data:
                labels = model_data['labels']
                idx1 = beverage1.index[0]
                idx2 = beverage2.index[0]
                cluster_assignments1[f'k{k}'] = int(labels[df.index.get_loc(idx1)])
                cluster_assignments2[f'k{k}'] = int(labels[df.index.get_loc(idx2)])

        # 거리 계산
        features1 = np.array([bev1[f] for f in features])
        features2 = np.array([bev2[f] for f in features])
        distance = float(np.linalg.norm(features1 - features2))

        # 유사도 백분율
        max_distance = np.linalg.norm([df[f].max() - df[f].min() for f in features])
        similarity_percentage = float((1 - distance / max_distance) * 100)

        # 피쳐 차이
        feature_differences = {}
        for feature in features:
            feature_differences[feature] = {
                'beverage1': float(bev1[feature]),
                'beverage2': float(bev2[feature]),
                'diff': float(bev2[feature] - bev1[feature])
            }

        result = {
            'beverages': [
                {
                    'beverage_name': bev1['식품명'],
                    'company': bev1['업체명'],
                    'food_type': bev1['대표식품명'],
                    'features': {f: float(bev1[f]) for f in features},
                    'cluster_assignment': cluster_assignments1
                },
                {
                    'beverage_name': bev2['식품명'],
                    'company': bev2['업체명'],
                    'food_type': bev2['대표식품명'],
                    'features': {f: float(bev2[f]) for f in features},
                    'cluster_assignment': cluster_assignments2
                }
            ],
            'distance': distance,
            'similarity_percentage': similarity_percentage,
            'feature_differences': feature_differences
        }

        return result
