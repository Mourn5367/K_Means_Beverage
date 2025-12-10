"""
인사이트 서비스
군집별 통계, 특성, 대표 음료 분석
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class InsightsService:
    """인사이트 서비스 클래스"""

    def __init__(self, model_cache):
        self.model_cache = model_cache

    def get_cluster_insights(self, beverage_type, k):
        """
        전체 군집 인사이트 조회

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k (int): 군집 수

        Returns:
            dict: 인사이트 데이터
        """
        model_data = self.model_cache.get(beverage_type, k)
        if model_data is None:
            raise ValueError(f"모델을 찾을 수 없습니다: {beverage_type} k={k}")

        df = model_data['df']
        labels = model_data['labels']
        features = model_data['features']
        centroids = model_data['centroids']

        insights = {
            'beverage_type': beverage_type,
            'k': k,
            'clusters': [],
            'global_statistics': self._get_global_statistics(df, features)
        }

        for cluster_id in range(k):
            cluster_insight = self._characterize_cluster(
                cluster_id, df, labels, features, centroids[cluster_id]
            )
            insights['clusters'].append(cluster_insight)

        return insights

    def _characterize_cluster(self, cluster_id, df, labels, features, centroid):
        """군집 특성 분석"""
        cluster_mask = labels == cluster_id
        cluster_df = df[cluster_mask]

        # 기본 정보 (나중에 name 업데이트)
        insight = {
            'cluster_id': cluster_id,
            'name': f"Cluster {cluster_id}",
            'size': int(cluster_mask.sum()),
            'percentage': float(cluster_mask.sum() / len(df) * 100),
            'representative_types': [],
            'feature_statistics': {},
            'centroid': {},
            'characteristics': [],
            'sample_beverages': []
        }

        # 대표 식품명 Top 5
        type_counts = cluster_df['대표식품명'].value_counts().head(5)
        for name, count in type_counts.items():
            insight['representative_types'].append({
                'name': name,
                'count': int(count),
                'percentage': float(count / len(cluster_df) * 100)
            })

        # 피쳐 통계
        for i, feature in enumerate(features):
            stats = cluster_df[feature].describe()
            insight['feature_statistics'][feature] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'variance': float(cluster_df[feature].var()),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'median': float(stats['50%']),
                'q1': float(stats['25%']),
                'q3': float(stats['75%']),
                'iqr': float(stats['75%'] - stats['25%'])
            }
            insight['centroid'][feature] = float(centroid[i])

        # 특성 생성 (전체 평균 대비)
        overall_means = df[features].mean()
        cluster_means = cluster_df[features].mean()

        # 가장 큰 차이를 보이는 특징 찾기
        max_diff_feature = None
        max_diff_pct = 0
        diff_direction = ""

        for feature in features:
            diff_pct = (cluster_means[feature] - overall_means[feature]) / overall_means[feature] * 100

            # 가장 큰 차이 추적
            if abs(diff_pct) > abs(max_diff_pct):
                max_diff_pct = diff_pct
                max_diff_feature = feature
                diff_direction = "높은" if diff_pct > 0 else "낮은"

            if abs(diff_pct) > 20:
                if diff_pct > 0:
                    insight['characteristics'].append(
                        f"높은 {feature} ({diff_pct:.1f}% 평균 이상)"
                    )
                else:
                    insight['characteristics'].append(
                        f"낮은 {feature} ({abs(diff_pct):.1f}% 평균 이하)"
                    )

        # 클러스터 이름 생성: "주요특징 대표식품명" (괄호 및 단위 제거)
        if max_diff_feature and len(insight['representative_types']) > 0:
            main_food_type = insight['representative_types'][0]['name']
            # 피쳐명에서 단위 제거 (예: "에너지(kcal)" -> "에너지")
            feature_name = max_diff_feature.split('(')[0]
            insight['name'] = f"{diff_direction} {feature_name} {main_food_type}"
        elif len(insight['representative_types']) > 0:
            insight['name'] = f"{insight['representative_types'][0]['name']} 중심"
        else:
            insight['name'] = f"Cluster {cluster_id}"

        # 샘플 음료 (중심에 가까운 순)
        cluster_features = cluster_df[features].values
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        closest_indices = distances.argsort()[:10]

        for idx in closest_indices:
            actual_idx = cluster_df.index[idx]
            beverage = cluster_df.loc[actual_idx]
            insight['sample_beverages'].append({
                '식품명': beverage['식품명'],
                '업체명': beverage['업체명'],
                '대표식품명': beverage['대표식품명'],
                'distance_to_centroid': float(distances[idx]),
                'features': {f: float(beverage[f]) for f in features}
            })

        return insight

    def _get_global_statistics(self, df, features):
        """전체 데이터 통계"""
        stats = {
            'total_beverages': len(df),
            'most_common_type': df['대표식품명'].value_counts().index[0],
            'feature_ranges': {}
        }

        for feature in features:
            stats['feature_ranges'][feature] = {
                'min': float(df[feature].min()),
                'max': float(df[feature].max()),
                'mean': float(df[feature].mean())
            }

        return stats

    def get_beverage_info(self, beverage_name, beverage_type):
        """특정 음료 정보 조회"""
        # 사용 가능한 모든 모델에서 해당 음료의 군집 찾기
        available_k = self.model_cache.list_available(beverage_type)

        beverage_info = None
        cluster_assignments = {}

        for k in available_k:
            model_data = self.model_cache.get(beverage_type, k)
            if model_data is None:
                continue

            df = model_data['df']
            labels = model_data['labels']
            features = model_data['features']

            # 음료 찾기
            matching = df[df['식품명'].str.contains(beverage_name, case=False, na=False)]

            if len(matching) > 0:
                beverage = matching.iloc[0]
                idx = matching.index[0]

                if beverage_info is None:
                    beverage_info = {
                        '식품명': beverage['식품명'],
                        '업체명': beverage['업체명'],
                        '대표식품명': beverage['대표식품명'],
                        'features': {f: float(beverage[f]) for f in features}
                    }

                cluster_assignments[f'k{k}'] = int(labels[df.index.get_loc(idx)])

        if beverage_info is None:
            raise ValueError(f"음료를 찾을 수 없습니다: {beverage_name}")

        beverage_info['cluster_assignments'] = cluster_assignments
        return beverage_info
