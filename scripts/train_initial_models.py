"""
초기 모델 학습 스크립트
k=2,3,4,5에 대한 모델을 사전 학습
"""

import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_preprocessor import DataPreprocessor
from app.services.model_cache import ModelCache
from app.services.clustering_service import ClusteringService


def main():
    print("=" * 70)
    print("초기 모델 학습 시작")
    print("=" * 70)

    # 1. 서비스 초기화
    print("\n1. 서비스 초기화 중...")
    data_preprocessor = DataPreprocessor()
    model_cache = ModelCache()
    clustering_service = ClusteringService(data_preprocessor, model_cache)

    # 2. 데이터 로드
    print("\n2. 데이터 로드 중...")
    caffeine_df, noncaffeine_df = data_preprocessor.load_and_split()
    print(f"   ✓ 카페인 음료: {len(caffeine_df)}개")
    print(f"   ✓ 논카페인 음료: {len(noncaffeine_df)}개")

    # 3. 모델 학습
    print("\n3. 모델 학습 시작...")
    k_values = [2, 3, 4, 5]

    for beverage_type in ['caffeine', 'noncaffeine']:
        print(f"\n   [{beverage_type.upper()}]")
        for k in k_values:
            try:
                print(f"   K={k} 학습 중...", end=" ")
                model_data = clustering_service.train_kmeans(beverage_type, k, force_retrain=True)

                print(f"완료")
                print(f"      - Inertia: {model_data['metrics']['inertia']:.2f}")
                print(f"      - Silhouette: {model_data['metrics']['silhouette_score']:.4f}")
                print(f"      - 학습 시간: {model_data['training_time']:.2f}초")

            except Exception as e:
                print(f"실패: {e}")

    # 4. 최적 K 탐색
    print("\n4. 최적 K 탐색 (참고용)...")
    for beverage_type in ['caffeine', 'noncaffeine']:
        print(f"\n   [{beverage_type.upper()}]")
        try:
            result = clustering_service.find_optimal_k(beverage_type, (2, 8))
            print(f"   ✓ Elbow method 추천: K={result['recommended_k_elbow']}")
            print(f"   ✓ Silhouette 추천: K={result['recommended_k_silhouette']}")
            print(f"   ✓ 전체 추천: K={result['recommended_k_overall']}")
        except Exception as e:
            print(f"   ✗ 실패: {e}")

    # 5. 캐시 상태 확인
    print("\n5. 캐시 상태 확인...")
    cache_info = model_cache.get_cache_info()
    print(f"   - 카페인 모델: {cache_info['caffeine']['disk_available']}")
    print(f"   - 논카페인 모델: {cache_info['noncaffeine']['disk_available']}")

    print("\n" + "=" * 70)
    print("초기 모델 학습 완료!")
    print("=" * 70)
    print("\n다음 명령으로 서버를 시작할 수 있습니다:")
    print("  uvicorn app.main:app --reload")
    print("  또는")
    print("  python3 -m uvicorn app.main:app --reload")
    print()


if __name__ == "__main__":
    main()
