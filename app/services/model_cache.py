"""
모델 캐시 관리 서비스
메모리 캐시 + 디스크 저장 하이브리드 전략
"""

import os
import joblib
import time
import logging
from datetime import datetime
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class ModelCache:
    """모델 캐싱 및 관리 클래스"""

    def __init__(self, model_dir="data/models", max_models_per_type=10):
        """
        Args:
            model_dir (str): 모델 저장 디렉토리
            max_models_per_type (int): 타입별 최대 캐시 모델 수
        """
        self.model_dir = model_dir
        self.max_models = max_models_per_type

        # 메모리 캐시 구조: {beverage_type: {k: model_data}}
        self.cache = {
            'caffeine': {},
            'noncaffeine': {}
        }

        # LRU를 위한 접근 시간 기록
        self.access_times = {
            'caffeine': {},
            'noncaffeine': {}
        }

        logger.info(f"ModelCache 초기화 완료 (max_models={max_models_per_type})")

    def _get_model_path(self, beverage_type: str, k: int) -> str:
        """모델 파일 경로 반환"""
        return os.path.join(self.model_dir, beverage_type, f"kmeans_k{k}.pkl")

    def get(self, beverage_type: str, k: int) -> Optional[Dict]:
        """
        모델 캐시에서 조회

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k (int): 군집 수

        Returns:
            dict or None: 모델 데이터 또는 None
        """
        # 1. 메모리 캐시 확인
        if k in self.cache[beverage_type]:
            self.access_times[beverage_type][k] = time.time()
            logger.info(f"메모리 캐시 히트: {beverage_type} k={k}")
            return self.cache[beverage_type][k]

        # 2. 디스크에서 로드 시도
        model_path = self._get_model_path(beverage_type, k)
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                logger.info(f"디스크에서 모델 로드: {model_path}")

                # 메모리 캐시에 추가
                self._add_to_cache(beverage_type, k, model_data)
                return model_data
            except Exception as e:
                logger.error(f"모델 로드 실패: {model_path}, {e}")
                return None

        # 3. 없음
        logger.info(f"캐시 미스: {beverage_type} k={k}")
        return None

    def set(self, beverage_type: str, k: int, model_data: Dict):
        """
        모델 캐시에 저장 (메모리 + 디스크)

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k (int): 군집 수
            model_data (dict): 저장할 모델 데이터
        """
        # 1. 디스크에 저장
        model_path = self._get_model_path(beverage_type, k)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        try:
            joblib.dump(model_data, model_path)
            logger.info(f"모델 디스크 저장 완료: {model_path}")
        except Exception as e:
            logger.error(f"모델 저장 실패: {model_path}, {e}")
            raise

        # 2. 메모리 캐시에 추가
        self._add_to_cache(beverage_type, k, model_data)

    def _add_to_cache(self, beverage_type: str, k: int, model_data: Dict):
        """
        메모리 캐시에 추가 (LRU 정책 적용)

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k (int): 군집 수
            model_data (dict): 모델 데이터
        """
        # LRU 정책: 캐시가 꽉 찼으면 가장 오래된 항목 제거
        if len(self.cache[beverage_type]) >= self.max_models:
            if k not in self.cache[beverage_type]:  # 새로운 항목인 경우만
                lru_k = min(
                    self.access_times[beverage_type],
                    key=self.access_times[beverage_type].get
                )
                logger.info(f"LRU 제거: {beverage_type} k={lru_k}")
                del self.cache[beverage_type][lru_k]
                del self.access_times[beverage_type][lru_k]

        self.cache[beverage_type][k] = model_data
        self.access_times[beverage_type][k] = time.time()
        logger.info(f"메모리 캐시 추가: {beverage_type} k={k}")

    def exists(self, beverage_type: str, k: int) -> bool:
        """
        모델 존재 여부 확인 (메모리 또는 디스크)

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            k (int): 군집 수

        Returns:
            bool: 존재 여부
        """
        # 메모리 확인
        if k in self.cache[beverage_type]:
            return True

        # 디스크 확인
        model_path = self._get_model_path(beverage_type, k)
        return os.path.exists(model_path)

    def list_available(self, beverage_type: str) -> List[int]:
        """
        사용 가능한 K값 리스트 반환

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'

        Returns:
            list: K값 리스트
        """
        k_values = set()

        # 메모리 캐시
        k_values.update(self.cache[beverage_type].keys())

        # 디스크 확인
        model_dir = os.path.join(self.model_dir, beverage_type)
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.startswith('kmeans_k') and filename.endswith('.pkl'):
                    try:
                        k = int(filename.replace('kmeans_k', '').replace('.pkl', ''))
                        k_values.add(k)
                    except ValueError:
                        continue

        return sorted(list(k_values))

    def list_all(self) -> Dict[str, List[int]]:
        """
        모든 타입의 사용 가능한 K값 리스트 반환

        Returns:
            dict: {beverage_type: [k값들]}
        """
        return {
            'caffeine': self.list_available('caffeine'),
            'noncaffeine': self.list_available('noncaffeine')
        }

    def clear(self, beverage_type: Optional[str] = None, k: Optional[int] = None):
        """
        캐시 삭제

        Args:
            beverage_type (str, optional): 특정 타입만 삭제
            k (int, optional): 특정 K값만 삭제
        """
        if beverage_type and k:
            # 특정 모델만 삭제
            if k in self.cache[beverage_type]:
                del self.cache[beverage_type][k]
                del self.access_times[beverage_type][k]
                logger.info(f"캐시 삭제: {beverage_type} k={k}")
        elif beverage_type:
            # 특정 타입 전체 삭제
            self.cache[beverage_type].clear()
            self.access_times[beverage_type].clear()
            logger.info(f"캐시 전체 삭제: {beverage_type}")
        else:
            # 모든 캐시 삭제
            for bev_type in ['caffeine', 'noncaffeine']:
                self.cache[bev_type].clear()
                self.access_times[bev_type].clear()
            logger.info("모든 캐시 삭제")

    def get_cache_info(self) -> Dict:
        """
        캐시 정보 반환

        Returns:
            dict: 캐시 통계
        """
        info = {
            'caffeine': {
                'memory_cached': list(self.cache['caffeine'].keys()),
                'disk_available': self.list_available('caffeine'),
                'cache_size': len(self.cache['caffeine'])
            },
            'noncaffeine': {
                'memory_cached': list(self.cache['noncaffeine'].keys()),
                'disk_available': self.list_available('noncaffeine'),
                'cache_size': len(self.cache['noncaffeine'])
            }
        }
        return info

    def preload_models(self, k_values: List[int] = [2, 3, 4, 5]):
        """
        지정된 K값의 모델들을 사전 로드

        Args:
            k_values (list): 로드할 K값 리스트
        """
        logger.info(f"모델 사전 로드 시작: {k_values}")

        for beverage_type in ['caffeine', 'noncaffeine']:
            for k in k_values:
                model_data = self.get(beverage_type, k)
                if model_data:
                    logger.info(f"  ✓ {beverage_type} k={k} 로드 완료")
                else:
                    logger.warning(f"  ✗ {beverage_type} k={k} 모델 없음")

        logger.info("모델 사전 로드 완료")
