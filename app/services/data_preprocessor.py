"""
데이터 전처리 서비스
CSV 로드, 피쳐 스케일링, 데이터 분리 등을 담당
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """데이터 전처리를 담당하는 클래스"""

    def __init__(self, data_dir="data/processed"):
        self.data_dir = data_dir
        self.caffeine_df = None
        self.noncaffeine_df = None

        # 피쳐 정의
        self.caffeine_features = [
            '에너지(kcal)', '단백질(g)', '당류(g)',
            '나트륨(mg)', '포화지방산(g)', '카페인(mg)'
        ]
        self.noncaffeine_features = [
            '에너지(kcal)', '단백질(g)', '당류(g)',
            '나트륨(mg)', '포화지방산(g)'
        ]
        self.metadata_columns = ['식품명', '업체명', '대표식품명']

    def load_and_split(self):
        """
        전처리된 CSV 파일 로드

        Returns:
            tuple: (caffeine_df, noncaffeine_df)
        """
        caffeine_path = os.path.join(self.data_dir, "caffeine_beverages.csv")
        noncaffeine_path = os.path.join(self.data_dir, "noncaffeine_beverages.csv")

        logger.info(f"카페인 음료 데이터 로드 중: {caffeine_path}")
        self.caffeine_df = pd.read_csv(caffeine_path, encoding='utf-8')
        logger.info(f"카페인 음료 로드 완료: {len(self.caffeine_df)}개")

        logger.info(f"논카페인 음료 데이터 로드 중: {noncaffeine_path}")
        self.noncaffeine_df = pd.read_csv(noncaffeine_path, encoding='utf-8')
        logger.info(f"논카페인 음료 로드 완료: {len(self.noncaffeine_df)}개")

        return self.caffeine_df, self.noncaffeine_df

    def get_features(self, beverage_type):
        """
        음료 타입에 따른 피쳐 리스트 반환

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'

        Returns:
            list: 피쳐 리스트
        """
        if beverage_type == 'caffeine':
            return self.caffeine_features
        elif beverage_type == 'noncaffeine':
            return self.noncaffeine_features
        else:
            raise ValueError(f"Invalid beverage_type: {beverage_type}")

    def get_metadata_columns(self):
        """메타데이터 컬럼 리스트 반환"""
        return self.metadata_columns

    def get_data(self, beverage_type):
        """
        음료 타입에 따른 데이터프레임 반환

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'

        Returns:
            pd.DataFrame: 해당 데이터프레임
        """
        if self.caffeine_df is None or self.noncaffeine_df is None:
            self.load_and_split()

        if beverage_type == 'caffeine':
            return self.caffeine_df
        elif beverage_type == 'noncaffeine':
            return self.noncaffeine_df
        else:
            raise ValueError(f"Invalid beverage_type: {beverage_type}")

    def scale_features(self, X, scaler=None):
        """
        피쳐 스케일링 (StandardScaler)

        Args:
            X (np.ndarray or pd.DataFrame): 스케일링할 데이터
            scaler (StandardScaler, optional): 기존 scaler (예측용)

        Returns:
            tuple: (X_scaled, scaler)
        """
        if scaler is None:
            # 새로운 scaler 생성 및 fit
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info("새 StandardScaler 생성 및 fit 완료")
        else:
            # 기존 scaler로 transform만 수행
            X_scaled = scaler.transform(X)
            logger.info("기존 StandardScaler로 transform 완료")

        return X_scaled, scaler

    def prepare_clustering_data(self, beverage_type):
        """
        군집화를 위한 데이터 준비

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'

        Returns:
            dict: {
                'X': 피쳐 데이터 (numpy array),
                'X_scaled': 스케일링된 피쳐 데이터,
                'scaler': StandardScaler 객체,
                'features': 피쳐 이름 리스트,
                'metadata': 메타데이터 (DataFrame),
                'df': 전체 데이터프레임
            }
        """
        df = self.get_data(beverage_type)
        features = self.get_features(beverage_type)

        # 피쳐 추출
        X = df[features].values

        # 스케일링
        X_scaled, scaler = self.scale_features(X)

        # 메타데이터 추출
        metadata = df[self.metadata_columns].copy()

        logger.info(f"{beverage_type} 데이터 준비 완료:")
        logger.info(f"  - 샘플 수: {len(X)}")
        logger.info(f"  - 피쳐 수: {len(features)}")

        return {
            'X': X,
            'X_scaled': X_scaled,
            'scaler': scaler,
            'features': features,
            'metadata': metadata,
            'df': df
        }

    def validate_feature_dict(self, beverage_type, features_dict):
        """
        사용자 입력 피쳐 딕셔너리 검증

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            features_dict (dict): 피쳐명: 값 딕셔너리

        Returns:
            tuple: (is_valid, error_message)
        """
        required_features = self.get_features(beverage_type)

        # 필수 피쳐 확인
        missing = set(required_features) - set(features_dict.keys())
        if missing:
            return False, f"누락된 피쳐: {missing}"

        # 음수 값 확인
        for feature, value in features_dict.items():
            if feature in required_features and value < 0:
                return False, f"{feature}는 음수일 수 없습니다: {value}"

        return True, None

    def features_dict_to_array(self, beverage_type, features_dict):
        """
        피쳐 딕셔너리를 numpy 배열로 변환

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'
            features_dict (dict): 피쳐명: 값 딕셔너리

        Returns:
            np.ndarray: 피쳐 배열 (1, n_features)
        """
        features = self.get_features(beverage_type)
        values = [features_dict[f] for f in features]
        return np.array([values])

    def get_feature_ranges(self, beverage_type):
        """
        각 피쳐의 최소/최대 범위 반환

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'

        Returns:
            dict: {feature_name: (min, max)}
        """
        df = self.get_data(beverage_type)
        features = self.get_features(beverage_type)

        ranges = {}
        for feature in features:
            ranges[feature] = (df[feature].min(), df[feature].max())

        return ranges

    def get_statistics(self, beverage_type):
        """
        데이터셋 통계 정보 반환

        Args:
            beverage_type (str): 'caffeine' or 'noncaffeine'

        Returns:
            dict: 통계 정보
        """
        df = self.get_data(beverage_type)
        features = self.get_features(beverage_type)

        stats = {
            'n_samples': len(df),
            'n_features': len(features),
            'features': features,
            'feature_stats': {}
        }

        for feature in features:
            stats['feature_stats'][feature] = {
                'mean': float(df[feature].mean()),
                'std': float(df[feature].std()),
                'min': float(df[feature].min()),
                'max': float(df[feature].max()),
                'median': float(df[feature].median())
            }

        # 대표식품명 분포
        stats['representative_types'] = df['대표식품명'].value_counts().head(10).to_dict()

        return stats
