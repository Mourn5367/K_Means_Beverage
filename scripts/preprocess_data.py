"""
데이터 전처리 스크립트
음료.CSV를 로드하여 UTF-8로 변환하고 카페인/논카페인으로 분리
"""

import pandas as pd
import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("데이터 전처리 시작")
    print("=" * 60)

    # 경로 설정
    raw_data_path = "data/raw/음료.CSV"
    processed_dir = "data/processed"

    if not os.path.exists(raw_data_path):
        print(f"오류: {raw_data_path} 파일을 찾을 수 없습니다!")
        return

    # 1. CSV 로드 (EUC-KR 인코딩)
    print(f"\n1. CSV 파일 로드 중: {raw_data_path}")
    try:
        df = pd.read_csv(raw_data_path, encoding='euc-kr')
        print(f"   ✓ 로드 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
    except Exception as e:
        print(f"   ✗ 로드 실패: {e}")
        return

    # 2. 컬럼 확인
    print(f"\n2. 컬럼 목록:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")

    # 3. 기본 통계
    print(f"\n3. 기본 통계:")
    print(f"   - 총 행 수: {len(df)}")
    print(f"   - 결측치 확인:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"     {col}: {count}개 ({count/len(df)*100:.2f}%)")

    # 4. 결측치 처리
    print(f"\n4. 결측치 처리 중...")
    numeric_cols = ['에너지(kcal)', '단백질(g)', '당류(g)', '나트륨(mg)', '포화지방산(g)', '카페인(mg)']

    for col in numeric_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                # 평균값으로 채우기
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                print(f"   ✓ {col}: {missing_count}개 결측치 → 평균값({mean_val:.2f})으로 대체")

    # 문자열 컬럼 결측치 처리
    string_cols = ['식품명', '업체명', '대표식품명']
    for col in string_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna('Unknown', inplace=True)
                print(f"   ✓ {col}: {missing_count}개 결측치 → 'Unknown'으로 대체")

    # 5. UTF-8로 저장
    utf8_path = os.path.join(processed_dir, "beverages_utf8.csv")
    df.to_csv(utf8_path, index=False, encoding='utf-8')
    print(f"\n5. UTF-8 파일 저장 완료: {utf8_path}")

    # 6. 카페인/논카페인 분리
    print(f"\n6. 카페인/논카페인 분리 중...")

    if '카페인(mg)' not in df.columns:
        print("   ✗ '카페인(mg)' 컬럼을 찾을 수 없습니다!")
        return

    # 카페인 음료 (카페인 > 0)
    caffeine_df = df[df['카페인(mg)'] > 0].copy()
    caffeine_path = os.path.join(processed_dir, "caffeine_beverages.csv")
    caffeine_df.to_csv(caffeine_path, index=False, encoding='utf-8')
    print(f"   ✓ 카페인 음료: {len(caffeine_df)}개 → {caffeine_path}")

    # 논카페인 음료 (카페인 = 0)
    noncaffeine_df = df[df['카페인(mg)'] == 0].copy()
    noncaffeine_path = os.path.join(processed_dir, "noncaffeine_beverages.csv")
    noncaffeine_df.to_csv(noncaffeine_path, index=False, encoding='utf-8')
    print(f"   ✓ 논카페인 음료: {len(noncaffeine_df)}개 → {noncaffeine_path}")

    # 7. 데이터 검증
    print(f"\n7. 데이터 검증:")
    total = len(caffeine_df) + len(noncaffeine_df)
    print(f"   - 전체: {len(df)}개")
    print(f"   - 카페인: {len(caffeine_df)}개 ({len(caffeine_df)/len(df)*100:.1f}%)")
    print(f"   - 논카페인: {len(noncaffeine_df)}개 ({len(noncaffeine_df)/len(df)*100:.1f}%)")
    print(f"   - 합계 확인: {total}개 (전체와 {'일치' if total == len(df) else '불일치'})")

    # 8. 카페인 통계
    print(f"\n8. 카페인 통계:")
    caffeine_stats = caffeine_df['카페인(mg)'].describe()
    print(f"   - 평균: {caffeine_stats['mean']:.2f}mg")
    print(f"   - 표준편차: {caffeine_stats['std']:.2f}mg")
    print(f"   - 최소: {caffeine_stats['min']:.2f}mg")
    print(f"   - 최대: {caffeine_stats['max']:.2f}mg")
    print(f"   - 중앙값: {caffeine_stats['50%']:.2f}mg")

    # 9. 대표식품명 분포
    print(f"\n9. 대표식품명 분포 (상위 10개):")
    top_types = df['대표식품명'].value_counts().head(10)
    for i, (name, count) in enumerate(top_types.items(), 1):
        print(f"   {i}. {name}: {count}개 ({count/len(df)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("데이터 전처리 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
