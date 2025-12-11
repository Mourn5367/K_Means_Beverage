# K-Means 음료 군집화 시스템

한국 음료 데이터를 K-means 알고리즘을 사용하여 군집화하고 분석하는 웹 기반 시스템입니다. 사용자는 표시할 영양 성분을 선택하고 K 값을 수정하여 시각화 할 수 있으며, 영양 성분을 입력하여 유사한 음료를 찾을 수 있습니다.

## 프로젝트 개요

- **총 데이터**: 1,935개 음료
- **카페인 음료**: 1,225개
- **논카페인 음료**: 710개
- **군집화 알고리즘**: K-means (scikit-learn)
- **API 프레임워크**: FastAPI
- **시각화**: Plotly.js

## 주요 기능

### 1. 음료 군집화
- **카페인/논카페인 음료 분류**: 두 가지 음료 타입에 대해 별도 군집화 수행
- **동적 K 값 설정**: 3~11개 군집 중 원하는 개수 선택 가능
- **자동 군집 명명**: 각 군집의 주요 특징과 대표 식품명을 조합하여 의미있는 이름 생성
  - 예: "높은 에너지 커피", "낮은 당류 스무디"
- **모델 캐싱**: 한 번 학습한 모델 재사용으로 빠른 응답

### 2. 대화형 시각화
- **산점도 (Scatter Plot)**
  - 축 선택 기능: X축과 Y축에 표시할 영양 성분 자유롭게 선택
  - 호버 정보: 제품명, 분류, 제조사, 모든 영양 성분 표시
  - 색상 구분: 군집별로 다른 색상으로 표시
  - 축 검증: X축과 Y축이 동일한 값이 되지 않도록 방지

- **분포 차트 (Distribution Charts)**
  - 각 영양 성분별 정규분포 곡선 표시
  - 전체 음료 분포 대비 해당 군집의 위치 시각화
  - 전체 평균선과 군집 평균선 비교
  - 호버 전용: 줌/팬 비활성화, 호버만 허용

### 3. 군집 인사이트
- **주요 특징 분석**
  - 각 영양 성분의 평균값 및 전체 평균 대비 비교
  - 분포 차트를 통한 시각적 위치 파악
  - 카페인/논카페인 음료 그룹 내 비율 표시

- **대표 음료 정보**
  - 각 군집의 중심에 가장 가까운 실제 음료 표시
  - 제품명, 제조사, 분류 및 모든 영양 성분 정보 제공

### 4. 유사 음료 검색
- **사용자 정의 입력**
  - 원하는 영양 성분 값을 직접 입력
  - 입력하지 않은 값은 자동으로 0으로 처리

- **유사도 점수**: Euclidean distance 기반 유사도 점수 제공
- **상세 정보**: 각 추천 음료의 모든 영양 성분 정보 표시

## 기술 스택

### Backend
- **FastAPI**: RESTful API 서버
- **scikit-learn**: K-means 군집화 알고리즘
- **pandas**: 데이터 처리 및 분석
- **numpy**: 수치 연산
- **Pydantic**: 데이터 검증 및 스키마
- **joblib**: 모델 영속성 관리

### Frontend
- **Plotly.js**: 대화형 차트 및 시각화
- **Vanilla JavaScript**: UI 상호작용
- **HTML/CSS**: 반응형 웹 인터페이스

### Data Processing
- **StandardScaler**: Z-score 정규화를 통한 특성 스케일링
- **openpyxl**: Excel 파일 처리

## 프로젝트 구조

```
K_Means_3/
├── app/
│   ├── api/
│   │   └── routes/          # API 엔드포인트
│   │       ├── clustering.py    # 군집화 API
│   │       ├── insights.py      # 인사이트 API
│   │       └── prediction.py    # 예측 및 유사 음료 검색 API
│   ├── models/
│   │   └── schemas.py       # Pydantic 스키마
│   ├── services/
│   │   ├── clustering_service.py    # 군집화 로직
│   │   ├── data_preprocessor.py     # 데이터 전처리
│   │   ├── insights_service.py      # 인사이트 생성
│   │   ├── model_cache.py           # 모델 캐싱
│   │   └── prediction_service.py    # 예측 서비스
│   ├── config.py            # 설정 파일
│   └── main.py              # FastAPI 애플리케이션
├── data/
│   ├── raw/                 # 원본 데이터 (Excel)
│   ├── processed/           # 전처리된 데이터 (CSV)
│   └── models/              # 학습된 모델 (joblib)
├── scripts/
│   └── preprocess_data.py   # 데이터 전처리 스크립트
├── templates/
│   └── index.html           # 웹 인터페이스
└── requirements.txt         # Python 패키지 의존성
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론 또는 다운로드
cd K_Means_3

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 원본 데이터를 data/raw/ 폴더에 배치
# 파일명: 카페인음료.xlsx, 논카페인음료.xlsx

# 데이터 전처리 실행
python scripts/preprocess_data.py
```

### 3. 서버 실행

```bash
# FastAPI 서버 시작
python app/main.py
```

서버가 시작되면 브라우저에서 `http://localhost:8000`으로 접속합니다.

모델은 가벼워서 첫 요청 시 자동으로 학습되므로 별도의 사전 학습이 필요하지 않습니다.

### 4. API 문서 확인

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### 군집화 (Clustering)
- `POST /api/clustering/train`: 새로운 K 값으로 군집화 모델 학습
- `GET /api/clustering/clusters`: 학습된 모델의 군집 정보 조회

### 인사이트 (Insights)
- `GET /api/insights/clusters`: 군집 인사이트 및 통계 정보 조회

### 예측 (Prediction)
- `POST /api/prediction/predict`: 입력 값 기반 군집 예측
- `POST /api/prediction/similar`: 유사 음료 검색

자세한 API 문서는 서버 실행 후 `http://localhost:8000/docs`에서 확인할 수 있습니다.

## 사용 방법

### 1. 음료 타입 및 군집 수 선택
- 화면 상단에서 카페인/논카페인 선택
- 군집 수(K) 선택 (1~11)
- "군집화 실행" 버튼 클릭

### 2. 결과 탐색
- **산점도**: X축과 Y축 선택하여 다양한 관점에서 데이터 탐색
  - 마우스 호버로 각 음료의 상세 정보 확인
  - 산점도의 점 클릭 시 해당 군집 상세 정보 모달 표시
- **사이드바**: 각 군집의 요약 정보 확인
  - 군집명, 음료 개수, 비율 표시
  - 클릭 시 상세 모달 표시
- **상세 모달**:
  - 주요 특징을 분포 차트로 시각화
  - 대표 음료 목록 및 영양 성분 정보

### 3. 유사 음료 찾기
- 하단의 "유사 음료 찾기" 섹션에서 원하는 영양 성분 값 입력
- 입력하지 않은 값은 자동으로 0으로 처리
- 검색 범위 선택:
  - "예측된 군집 내에서만": 입력 값과 같은 군집의 음료만 검색
  - "모든 군집에서": 전체 음료 데이터에서 검색
- "유사 음료 찾기" 버튼 클릭하여 추천 결과 확인

## 핵심 알고리즘

### K-means 군집화

```python
# K-means 설정
KMeans(
    n_clusters=k,
    init='k-means++',      # 초기 중심점 선택
    n_init=10,             # 초기화 시도 횟수
    max_iter=300,          # 최대 반복 횟수
    random_state=42        # 재현성
)

# StandardScaler를 사용한 특성 정규화
X_scaled = (X - mean) / std
```

### 유사도 계산

```python
# Euclidean distance 기반
distance = np.linalg.norm(X1 - X2, axis=1)
distance = sqrt(sum((x_i - y_i)^2))

# 유사도 점수 (0~1 범위, 1에 가까울수록 유사)
similarity = 1.0 / (1.0 + distance)
```

### 분포 시각화

```python
# 정규분포 근사 (표준편차 추정)
std ≈ range / 6

# 확률 밀도 함수
PDF(x) = exp(-0.5 * ((x - mean) / std)^2) / (std * sqrt(2π))
```

### 군집 명명 로직

```python
# 1. 전체 평균 대비 가장 차이가 큰 특징 찾기
max_diff_feature = feature with max(abs(cluster_mean - global_mean))

# 2. 높음/낮음 방향 결정
direction = "높은" if cluster_mean > global_mean else "낮은"

# 3. 특징명에서 단위 제거
feature_name = max_diff_feature.split('(')[0]  # "에너지(kcal)" -> "에너지"

# 4. 대표 식품명 가져오기
representative_food = most_common_food_type_in_cluster

# 5. 군집명 생성
cluster_name = f"{direction} {feature_name} {representative_food}"
# 예: "높은 에너지 커피", "낮은 나트륨 주스"
```

## 영양 성분

### 카페인 음료
- 에너지(kcal)
- 단백질(g)
- 당류(g)
- 나트륨(mg)
- 포화지방산(g)
- 카페인(mg)

### 논카페인 음료
- 에너지(kcal)
- 단백질(g)
- 당류(g)
- 나트륨(mg)
- 포화지방산(g)

## 주요 특징

### 입력 처리
- **빈 값**: 자동으로 0으로 처리
- **실시간 검증**: X축과 Y축에 동일한 값 선택 방지

### 시각화
- **반응형 디자인**: 다양한 화면 크기 지원
- **호버 상호작용**: 데이터 포인트 위에 마우스를 올려 상세 정보 확인
- **고정된 분포 차트**: 줌/팬 비활성화, 호버만 허용
- **숨겨진 스크롤바**: 모달 창의 스크롤바 시각적으로 숨김

### 성능 최적화
- **모델 캐싱**: 한 번 학습한 모델 메모리에 저장하여 재사용
- **안전한 JSON 직렬화**: NaN, Inf 값 안전하게 처리
- **효율적인 데이터 구조**: numpy 배열 기반 벡터 연산

### 군집 품질 메트릭
- **Inertia**: 군집 내 제곱합 (낮을수록 좋음)
- **Silhouette Score**: -1~1 범위 (높을수록 좋음)
- **Calinski-Harabasz Score**: 군집 간/내 분산 비율 (높을수록 좋음)
- **Davies-Bouldin Score**: 군집 유사도 (낮을수록 좋음)

## 사용 예시 (API)

### 1. 군집화 수행

```bash
curl -X POST "http://localhost:8000/api/clustering/train" \
  -H "Content-Type: application/json" \
  -d '{
    "beverage_type": "caffeine",
    "k": 3
  }'
```

### 2. 군집 인사이트 조회

```bash
curl "http://localhost:8000/api/insights/clusters?beverage_type=caffeine&k=3"
```

### 3. 군집 예측

```bash
curl -X POST "http://localhost:8000/api/prediction/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "beverage_type": "caffeine",
    "k": 3,
    "features": {
      "에너지(kcal)": 65.0,
      "단백질(g)": 1.2,
      "당류(g)": 12.0,
      "나트륨(mg)": 45.0,
      "포화지방산(g)": 0.5,
      "카페인(mg)": 80.0
    }
  }'
```

### 4. 유사 음료 찾기

```bash
curl -X POST "http://localhost:8000/api/prediction/similar" \
  -H "Content-Type: application/json" \
  -d '{
    "beverage_type": "caffeine",
    "k": 3,
    "features": {
      "에너지(kcal)": 65.0,
      "카페인(mg)": 80.0
    },
    "n_neighbors": 10,
    "search_scope": "predicted_cluster"
  }'
```


### 서버 실행 오류
- 포트 8000이 이미 사용 중인 경우: `--port` 옵션으로 다른 포트 지정
- 의존성 패키지 설치 확인: `pip install -r requirements.txt`


## 시연

### 시연 이미지


#### 메인 화면
![메인 화면](https://velog.velcdn.com/images/mourn5367/post/db80e684-e9db-4de2-8999-ebcc75f018b2/image.png)

#### 음료 타입 변경
![음료 타입 변경](https://velog.velcdn.com/images/mourn5367/post/e3931b50-581c-49d9-95a1-70c0a6d1d010/image.png)

#### K 값 변경
![K 값 변경](https://velog.velcdn.com/images/mourn5367/post/2908ea66-59f8-4a96-bb40-4b4544cc5767/image.png)

#### 차트 중복 축 선택
![차트 중복 축 선택](https://velog.velcdn.com/images/mourn5367/post/7d104948-5b90-4209-bc76-d39b400a8ca9/image.png)

#### 군집 정보 - 1
![군집 정보 - 1](https://velog.velcdn.com/images/mourn5367/post/10a6b6f6-87fc-48b7-bd10-06c0702b32b5/image.png)

#### 군집 정보 - 2
![군집 정보 - 2](https://velog.velcdn.com/images/mourn5367/post/98d14d62-f081-48e9-b4b7-35b5ca8937eb/image.png)

#### 새 음료 군집 예측
![새 음료 군집 예측](https://velog.velcdn.com/images/mourn5367/post/4ae738c9-bda3-4efb-9bb6-5de5bd09bc0a/image.png)


### 시연 영상

[![프로젝트 시연 영상](https://img.youtube.com/vi/CwXFU-wyZ98/0.jpg)](https://youtu.be/CwXFU-wyZ98)
