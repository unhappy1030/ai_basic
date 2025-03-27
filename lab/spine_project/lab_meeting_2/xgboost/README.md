# XGBoost (eXtreme Gradient Boosting)

## XGBoost란?

XGBoost는 2014년 Tianqi Chen에 의해 개발된 앙상블 기계학습 알고리즘으로, Gradient Boosting 프레임워크를 기반으로 하며, 특히 속도와 성능 면에서 뛰어난 성능을 보여주는 알고리즘입니다. 'eXtreme Gradient Boosting'의 약자로, 기존의 Gradient Boosting 알고리즘을 개선하여 더 빠르고 효율적으로 만들었습니다.

## 주요 특징

1. **정규화된 Gradient Boosting**:

   - L1, L2 정규화를 통한 과적합 방지
   - 각 트리의 가중치에 대한 정규화 항 추가
   - 모델의 복잡도 제어 가능

2. **효율적인 트리 학습**:

   - 가중치가 있는 데이터 처리 가능
   - 결측치 자동 처리
   - 병렬 처리 지원

3. **시스템 최적화**:
   - 캐시 인식 알고리즘
   - 데이터 압축
   - 분산 처리 지원

## 작동 방식

1. **기본 원리**:

   - 이전 모델의 오차를 보완하는 새로운 모델을 순차적으로 추가
   - 각 단계에서 잔차(residual)를 예측
   - 모든 모델의 예측을 가중 합산하여 최종 예측

2. **목적 함수**:

   - 학습 목적: 예측 정확도 최대화
   - 정규화 목적: 모델 복잡도 최소화
   - 최종 목적 함수 = 학습 목적 + 정규화 항

3. **트리 구조 학습**:
   - 최적의 분할점 찾기
   - 정보 이득 계산
   - 트리 구조 최적화

## 장점

1. **높은 예측 정확도**:

   - 정규화를 통한 과적합 방지
   - 효율적인 특성 선택
   - 결측치 자동 처리

2. **빠른 학습 속도**:

   - 병렬 처리 지원
   - 캐시 인식 알고리즘
   - 데이터 압축

3. **유연한 사용**:

   - 분류 및 회귀 문제 모두 해결 가능
   - 다양한 하이퍼파라미터 조정 가능
   - 다양한 평가 지표 지원

4. **결측치 처리**:

   - 자동으로 결측치 처리
   - 결측치가 있는 데이터도 학습 가능

5. **특성 중요도**:
   - 각 특성의 중요도 계산 가능
   - 특성 선택에 활용 가능

## 단점

1. **하이퍼파라미터 튜닝의 복잡성**:

   - 많은 하이퍼파라미터 존재
   - 최적값 찾기가 어려움
   - 시간 소요가 많음

2. **메모리 사용량**:

   - 대규모 데이터셋에서 메모리 부담
   - 학습 과정에서 많은 메모리 필요

3. **학습 시간**:
   - 데이터 크기가 클수록 학습 시간 증가
   - 하이퍼파라미터 튜닝에 시간 소요

## 주요 하이퍼파라미터

1. **기본 파라미터**:

   - `booster`: 부스터 타입 (gbtree, gblinear, dart)
   - `objective`: 학습 목적 (분류/회귀)
   - `eval_metric`: 평가 지표

2. **트리 관련 파라미터**:

   - `max_depth`: 트리의 최대 깊이
   - `min_child_weight`: 자식 노드에 필요한 최소 가중치 합
   - `gamma`: 분할에 필요한 최소 손실 감소량
   - `subsample`: 각 트리에서 사용할 샘플 비율
   - `colsample_bytree`: 각 트리에서 사용할 특성 비율

3. **학습 관련 파라미터**:
   - `learning_rate`: 학습률
   - `n_estimators`: 부스팅 라운드 수
   - `lambda`: L2 정규화 파라미터
   - `alpha`: L1 정규화 파라미터

## 구현 예시

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 데이터 준비
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 형식 변환
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 파라미터 설정
params = {
    'objective': 'multi:softmax',  # 다중 분류
    'num_class': 3,                # 클래스 수
    'max_depth': 6,                # 트리 최대 깊이
    'learning_rate': 0.1,          # 학습률
    'subsample': 0.8,              # 샘플링 비율
    'colsample_bytree': 0.8,       # 특성 샘플링 비율
    'min_child_weight': 1,         # 최소 자식 노드 가중치
    'gamma': 0,                    # 분할에 필요한 최소 손실 감소량
    'lambda': 1,                   # L2 정규화
    'alpha': 0,                    # L1 정규화
    'eval_metric': 'mlogloss'      # 평가 지표
}

# 모델 학습
num_rounds = 100
model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=True
)

# 예측
predictions = model.predict(dtest)

# 성능 평가
accuracy = accuracy_score(y_test, predictions)
print(f"정확도: {accuracy:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, predictions))

# 특성 중요도 확인
importance = model.get_score(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': list(importance.keys()),
    'importance': list(importance.values())
})
print("\n특성 중요도:")
print(feature_importance.sort_values('importance', ascending=False))
```

## 활용 분야

1. **분류 문제**:

   - 이진 분류
   - 다중 클래스 분류
   - 불균형 데이터 분류

2. **회귀 분석**:

   - 연속형 변수 예측
   - 시계열 예측

3. **순위 예측**:
   - 검색 엔진 순위
   - 추천 시스템

## 참고 자료

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
- XGBoost 공식 문서
- scikit-learn 공식 문서
- Kaggle XGBoost 가이드
