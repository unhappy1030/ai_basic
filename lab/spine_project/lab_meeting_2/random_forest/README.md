# 랜덤 포레스트 (Random Forest)

## 랜덤 포레스트란?

랜덤 포레스트는 여러 개의 의사결정 나무(Decision Tree)를 만들고, 이들의 예측을 종합하여 최종 예측을 하는 앙상블 기계학습 알고리즘입니다. 2001년 Leo Breiman에 의해 제안되었으며, 현재까지도 가장 널리 사용되는 머신러닝 알고리즘 중 하나입니다. 이 알고리즘은 'Bagging'과 'Random Feature Selection'이라는 두 가지 핵심 개념을 결합하여 만들어졌습니다.

## 주요 특징

1. **앙상블 학습**:

   - 여러 개의 의사결정 나무를 만들어 각각의 예측을 종합합니다.
   - 각 나무는 독립적으로 학습되며, 서로 다른 데이터와 특성을 사용합니다.
   - 최종 예측은 모든 나무의 예측을 종합하여 결정됩니다.

2. **랜덤성**:

   - 데이터의 랜덤 샘플링 (Bootstrap)
     - 원본 데이터셋에서 무작위로 샘플을 추출
     - 각 샘플은 원본 크기와 동일 (중복 허용)
     - 약 1/3의 데이터는 Out-of-Bag(OOB) 샘플로 남음
   - 특성(feature)의 랜덤 선택
     - 각 분기점에서 전체 특성 중 일부만 무작위로 선택
     - 일반적으로 전체 특성 수의 제곱근만큼 선택
     - 이는 나무 간의 상관관계를 줄이는 효과가 있음

3. **과적합 방지**:

   - 여러 나무의 예측을 평균화하여 과적합을 줄입니다.
   - 각 나무는 서로 다른 데이터로 학습되어 다양한 관점에서 예측
   - OOB 샘플을 통한 성능 평가 가능

4. **특성 중요도**:
   - 각 특성이 예측에 얼마나 중요한지 평가 가능
   - Gini 불순도 감소량 기반
   - 특성 선택 및 데이터 분석에 활용

## 작동 방식

1. **데이터 샘플링**:

   - 원본 데이터셋에서 무작위로 샘플을 추출하여 여러 개의 학습 데이터셋 생성
   - 각 샘플은 원본 데이터셋과 동일한 크기 (중복 허용)
   - Bootstrap 과정을 통해 각 나무마다 다른 학습 데이터 사용
   - OOB 샘플은 모델 평가에 활용

2. **나무 생성**:

   - 각 샘플 데이터셋에 대해 의사결정 나무 생성
   - 각 분기점에서 전체 특성 중 일부만 무작위로 선택하여 최적의 분기점 결정
   - 나무는 완전히 자라날 때까지 성장 (일반적으로)
   - 가지치기(pruning) 없이 과적합을 방지

3. **예측**:
   - 새로운 데이터에 대해 모든 나무의 예측을 수행
   - 분류 문제: 다수결 투표 (각 나무의 예측 클래스 중 가장 많은 표를 받은 클래스 선택)
   - 회귀 문제: 평균값 사용 (모든 나무의 예측값의 평균)

## 장점

1. **높은 예측 정확도**:

   - 여러 나무의 예측을 종합하여 안정적인 결과 제공
   - 노이즈에 강한 예측 성능

2. **과적합에 강함**:

   - 랜덤성과 앙상블 효과로 과적합 위험 감소
   - OOB 샘플을 통한 성능 평가 가능

3. **특성 중요도 평가 가능**:

   - 각 특성의 중요도를 정량적으로 평가
   - 특성 선택에 활용 가능

4. **대용량 데이터 처리 가능**:

   - 병렬 처리 지원
   - 대규모 데이터셋에서도 효율적인 학습

5. **결측치와 이상치에 강함**:

   - 여러 나무의 예측을 종합하여 노이즈에 강함
   - 결측치가 있어도 학습 가능

6. **병렬 처리 가능**:
   - 각 나무를 독립적으로 학습하여 병렬 처리 가능
   - 학습 시간 단축

## 단점

1. **모델 해석이 복잡함**:

   - 여러 나무의 예측을 종합하여 최종 결과 도출
   - 단일 의사결정 나무보다 해석이 어려움

2. **메모리 사용량이 많음**:

   - 여러 개의 나무를 저장해야 함
   - 대규모 데이터셋에서 메모리 부담

3. **학습 시간이 상대적으로 김**:
   - 여러 나무를 생성하고 학습해야 함
   - 데이터 크기가 클수록 학습 시간 증가

## 활용 분야

1. **분류 문제**:

   - 이진 분류
   - 다중 클래스 분류
   - 불균형 데이터 분류

2. **회귀 분석**:

   - 연속형 변수 예측
   - 시계열 예측

3. **특성 선택**:

   - 특성 중요도 기반 특성 선택
   - 차원 축소

4. **이상치 탐지**:
   - OOB 샘플 활용
   - 이상치 점수 계산

## 구현 예시

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 데이터 준비
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성
rf_model = RandomForestClassifier(
    n_estimators=100,    # 나무의 개수
    max_depth=None,      # 나무의 최대 깊이
    min_samples_split=2, # 분기점 생성에 필요한 최소 샘플 수
    min_samples_leaf=1,  # 리프 노드에 필요한 최소 샘플 수
    max_features='sqrt', # 각 분기점에서 사용할 특성 수
    bootstrap=True,      # 부트스트랩 샘플링 사용
    random_state=42      # 재현성을 위한 시드값
)

# 모델 학습
rf_model.fit(X_train, y_train)

# 예측
predictions = rf_model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, predictions)
print(f"정확도: {accuracy:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, predictions))

# 특성 중요도 확인
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
})
print("\n특성 중요도:")
print(feature_importance.sort_values('importance', ascending=False))
```

## 하이퍼파라미터 튜닝

랜덤 포레스트의 주요 하이퍼파라미터:

1. **n_estimators**: 나무의 개수

   - 더 많은 나무 = 더 안정적인 예측
   - 학습 시간과 메모리 사용량 증가

2. **max_depth**: 나무의 최대 깊이

   - None: 완전히 자라날 때까지
   - 제한: 과적합 방지

3. **min_samples_split**: 분기점 생성에 필요한 최소 샘플 수

   - 더 큰 값 = 과적합 방지
   - 더 작은 값 = 더 세밀한 분할

4. **max_features**: 각 분기점에서 사용할 특성 수
   - 'sqrt': 전체 특성 수의 제곱근
   - 'log2': 전체 특성 수의 로그
   - 정수: 사용할 특성 수

## 참고 자료

- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- scikit-learn 공식 문서
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning
