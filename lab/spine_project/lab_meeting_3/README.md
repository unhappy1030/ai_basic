# Support Vector Machine (SVM)

## 개요
Support Vector Machine(SVM)은 지도 학습 기반의 강력한 분류 알고리즘으로, 고차원 공간에서 데이터를 분류하는 데 사용됩니다. SVM은 특히 복잡한 분류 문제에서 뛰어난 성능을 보여주며, 선형 및 비선형 분류 모두에 적용할 수 있습니다.

## 핵심 개념

### 1. 마진과 서포트 벡터
- **마진(Margin)**: 결정 경계(decision boundary)와 가장 가까운 데이터 포인트들 사이의 거리
- **서포트 벡터(Support Vectors)**: 마진을 결정하는 가장 가까운 데이터 포인트들
- SVM의 목표는 마진을 최대화하는 결정 경계를 찾는 것입니다.

### 2. 선형 SVM
- 선형적으로 분리 가능한 데이터에 적용
- 결정 경계는 다음과 같은 형태:
  ```
  w·x + b = 0
  ```
  여기서 w는 가중치 벡터, b는 편향, x는 입력 벡터

### 3. 비선형 SVM
- 커널 트릭(Kernel Trick)을 사용하여 비선형 분류 수행
- 주요 커널 함수:
  - RBF(Radial Basis Function) 커널
  - 다항식 커널
  - 시그모이드 커널

## SVM의 장점
1. 고차원 공간에서 효과적
2. 과적합에 강건
3. 비선형 분류 가능
4. 메모리 효율적 (서포트 벡터만 저장)

## SVM의 단점
1. 대규모 데이터셋에서 계산 비용이 높음
2. 하이퍼파라미터 튜닝이 중요
3. 노이즈에 민감할 수 있음

## 하이퍼파라미터
주요 하이퍼파라미터:
- C: 규제 파라미터
- γ (gamma): RBF 커널의 파라미터
- degree: 다항식 커널의 차수

## 구현 예시 (Python)
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 데이터 전처리
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM 모델 생성 및 학습
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)

# 예측
y_pred = svm.predict(X_test_scaled)
```

## 참고 자료
1. Vapnik, V. N. (1995). The Nature of Statistical Learning Theory
2. Cortes, C., & Vapnik, V. (1995). Support-vector networks
3. scikit-learn documentation: https://scikit-learn.org/stable/modules/svm.html 