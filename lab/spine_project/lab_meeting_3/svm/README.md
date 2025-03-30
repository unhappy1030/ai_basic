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

# 학습 계획

## 공부 순서와 학습 내용

### 1. 분류(Classification)와 회귀(Regression)
- 지도학습의 두 가지 주요 유형 이해

### 분류(Classification)
분류는 데이터를 미리 정의된 클래스나 카테고리로 구분하는 문제이다.

#### 특징
- 출력이 이산적(discrete)값
- 예시 :
    - 이메일 스팸 분류 (스팸/정상)
    - 이미지 분류 (고양이/강아지)
    - 질병 진단(양성/음성)
#### 주요 알고리즘
- 로지스틱 회귀
- 결정 트리
- SVM
- 신경망

### 회귀(Regression)
회귀는 연속적인 값을 예측하는 문제이다.

#### 특징
- 출력이 연속적(continuous)
- 예시:
    - 집 가격 예측
    - 날씨 온도 예측
    - 주식 가격 예측

#### 주요 알고리즘
- 선형 회귀
- 다항 회귀 
- 릿지 회귀
- 라쏘 회귀

### 차이점 비교
#### 1) 출력값의 특성
- 분류 : 이산적 값(예: 0 또는 1, 클래스 레이블)
- 회귀 : 연속적 값(예: 실수값)

#### 2) 평가 지표
- 분류: 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-score
- 회귀: MSE(Mean Squared Error), RMSE, MAE(Mean Absolute Error), $R^2$

#### 3) 모델의 목적
- 분류: 데이터를 미리 정의된 클래스로 구분
- 회귀: 연속적인 값의 패턴을 찾아 예측

### 2. 베이즈 정리(Bayesian Theorem)
- 확률론의 기초 개념
- 조건부 확률과 베이즈 정리
- 실습: 간단한 확률 계산 예제

### 3. 생성 모델(Generative Model)과 가우시안 모델
- 생성 모델의 기본 개념
- 정규분포와 가우시안 모델
- 실습: 데이터 분포 시각화

### 4. 판별 함수와 선형 분류기
- 선형 분류의 기본 개념
- 결정 경계의 이해
- 실습: 2차원 데이터로 선형 분류기 구현

### 5. K-최근접 이웃 알고리즘
- KNN의 작동 원리
- 거리 측정 방법
- 실습: scikit-learn으로 KNN 구현

### 6. 파라미터 추정 방법
- 최대 가능도 추정(MLE)
- 최대 사후 확률 추정(MAP)
- 실습: 간단한 파라미터 추정 예제

### 7. 로지스틱 회귀
- 로지스틱 함수의 이해
- 이진 분류에서의 로지스틱 회귀
- 실습: scikit-learn으로 로지스틱 회귀 구현

### 8. 서포트 벡터 머신
- SVM의 기본 개념
- 커널 트릭의 이해
- 실습: scikit-learn으로 SVM 구현

## 학습 방법
1. **이론 학습**
   - 교재나 온라인 강의로 개념 이해
   - 수학적 기초가 필요한 부분은 수식 유도 과정도 함께 학습

2. **코드 실습**
   - 각 주제별로 간단한 예제 코드 작성
   - scikit-learn 라이브러리 활용
   - 결과 시각화를 통한 이해

3. **실제 데이터 적용**
   - 각 알고리즘을 실제 데이터셋에 적용
   - 성능 평가 및 비교

4. **정리 및 복습**
   - 각 주제별로 핵심 개념 정리
   - 다른 알고리즘들과의 비교 분석

# Support Vector Machine : SVM
- 국경 만들기 -> 이진분류 문제

경계 : $ax + by = c$

