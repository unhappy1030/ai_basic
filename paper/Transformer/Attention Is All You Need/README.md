# Attention Is All You Need

<https://arxiv.org/pdf/1706.03762>

## Abstract

현재 주류를 이루는 sequence transduction model들은 인코더와 디코더를 포함하는 복잡한 recurrent, convolution network를 기반으로 한다.
가장 성능이 뛰어난 모델들은 또한 어텐션 메커니즘을 통해서 인코더와 디코더를 연결한다.
우리는 순환과 합성곱을 완전히 배제하고 오직 어텐션 메커니즘만을 기반으로 하는 단순 네트워크 구조인 Transformer를 제안한다.

- 이전에는 attention 메커니즘을 적용하는 데에 recurrent or convolution network가 포함된 인코더, 디코더를 연결해서 사용했지만 여기서는 Transformer라는 recurrent or convolution network가 없는 단순 네트워크 구조를 사용한 어텐션 메커니즘을 제안한다.

## Introduction

RNN, LSTM, GRU 등의 신경망은 언어 모델링과 기계 번역과 같은 시퀀스 모델링 및 변환 문제에서 최첨단 접근방식으로 확고히 자리 잡았다.
어텐션 메커니즘은 다양한 작업에서 효과적인 시퀀스 모델링과 변환 작업에 꼭 필요한 요소가 되었다. 이 메커니즘을 사용하면, 입력이나 출력의 길이나 단어들 사이의 거리에 상관없이 서로의 관계를 쉽게 이해하고 모델링 할 수 있다.
순환 모델은 입력과 출력 시퀀스의 각 위치에서 개별적인 계산을 수행함으로써 전체 연산을 구성한다.

```
symbol or token은 시퀀스를 구성하는 가장 작은 단위들을 의미한다.

    단어 토큰: 문장을 단어 단위로 나누어 각 단어를 하나의 토큰으로 취급하는 방식

    서브워드 토큰: 단어를 더 작은 의미 단위(예: 형태소)로 나누어 처리하는 방식

    문자 토큰: 각각의 문자를 하나의 토큰으로 보는 방식

시퀀스는 숫서가 있는 데이터 집합을 의미한다. 이는 입력 데이터(예: 문장에서 각 단어의 순서) 또는 출력 결과물(예: 번여고딘 문장의 단어 순서)이 될 수 있다.
```

## Model Achitecture

가장 경쟁력있는 뉴런 변환 모델은 인코더와 디코더를 포함하는 구조를 가진다.

### Encorder and Decoder Stacks

> Encoder: N = 6, identical layers, each layer has sub-two layers

- first layer : multi-head self-attention mechanism
  - multi-head self-attention mechanism : 이 메커니즘은 입력 시퀀스 내의 각 요소가 다른 모든 요소와의 관계를 학습할 수 있도록 설계된 핵심 신경망 구조이다. 주요 구성요소로는 Query, Key, Value라는 세가지 주요 요소를 포함하며, 각 요소는 입력 임베딩에서 선형 변환을 통해 도출된다.
