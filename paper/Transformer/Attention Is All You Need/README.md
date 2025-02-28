# Attention Is All You Need

<https://arxiv.org/pdf/1706.03762>

## Abstract

현재 주류를 이루는 sequence transduction model들은 인코더와 디코더를 포함하는 복잡한 recurrent, convolution network를 기반으로 한다.
가장 성능이 뛰어난 모델들은 또한 어텐션 메커니즘을 통해서 인코더와 디코더를 연결한다.
우리는 순환과 합성곱을 완전히 배제하고 오직 어텐션 메커니즘만을 기반으로 하는 단순 네트워크 구조인 Transformer를 제안한다.

- 이전에는 attention 메커니즘을 적용하는 데에 recurrent or convolution network가 포함된 인코더, 디코더를 연결해서 사용했지만 여기서는 Transformer라는 recurrent or convolution network가 없는 단순 네트워크 구조를 사용한 어텐션 메커니즘을 제안한다.

## Introduction

RNN, LSTM, GRU 등의 신경망은 언어 모델링과 기계 번역과 같은 시퀀스 모델링 및 변환 문제에서 최첨단 접근방식으로 확고히 자리 잡았다.
