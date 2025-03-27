# 김인중 교수님 Transformer 강의

## Attention model

### Sequence-to Sequence Model

- Encoder-decorder architecture
  - Encoder: input sequence -> **context vector**
  - Decoder: **context vector**(+ previous output) -> new outoput
  - 여기서 Decorder는 auto regressive model의 형태를 가지고 있다.

### Attention model

- 디코더는 weighted sum의 형태로 인코더의 hidden state를 가져온다.
  - weighted sum : 가중합
    $$
    c_i = \displaystyle \sum_{j=1}^{T_x} \alpha_{ij}h_{j}
    $$
  - 여기서 $\alpha_{ij}$가 attention weight이다.
  - 그러면 $\alpha_{ij}$는 hidden state의 정보인 $h_j$가 얼마나 중요한지를 나타내는 역할을 한다.
