#  NLP HW4 -- Character RNN & Mini Transformer Encoder

This repository contains implementations of core neural
language-modeling components, including:

-   A **Character-level RNN Language Model**
-   A **Scaled Dot-Product Attention module**
-   A **Mini Transformer Encoder** block

These implementations are designed for educational purposes and follow
the structure of typical NLP coursework involving sequence modeling,
attention mechanisms, and transformer architectures.

## Project Structure

    .
    ├── char_rnn_language_model.py        # Character-level RNN language model
    ├── scaled_dot_product_attention.py   # Attention mechanism
    ├── mini_transformer_encoder.py       # Lightweight Transformer encoder block

## Features

### 1. **Character RNN Language Model**

-   Implements a recurrent neural network operating on character
    embeddings\
-   Predicts next characters in a sequence\
-   Uses PyTorch modules such as `nn.RNN` / `nn.LSTM` / `nn.GRU`\
-   Supports sampling text from the trained model

### 2. **Scaled Dot-Product Attention**

-   Computes query--key similarity\
-   Applies scaling factor `1/√d_k`\
-   Includes optional masking for autoregressive tasks\
-   Forms the foundation of multi-head attention

### 3. **Mini Transformer Encoder**

-   Includes:
    -   Scaled dot-product attention\
    -   Feedforward network\
    -   Layer normalization\
    -   Residual connections\
-   Represents the core building block of modern transformer
    architectures

## Requirements

    pip install torch numpy

## Usage

### Character RNN Language Model

``` python
from char_rnn_language_model import CharRNNLanguageModel

model = CharRNNLanguageModel(...)
output = model(inputs)
```

### Scaled Dot-Product Attention

``` python
from scaled_dot_product_attention import scaled_dot_product_attention

attn_output, attn_weights = scaled_dot_product_attention(Q, K, V)
```

### Mini Transformer Encoder

``` python
from mini_transformer_encoder import MiniTransformerEncoder

encoder = MiniTransformerEncoder(...)
output = encoder(x)
```

## Learning Objectives

This project helps reinforce understanding of:

-   Sequence modeling with recurrent neural networks\
-   Attention mechanisms and their mathematical foundations\
-   Transformer architecture internals\
-   Building and debugging modular deep-learning code

## License

This project is for educational purposes.\
You may adapt or reuse the code with proper attribution.
