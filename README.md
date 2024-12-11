# T5_SO3

SO3-equivariant T5 transformer for conditional 3D graph generation, with a detailed example of protein design

## Model detail

Encoder-decoder model inspired by [T5](https://arxiv.org/abs/1910.10683). [Frame-averaging](https://arxiv.org/abs/2110.03336) method is used to provide equivariance and invariance to any SO3 transformation. We also include proven techniques such as RoPE, RMS norm and kv cache. As a prototype, kv cache is in a relatively naive implementation.
