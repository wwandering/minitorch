# minitorch
A minimalistic Torch library reimplementation based on Sasha Rush's (https://github.com/srush) materials.

## Notes
- Autodiff: scalars, chain rule, backprop.
- Tensors: core backend, broadcasting, operations, gradient & autograd.
- Some operations (like map-scan-reduce and matrix multiplication) enhanced with numba & CUDA.
- For sake of testing actually if we can do anything with this library, core network code for CNNs (like convolutions, pooling, etc.) is written and tested.
- Due to numba and its GPU shenanigans, some tests are failing on my local machine. However, Google Colab is good to go and everything works with it.
