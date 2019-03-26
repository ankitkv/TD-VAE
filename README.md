# TD-VAE

TD-VAE implementation in PyTorch 1.0.

This code implements the ideas presented in the paper [Temporal Difference Variational Auto-Encoder (Gregor et al)][2]. This implementation includes configurable number of stochastic layers as well as the specific multilayer RNN design proposed in the paper.

**NOTE**: This implementation also makes use of [`pylego`][1], which is a minimal library to write easily extendable experimental machine learning code.

## Results

Results on Moving MNIST will be posted here soon.

[1]: https://github.com/ankitkv/pylego
[2]: https://arxiv.org/abs/1806.03107
