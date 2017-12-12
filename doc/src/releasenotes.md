
## MIOpen Release notes


### 12/15/2017 [ 1.2.0 ]

Notes:

- This release includes recurrent neural networks as a new module. Support for RNNs with tanh and relu activations, GRUs, and LSTMs.

Changes:
- 50% improvement in ResNet performance
- Tensorflow padding modes now supported (same & valid) for convolutions and pooling
- RNN, GRU, and LSTM recurrent network support added
- LRN bug fixes and tests added
- Winograd convolution backwards data assembly kernel added
- New method for searching the performance database (see documentation)
- Tensor Ops now allows for beta and alpha scaling values and supports up to 5 dimensions with strides and offsets
- Copy tensor allows for up to 5 dimesnional copies with strides and offsets

Known issues:
- RNNs may give incorrect result due to a known compiler bug, issue may arise during some recurrent network configs with GEMM of size power of 4
- Potential issue where OpenCL resources will be exhausted for large RNN



### 09/08/2017 [ 1.1.0 ]

Notes: 

- The scaling parameter alpha and shift parameter beta for layers kernels are only supported for alpha = 1 and beta = 0.
The exceptions to this are for miopenOptTensor, miopenConvolutionForwardBias, and miopenConvolutionBackwardBias.

- Currently, only 32-bit floats are supported in MIOpen.

- MIOpen only supports tensor layout NCHW.

Changes:
- Added persistent cache for compiled GPU kernels
- Performance improvements for batch normalization kernels
- Performance improvements for all types of convolutions for 1x1 filters
- Performance improvements for all types of convolutions with non-unit strides
- Performance improvements for backward-weights convolutions for 3x3 filters
- Performance improvements for the AddTensor operation
- Various bug fixes for Winograd convolutions 


### 08/27/2017 [ 1.0.2 ]
- Fixed 1x1 forward and backward convolutions for large input
- Fixed pooling MIOpendriver
- Disabled 1x1 Winograd convolution for HIP
- Disabled asm. backward-weights convolutions for input width == 175 
 

### 07/26/2017 [ 1.0.1 ] 
- Added dilation support for convolutions 
- Added unit-tests for Softmax
- Added miopengemm as a required dependency for MIOpen build
- Performance improvements for batch normalization via activation of data-parallel primitives (DPP) hardware instructions
- Fixed documentation to remove GEMM API interface
- Fixed Bwd-Weights Convolutions with 1x1 filters with stride=2
- Fixed Softmax grid-size selection
- Fixed debug prints of kernel launch parameters.
- Removed GEMM interface from the MIOpen API


### 06/30/2017 [ 1.0.0 ] Initial release  
