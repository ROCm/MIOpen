
## MIOpen Release notes

### 03/30/2018 [ 1.3.0 ]

Notes: 

- Performance improvements for RNNs
- Performance improvements for convolutions using 1x1 filters
- Performance improvement for Batch Normalization
- This release adds preliminary fp16 support for Inference using CNNs
- Bug fixes for various components of MIOpen

Changes:

- Added 2 new API for RNNs: miopenGetRNNLayerParamOffset and miopenGetRNNLayerBiasOffset
- Added support for uninitialized hidden states and nullptr outputs in RNNs
- Added support for Set and Scale operations for strided tensors with dimensions 1 to 5
- Added multi-thread and multi-process support for the performance database
- Improved performance for OpTensor
- Fixed bug in convolutions for backward bias
- Fixed logic issues in get and set layer functions and related w_supertensor test
- Fixed hang in batch norm with batch sizes greater than 256

Known Issues:

- RNNs do not support fp16
- Training with CNNs does not support fp16


### 03/08/2018 [ 1.2.1 ]

Notes:

- This release adds support for ROCm 1.7.1.


### 12/15/2017 [ 1.2.0 ]

Notes:

- This release adds the support for recurrent neural networks (RNNs) for three flavors - Vanilla, LSTMs, and GRU
- Users can now themselves update the perf-db file, which hosts the tuning parameters for convolutions, by setting appropriate environment variables

Changes:

- Over 50% improvement in ResNet performance since the last release
- Multiple padding modes like Same and Valid added
- Winograd convolution kernels added for strided bwd-data convolutions
- Tensor Ops allow for beta and alpha scaling values and support up to 5 dimensions with strides and offsets
- Tensor Copy supports up to 5 dimesnional copies with strides and offsets
- Unit-tests for LRN are added
- Several bug fixes for all the layers of the library

Known issues:

- RNNs may give incorrect result due to a known compiler bug; issue may particulary arise during some RNNs configs with GEMM of size power of 4
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
