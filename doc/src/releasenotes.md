
## MIOpen Release notes


### 05/03/2019 [ 1.8.1 ]

- This release contains minor bug fixes and additional performance database improvements.

Changes:

- Fixed accuracy issue with backwards weights
- Fixed issue with name parsing for newer architectures
- Added narrow workaround for 5x10 and 5x20 filter performance regression
- Improved support in performance database for Radeon VII



### 04/11/2019 [ 1.8.0 ]

- This release contains full 3-D convolution support and int8 support for inference. 
- Additionally, there are updates in the performance database for major models including those found in Torchvision. 
- An assortment of bugs have been resolved in this release.

  
Changes:

- Fixed various issues in assembly kernels
- Fixed issue #92 and #79 for miopenOpTensor
- Fixed issue #88 for bzip2
- Fixed issue #77 algorithm mismatch
- Added Winograd support for fp32 backwards weights
- Added pooling inclusive mode
- Added tuning for direct group convolution algorithms
- Added additional kernel support for group convolutions
- Added API for 3-D convolutions
- Added support for int8 inference convolutions
- Added integer selection for pooling indexing
- Added minimum dependencies support
- Added RNN fp16 support on the MIOpen-HIP backend
- Added 1x1 convolution + bias + activation fusions
- Added workaround for issue #84 GPU memory access fault
- Added performance tuning for direct backwards weights
- Improved performance database coverage
- Improved internal quality by reducing redunant code
- Improved build instructions in README.md
- Improved performance database coverage for fusions
- Updated Docker components and requirements


Known Issues:

- RNNs do not support fp16 on the MIOpen-OpenCL backend
- OpenCL backend does not support GEMM convolutions in fp16



### 02/06/2019 [ 1.7.1 ]

- This release contains minor bug fixes and performance improvements.
  

Changes:

- Fixed corrupt and obsolete performance database entries
- Fixed issue #70, "SIGFPE (DIV/0) in ConvOclBwdWrW2::GetSolution()"
- Fixed issue #72, "workSpaceSize check assertion fails in ConvolutionBackwardWeights() - DEBUG builds only"
- Fixed issue #77, "Results of ConvBwdWeightsAlgoDirect and ConvBwdWeightsAlgoGEMM mismatch for some specific parameters"
- Removed default dependency of RNNs on rocBLAS
- Added a workaround for softmax fp16 correctness issue
- Added check to only make MIOpen with static boost libraries
- Improved performance database coverage

Known Issues:

- RNNs do not support fp16
- OpenCL backend does not support GEMM convolutions in fp16
- Layer fusions for convolution 1x1 fp16 are not supported
- Layer fusions for large image 1x1 convolutions may cause an exception instead of a warning during compile phase if plan is not supported


### 12/19/2018 [ 1.7.0 ]

- This release contains general bug fixes and an updated performance database
- Group convolutions backwards weights performance has been improved
- Logging across the library has been improved
- Performance database has been updated

  
Changes:

- Fixed logging issues with group convolution and pooling
- Fixed sphinx version issue in document generation
- Fixed issues with corrupt entries in performance database
- Removed external dependency on libSSL and libCrypto
- Added support for large image backwards weights in direct convolution
- Added fp16 support for RNNs on the HIP backend
- Improved performance database coverage

Known Issues:

- RNNs do not support fp16
- OpenCL backend does not support GEMM convolutions in fp16
- Layer fusions for convolution 1x1 fp16 are not supported
- Layer fusions for large image 1x1 convolutions may cause an exception instead of a warning during compile phase if plan is not supported


### 11/18/2018 [ 1.6.0 ]

- Training in fp16 (half precision) including mixed-precision is now fully supported
- Batch Normalization in fp16 (half precision) including mixed-precision are now available
- Performance improvements for 3x3 and 1x1 single-precision convolutions
- Layer fusions for BatchNorm+Activation are now available
- Layer fusions with convolutions now support varying strides and padding configurations

Changes: 

- rocBLAS is now used as the default BLAS library for the HIP backend (minimum version 14.3.0)
- Fixed various bugs in convolution kernels
- Fixed issues with bad references in layer fusion 
- Fixed gfx803 assembily issues
- Added support fp16 Winograd convolutions
- Added support for fp16 pooling
- Improved error reporting for convolutions and layer fusions
- Improved documentation

Known Issues:

- RNNs do not support fp16
- OpenCL backend does not have full fp16 support
- Layer fusions for convolution 1x1 fp16 are not supported


### 09/14/2018 [ 1.5.0 ]

Notes:

- A new kernel fusion API is now available for inference for convolution, bias, 
  batch normalization, and activations.
- This release includes new features and bug fixes
- Group and Depthwise convolutions are now available
- 3D Batch Normalization has been implemented for fully packed tensors
- Dilation for convolutions have been implemented

Changes:

- Fixed bugs in direct convolutions
- Fixed issue with paths when $HOME variable is not set
- Fixed padding issues with 1x1 convolutions
- Added incremental support for fp16
- Added fused kernels for Winograd and direct with bias and activations
- Added a getting started guide for kernel fusion.
- Added group and depthwise API for convolutions
- Added 3-D batch normalization support with 5-D tensors
- Improved max pooling performance
- Improved debug and error reporting information
- Improved documentation for convolutions

Known Issues:

- RNNs do not support fp16
- Training with CNNs does not support fp16


### 07/30/2018 [ 1.4.2 ]

Notes: 

- This release is a hot-fix to enable ICNet and PSPNet

Known Issues:

- RNNs do not support fp16
- Training with CNNs does not support fp16
- Users may encounter a warning that their performance database is out of date. The performance database can be updated by setting the environment variable for just the initial run of an application: `MIOPEN_FIND_ENFORCE=search`
For more information on the performance database, see: https://rocmsoftwareplatform.github.io/MIOpen/doc/html/perfdatabase.html#

### 07/19/2018 [ 1.4.1 ]

Notes: 

- This release includes a bug fix for 3x3 convolutions
- Updated README file configuration instructions

Known Issues:

- RNNs do not support fp16
- Training with CNNs does not support fp16
- Users may encounter a warning that their performance database is out of date. The performance database can be updated by setting the environment variable for just the initial run of an application: `MIOPEN_FIND_ENFORCE=search`
For more information on the performance database, see: https://rocmsoftwareplatform.github.io/MIOpen/doc/html/perfdatabase.html#

### 07/06/2018 [ 1.4.0 ]

Notes:

- This release includes a number of performance improvements and bug fixes
- New features have been added to convolutions for auto-tuning kernels
- Activations now have new modes available
- Documentation has been updated and corrected

Changes:

- Fixed documentation errors
- Fixed bug in activations with pass-through mode
- Fixed performance database locking issues
- Fixed Winograd kernel behavior for stride 2 backwards data
- Fixed a bug in OpTensor layer
- Fixed a timing issue with batch normalization inline assembly 
- Fixed issue with an unnecessary binary creation in assembly bug detection
- Fixed issue with disk program cache directory not being created
- Fixed a bug with convolution+bias
- Added to performance database functionality
- Added leaky-ReLU, clipped, and exponential-ReLU modes to activation
- Added documentation for performance database usage
- Added support for 1x1 convolutions with non-zero padding
- Added API for printing status codes as strings
- Added auto-tuning feature for convolutions
- Improved LSTM and GRU backwards pass performance
- Improved debug and error reporting information
- Improved performance of batch normalization spatial mode
- Improved find stage for convolutions
- Improved readability for user database file

Known Issues:

- RNNs do not support fp16
- Training with CNNs does not support fp16

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
