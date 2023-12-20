
## MIOpen Release notes

### 08/03/2021 [2.12.0]

 - This release includes support for Navi21 and various other bug fixes and performance improvements

 - MIOpen now supports Navi21!! (via MIOpen PRs 973, 780, 764, 740, 739, 677, 660, 653, 493, 498)
 - Fixed a correctness issue with ImplicitGemm algorithm 
 - Updated the performance data for new kernel versions
 - Improved MIOpen build time by splitting large kernel header files
 - Fixed an issue in reduction kernels for padded tensors
 - Various other bug fixes and performance improvements

### 05/17/2021 [2.11.0]

 - This release contains various bug fixes and performance improvements.
 
 - Updates for Target ID features in ROCm stack
 - Correctness fix in Batchnorm kernels
 - Various bug fixes for MIOpenGEMM on the OpenCL backend
 - Various bug fixes in 3x3 assembly kernels

### 03/25/2021 [2.10.0]

 - This release contains new reduction operations, Winograd algorithm performance improvements as well as bug fixes. Various host side performance improvements have been added as well.

 - Added a GPU reference kernel implementation for faster testing.
 - Add TargetID support for new AMD GPU architectures.
 - Implementation of four additional generic tensor reduction operations (AVG, AMAX, NORM1, NORM2).
 - Fixed a bug where Batchnorm would give incorrect results when the product of image height and image width is not a factor of four.
 - Various host side improvements for better find and tuning performance.
 - Added support for AMD Code Object V4.

### 12/01/2020 [ 2.9.0 ]

 - This release contains implicit GEMM algorithm performance updates and bug fixes. Additional performance improvements have been implemented for batch normalization.

 - Added new assembly implicit GEMM kernels
 - Added batch normalization optimizations
 - Added missing tunings from 2.8.0 release cycle
 - Fixed issue where miopen-hip backend install would not search for rocBLAS dependency 
 - Removed deprecated implicit GEMM xDLOPs solvers
 - Removed incorrect error messages from implicit GEMM solvers
 - Disabled ConvAsmBwdWrW3x3 solver for stride > 1 cases
 - Disabled bidirectional multi-pass Winograd kernels due to stability issues


### 10/28/2020 [ 2.8.0 ]

This release provides additional bug fixes and support for embedded build using MIOpen as a static library. 

- Fixed workspace size calculation for GEMM group convolutions
- Fixed performance regression for M/N
- Fixed issue with faulty compiler option
- Fixed typo in components dependency variable in CMakeLists.txt
- Fixed issues with COMgr backed online compilation for HIP kernels
- Added cmake flag for embedding system databases when building a static library
- Added a way to disable building MIOpenDriver when building a static library
- Added CC compiler detection in ROCm environment
- Known issue: This release may show warnings for "obsolete configs" in the performance database. This can be fixed by rerunning tuning on a specfic network; [see tuning documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/perfdatabase.html#miopen-find-enforce)



### 09/18/2020 [ 2.7.0 ]

- This release contains a new reduction API; see [API documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/apireference.html) for more information. Additional features for embedded builds have been added, and further support for 3D convolutional networks. 

- Added additional tunings into performance database
- Added general reduction API
- Added cmake flag for embedding binary database into a static MIOpen build
- Added cmake flag for embedding system find-db text files into static MIOpen build
- Fixed issue with GEMM workspace size calculation for backwards data convolutions [#381](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/381)
- Fixed issue with 3D pooling indexing [#365](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/365)


### 08/20/2020 [ 2.6.0 ]

- This release contains convolution performance improvements, improved multi-threading behavior, and improved stability for half precision convolutions. Initial iteration time has been reduced with the introduction of hybrid find mode. Builds for a static library have been refined for this release.

- Added MIOPEN_FIND_MODE=3 as the new default convolution Find mode; see documentation [here](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/find_and_immediate.html#find-modes) for details
- Added a more runtime-parameterized version of pooling to reduce the number of online compilations
- Improved the performance of backwards spatial batch normalization for small images
- Fixed issue with std::logic_error in SQLite deleter [#306](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/306)
- Fixed issues with half precision stability for convolutions
- Fixed issues with multi-threaded SQLite database accesses
- Fixed issues with 3-D convolutions and incorrect parameters
- Fixed various issues with implicit GEMM static assert failures
- Removed inactive implicit GEMM convolution solvers
- Removed SCGEMM convolutional algorithm from MIOpen


### 07/10/2020 [ 2.5.0 ]

- This release contains convolution performance improvements, various minor fixes and documentation updates.

- Added a script to detect and install appropriate precompiled kernels
- Added 3D convolution backwards weights implicit GEMM implementation 
- Improve performance of convolution implicit GEMM algorithm
- Improved database coverage for batch size 1
- Improved logging and error reporting
- Improved documentation for debugging with numeric checks
- Fixed issue with potential infinities and NaNs appearing during low precision training on CNNs


### 06/02/2020 [ 2.4.0 ]

- This release contains new implementations of 3D convolutions using implicitGEMM, general performance improvements for convolutions, bug fixes, better versioning in directories, integration with the new rocclr, and dropout support in RNNs.

- Added 3D convolutions for the implicitGEMM algorithm in the forward and backward-data passes
- Added dropout support for RNN layer; e.g., RNN-vanilla, GRU, and LSTM
- Added support for AMD's rocclr runtime and compiler
- Improved performance for implicitGEMM and Winograd algorithms
- Improved database locking
- Fixed issue with GPU memory segmentation fault on asymmetric padding [#142](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/142)


### 03/01/2020 [ 2.3.0 ]

- This release contains new implementations of the implicitGEMM and Winograd algorithms, performance improvements for convolutions, further support for 3D convolutional networks, and various bug fixes.

- Added 3D Pooling layers
- Added backwards data algorithm for implicitGEMM
- Added GEMM performance improvements via relaxed constraints in rocBLAS-Tensile
- Added full CO v3 support for all kernels in MIOpen
- Added new Winograd group convolution kernels
- Added an API to query MIOpen's version
- Added parallel compilation in initial convolutional algorithm search; partial solution to [#130](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/130)
- Added SQLite binary program cache
- Improved logging across all layers
- Improved MIOpen's internal design for calling convolutional solvers
- Fixed various bugs for the implicitGEMM algorithm


### 01/24/2020 [ 2.2.1 ]

- This release contains bug fixes, documentation updates, and further code object version 3 support


Changes:

- Added support for multiple ROCm installations
- Added additional support for code object v3
- Fixed issue with incorrect LRN calculation [#127](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/127)
- Fixed incorrect performance database documentation
- Fixed issue with incorrect workspace calculation in group convolutions
- Fixed issue with unsupported hardware instructions used with inline assembly


### 12/19/2019 [ 2.2.0 ]

- This release contains bug fixes, performance improvements, and expanded applicability for specific convolutional algorithms.
- MIOpen has posted a citable paper on ArXiv [here](https://arxiv.org/abs/1910.00078).
- An SQLite database has been added to replace the text-based performance database. While the text file still exists, by default SQLite is used over the text-based performance database; see [documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/perfdatabase.html) from more details. 


Changes:

- Added per solution algorithm filtering environmental variable for debugging
- Added SQLite3 database and build dependency. The text-based performance database support is deprecated and will be removed in the next release.
- Added citation page to documentation pointing to [MIOpen's paper](https://arxiv.org/abs/1910.00078)
- Added to the overall documentation
- Fixed fusion compilation check issue
- Fixed fusion group convolution warning
- Improved performance of forward pooling
- Improved performance of convolutions
- Improved performance of spatial training batch normalization for some large batch size input configurations
- Improved applicability of implicit GEMM convolution algorithm
- Improved performance of calls to miopenConvolutionXXXGetWorkSpaceSize() functions
- Improved conformance to code object version 3
- Removed SCGEMM convolution algorithm by default; this algorithm is deprecated and will be removed in future releases
- Changed "hip_hcc" to "hip-hcc" for the MIOpen package requirements in CMakeLists.txt


### 09/25/2019 [ 2.1.0 ]

- This release contains new layers, bug fixes, and a new convolution algorithm.

Changes:

- Added a dropout layer API for training
- Added a new SCGEMM algorithm for convolutions
- Added further support for bfp16 in convolutions
- Added a [docker hub link](https://hub.docker.com/r/rocm/miopen/tags) for MIOpen docker images.
- Fixed issue with NaN appearing on batch normalization backwards pass in fp16
- Fixed softmax kernel bug in log mode [#112](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/112)
- Fixed ROCm gfx803 support issue [#869](https://github.com/RadeonOpenCompute/ROCm/issues/869)
- Improved performance of batch normalization fp16 forward training layers
- Improved performance of convolutions layers
- Removed MIOpenGEMM as a requirement for the HIP backend. It is now optional.



### 08/13/2019 [ 2.0.1 ]

- This release contains bug fixes and performance improvements.
- Additionally, the convolution algorithm Implicit GEMM is now enabled by default
- Known issues: 
    - Backward propagation for batch normalization in fp16 mode may trigger NaN in some cases
    - Softmax Log mode may produce an incorrect result in back propagation

Changes:

- Added Winograd multi-pass convolution kernel
- Fixed issue with hip compiler paths
- Fixed immediate mode behavior with auto-tuning environment variable
- Fixed issue with system find-db in-memory cache, the fix enable the cache by default
- Improved logging
- Improved how symbols are hidden in the library
- Updated default behavior to enable implicit GEMM



### 07/08/2019 [ 2.0.0 ]

- This release contains several new features including an immediate mode for selecting convolutions, bfloat16 support, new layers, modes, and algorithms.
- MIOpenDriver, a tool for benchmarking and developing kernels is now shipped with MIOpen.
- BFloat16 now supported in HIP requires an updated rocBLAS as a GEMM backend.
- Immediate mode API now provides the ability to quickly obtain a convolution kernel. 
- MIOpen now contains HIP source kernels and implements the ImplicitGEMM kernels. This is a new feature and is currently disabled by default. Use the environmental variable "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1" to activation this feature. ImplicitGEMM requires an up to date HIP version of at least 1.5.9211.
- A new "loss" catagory of layers has been added, of which, CTC loss is the first. See the API reference for more details.
- 2.0 is the last release of active support for gfx803 architectures. In future releases, MIOpen will not actively debug and develop new features specifically for gfx803.
- System Find-Db in memory cache is disabled by default. Please see build instructions to enable this feature.


Changes:

- Added support for bfloat16 datatype in convolutions
- Added softmax channel mode and new softmax version 2 API
- Added fast / accurate / log softmax algorithms 
- Added new implicit GEMM convolution algorithm for forward and backwards data passes, disabled by default
- Added int32 datatype support for output tensors in int8 convolutions
- Added immediate mode for finding the best convolution kernel for a given configuration
- Added a Find-Db infrastructure which stashes results of find on a user's system
- Added a shipped System Find-Db containing offline run Find() results
- Added an additional, faster batch norm assembly kernel for fp16
- Added CTC loss layer
- Added MIOpenDriver as a default component in MIOpen's build [#34](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/34)
- Fixed C compatability for boolean types in C API [#103](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/103)
- Fixed incorrect calculation in per-activation batch norm backwards pass [#104](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/104)
- Fixed bug [#95](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/95) with asm batch norm ISA 
- Fixed IsApplicable bug in Conv3x3Asm for group convolutions
- Improved performance of 1x1 stride 2 fp32 convolutions in the forward and backwards data passes
- Improved 3-D convolution stability
- Improved applicability of direct convolution backwards weights for 2x2, 5x10, and 5x20 filter sizes
- Improved maintainability in kernels and cpp code
- Updated rocBLAS minimum version to branch [master-rocm-2.6](https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/master-rocm-2.6)


### 05/03/2019 [ 1.8.1 ]

- This release contains minor bug fixes and additional performance database improvements.

Changes:

- Fixed accuracy issue with backwards weights
- Fixed issue with name parsing for newer architectures
- Added narrow workaround for 5x10 and 5x20 filter performance regression
- Improved support in performance database for Radeon VII


### 04/11/2019 [ 1.8.0 ]

- This release contaings full 3-D convolution support and int8 support for interfence. 
- Additionally, there are major updates in the performance database for major models including those found in Torchvision. 
- This release contains full 3-D convolution support and int8 support for inference. 
- Additionally, there are updates in the performance database for major models including those found in Torchvision. 
- An assortment of bugs have been resolved in this release.


Changes:
- Fixed various issues in assembly kernels
- Fixed issue #92 and #79 for miopenOpTensor
- Fixed issue #88 for bzip2
- Fixed issue #77 algorithm mismatch
- Added Winograd suport for fp32 backwards weights
- Added Winograd support for fp32 backwards weights
- Added pooling inclusive mode
- Added tuning for direct group convolution algorithms
- Added additional kernel supoort for group convolutions
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
 
