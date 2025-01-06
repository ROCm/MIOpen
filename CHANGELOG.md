
# Change Log for MIOpen

Full documentation for MIOpen is available [here](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
## MIOpen 3.3.0 for ROCm 6.3.0
### Added

* [RNN] LSTM fwd
* [Mha] Mask is added for Forward pass 
* [GLU] Gated Linear Unit (this is an experimental feature)
* [PReLU] Implemented PReLU bwd (this is an experimental feature)
### Optimized

- MI300 TunaNet Update: CK FWD and WRW Solvers Updated 
### Resolved issues

- Fixed unset stream when calling `hipMemsetAsync`
- Fixed a memory leak issue caused by an incorrect transpose in find 2.0 (see PR #3285 on GitHub)
- Fixed a `memcopy` data race by replacing `hipMemcpy` with `hipMemcpyWithStream`


## MIOpen 3.2.0 for ROCm 6.2.0
### Added
- [Conv] bilinear (alpha beta) solvers
- [Conv] enable bf16 for ck-based solvers
- [Conv] Add split_k tuning to 2d wrw ck-based solver
- [MHA] graph API fp8 fwd
- [RNN] multi-stream as default solution.
- TunaNetv2.0 for MI300
- Added adam and amp adam optimizer
### Fixed
- Memory access fault caused by GemmBwdRest
- Context configuration in GetWorkSpaceSize
- Fixes to support huge tensors

### Performance
- Find: Improve precision of benchmarking

## MIOpen 3.1.0 for ROCm 6.1.0
### Added
- CK-based 2d/3d convolution solvers to support nchw/ncdhw layout
- Fused solver for Fwd Convolution with Residual, Bias and activation
- AI Based Parameter Prediction Model for conv_hip_igemm_group_fwd_xdlops Solver
- Forward, backward data and backward weight convolution solver with fp8/bfp8
- check for packed tensors for convolution solvers
- Integrate CK's layer norm
- Combine gtests into single binary
### Fixed
- fix for backward passes bwd/wrw for CK group conv 3d
- Fixed out-of-bounds memory access : ConvOclDirectFwdGen
- fixed build failure due to hipRTC
### Changed
- Standardize workspace abstraction
- Use split CK libraries
### Removed
- clamping to MAX from CastTensor used in Bwd and WrW convolution

## MIOpen 3.0.0 for ROCm 6.0.0
- This release adds 3D convolution, enablement of fp8 convolution, NHWC batch norm, RNN padding support. It also removes
INI8x4 support and fix minor issues and bugs.
### Notes
### Added
- 3D forward convolution solver with non-packed input tensors
- 3D group backward weight convolution solver
- 3D group backward data convolution solver
- 3D group forward convolution solver
- winograd fury convolution support
- NHWC Batchnorm support
- RNN padding and SeqTensorDescriptor
- FP8 and BFP8 convolution enablement
### Fixed
- transposed convolutions
- issue with parameter truncation for CK solvers
### Changed
- Ck kernels invocation refactoring
- Replace miopen::ProblemDescription with conv::ProblemDescription
### Removed
- Remove INT8x4 support
- Remove target ids from kdb args

## MIOpen 2.21.0 for ROCm 5.7.0
### Added
- AI Heuristic for Immediate Mode Fallback
- CK group forward convolution integration
- additional tunings into performance database
### Fixed
- [HOTFIX] Workaround for HIP iGEMM in buffer_load_max_length
### Changed
- Update fdb data to use solver keys [MI100][MI200]

## MIOpen 2.20.0 for ROCm 5.6.0
### Added
- AI Based Heuristic for Kernel Parameter Prediction
- LSTM multi-stream solver
### Fixed
- Tuning fails for ResNet50
- FP16 precision issues in pooling
- Winograd kernel failure
- Perf DB updates for gfx908 and gfx90a

## MIOpen 2.19.0 for ROCm 5.5.0
### Added
- ROCm 5.5 support for gfx1101 (Navi32)
### Changed
- Tuning results for MLIR on ROCm 5.5
- Bumping MLIR commit to 5.5.0 release tag
### Fixed
- Fix 3d convolution Host API bug
- [HOTFIX][MI200][FP16] Disabled ConvHipImplicitGemmBwdXdlops when FP16_ALT is required. 

## MIOpen 2.16.0 for ROCm 5.1.1
### Notes
- This release includes enhanced support for MI210 and MI250 along with various other improvements.
- This release consists of various bug fixes and performance improvements
### Added
- Improved support for Navi21
- Performance improvements via performance database updates
### Fixed
- various issues in convolution kernels specific to certain ASICs
- an accuracy issue in reduction kernels
- an accuracy issue in Batchnormalization kernels

## MIOpen 2.14.0 for ROCm 4.5.2
### Notes
- This release consists of various bug fixes and performance improvements
### Added
- Improved support for Navi21
- Performance improvements via performance database updates
### Fixed
- various issues in convolution kernels specific to certain ASICs
- an accuracy issue in reduction kernels
- an accuracy issue in Batchnormalization kernels

## MIOpen 2.12.0 for ROCm 4.3.1
### Notes
- This release includes support for Navi21 and various other bug fixes and performance improvements
### Added
- MIOpen now supports Navi21!! (via MIOpen PRs 973, 780, 764, 740, 739, 677, 660, 653, 493, 498)
- Updated the performance data for new kernel versions
- Improved MIOpen build time by splitting large kernel header files
### Fixed
- a correctness issue with ImplicitGemm algorithm 
- an issue in reduction kernels for padded tensors
- Various other bug fixes and performance improvements

## MIOpen 2.11.0 for ROCm 4.2.0
- This release contains various bug fixes and performance improvements.
### Added
- Updates for Target ID features in ROCm stack
### Fixed 
- Various bug for MIOpenGEMM on the OpenCL backend
- Various bug in 3x3 assembly kernels
- Correctness fix in Batchnorm kernels

## MIOpen 2.10.0 for ROCm 4.1.0
### Notes
- This release contains new reduction operations, Winograd algorithm performance improvements as well as bug fixes. Various host side performance improvements have been added as well.
### Added
- a GPU reference kernel implementation for faster testing.
- TargetID support for new AMD GPU architectures.
- Implementation of four additional generic tensor reduction operations (AVG, AMAX, NORM1, NORM2).
- support for AMD Code Object V4.
- various host side improvements for better find and tuning performance.
### Fixed
- a bug where Batchnorm would give incorrect results when the product of image height and image width is not a factor of four.

## MIOpen 2.9.0 for ROCm 4.0.0
### Notes 
- This release contains implicit GEMM algorithm performance updates and bug fixes. Additional performance improvements have been implemented for batch normalization.
### Added
- new assembly implicit GEMM kernels
- batch normalization optimizations
- missing tunings from 2.8.0 release cycle
### Fixed
- issue where miopen-hip backend install would not search for rocBLAS dependency 
### Removed
- deprecated implicit GEMM xDLOPs solvers
- incorrect error messages from implicit GEMM solvers
- Disabled ConvAsmBwdWrW3x3 solver for stride > 1 cases
- Disabled bidirectional multi-pass Winograd kernels due to stability issues

## MIOpen 2.8.0 for ROCm 3.9.0
### Notes
- This release provides additional bug fixes and support for embedded build using MIOpen as a static library. 
### Added
- cmake flag for embedding system databases when building a static library
- a way to disable building MIOpenDriver when building a static library
- CC compiler detection in ROCm environment
### Fixed
- workspace size calculation for GEMM group convolutions
- performance regression for M/N
- issue with faulty compiler option
- typo in components dependency variable in CMakeLists.txt
- issues with COMgr backed online compilation for HIP kernels
### Known issues
- This release may show warnings for "obsolete configs" in the performance database. This can be fixed by rerunning tuning on a specfic network; [see tuning documentation](https://ROCm.github.io/MIOpen/doc/html/perfdatabase.html#miopen-find-enforce)

## MIOpen 2.7.0 for ROCm 3.8.0
- This release contains a new reduction API; see [API documentation](https://ROCm.github.io/MIOpen/doc/html/apireference.html) for more information. Additional features for embedded builds have been added, and further support for 3D convolutional networks. 
### Added
- additional tunings into performance database
- general reduction API
- cmake flag for embedding binary database into a static MIOpen build
- cmake flag for embedding system find-db text files into static MIOpen build
### Fixed
- issue with GEMM workspace size calculation for backwards data convolutions [#381](https://github.com/ROCm/MIOpen/issues/381)
- issue with 3D pooling indexing [#365](https://github.com/ROCm/MIOpen/issues/365)

## MIOpen 2.6.0 for ROCm 3.7.0
### Notes
- This release contains convolution performance improvements, improved multi-threading behavior, and improved stability for half precision convolutions. Initial iteration time has been reduced with the introduction of hybrid find mode. Builds for a static library have been refined for this release.
### Added
- MIOPEN_FIND_MODE=3 as the new default convolution Find mode; see documentation [here](https://ROCm.github.io/MIOpen/doc/html/find_and_immediate.html#find-modes) for details
- a more runtime-parameterized version of pooling to reduce the number of online compilations
- Improved the performance of backwards spatial batch normalization for small images
### Fixed
- issue with std::logic_error in SQLite deleter [#306](https://github.com/ROCm/MIOpen/issues/306)
- issues with half precision stability for convolutions
- issues with multi-threaded SQLite database accesses
- issues with 3-D convolutions and incorrect parameters
- various issues with implicit GEMM static assert failures
### Removed
- inactive implicit GEMM convolution solvers
- SCGEMM convolutional algorithm from MIOpen

## MIOpen 2.5.0 for ROCm 3.5.0
### Notes
- This release contains convolution performance improvements, various minor fixes and documentation updates.
### Added
- a script to detect and install appropriate precompiled kernels
- 3D convolution backwards weights implicit GEMM implementation 
- Improve performance of convolution implicit GEMM algorithm
- Improved database coverage for batch size 1
- Improved logging and error reporting
- Improved documentation for debugging with numeric checks
### Fixed
- issue with potential infinities and NaNs appearing during low precision training on CNNs

## MIOpen 2.4.0 for ROCm 3.5.0
### Notes
- This release contains new implementations of 3D convolutions using implicitGEMM, general performance improvements for convolutions, bug fixes, better versioning in directories, integration with the new rocclr, and dropout support in RNNs.
### Added
- 3D convolutions for the implicitGEMM algorithm in the forward and backward-data passes
- dropout support for RNN layer; e.g., RNN-vanilla, GRU, and LSTM
- support for AMD's rocclr runtime and compiler
- Improved performance for implicitGEMM and Winograd algorithms
- Improved database locking
### Fixed
- issue with GPU memory segmentation fault on asymmetric padding [#142](https://github.com/ROCm/MIOpen/issues/142)

## MIOpen 2.3.0 for ROCm 3.1.0
### Notes
- This release contains new implementations of the implicitGEMM and Winograd algorithms, performance improvements for convolutions, further support for 3D convolutional networks, and various bug fixes.
### Added
- 3D Pooling layers
- backwards data algorithm for implicitGEMM
- GEMM performance improvements via relaxed constraints in rocBLAS-Tensile
- full CO v3 support for all kernels in MIOpen
- new Winograd group convolution kernels
- an API to query MIOpen's version
- parallel compilation in initial convolutional algorithm search; partial solution to [#130](https://github.com/ROCm/MIOpen/issues/130)
- SQLite binary program cache
- Improved logging across all layers
- Improved MIOpen's internal design for calling convolutional solvers
### Fixed
- various bugs for the implicitGEMM algorithm

## MIOpen 2.2.1 for ROCm 3.1.0
### Notes
- This release contains bug fixes, documentation updates, and further code object version 3 support
### Added
- support for multiple ROCm installations
- additional support for code object v3
### Fixed
- issue with incorrect LRN calculation [#127](https://github.com/ROCm/MIOpen/issues/127)
- incorrect performance database documentation
- issue with incorrect workspace calculation in group convolutions
- issue with unsupported hardware instructions used with inline assembly

## MIOpen 2.2.0 for ROCm 3.0.0
### Notes
- This release contains bug fixes, performance improvements, and expanded applicability for specific convolutional algorithms.
- MIOpen has posted a citable paper on ArXiv [here](https://arxiv.org/abs/1910.00078).
- An SQLite database has been added to replace the text-based performance database. While the text file still exists, by default SQLite is used over the text-based performance database; see [documentation](https://ROCm.github.io/MIOpen/doc/html/perfdatabase.html) from more details. 
### Added
- per solution algorithm filtering environmental variable for debugging
- SQLite3 database and build dependency. The text-based performance database support is deprecated and will be removed in the next release.
- citation page to documentation pointing to [MIOpen's paper](https://arxiv.org/abs/1910.00078) overall documentation
- Improved applicability of implicit GEMM convolution algorithm
- Improved performance of calls to miopenConvolutionXXXGetWorkSpaceSize() functions
- Improved conformance to code object version 3
- Improved performance of forward pooling
- Improved performance of convolutions
- Improved performance of spatial training batch normalization for some large batch size input configurations
### Changed
- "hip_hcc" to "hip-hcc" for the MIOpen package requirements in CMakeLists.txt
### Removed
- SCGEMM convolution algorithm by default; this algorithm is deprecated and will be removed in future releases
### Fixed
- fusion compilation check issue
- fusion group convolution warning

## MIOpen 2.1.0 for ROCm 2.10.0
### Notes
- This release contains new layers, bug fixes, and a new convolution algorithm.
### Added
- a dropout layer API for training
- a new SCGEMM algorithm for convolutions
- further support for bfp16 in convolutions
- a [docker hub link](https://hub.docker.com/r/rocm/miopen/tags) for MIOpen docker images.
- Improved performance of batch normalization fp16 forward training layers
- Improved performance of convolutions layers
### Fixed
- issue with NaN appearing on batch normalization backwards pass in fp16
- softmax kernel bug in log mode [#112](https://github.com/ROCm/MIOpen/issues/112)
- ROCm gfx803 support issue [#869](https://github.com/RadeonOpenCompute/ROCm/issues/869)
### Removed
- MIOpenGEMM as a requirement for the HIP backend. It is now optional.

## MIOpen 2.0.1 for ROCm 2.7.0
### Notes
- This release contains bug fixes and performance improvements.
- Additionally, the convolution algorithm Implicit GEMM is now enabled by default
### Added
- Winograd multi-pass convolution kernel
- Improved logging
- Improved how symbols are hidden in the library
### Fixed
- issue with hip compiler paths
- immediate mode behavior with auto-tuning environment variable
- issue with system find-db in-memory cache, the fix enable the cache by default
### Changed
- Updated default behavior to enable implicit GEMM
### Known issues
- Backward propagation for batch normalization in fp16 mode may trigger NaN in some cases
- Softmax Log mode may produce an incorrect result in back propagation

## MIOpen 2.0.0 for ROCm 2.6.0
### Notes
- This release contains several new features including an immediate mode for selecting convolutions, bfloat16 support, new layers, modes, and algorithms.
- MIOpenDriver, a tool for benchmarking and developing kernels is now shipped with MIOpen.
- BFloat16 now supported in HIP requires an updated rocBLAS as a GEMM backend.
- Immediate mode API now provides the ability to quickly obtain a convolution kernel. 
- MIOpen now contains HIP source kernels and implements the ImplicitGEMM kernels. This is a new feature and is currently disabled by default. Use the environmental variable "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1" to activation this feature. ImplicitGEMM requires an up to date HIP version of at least 1.5.9211.
- A new "loss" catagory of layers has been added, of which, CTC loss is the first. See the API reference for more details.
- 2.0 is the last release of active support for gfx803 architectures. In future releases, MIOpen will not actively debug and develop new features specifically for gfx803.
- System Find-Db in memory cache is disabled by default. Please see build instructions to enable this feature.
### Added
- support for bfloat16 datatype in convolutions
- softmax channel mode and new softmax version 2 API
- fast / accurate / log softmax algorithms 
- new implicit GEMM convolution algorithm for forward and backwards data passes, disabled by default
- int32 datatype support for output tensors in int8 convolutions
- immediate mode for finding the best convolution kernel for a given configuration
- a Find-Db infrastructure which stashes results of find on a user's system
- a shipped System Find-Db containing offline run Find() results
- an additional, faster batch norm assembly kernel for fp16
- CTC loss layer
- MIOpenDriver as a default component in MIOpen's build [#34](https://github.com/ROCm/MIOpen/issues/34)
- Improved performance of 1x1 stride 2 fp32 convolutions in the forward and backwards data passes
- Improved 3-D convolution stability
- Improved applicability of direct convolution backwards weights for 2x2, 5x10, and 5x20 filter sizes
- Improved maintainability in kernels and cpp code
### Changed
- Updated rocBLAS minimum version to branch [master-rocm-2.6](https://github.com/ROCm/rocBLAS/tree/master-rocm-2.6)
### Fixed
- C compatability for boolean types in C API [#103](https://github.com/ROCm/MIOpen/issues/103)
- incorrect calculation in per-activation batch norm backwards pass [#104](https://github.com/ROCm/MIOpen/issues/104)
- bug [#95](https://github.com/ROCm/MIOpen/issues/95) with asm batch norm ISA 
- IsApplicable bug in Conv3x3Asm for group convolutions

## MIOpen 1.8.1 for ROCm 2.5.0
### Notes
- This release contains minor bug fixes and additional performance database improvements.
### Added
- workaround for 5x10 and 5x20 filter performance regression
- Improved support in performance database for Radeon VII
### Fixed
- accuracy issue with backwards weights
- issue with name parsing for newer architectures

## MIOpen 1.8.0 for ROCm 2.3.0
### Notes
- This release contaings full 3-D convolution support and int8 support for interfence. 
- Additionally, there are major updates in the performance database for major models including those found in Torchvision. 
- This release contains full 3-D convolution support and int8 support for inference. 
- Additionally, there are updates in the performance database for major models including those found in Torchvision. 
- An assortment of bugs have been resolved in this release.
### Added
- Winograd suport for fp32 backwards weights
- Winograd support for fp32 backwards weights
- pooling inclusive mode
- tuning for direct group convolution algorithms
- additional kernel supoort for group convolutions
- additional kernel support for group convolutions
- API for 3-D convolutions
- support for int8 inference convolutions
- integer selection for pooling indexing
- minimum dependencies support
- RNN fp16 support on the MIOpen-HIP backend
- 1x1 convolution + bias + activation fusions
- workaround for issue #84 GPU memory access fault
- performance tuning for direct backwards weights
- Improved performance database coverage
- Improved internal quality by reducing redunant code
- Improved build instructions in README.md
- Improved performance database coverage for fusions
### Changed
- Updated Docker components and requirements
### Fixed
- various issues in assembly kernels
- issue #92 and #79 for miopenOpTensor
- issue #88 for bzip2
- issue #77 algorithm mismatch
### Known Issues
- RNNs do not support fp16 on the MIOpen-OpenCL backend
- OpenCL backend does not support GEMM convolutions in fp16

## MIOpen 1.7.1 for ROCm 2.3.0
### Notes
- This release contains minor bug fixes and performance improvements.
### Added
- a workaround for softmax fp16 correctness issue
- check to only make MIOpen with static boost libraries
- Improved performance database coverage
### Fixed
- corrupt and obsolete performance database entries
- issue #70, "SIGFPE (DIV/0) in ConvOclBwdWrW2::GetSolution()"
- issue #72, "workSpaceSize check assertion fails in ConvolutionBackwardWeights() - DEBUG builds only"
- issue #77, "Results of ConvBwdWeightsAlgoDirect and ConvBwdWeightsAlgoGEMM mismatch for some specific parameters"
### Removed
- default dependency of RNNs on rocBLAS
### Known Issues
- RNNs do not support fp16
- OpenCL backend does not support GEMM convolutions in fp16
- Layer fusions for convolution 1x1 fp16 are not supported
- Layer fusions for large image 1x1 convolutions may cause an exception instead of a warning during compile phase if plan is not supported

## MIOpen 1.7.0 for ROCm ROCm 2.1.0
### Notes
- This release contains general bug fixes and an updated performance database
- Group convolutions backwards weights performance has been improved
- Logging across the library has been improved
- Performance database has been updated
### Added
- support for large image backwards weights in direct convolution
- fp16 support for RNNs on the HIP backend
- Improved performance database coverage
### Fixed
- logging issues with group convolution and pooling
- sphinx version issue in document generation
- issues with corrupt entries in performance database
### Removed
- external dependency on libSSL and libCrypto
### Known issues
- RNNs do not support fp16
- OpenCL backend does not support GEMM convolutions in fp16
- Layer fusions for convolution 1x1 fp16 are not supported
- Layer fusions for large image 1x1 convolutions may cause an exception instead of a warning during compile phase if plan is not supported

## MIOpen 1.6.0 for ROCm 1.0.0
### Added
- support fp16 Winograd convolutions
- support for fp16 pooling
- Training in fp16 (half precision) including mixed-precision
- Batch Normalization in fp16 (half precision) including mixed-precision
- Layer fusions for BatchNorm+Activation
- Layer fusions with convolutions now support varying strides and padding configurations
- Improved error reporting for convolutions and layer fusions
- Improved documentation
### Changed
- rocBLAS is now used as the default BLAS library for the HIP backend (minimum version 14.3.0)
### Fixed
- various bugs in convolution kernels
- issues with bad references in layer fusion 
- gfx803 assembily issues
### Known issues
- RNNs do not support fp16
- OpenCL backend does not have full fp16 support
- Layer fusions for convolution 1x1 fp16 are not supported

## MIOpen 1.5.0 for ROCm 1.0.0
### Added
- A new kernel fusion API is now available for inference for convolution, bias, batch normalization, and activations.
- fused kernels for Winograd and direct with bias and activations
- Getting started guide for kernel fusion.
- Group and depthwise API for convolutions
- 3-D batch normalization support with 5-D tensors
- new features and bug fixes
- Group and Depthwise convolutions are now available
- 3D Batch Normalization has been implemented for fully packed tensors
- Dilation for convolutions have been implemented
- incremental support for fp16
- debug and error reporting information
- documentation for convolutions
- improved max pooling performance
### Fixed
- bugs in direct convolutions
- issue with paths when $HOME variable is not set
- padding issues with 1x1 convolutions

### Known issues
- RNNs do not support fp16
- Training with CNNs does not support fp16
 
## MIOpen 1.4.2 for ROCm 1.0.0
### Fixed
- This release is a hot-fix to enable ICNet and PSPNet
### Known issues
- RNNs do not support fp16
- Training with CNNs does not support fp16
- Users may encounter a warning that their performance database is out of date. The performance database can be updated by setting the environment variable for just the initial run of an application: `MIOPEN_FIND_ENFORCE=search`. For more information on the performance database, see: https://ROCm.github.io/MIOpen/doc/html/perfdatabase.html#

## MIOpen 1.4.1 for ROCm 1.0.0
### Added: 
- This release includes a bug fix for 3x3 convolutions Changed 
- Updated README file configuration instructions
### Known issues
- RNNs do not support fp16
- Training with CNNs does not support fp16
- Users may encounter a warning that their performance database is out of date. The performance database can be updated by setting the environment variable for just the initial run of an application: `MIOPEN_FIND_ENFORCE=search`. For more information on the performance database, see: https://ROCm.github.io/MIOpen/doc/html/perfdatabase.html#

## MIOpen 1.4.0 for ROCm 1.0.0
### Notes:
- This release includes a number of performance improvements and bug fixes
- New features have been added to convolutions for auto-tuning kernels
- Activations now have new modes available
- Documentation has been updated and corrected
### Added
- to performance database functionality
- leaky-ReLU, clipped, and exponential-ReLU modes to activation
- documentation for performance database usage
- support for 1x1 convolutions with non-zero padding
- API for printing status codes as strings
- auto-tuning feature for convolutions
- LSTM and GRU backwards pass performance
- debug and error reporting information
- performance of batch normalization spatial mode
- find stage for convolutions
- readability for user database file
### Fixed 
- documentation errors
- bug in activations with pass-through mode
- performance database locking issues
- Winograd kernel behavior for stride 2 backwards data
- a bug in OpTensor layer
- a timing issue with batch normalization inline assembly 
- issue with an unnecessary binary creation in assembly bug detection
- issue with disk program cache directory not being created
- a bug with convolution+bias
### Known issues
- RNNs do not support fp16
- Training with CNNs does not support fp16

## MIOpen 1.3.0 for ROCm 1.0.0
### Added
- This release adds preliminary fp16 support for Inference using CNNs
- 2 new API for RNNs: miopenGetRNNLayerParamOffset and miopenGetRNNLayerBiasOffset
- support for uninitialized hidden states and nullptr outputs in RNNs
- support for Set and Scale operations for strided tensors with dimensions 1 to 5
- multi-thread and multi-process support for the performance database
- performance improvements for RNNs
- performance improvements for convolutions using 1x1 filters
- performance improvement for Batch Normalization
- performance Improved performance for OpTensor
### Fixed
- bug in convolutions for backward bias
- logic issues in get and set layer functions and related w_supertensor test
- hang in batch norm with batch sizes greater than 256
- Bug fixes for various components of MIOpen
### Known Issues
- RNNs do not support fp16
- Training with CNNs does not support fp16

## MIOpen 1.2.1 for ROCm 1.0.0
### Added
- adds support for ROCm 1.7.1.

## MIOpen 1.2.0 for ROCm 1.0.0
### Notes
- This release adds the support for recurrent neural networks (RNNs) for three flavors - Vanilla, LSTMs, and GRU
- Users can now themselves update the perf-db file, which hosts the tuning parameters for convolutions, by setting appropriate environment variables
### Added
- Multiple padding modes like Same and Valid added
- Winograd convolution kernels added for strided bwd-data convolutions
- Tensor Ops allow for beta and alpha scaling values and support up to 5 dimensions with strides and offsets
- Tensor Copy supports up to 5 dimesnional copies with strides and offsets
- Unit-tests for LRN are added
- Over 50% improvement in ResNet performance since the last release
### Fixed
- Several bug fixes for all the layers of the library
### Known issues
- RNNs may give incorrect result due to a known compiler bug; issue may particulary arise during some RNNs configs with GEMM of size power of 4
- Potential issue where OpenCL resources will be exhausted for large RNN

## MIOpen 1.1.0 for ROCm 1.0.0
### Notes 
- The scaling parameter alpha and shift parameter beta for layers kernels are only supported for alpha = 1 and beta = 0.
- The exceptions to this are for miopenOptTensor, miopenConvolutionForwardBias, and miopenConvolutionBackwardBias.
- Currently, only 32-bit floats are supported in MIOpen.
- MIOpen only supports tensor layout NCHW.
### Added
- persistent cache for compiled GPU kernels
- performance improvements for batch normalization kernels
- performance improvements for all types of convolutions for 1x1 filters
- performance improvements for all types of convolutions with non-unit strides
- performance improvements for backward-weights convolutions for 3x3 filters
- performance improvements for the AddTensor operation
### Fixed
- Various bug fixes for Winograd convolutions 

## MIOpen 1.0.2 for ROCm 1.0.0
### Fixed
- 1x1 forward and backward convolutions for large input
- pooling MIOpendriver
### Disabled
- 1x1 Winograd convolution for HIP
- asm. backward-weights convolutions for input width == 175 

## MIOpen 1.0.1 for ROCm 1.0.0
### Added
- dilation support for convolutions 
- unit-tests for Softmax
- miopengemm as a required dependency for MIOpen build
- Performance improvements for batch normalization via activation of data-parallel primitives (DPP) hardware instructions
### Fixed
- documentation to remove GEMM API interface
- Bwd-Weights Convolutions with 1x1 filters with stride=2
- Softmax grid-size selection
- debug prints of kernel launch parameters.
### Removed
- GEMM interface from the MIOpen API

## MIOpen 1.0.0 for ROCm 1.0.0
### Added
- Initial release 

