
## MIOPEN ChangeLog

### 09/08/2017 [ 1.1.0 ]
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
