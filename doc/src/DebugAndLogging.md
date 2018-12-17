Debugging and Logging
=====================

## Logging

* `MIOPEN_ENABLE_LOGGING` - Enables printing the basic layer by layer MIOpen API call information with actual parameters (configurations). Important for debugging. Disabled by default.

* `MIOPEN_ENABLE_LOGGING_CMD` - A user can use this environmental variable to output the associated `MIOpenDriver` command line(s) onto console. Disabled by default.

> **_NOTE:_ These two and other two-state ("boolean") environment variables can be set to the following values:**
> ```
> 1, yes, true, enable, enabled - to enable feature
> 0, no, false, disable, disabled - to disable feature
> ```

## Log Levels
The `MIOPEN_LOG_LEVEL` environment variable controls the verbosity of the messages printed by MIOpen onto console. Allowed values are:
* 0 - Default. Works as level 4 for Release builds, level 5 for Debug builds.
* 1 - Quiet. No logging messages.
* 2 - Fatal errors only (not used yet).
* 3 - Errors and fatals.
* 4 - All errors and warnings.
* 5 - Info. All the above plus information for debugging purposes.
* 6 - Detailed info. All the above plus more detailed information for debugging.
* 7 - Trace: the most detailed debugging info plus all above (not used so far).

All messages output via `stderr`.

`MIOPEN_ENABLE_LOGGING` and `MIOPEN_LOG_LEVEL` are independent from each other.

> **_NOTE:_ When asking for technical support, please include the console log obtained with the following settings:**
> ```
> export MIOPEN_ENABLE_LOGGING=1
> export MIOPEN_LOG_LEVEL=5
> ```

## Layer Filtering

The following list of environment variables allow for enabling/disabling various kinds of kernels and algorithms. This can be helpful for both debugging MIOpen and integration with frameworks.

> **_NOTE:_ These variables can be set to the following values:**
> ```
> 1, yes, true, enable, enabled - to enable kernels/algorithm
> 0, no, false, disable, disabled - to disable kernels/algorithm
> ```

If a variable is not set, then MIOpen behaves as if it is set to `enabled`, unless otherwise specified. So all kinds of kernels/algorithms are enabled by default and variables can be used for disabling them.

* `MIOPEN_DEBUG_CONV_FFT` – FFT convolution algorithm. 
* `MIOPEN_DEBUG_CONV_DIRECT` – Direct convolution algorithm.
* `MIOPEN_DEBUG_CONV_GEMM` - GEMM convolution algorithm. These are implemented on top of miopengemm or rocBlas.
* `MIOPEN_DEBUG_GCN_ASM_KERNELS` – Kernels written in assembly language. So far, the most of the assembly kernels are implementing the Direct convolution algorithm.
* `MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES` - Binary kernels. Right now all the binary kernels are Winograd ones, however, not all Winograds are binaries. To disable all Winograd algorithms, the following two vars can be used:
* `MIOPEN_DEBUG_AMD_WINOGRAD_3X3` - FP32 Winograd Fwd/Bwd, filter size fixed to 3x3.
* `MIOPEN_DEBUG_AMD_WINOGRAD_RXS` - FP32 and FP16 Winograd Fwd/Bwd, variable filter size.
* `MIOPEN_DEBUG_AMD_FUSED_WINOGRAD` - Fused FP32 Winograd kernels, variable filter size.

## rocBlas Logging and Behavior
The `ROCBLAS_LAYER` environmental variable can be set to output GEMM information:
* `ROCBLAS_LAYER=`  - is not set, there is no logging
* `ROCBLAS_LAYER=1` - is set to 1, then there is trace logging
* `ROCBLAS_LAYER=2` - is set to 2, then there is bench logging
* `ROCBLAS_LAYER=3` - is set to 3, then there is both trace and bench logging

Additionally, using environment variable "MIOPEN_GEMM_ENFORCE_BACKEND", can override the default behavior. The default behavior which is to use
both MIOpenGEMM and rocBlas depending on the input configuration:

* `MIOPEN_GEMM_ENFORCE_BACKEND=1`, use rocBLAS if enabled
* `MIOPEN_GEMM_ENFORCE_BACKEND=2`, use MIOpenGEMM for FP32, use rocBLAS for FP16 if enabled
* `MIOPEN_GEMM_ENFORCE_BACKEND=3`, no gemm will be called
* `MIOPEN_GEMM_ENFORCE_BACKEND=<any other value>`, use default behavior

To disable using rocBlas entirely, set the configuration flag `-DMIOPEN_USE_ROCBLAS=Off` during MIOpen configuration.

More information on logging with RocBlas can be found [here](https://github.com/ROCmSoftwarePlatform/rocBLAS/wiki/5.Logging).
