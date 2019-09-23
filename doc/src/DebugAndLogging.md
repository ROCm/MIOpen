Debugging and Logging
=====================

## Logging

All logging messages output to standard error stream (`stderr`). The following environment variables can be used to control logging:

* `MIOPEN_ENABLE_LOGGING` - Enables printing the basic layer by layer MIOpen API call information with actual parameters (configurations). Important for debugging. Disabled by default.

* `MIOPEN_ENABLE_LOGGING_CMD` - A user can use this environmental variable to output the associated `MIOpenDriver` command line(s) onto console. Disabled by default.

> **_NOTE:_ These two and other two-state ("boolean") environment variables can be set to the following values:**
> ```
> 1, yes, true, enable, enabled - to enable feature
> 0, no, false, disable, disabled - to disable feature
> ```

* `MIOPEN_LOG_LEVEL` - In addition to API call information and driver commands, MIOpen prints various information related to the progress of its internal operations. This information can be useful both for debugging and for understanding the principles of operation of the library. The `MIOPEN_LOG_LEVEL` environment variable controls the verbosity of these messages. Allowed values are:
  * 0 - Default. Works as level 4 for Release builds, level 5 for Debug builds.
  * 1 - Quiet. No logging messages.
  * 2 - Fatal errors only (not used yet).
  * 3 - Errors and fatals.
  * 4 - All errors and warnings.
  * 5 - Info. All the above plus information for debugging purposes.
  * 6 - Detailed info. All the above plus more detailed information for debugging.
  * 7 - Trace: the most detailed debugging info plus all above.

> **_NOTE:_ When asking for technical support, please include the console log obtained with the following settings:**
> ```
> export MIOPEN_ENABLE_LOGGING=1
> export MIOPEN_ENABLE_LOGGING_CMD=1
> export MIOPEN_LOG_LEVEL=6
> ```

* `MIOPEN_ENABLE_LOGGING_MPMT` - When enabled, each log line is prefixed with information which allows the user to identify records printed from different processes and/or threads. Useful for debugging multi-process/multi-threaded apps.

* `MIOPEN_ENABLE_LOGGING_ELAPSED_TIME` - Adds a timestamp to each log line. Indicates the time elapsed since the previous log message, in milliseconds.

## Layer Filtering

The following list of environment variables allow for enabling/disabling various kinds of kernels and algorithms. This can be helpful for both debugging MIOpen and integration with frameworks.

> **_NOTE:_ These variables can be set to the following values:**
> ```
> 1, yes, true, enable, enabled - to enable kernels/algorithm
> 0, no, false, disable, disabled - to disable kernels/algorithm
> ```

If a variable is not set, then MIOpen behaves as if it is set to `enabled`, unless otherwise specified. So all kinds of kernels/algorithms are enabled by default and the below variables can be used for disabling them.

### Filtering by algorithm

These variables control the sets (families) of convolution Solvers. For example, Direct algorithm is implemented by several Solvers written in openCL and assembly languages. The corresponding variable can disable them all.
* `MIOPEN_DEBUG_CONV_FFT` - FFT convolution algorithm. 
* `MIOPEN_DEBUG_CONV_DIRECT` - Direct convolution algorithm.
* `MIOPEN_DEBUG_CONV_GEMM` - GEMM convolution algorithm.
* `MIOPEN_DEBUG_CONV_WINOGRAD` - Winograd convolution algorithm.
* `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM` – Implicit GEMM convolution algorithm.
* `MIOPEN_DEBUG_CONV_SCGEMM` – Statically Compiled GEMM convolution algorithm.

### Filtering by build method

* `MIOPEN_DEBUG_GCN_ASM_KERNELS` - Kernels written in assembly language. Currently these include some Direct solvers, Winograd kernels and SCGEMM.
* `MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES` - Binary kernels. Right now the library does not use binaries.

### Controlling the Solvers on individual basis

Some of the available Solvers have individual controls:
* `MIOPEN_DEBUG_AMD_WINOGRAD_3X3` - FP32 Winograd Fwd/Bwd, filter size fixed to 3x3. Solver name: `ConvBinWinograd3x3U`.
* `MIOPEN_DEBUG_AMD_WINOGRAD_RXS` - FP32 and FP16 Winograd Fwd/Bwd/WrW. Solver name: `ConvBinWinogradRxS`.
* `MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW` - Subset of previous, controls only WrW (backward weights) convolutions of the `ConvBinWinogradRxS` solver.
* `MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2` - FP32 and FP16 Fwd/Bwd F(3,2) Winograd. Solver name: `ConvBinWinogradRxSf3x2`.
* `MIOPEN_DEBUG_AMD_FUSED_WINOGRAD` - Fused FP32 Winograd kernels, variable filter size.

Family of Multi-pass Winograd Solvers:
* `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2`
* `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3`
* `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4`
* `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5`
* `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6`
* `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2`
* `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3`

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

## Experimental controls

> **_NOTE: Using experimental controls may result in:_**
> * Performance drops
> * Computation inaccuracies
> * Run-time errors
> * Other kinds of unexpected behavior
>
> **_It is strongly recommended to use them only with the explicit permission or request of the library developers._**

### Code Object (CO) version selection (EXPERIMENTAL)

currently, ROCm fully supports Code Object version 2 (Co v2). The support for version 3 (CO v3) is being gradually introduced. These variables allows for experimenting and triaging problems related to CO version:
* `MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE` - Overrides CO version auto-detection implemented in the library. `0` or unset - disable overriding (the default), `1` - enforces CO v2, `2` - behave as if both CO v2 and v3 are supported, `2` - enforces CO v3.
* `MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_NEWER` - This variable affects only Solvers that able to produce either v2 or v3 code objects, and is intended to use only when ROCm supports both CO v2 and CO v3. By default, the older format is used (CO v2) by Solvers. When this variable is _enabled_, the behavior is reversed.
* `MIOPEN_DEBUG_AMD_OPENCL_ENFORCE_COV3` - Enforces CO v3 for OpenCL kernels.

### `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX`

Syntax of value:
* decimal or hex (with `0x` prefix) value that should fit into `unsigned long` (64 bits).
* If syntax is violated, then the behavior is unspecified.

Semantics:
* Sets the **_limit_** (max allowed workspace size) for Multi-pass (MP) Winograd Solvers, in bytes.
* Affects all MP Winograd Solvers. If a solver needs more workspace than the limit, then it does not apply.
* If unset, then _the default_ limit is used. Current default is `2000000000` (~1.862 GiB) for gfx900 and gfx906/60 (or less CUs). No default limit is set for other GPUs and MP Winograd Solvers will try to make the most of the workspace.
* Special values:
```
 0 - Use the default limit, as if the variable is unset.
 1 - Completely prohibit the use of workspace.
-1 - Remove the default limit.
```
