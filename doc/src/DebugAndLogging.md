Debugging and Logging
=====================

## Logging

All logging messages output to standard error stream (`stderr`). The following environment variables can be used to control logging:

* `MIOPEN_ENABLE_LOGGING` - Enables printing the basic layer by layer MIOpen API call information with actual parameters (configurations). Important for debugging. Disabled by default.

* `MIOPEN_ENABLE_LOGGING_CMD` - A user can use this environmental variable to output the associated `MIOpenDriver` command line(s) onto console. Disabled by default.

> **_NOTE 1:_ These two and other two-state ("boolean") environment variables can be set to the following values:**
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

> **_NOTE 2:_ When asking for technical support, please include the console log obtained with the following settings:**
> ```
> export MIOPEN_ENABLE_LOGGING=1
> export MIOPEN_ENABLE_LOGGING_CMD=1
> export MIOPEN_LOG_LEVEL=6
> ```

* `MIOPEN_ENABLE_LOGGING_MPMT` - When enabled, each log line is prefixed with information which allows the user to identify records printed from different processes and/or threads. Useful for debugging multi-process/multi-threaded apps.

* `MIOPEN_ENABLE_LOGGING_ELAPSED_TIME` - Adds a timestamp to each log line. Indicates the time elapsed since the previous log message, in milliseconds.

## Layer Filtering

The following list of environment variables allow for enabling/disabling various kinds of kernels and algorithms. This can be helpful for both debugging MIOpen and integration with frameworks.

> **_NOTE 3:_ These variables can be set to the following values:**
> ```
> 1, yes, true, enable, enabled - to enable kernels/algorithm
> 0, no, false, disable, disabled - to disable kernels/algorithm
> ```

If a variable is not set, then MIOpen behaves as if it is set to `enabled`, unless otherwise specified. So all kinds of kernels/algorithms are enabled by default and the below variables can be used for disabling them.

> **_WARNING:_** **When the library is used with layer filtering, the results of `Find()` calls become narrower than during normal operation. This means that relevant find-db entries would not include some solutions that normally should be there.** **_Therefore the subsequent Immediate mode `Get()` calls may return incomplete information or even run into Fallback path._**

In order to rehabilitate the Immediate mode, the user can:
- Re-enable all solvers and re-run the same `Find()` calls that have been run before,
- Or, completely remove the User find-db.

### Filtering by algorithm

These variables control the sets (families) of convolution Solutions. For example, Direct algorithm is implemented in several Solutions that use OpenCL, GCN assembly etc. The corresponding variable can disable them all.
* `MIOPEN_DEBUG_CONV_FFT` - FFT convolution algorithm. 
* `MIOPEN_DEBUG_CONV_DIRECT` - Direct convolution algorithm.
* `MIOPEN_DEBUG_CONV_GEMM` - GEMM convolution algorithm.
* `MIOPEN_DEBUG_CONV_WINOGRAD` - Winograd convolution algorithm.
* `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM` - Implicit GEMM convolution algorithm.
* `MIOPEN_DEBUG_CONV_SCGEMM` - Statically Compiled GEMM convolution algorithm.

### Filtering by build method

* `MIOPEN_DEBUG_GCN_ASM_KERNELS` - Kernels written in assembly language. Currently these include some Direct solvers, Winograd kernels and SCGEMM.
* `MIOPEN_DEBUG_OPENCL_CONVOLUTIONS` - Convolution kernels written in OpenCL (note that _only_ convolutions affected).
* `MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES` - Binary kernels. Right now the library does not use binaries.

### Filtering out all Solutions except one

* `MIOPEN_DEBUG_FIND_ONLY_SOLVER=solution_id`, where `solution_id` should be either numeric or string identifier of some Solution. Directly affects only `Find()` calls _(however there is some indirect connection to Immediate mode; please see the "Warning" above.)_
  - If `solution_id` denotes some applicable Solution, then only that Solution will be found (plus GEMM and FFT, if these applicable, see _Note 4_).
  - Else, if `solution_id` is valid but not applicable, then `Find()` would fail with all algorithms (again, except GEMM and FFT, see _Note 4_)
  - Otherwise the `solution_id` is invalid (i.e. it doesn't match any existing Solution), and the `Find()` call would fail.

> **_NOTE 4:_** This env. variable does not affect the "gemm" and "fft" solutions. For now, GEMM and FFT can be disabled only at algorithm level (see above).

### Filtering the Solutions on individual basis

Some of the Solutions have individual controls available. These affect both Find and Immediate modes. _Note the "Warning" above._

Direct Solutions:
* `MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U` - `ConvAsm3x3U`.
* `MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U` - `ConvAsm1x1U`.
* `MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2` - `ConvAsm1x1UV2`.
* `MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2` - `ConvAsm5x10u2v2f1`, `ConvAsm5x10u2v2b1`.
* `MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224` - `ConvAsm7x7c3h224w224k64u2v2p3q3f1`.
* `MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3` - `ConvAsmBwdWrW3x3`.
* `MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1` - `ConvAsmBwdWrW1x1`.
* `MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11` - `ConvOclDirectFwd11x11`.
* `MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN` - `ConvOclDirectFwdGen`.
* `MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD3X3` - `ConvOclDirectFwd3x3`.
* `MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD` - `ConvOclDirectFwd`.
* `MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1` - `ConvOclDirectFwd`.
* `MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2` - `ConvOclBwdWrW2<n>` (where n = `{1,2,4,8,16}`), and `ConvOclBwdWrW2NonTunable`.
* `MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53` - `ConvOclBwdWrW53`.
* `MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1` - `ConvOclBwdWrW1x1`

Winograd  Solutions:
* `MIOPEN_DEBUG_AMD_WINOGRAD_3X3` - `ConvBinWinograd3x3U`, FP32 Winograd Fwd/Bwd, filter size fixed to 3x3.
* `MIOPEN_DEBUG_AMD_WINOGRAD_RXS` - `ConvBinWinogradRxS`, FP32/FP16 F(3,3) Fwd/Bwd and FP32 F(3,2) WrW Winograd. Subsets:
  * `MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW` - FP32 F(3,2) WrW convolutions only.
  * `MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD` - FP32/FP16 F(3,3) Fwd/Bwd.
* `MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2` - `ConvBinWinogradRxSf3x2`, FP32/FP16 Fwd/Bwd F(3,2) Winograd.
* Multi-pass Winograd:
  * `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2` - `ConvWinograd3x3MultipassWrW<3-2>`, WrW F(3,2), stride 2 only.
  * `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3` - `ConvWinograd3x3MultipassWrW<3-3>`, WrW F(3,3), stride 2 only.
  * `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4` - `ConvWinograd3x3MultipassWrW<3-4>`, WrW F(3,4).
  * `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5` - `ConvWinograd3x3MultipassWrW<3-5>`, WrW F(3,5).
  * `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6` - `ConvWinograd3x3MultipassWrW<3-6>`, WrW F(3,6).
  * `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3` - `ConvWinograd3x3MultipassWrW<5-3>`, WrW F(5,3).
  * `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4` - `ConvWinograd3x3MultipassWrW<5-4>`, WrW F(5,4).
  * `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2`:
    * `ConvWinograd3x3MultipassWrW<7-2>`, WrW F(7,2)
    * `ConvWinograd3x3MultipassWrW<7-2-1-1>`, WrW F(7x1,2x1)
    * `ConvWinograd3x3MultipassWrW<1-1-7-2>`, WrW F(1x7,1x2)
  * `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3`:
    * `ConvWinograd3x3MultipassWrW<7-3>`, WrW F(7,3)
    * `ConvWinograd3x3MultipassWrW<7-3-1-1>`, WrW F(7x1,3x1)
    * `ConvWinograd3x3MultipassWrW<1-1-7-3>`, WrW F(1x7,1x3)
* `MIOPEN_DEBUG_AMD_FUSED_WINOGRAD` - Fused FP32 F(3,3) Winograd, variable filter size.

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

> **_NOTE 5: Using experimental controls may result in:_**
> * Performance drops
> * Computation inaccuracies
> * Run-time errors
> * Other kinds of unexpected behavior
>
> **_It is strongly recommended to use them only with the explicit permission or request of the library developers._**

### Code Object (CO) version selection (EXPERIMENTAL)

currently, ROCm fully supports Code Object version 2 (Co v2). The support for version 3 (CO v3) is being gradually introduced. These variables allows for experimenting and triaging problems related to CO version:
* `MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE` - Overrides CO version auto-detection implemented in the library. `0` or unset - disable overriding (the default), `1` - enforces CO v2, `2` - behave as if both CO v2 and v3 are supported, `2` - enforces CO v3.
* `MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_NEWER` - This variable affects only Solutions available in both v2 and v3 code objects, and is intended to use only when ROCm supports both CO v2 and CO v3. By default, the older format is used (CO v2). When this variable is _enabled_, the behavior is reversed.
* `MIOPEN_DEBUG_AMD_OPENCL_ENFORCE_COV3` - Enforces CO v3 for OpenCL kernels.

### `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX`

Syntax of value:
* decimal or hex (with `0x` prefix) value that should fit into `unsigned long` (64 bits).
* If syntax is violated, then the behavior is unspecified.

Semantics:
* Sets the **_limit_** (max allowed workspace size) for Multi-pass (MP) Winograd Solutions, in bytes.
* Affects all MP Winograd Solutions. If a Solution needs more workspace than the limit, then it does not apply.
* If unset, then _the default_ limit is used. Current default is `2000000000` (~1.862 GiB) for gfx900 and gfx906/60 (or less CUs). No default limit is set for other GPUs.
* Special values:
```
 0 - Use the default limit, as if the variable is unset.
 1 - Completely prohibit the use of workspace.
-1 - Remove the default limit.
```
