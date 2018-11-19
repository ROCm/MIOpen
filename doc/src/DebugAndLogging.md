Debugging and Logging
=====================

## Logging
The most basic enviromental variable for debugging purposes is `MIOPEN_ENABLE_LOGGING=1`. This will give the user basic layer by layer call and configurations. If bulding from source, a user can use the environmental variable `MIOPEN_ENABLE_LOGGING_CMD=1` to output the associated `MIOpenDriver` command line.


## Log Levels
The `MIOPEN_LOG_LEVEL` environment variable controls the verbosity of the messages printed by MIOpen onto console. Allowed values are:
* 0 - Default. Works as level 4 for Release builds, level 5 for Debug builds.
* 1 - Quiet. No logging messages (except those controlled by MIOPEN_ENABLE_LOGGING).
* 2 - Fatal errors only (not used yet).
* 3 - Errors and fatals.
* 4 - All errors and warnings.
* 5 - Info. All the above plus information for debugging purposes.
* 6 - Detailed info. All the above plus more detailed information for debugging.
* 7 - Trace: the most detailed debugging info plus all above (not used so far).

All messages output via `stderr`.


## Layer Filtering
The following list of environment variables can be helpful for both debugging MIOpen as well integration with frameworks.

* `MIOPEN_ENABLE_LOGGING=1` – log all the MIOpen APIs called including the parameters passed to those APIs.
* `MIOPEN_DEBUG_GCN_ASM_KERNELS=0` – disable hand-tuned asm. kernels for Direct convolution algorithm. Fall-back to kernels written in high-level language.
* `MIOPEN_DEBUG_CONV_FFT=0` – disable FFT convolution algorithm. 
* `MIOPEN_DEBUG_CONV_DIRECT=0` – disable Direct convolution algorithm.
* `MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES=0` - this disables binary Winograd kernels, however, not all Winograds are binaries. To disable all Winograd algorithms, the following two vars can be used:
* MIOPEN_DEBUG_AMD_WINOGRAD_3X3=0 - FP32 Winograd Fwd/Bwd, filter size fixed to 3x3.
* MIOPEN_DEBUG_AMD_WINOGRAD_RXS=0 - FP32 and FP16 Winograd Fwd/Bwd, variable filter size.

## rocBlas Logging
The `ROCBLAS_LAYER` environmental variable can be set to output GEMM information:
* `ROCBLAS_LAYER=` - is not set, there is no logging
* `ROCBLAS_LAYER=1` - is set to 1, then there is trace logging
* `ROCBLAS_LAYER=2` - is set to 2, then there is bench logging
* `ROCBLAS_LAYER=3` - is set to 3, then there is both trace and bench logging

 To disable using rocBlas entirely set the configuration flag `-DMIOPEN_USE_ROCBLAS=Off` during MIOpen configuration.


More information on logging with RocBlas can be found [here](https://github.com/ROCmSoftwarePlatform/rocBLAS/wiki/5.Logging).
