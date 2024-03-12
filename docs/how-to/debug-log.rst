.. meta::
  :description: MIOpen documentation
  :keywords: MIOpen, ROCm, API, documentation

********************************************************************
Logging & debugging
********************************************************************

All logging messages are output to the standard error stream (``stderr``). You can use the following
environmental variables to control logging. Both variables are disabled by default.

* ``MIOPEN_ENABLE_LOGGING``: Print the basic layer-by-layer MIOpen API call
  information with actual parameters (configurations). This information is important for debugging.

  * To enable feature: ``1``, ``on``, ``yes``, ``true``, ``enable``, ``enabled``
  * To disable feature: ``0``, ``off``, ``no``, ``false``, ``disable``, ``disabled``

* ``MIOPEN_ENABLE_LOGGING_CMD``: Output the associated ``MIOpenDriver`` command lines into the
  console.

  * To enable feature: ``1``, ``on``, ``yes``, ``true``, ``enable``, ``enabled``
  * To disable feature: ``0``, ``off``, ``no``, ``false``, ``disable``, ``disabled``

* ``MIOPEN_LOG_LEVEL``: In addition to API call information and driver commands, MIOpen prints
  information related to the progress of its internal operations. This information can be useful for
  debugging and understanding the library's principles of operation. `MIOPEN_LOG_LEVEL` controls the
  verbosity of these messages. Allowed values are:

  * ``0``: Default. Works as level 4 for release builds and level 5 for debug builds.
  * ``1``: Quiet. No logging messages.
  * ``2``: Fatal errors only (not used yet).
  * ``3``: Errors and fatals.
  * ``4``: All errors and warnings.
  * ``5``: Info. All the preceding information, plus information for debugging purposes.
  * ``6``: Detailed information. All the preceding information, plus more detailed information for
    debugging.
  * ``7``: Trace. All the preceding information, plus additional details.

* ``MIOPEN_ENABLE_LOGGING_MPMT``: When enabled, each log line is prefixed with information you
  can use to identify records printed from different processes or threads. This is useful for debugging
  multi-process/multi-threaded applications.

* ``MIOPEN_ENABLE_LOGGING_ELAPSED_TIME``: Adds a timestamp to each log line that indicates the
  time elapsed (in milliseconds) since the previous log message.

.. tip::

  If you require technical support, include the console log that is produced from:

  .. code:: cpp

    export MIOPEN_ENABLE_LOGGING=1
    export MIOPEN_ENABLE_LOGGING_CMD=1
    export MIOPEN_LOG_LEVEL=6

Layer filtering
===================================================

The following sections contain environment variables that you can use to enable or disable various
kinds of kernels and algorithms. These are helpful for debugging MIOpen and framework integrations.

For these environment variables, you can use the following values:

* To enable kernel/algorithm: ``1``, ``yes``, ``true``, ``enable``, ``enabled``
* To disable kernel/algorithm: ``0``, ``no``, ``false``, ``disable``, ``disabled``

.. warning::

  When you use the library with layer filtering, the results of ``*Find()`` calls become narrower than
  during normal operation. This means that relevant FindDb entries won't include all the solutions that
  are normally there. Therefore, the subsequent immediate mode ``*Get()`` calls may return incomplete
  information or run into fallback path.

In order to rehabilitate immediate mode, you can:

* Re-enable all solvers and re-run the same ``*Find()`` calls you previously ran
* Completely remove the User FindDb

If no variable is set, MIOpen behaves as if the variable is set to ``enabled``. This means that kernels and
algorithms are enabled by default.

Filtering by algorithm
--------------------------------------------------------------------------------------------------------------

These variables control the sets (families) of convolution solutions. For example, the direct algorithm
is implemented in several solutions that use OpenCL and GCN assembly. The corresponding variable
can disable them all.

* ``MIOPEN_DEBUG_CONV_FFT``: FFT convolution algorithm.
* ``MIOPEN_DEBUG_CONV_DIRECT``: Direct convolution algorithm.
* ``MIOPEN_DEBUG_CONV_GEMM``: GEMM convolution algorithm.
* ``MIOPEN_DEBUG_CONV_WINOGRAD``: Winograd convolution algorithm.
* ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM``: Implicit GEMM convolution algorithm.

Filtering by build method
--------------------------------------------------------------------------------------------------------------

* ``MIOPEN_DEBUG_GCN_ASM_KERNELS``: Kernels written in assembly language. These are used in
  many convolutions (some direct solvers, Winograd kernels, and fused convolutions) and batch
  normalization.
* ``MIOPEN_DEBUG_HIP_KERNELS``: Convolution kernels written in HIP. These implement the
  ImplicitGemm algorithm.
* ``MIOPEN_DEBUG_OPENCL_CONVOLUTIONS``: Convolution kernels written in OpenCL; this only
  affects convolutions.
* ``MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES``: Binary kernels. The library does not
  currently use binaries.

Filtering out all but one solution
--------------------------------------------------------------------------------------------------------------

* ``MIOPEN_DEBUG_FIND_ONLY_SOLVER=solution_id``: Directly affects only ``*Find()`` calls. However,
  there is an indirect connection to immediate mode (per the previous warning).
  * ``solution_id`` must be a numeric or a string identifier of some solution.
  * If ``solution_id`` denotes some applicable solution, then only that solution is found (in addition to
    GEMM and FFT, if applicable--refer to the following note).
  * If ``solution_id`` is valid, but not applicable, then ``*Find()`` fails with all algorithms (except for GEMM
    and FFT,--refer to the following note).
  * Otherwise, the ``solution_id`` is invalid (i.e., it doesn't match any existing solution) and the ``*Find()``
   call fails.

.. note::

  This environmental variable doesn't affect the GEMM and FFT solutions. For now, GEMM and FFT can
  only be disabled at the algorithm level.

Filtering the solutions on an individual basis
--------------------------------------------------------------------------------------------------------------

Some of the solutions have individual controls, which affect both find and immediate modes.

* Direct solutions:

  * ``MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U`` -- ``ConvAsm3x3U``
  * ``MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U`` -- ``ConvAsm1x1U``
  * ``MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2`` -- ``ConvAsm1x1UV2``
  * ``MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2`` -- ``ConvAsm5x10u2v2f1`, `ConvAsm5x10u2v2b1``
  * ``MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224`` -- ``ConvAsm7x7c3h224w224k64u2v2p3q3f1``
  * ``MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3`` -- ``ConvAsmBwdWrW3x3``
  * ``MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1`` -- ``ConvAsmBwdWrW1x1``
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11`` -- ``ConvOclDirectFwd11x11``
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN`` -- ``ConvOclDirectFwdGen``
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD`` -- ``ConvOclDirectFwd``
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1`` -- ``ConvOclDirectFwd1x1``
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2`` -- ``ConvOclBwdWrW2<n>`` (where n = ``{1,2,4,8,16}``)
    and ``ConvOclBwdWrW2NonTunable``
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53`` -- ``ConvOclBwdWrW53``
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1`` -- ``ConvOclBwdWrW1x1``

* Winograd solutions:

  * ``MIOPEN_DEBUG_AMD_WINOGRAD_3X3`` -- ``ConvBinWinograd3x3U``, FP32 Winograd Fwd/Bwd,
    filter size fixed to 3x3
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS`` -- ``ConvBinWinogradRxS``, FP32/FP16 F(3,3) Fwd/Bwd
    and FP32 F(3,2) WrW Winograd. Subsets:

    * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW`` -- FP32 F(3,2) WrW convolutions only
    * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD`` -- FP32/FP16 F(3,3) Fwd/Bwd

  * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2`` -- ``ConvBinWinogradRxSf3x2``, FP32/FP16
    Fwd/Bwd F(3,2) Winograd
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3`` -- ``ConvBinWinogradRxSf2x3``, FP32/FP16
    Fwd/Bwd F(2,3) Winograd, serves group convolutions only
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1`` -- ``ConvBinWinogradRxSf2x3g1``, FP32/FP16
    Fwd/Bwd F(2,3) Winograd, for non-group convolutions

* Multi-pass Winograd:

  * ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2`` -- ``ConvWinograd3x3MultipassWrW<3-2>``,
    WrW F(3,2), stride 2 only
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3`` -- ``ConvWinograd3x3MultipassWrW<3-3>``,
   WrW F(3,3), stride 2 only
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4`` -- ``ConvWinograd3x3MultipassWrW<3-4>``,
   WrW F(3,4)
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5`` -- ``ConvWinograd3x3MultipassWrW<3-5>``,
   WrW F(3,5)
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6`` -- ``ConvWinograd3x3MultipassWrW<3-6>``,
   WrW F(3,6)
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3`` -- ``ConvWinograd3x3MultipassWrW<5-3>``,
   WrW F(5,3)
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4`` -- ``ConvWinograd3x3MultipassWrW<5-4>``,
   WrW F(5,4)
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2``:

    * ``ConvWinograd3x3MultipassWrW<7-2>``, WrW F(7,2)
    * ``ConvWinograd3x3MultipassWrW<7-2-1-1>``, WrW F(7x1,2x1)
    * ``ConvWinograd3x3MultipassWrW<1-1-7-2>``, WrW F(1x7,1x2)

  * ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3``:

    * ``ConvWinograd3x3MultipassWrW<7-3>``, WrW F(7,3)
    * ``ConvWinograd3x3MultipassWrW<7-3-1-1>``, WrW F(7x1,3x1)
    * ``ConvWinograd3x3MultipassWrW<1-1-7-3>``, WrW F(1x7,1x3)

  * ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3`` -- ``ConvMPBidirectWinograd<2-3>``,
   FWD/BWD F(2,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3`` -- ``ConvMPBidirectWinograd<3-3>``,
   FWD/BWD F(3,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3`` -- ``ConvMPBidirectWinograd<4-3>``,
   FWD/BWD F(4,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3`` -- ``ConvMPBidirectWinograd<5-3>``,
   FWD/BWD F(5,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3`` -- ``ConvMPBidirectWinograd<6-3>``,
   FWD/BWD F(6,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F2X3`` --
   ``ConvMPBidirectWinograd_xdlops<2-3>``, FWD/BWD F(2,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F3X3`` --
   ``ConvMPBidirectWinograd_xdlops<3-3>``, FWD/BWD F(3,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F4X3`` --
   ``ConvMPBidirectWinograd_xdlops<4-3>``, FWD/BWD F(4,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F5X3`` --
   ``ConvMPBidirectWinograd_xdlops<5-3>``, FWD/BWD F(5,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F6X3`` --
   ``ConvMPBidirectWinograd_xdlops<6-3>``, FWD/BWD F(6,3)
  * ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM`` --
   ``ConvMPBidirectWinograd*``, FWD/BWD FP16 experimental mode (use at your own risk). Disabled
   by default.
  * ``MIOPEN_DEBUG_AMD_FUSED_WINOGRAD`` -- Fused FP32 F(3,3) Winograd, variable filter size.

Implicit GEMM solutions:

* ASM implicit GEMM

  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1`` --
 ``ConvAsmImplicitGemmV4R1DynamicFwd``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1`` --
   ``ConvAsmImplicitGemmV4R1DynamicFwd_1x1``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_V4R1`` --
   ``ConvAsmImplicitGemmV4R1DynamicBwd``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1`` --
   ``ConvAsmImplicitGemmV4R1DynamicWrw``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS`` --
   ``ConvAsmImplicitGemmGTCDynamicFwdXdlops``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS`` --
   ``ConvAsmImplicitGemmGTCDynamicBwdXdlops``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS`` --
   ``ConvAsmImplicitGemmGTCDynamicWrwXdlops``

* HIP implicit GEMM

  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1`` --
   ``ConvHipImplicitGemmV4R1Fwd``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4`` --
    ``ConvHipImplicitGemmV4R4Fwd``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1`` --
   ``ConvHipImplicitGemmBwdDataV1R1``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1`` --
    ``ConvHipImplicitGemmBwdDataV4R1``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1`` --
   ``ConvHipImplicitGemmV4R1WrW``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4`` --
   ``ConvHipImplicitGemmV4R4WrW``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_XDLOPS`` --
   ``ConvHipImplicitGemmForwardV4R4Xdlops``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R5_XDLOPS`` --
    ``ConvHipImplicitGemmForwardV4R5Xdlops``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS`` --
    ``ConvHipImplicitGemmBwdDataV1R1Xdlops``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS`` --
   ``ConvHipImplicitGemmBwdDataV4R1Xdlops``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4_XDLOPS`` --
   ``ConvHipImplicitGemmWrwV4R4Xdlops``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_PADDED_GEMM_XDLOPS`` --
   ``ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm``
  * ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4_PADDED_GEMM_XDLOPS`` --
   ``ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm``

rocBlas logging and behavior
============================================================

You can set the ``ROCBLAS_LAYER`` environmental variable to output GEMM information:

* ``ROCBLAS_LAYER=``: Not set--there is no logging
* ``ROCBLAS_LAYER=1``: Trace logging
* ``ROCBLAS_LAYER=2``: Bench logging
* ``ROCBLAS_LAYER=3``: Both trace and bench logging

Additionally, you can set the ``MIOPEN_GEMM_ENFORCE_BACKEND`` environment variable to override
the default GEMM backend (rocBLAS):

* ``MIOPEN_GEMM_ENFORCE_BACKEND=1`, use rocBLAS if enabled
* ``MIOPEN_GEMM_ENFORCE_BACKEND=2`, reserved
* ``MIOPEN_GEMM_ENFORCE_BACKEND=3`, no gemm will be called
* ``MIOPEN_GEMM_ENFORCE_BACKEND=4`, reserved
* ``MIOPEN_GEMM_ENFORCE_BACKEND=<any other value>`, use default behavior

To disable using rocBlas entirely, set the configuration flag `-DMIOPEN_USE_ROCBLAS=Off` during MIOpen configuration.

More information on logging with rocBlas can be found [here](https://github.com/ROCm/rocBLAS/wiki/5.Logging).


## Numerical Checking

MIOpen provides the environmental variable `MIOPEN_CHECK_NUMERICS` to allow users to debug potential numerical abnormalities. Setting this variable will scan all inputs and outputs of each kernel called and attempt to detect infinities (infs), not-a-number (NaN), or all zeros. The environment variable has several settings that will help with debugging:

* `MIOPEN_CHECK_NUMERICS=0x01`: Fully informative, prints results from all checks to console
* `MIOPEN_CHECK_NUMERICS=0x02`: Warning information, prints results only if abnormality detected
* `MIOPEN_CHECK_NUMERICS=0x04`: Throw error on detection, MIOpen execute MIOPEN_THROW on abnormal result
* `MIOPEN_CHECK_NUMERICS=0x08`: Abort on abnormal result, this will allow users to drop into a debugging session
* `MIOPEN_CHECK_NUMERICS=0x10`: Print stats, this will compute and print mean/absmean/min/max (note, this is much slower)

.. _control-parallel-compilation:

## Controlling Parallel Compilation

MIOpen's Convolution Find() calls will compile and benchmark a set of `solvers` contained in `miopenConvAlgoPerf_t` this is done in parallel per `miopenConvAlgorithm_t`. Parallelism per algorithm is set to 20 threads. Typically there are far fewer threads spawned due to the limited number of kernels under any given algorithm. The level of parallelism can be controlled using the environment variable `MIOPEN_COMPILE_PARALLEL_LEVEL`. 

For example, to disable multi-threaded compilation:
```
export MIOPEN_COMPILE_PARALLEL_LEVEL=1
```


## Experimental controls

> **_NOTE 5: Using experimental controls may result in:_**
> * Performance drops
> * Computation inaccuracies
> * Run-time errors
> * Other kinds of unexpected behavior
>
> **_It is strongly recommended to use them only with the explicit permission or request of the library developers._**

### Code Object (CO) version selection (EXPERIMENTAL)

Different ROCm versions use Code Object files of different versions (or, in other words, formats). The library uses suitable version automatically. The following variables allow for experimenting and triaging possible problems related to CO version:
* `MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE` - Affects kernels written in GCN assembly language.
  * `0` or unset - Automatically detect the required CO version and assemble to that version. This is the default.
  * `1` - Do not auto-detect Code Object version, always assemble v2 Code Objects.
  * `2` - Behave as if both CO v2 and v3 are supported (see `MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER`).
  * `3` - Always assemble v3 Code Objects.
* `MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER` - This variable affects only assembly kernels, and only when ROCm supports both CO v2 and CO v3 (like ROCm 2.10). By default, the newer format is used (CO v3). When this variable is _enabled_, the behavior is reversed.
* `MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_VERSION` - Enforces Code Object format for OpenCL kernels. Works with HIP backend only (`cmake ... -DMIOPEN_BACKEND=HIP...`).
  * Unset - Automatically detect the required CO version. This is the default.
  * `2` - Always build to CO v2.
  * `3` - Always build to CO v3.
  * `4` - Always build to CO v4.

### Winograd Multi-pass Maximum Workspace throttling

`MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX` - `ConvWinograd3x3MultipassWrW`, WrW
`MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX` - `ConvMPBidirectWinograd*`, FWD BWD

Syntax of value:
* decimal or hex (with `0x` prefix) value that should fit into 64-bit unsigned integer.
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
