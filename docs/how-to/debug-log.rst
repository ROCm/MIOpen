.. meta::
  :description: How to log and debug MIOpen
  :keywords: MIOpen, ROCm, API, documentation, logging, debugging

********************************************************************
Logging and debugging
********************************************************************

All logging messages are output to the standard error stream (``stderr``). You can use the following
environmental variables to control logging. Both variables are disabled by default.

* ``MIOPEN_ENABLE_LOGGING``: Prints the basic layer-by-layer MIOpen API call
  information with the actual parameters and configurations. This information is important for debugging.

  * To enable the feature: ``1``, ``on``, ``yes``, ``true``, ``enable``, or ``enabled``
  * To disable the feature: ``0``, ``off``, ``no``, ``false``, ``disable``, or ``disabled``

* ``MIOPEN_ENABLE_LOGGING_CMD``: Outputs the associated ``MIOpenDriver`` command lines to the
  console.

  * To enable the feature: ``1``, ``on``, ``yes``, ``true``, ``enable``, or ``enabled``
  * To disable the feature: ``0``, ``off``, ``no``, ``false``, ``disable``, or ``disabled``

* ``MIOPEN_LOG_LEVEL``: In addition to API call information and driver commands, MIOpen logs
  information related to the progress of its internal operations. This information can be useful for
  debugging and understanding how the library works. ``MIOPEN_LOG_LEVEL`` controls the
  verbosity of these messages. The allowed values are:

  * ``0``: Default. Works as level ``4`` for release builds and level ``5`` for debug builds.
  * ``1``: Quiet. No logging messages.
  * ``2``: Fatal errors only (unused).
  * ``3``: Errors, including fatal errors.
  * ``4``: All errors and warnings.
  * ``5``: Info. All the preceding information, plus information for debugging purposes.
  * ``6``: Detailed information. All the preceding information, plus more detailed information for
    debugging.
  * ``7``: Trace. All the preceding information, plus additional details.

* ``MIOPEN_ENABLE_LOGGING_MPMT``: Each log line is prefixed with information
  to identify records printed from different processes or threads. This is useful for debugging
  multi-process and multi-threaded applications.

* ``MIOPEN_ENABLE_LOGGING_ELAPSED_TIME``: Adds a timestamp to each log line that indicates the
  time elapsed (in milliseconds) since the previous log message.

.. note::

  If you require technical support, include the console log that is produced from these settings:

  .. code:: cpp

    export MIOPEN_ENABLE_LOGGING=1
    export MIOPEN_ENABLE_LOGGING_CMD=1
    export MIOPEN_LOG_LEVEL=6

Layer filtering
===================================================

The following sections describe environment variables for enabling or disabling various
types of kernels and algorithms. These are helpful for debugging MIOpen and framework integrations.

For these environment variables, you can use the following values:

* To enable the kernel/algorithm: ``1``, ``yes``, ``true``, ``enable``, or ``enabled``
* To disable the kernel/algorithm: ``0``, ``no``, ``false``, ``disable``, or ``disabled``

.. warning::

  When you use the library with layer filtering, the results of ``*Find()`` calls become narrower than
  during normal operation. This means that relevant FindDb entries won't include all the solutions that
  are normally present. Therefore, the subsequent immediate mode ``*Get()`` calls may return incomplete
  information or run into the fallback path.

In order to repair immediate mode, you can:

* Re-enable all solvers and rerun the same ``*Find()`` calls from before
* Completely remove the User FindDb

If a variable is not set, MIOpen treats it as ``enabled``. This means that kernels and
algorithms are enabled by default.

Filtering by algorithm
--------------------------------------------------------------------------------------------------------------

These variables control the sets (families) of convolution solutions. For example, the direct algorithm
is implemented in several solutions that use OpenCL and GCN assembly. The corresponding variable
is used to disable them.

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
* ``MIOPEN_DEBUG_OPENCL_CONVOLUTIONS``: Convolution kernels written in OpenCL. This only
  affects convolutions.

Filtering out all but one solution
--------------------------------------------------------------------------------------------------------------

* ``MIOPEN_DEBUG_FIND_ONLY_SOLVER=solution_id``: Only directly affects ``*Find()`` calls. However,
  there is an indirect connection to immediate mode (see the previous warning).

  * ``solution_id`` must be a numeric or a string identifier of some solution.
  * If the ``solution_id`` denotes an applicable solution, then only that solution is found (in addition to
    GEMM and FFT, if applicable). See the note in this section.
  * If the ``solution_id`` is valid, but not applicable, then ``*Find()`` fails with all algorithms (except for GEMM
    and FFT). See the note in this section.
  * Otherwise, the ``solution_id`` is invalid (for instance, it doesn't match any existing solution) and the ``*Find()``
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
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2`` -- ``ConvOclBwdWrW2<n>`` (where n =
    ``{1,2,4,8,16}``) and ``ConvOclBwdWrW2NonTunable``
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53`` -- ``ConvOclBwdWrW53``
  * ``MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1`` -- ``ConvOclBwdWrW1x1``

* Winograd solutions:

  * ``MIOPEN_DEBUG_AMD_WINOGRAD_3X3`` -- ``ConvBinWinograd3x3U``, ``FP32`` Winograd Fwd/Bwd,
    filter size fixed to 3x3
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS`` -- ``ConvBinWinogradRxS``, ``FP32``/``FP16`` F(3,3) Fwd/Bwd
    and ``FP32`` F(3,2) WrW Winograd. Subsets include:

    * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW`` -- ``FP32`` F(3,2) WrW convolutions only
    * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD`` -- ``FP32``/``FP16`` F(3,3) Fwd/Bwd

  * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2`` -- ``ConvBinWinogradRxSf3x2``, ``FP32``/``FP16``
    Fwd/Bwd F(3,2) Winograd
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3`` -- ``ConvBinWinogradRxSf2x3``, ``FP32``/``FP16``
    Fwd/Bwd F(2,3) Winograd, serves group convolutions only
  * ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1`` -- ``ConvBinWinogradRxSf2x3g1``, ``FP32``/``FP16``
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
  * ``MIOPEN_DEBUG_AMD_FUSED_WINOGRAD`` -- Fused ``FP32`` F(3,3) Winograd, variable filter size.

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

GEMM logging and behavior
==========================================================

The ``ROCBLAS_LAYER`` environmental variable can be set to output GEMM information when using the rocBLAS GEMM backend:

* ``ROCBLAS_LAYER=``: No logging (not set)
* ``ROCBLAS_LAYER=1``: Trace logging
* ``ROCBLAS_LAYER=2``: Bench logging
* ``ROCBLAS_LAYER=3``: Trace and bench logging

The ``HIPBLASLT_LOG_LEVEL`` environmental variable can be set to output GEMM information when using the hipBLASLt GEMM backend:

* ``HIPBLASLT_LOG_LEVEL=0``: Off -- no logging (default)
* ``HIPBLASLT_LOG_LEVEL=1``: Error logging
* ``HIPBLASLT_LOG_LEVEL=2``: Trace - API calls that launch HIP kernels log their parameters and important information
* ``HIPBLASLT_LOG_LEVEL=3``: Hints - Hints that can potentially improve the application’s performance
* ``HIPBLASLT_LOG_LEVEL=4``: Info - Provides general information about the library execution. Can contain details about heuristic status
* ``HIPBLASLT_LOG_LEVEL=5``: API Trace - API calls log their parameters and important information

You can also set the ``MIOPEN_GEMM_ENFORCE_BACKEND`` environment variable to override the
default GEMM backend, which is rocBLAS:

* ``MIOPEN_GEMM_ENFORCE_BACKEND=1``: Use rocBLAS if enabled
* ``MIOPEN_GEMM_ENFORCE_BACKEND=2``: Reserved
* ``MIOPEN_GEMM_ENFORCE_BACKEND=3``: No GEMM is called
* ``MIOPEN_GEMM_ENFORCE_BACKEND=4``: Reserved
* ``MIOPEN_GEMM_ENFORCE_BACKEND=5``: Use hipBLASLt if enabled
* ``MIOPEN_GEMM_ENFORCE_BACKEND=<any other value>``: Use default behavior

To disable the use of rocBLAS entirely, set the ``-DMIOPEN_USE_ROCBLAS=Off`` configuration flag during
MIOpen configuration. To disable the use of hipBLASLt entirely, set the ``-DMIOPEN_USE_HIPBLASLT=Off`` configuration flag during
MIOpen configuration.

For more information on how to use logging with rocBLAS, see the
:doc:`rocBLAS programmer guide <rocblas:how-to/Programmers_Guide>`.

Numerical checking
==========================================================

Use the ``MIOPEN_CHECK_NUMERICS`` environmental variable to debug potential numerical
abnormalities. When this variable is set, MIOpen scans all inputs and outputs for each kernel called and attempts to
detect infinities (``infs``), not-a-number (``NaN``), and all zeros. This environment variable has several
settings to help with debugging:

* ``MIOPEN_CHECK_NUMERICS=0x01``: Fully informative. Prints results from all checks to console.
* ``MIOPEN_CHECK_NUMERICS=0x02``: Warning information. Prints results only if an abnormality is
  detected.
* ``MIOPEN_CHECK_NUMERICS=0x04``: Throw error on detection. MIOpen runs ``MIOPEN_THROW``
  upon an abnormal result.
* ``MIOPEN_CHECK_NUMERICS=0x08``: Abort upon an abnormal result. Lets you drop into a
  debugging session.
* ``MIOPEN_CHECK_NUMERICS=0x10``: Print stats. Computes and prints mean/absmean/min/max
  (note that this is slow).

.. _control-parallel-compilation:

Check for GPU sub-buffer out-of-bounds memory accesses
==========================================================
Use the ``MIOPEN_DEBUG_CHECK_SUB_BUFFER_OOB_MEMORY_ACCESS`` to help debug potential out-of-bounds (OOBs)
memory access errors on GPU sub-buffers.  If the environment variable is undefined or set to zero, then no
sub-buffer out-of-bounds detection is performed. To verify the memory accesses, use one of these values:

* ``MIOPEN_DEBUG_CHECK_SUB_BUFFER_OOB_MEMORY_ACCESS=1``: Check for OOBs before the start of the sub-buffers.
* ``MIOPEN_DEBUG_CHECK_SUB_BUFFER_OOB_MEMORY_ACCESS=2``: Check for OOBs after the end of the sub-buffers.

Note that these checks are not intended for production use because they might cause a performance hit.

Controlling parallel compilation
==========================================================

MIOpen's convolution ``*Find()`` calls compile and benchmark a set of ``solvers`` contained in
``miopenConvAlgoPerf_t``. This is done in parallel with ``miopenConvAlgorithm_t``. Parallelism per
algorithm is set to 20 threads. Typically, there are far fewer threads spawned due to the limited number
of kernels under any given algorithm.

You can control the level of parallelism using the ``MIOPEN_COMPILE_PARALLEL_LEVEL`` environment
variable.

To disable multi-threaded compilation, run:

.. code:: cpp

  export MIOPEN_COMPILE_PARALLEL_LEVEL=1

Experimental controls
==========================================================

Using experimental controls might result in:

* Performance drops
* Computation inaccuracies
* Runtime errors
* Other kinds of unexpected behavior

We strongly recommended only using these controls at the explicit request of the library developers.

Code Object version selection (experimental)
-------------------------------------------------------------------------------------------------------------

Different ROCm versions use Code Object (CO) files from different versions (formats). The library
automatically uses the most suitable version. The following variables allow for experimenting and
triaging possible problems related to the CO version:

* ``MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE``: Affects kernels written in GCN assembly
  language.

  * ``0`` (or unset): Automatically detects the required CO version and assembles the files to that version. This is
    the default.
  * ``1``: Do not auto-detect the CO version and always assemble v2 COs.
  * ``2``: Behave as if both v2 and v3 COs are supported (see
    ``MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER``).
  * ``3``: Always assemble v3 COs.

* ``MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER``: This variable only affects assembly
  kernels and only applies when ROCm supports both v2 and v3 COs. By default, the newer
  format is used (v3 CO). When this variable is enabled, the behavior is reversed.
* ``MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_VERSION``: Enforces the CO format for OpenCL
  kernels. This only works with the HIP backend, when ``cmake ... -DMIOPEN_BACKEND=HIP...`` is used.

  * Unset - Automatically detects the required CO version. This is the default.
  * ``2``: Always build to v2 CO.
  * ``3``: Always build to v3 CO.
  * ``4``: Always build to v4 CO.

Winograd multi-pass maximum workspace throttling
-------------------------------------------------------------------------------------------------------------

* ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX``:
  ``ConvWinograd3x3MultipassWrW``, WrW
* ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX``: ``ConvMPBidirectWinograd*``,
  FWD BWD

Syntax of value:

* A decimal or hex (with ``0x`` prefix) value that must fit into a 64-bit unsigned integer
* If the syntax is invalid, then the behavior is unspecified

Usage notes:

* Sets the limit (max allowed workspace size) in bytes for multi-pass (MP) Winograd solutions.
* The value affects all MP Winograd solutions. If a solution needs more workspace than the limit, it doesn't apply.
* If the value is not set, then the default limit is used. The current default is ``2000000000`` (~1.862 GiB) for the gfx900
  and gfx906/60 (or fewer CUs). No default limit is defined for other GPUs.
* Special values:

  * ``0``: Use the default limit, as if the variable is unset
  * ``1``: Completely prohibit the use of workspace
  * ``-1``: Remove the default limit
