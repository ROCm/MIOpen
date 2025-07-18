.. meta::
  :description: MIOpen environment variables reference
  :keywords: MIOpen, ROCm, API, environment variables, environment, reference

********************************************************************
MIOpen environment variables
********************************************************************

This section describes the important MIOpen environment variables,
which are grouped by functionality.

Logging and debugging
======================

The logging and debugging environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_ENABLE_LOGGING``
        | Prints basic layer-by-layer MIOpen API call information with parameters and configurations.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_ENABLE_LOGGING_CMD``
        | Outputs associated MIOpenDriver command lines to console.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_LOG_LEVEL``
        | Controls verbosity of internal operation logging messages.
      - | 0: Default (level 4 for release, level 5 for debug builds)
        | 1: Quiet (no logging)
        | 2: Fatal errors only (unused)
        | 3: Errors including fatal errors
        | 4: All errors and warnings
        | 5: Info level debugging
        | 6: Detailed debugging information
        | 7: Trace level with additional details

    * - | ``MIOPEN_ENABLE_LOGGING_MPMT``
        | Prefixes each log line with process/thread identification for multi-process/multi-threaded debugging.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_ENABLE_LOGGING_ELAPSED_TIME``
        | Adds timestamp showing elapsed time in milliseconds since previous log message.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_CHECK_NUMERICS``
        | Scans inputs/outputs for numerical abnormalities (inf, NaN, zeros).
      - | 0x01: Fully informative (print all check results)
        | 0x02: Warning information (print only abnormalities)
        | 0x04: Throw error on detection
        | 0x08: Abort on abnormal result
        | 0x10: Print statistics (mean/absmean/min/max)

    * - | ``MIOPEN_DEBUG_CHECK_SUB_BUFFER_OOB_MEMORY_ACCESS``
        | Checks for GPU sub-buffer out-of-bounds memory access errors.
      - | 0 or unset: No OOB detection
        | 1: Check for OOBs before sub-buffer start
        | 2: Check for OOBs after sub-buffer end

Find mode configuration
=======================

The find mode configuration environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Find database <../conceptual/finddb>`, :doc:`Use the find APIs and immediate mode <../how-to/find-and-immediate>`
and :doc:`Performance database <../conceptual/perfdb>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_FIND_MODE``
        | Sets find mode to accelerate find API calls.
      - | "NORMAL" or 1: Full find mode (benchmarks all solvers)
        | "FAST" or 2: Fast find (use FindDb or immediate fallback)
        | "HYBRID" or 3: Hybrid find (FindDb hit or full find)
        | 4: Reserved (do not use)
        | "DYNAMIC_HYBRID" or 5: Dynamic hybrid (default, skip non-dynamic kernels)

    * - | ``MIOPEN_FIND_ENFORCE``
        | Controls auto-tune behavior and database updates.
      - | "NONE" or 1: No change in default behavior
        | "DB_UPDATE" or 2: Always perform auto-tune and update PerfDb
        | "SEARCH" or 3: Auto-tune even if not requested via API
        | "SEARCH_DB_UPDATE" or 4: Combination of DB_UPDATE and SEARCH
        | "DB_CLEAN" or 5: Remove optimized values from User PerfDb

    * - | ``MIOPEN_DEBUG_DISABLE_FIND_DB``
        | Disables FindDb functionality.
      - | 1: Disable FindDb
        | 0 or unset: Enable FindDb

Algorithm control
=================

The algorithm control environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_CONV_FFT``
        | Controls FFT convolution algorithm.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT``
        | Controls direct convolution algorithm.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_GEMM``
        | Controls GEMM convolution algorithm.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_WINOGRAD``
        | Controls Winograd convolution algorithm.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM``
        | Controls implicit GEMM convolution algorithm.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMMED_FALLBACK``
        | Controls immediate fallback for convolution algorithms.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK``
        | Controls AI immediate mode fallback behavior.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK``
        | Forces immediate mode fallback for convolution operations.
      - | 0: Disable
        | 1: Enable

Kernel build method control
===========================

The kernel build method control environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_GCN_ASM_KERNELS``
        | Controls assembly language kernels for convolutions and batch normalization.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_HIP_KERNELS``
        | Controls HIP-written convolution kernels (ImplicitGemm algorithm).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_OPENCL_CONVOLUTIONS``
        | Controls OpenCL-written convolution kernels.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP``
        | Controls OpenCL Wave64 without workgroup behavior.
      - | 0: Disable
        | 1: Enable

Solution selection
==================

The solution selection environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_FIND_ONLY_SOLVER``
        | Forces use of only one specific solution. Affects ``*Find()`` calls only.
      - | Numeric or string solution identifier
        | If valid and applicable: only that solution is found
        | If valid but not applicable: ``*Find()`` fails
        | If invalid: ``*Find()`` call fails

Direct solution control
=======================

The direct solution control environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_3X3U``
        | Controls ConvAsm3x3U direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U``
        | Controls ConvAsm1x1U direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1UV2``
        | Controls ConvAsm1x1UV2 direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_5X10U2V2``
        | Controls ConvAsm5x10u2v2f1 and ConvAsm5x10u2v2b1 direct solutions.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_7X7C3H224W224``
        | Controls ConvAsm7x7c3h224w224k64u2v2p3q3f1 direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW3X3``
        | Controls ConvAsmBwdWrW3x3 direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1``
        | Controls ConvAsmBwdWrW1x1 direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11``
        | Controls ConvOclDirectFwd11x11 direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN``
        | Controls ConvOclDirectFwdGen direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD``
        | Controls ConvOclDirectFwd direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1``
        | Controls ConvOclDirectFwd1x1 direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2``
        | Controls ConvOclBwdWrW2<n> (n={1,2,4,8,16}) and ConvOclBwdWrW2NonTunable solutions.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53``
        | Controls ConvOclBwdWrW53 direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1``
        | Controls ConvOclBwdWrW1x1 direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_PERF_VALS``
        | Controls performance values for ConvAsm1x1U direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_SEARCH_OPTIMIZED``
        | Controls optimized search for ConvAsm1x1U direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_ASM_1X1U_AI_HEUR``
        | Controls AI heuristics for ConvAsm1x1U direct solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD``
        | Controls naive convolution forward direct solution.
      - | 0: Disable
        | 1: Enable

Winograd solution control
=========================

The Winograd solution control environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_3X3``
        | Controls ConvBinWinograd3x3U FP32 Winograd Fwd/Bwd (filter size 3x3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS``
        | Controls ConvBinWinogradRxS FP32/FP16 F(3,3) Fwd/Bwd and FP32 F(3,2) WrW Winograd.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_WRW``
        | Controls FP32 F(3,2) WrW convolutions only (subset of ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS``).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FWD_BWD``
        | Controls FP32/FP16 F(3,3) Fwd/Bwd (subset of ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS``).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2``
        | Controls ConvBinWinogradRxSf3x2 FP32/FP16 Fwd/Bwd F(3,2) Winograd.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3``
        | Controls ConvBinWinogradRxSf2x3 FP32/FP16 Fwd/Bwd F(2,3) Winograd (group convolutions only).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_G1``
        | Controls ConvBinWinogradRxSf2x3g1 FP32/FP16 Fwd/Bwd F(2,3) Winograd (non-group convolutions).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_FUSED_WINOGRAD``
        | Controls Fused FP32 F(3,3) Winograd with variable filter size.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F2X3_PERF_VALS``
        | Controls performance values for Winograd RxS F(2,3) solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3``
        | Controls Winograd Fury RxS F(2,3) solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2``
        | Controls Winograd Fury RxS F(3,2) solution.
      - | 0: Disable
        | 1: Enable

Multi-pass Winograd solution control
====================================

The multi-pass Winograd solution control environment variables for MIOpen are collected in the
following table. For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X2``
        | Controls ConvWinograd3x3MultipassWrW<3-2> WrW F(3,2), stride 2 only.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3``
        | Controls ConvWinograd3x3MultipassWrW<3-3> WrW F(3,3), stride 2 only.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4``
        | Controls ConvWinograd3x3MultipassWrW<3-4> WrW F(3,4).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5``
        | Controls ConvWinograd3x3MultipassWrW<3-5> WrW F(3,5).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6``
        | Controls ConvWinograd3x3MultipassWrW<3-6> WrW F(3,6).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3``
        | Controls ConvWinograd3x3MultipassWrW<5-3> WrW F(5,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X4``
        | Controls ConvWinograd3x3MultipassWrW<5-4> WrW F(5,4).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X2``
        | Controls ConvWinograd3x3MultipassWrW<7-2>, <7-2-1-1>, and <1-1-7-2> WrW F(7,2) variants.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F7X3``
        | Controls ConvWinograd3x3MultipassWrW<7-3>, <7-3-1-1>, and <1-1-7-3> WrW F(7,3) variants.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3``
        | Controls ConvMPBidirectWinograd<2-3> FWD/BWD F(2,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3``
        | Controls ConvMPBidirectWinograd<3-3> FWD/BWD F(3,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3``
        | Controls ConvMPBidirectWinograd<4-3> FWD/BWD F(4,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3``
        | Controls ConvMPBidirectWinograd<5-3> FWD/BWD F(5,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3``
        | Controls ConvMPBidirectWinograd<6-3> FWD/BWD F(6,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F2X3``
        | Controls ConvMPBidirectWinograd_xdlops<2-3> FWD/BWD F(2,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F3X3``
        | Controls ConvMPBidirectWinograd_xdlops<3-3> FWD/BWD F(3,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F4X3``
        | Controls ConvMPBidirectWinograd_xdlops<4-3> FWD/BWD F(4,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F5X3``
        | Controls ConvMPBidirectWinograd_xdlops<5-3> FWD/BWD F(5,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F6X3``
        | Controls ConvMPBidirectWinograd_xdlops<6-3> FWD/BWD F(6,3).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM``
        | Controls ConvMPBidirectWinograd* FWD/BWD FP16 experimental mode (use at your own risk).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX``
        | Sets workspace size limit for ConvWinograd3x3MultipassWrW solutions.
      - | Decimal or hex value (64-bit unsigned integer) in bytes
        | Default: 2000000000 (~1.862 GiB) for gfx900 and gfx906/60
        | 0: Use default limit
        | 1: Prohibit workspace use
        | -1: Remove default limit

    * - | ``MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX``
        | Sets workspace size limit for ConvMPBidirectWinograd solutions.
      - | Decimal or hex value (64-bit unsigned integer) in bytes
        | 0: Use default limit
        | 1: Prohibit workspace use
        | -1: Remove default limit

ASM implicit GEMM solution control
==================================

The ASM implicit GEMM solution control environment variables for MIOpen are collected in the
following table. For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1``
        | Controls ConvAsmImplicitGemmV4R1DynamicFwd solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1``
        | Controls ConvAsmImplicitGemmV4R1DynamicFwd_1x1 solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_V4R1``
        | Controls ConvAsmImplicitGemmV4R1DynamicBwd solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_V4R1``
        | Controls ConvAsmImplicitGemmV4R1DynamicWrw solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS``
        | Controls ConvAsmImplicitGemmGTCDynamicFwdXdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS``
        | Controls ConvAsmImplicitGemmGTCDynamicBwdXdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS``
        | Controls ConvAsmImplicitGemmGTCDynamicWrwXdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC``
        | Controls ConvAsmImplicitGemmGTCFwdXdlopsNHWC solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC``
        | Controls ConvAsmImplicitGemmGTCBwdXdlopsNHWC solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_WRW_GTC_XDLOPS_NHWC``
        | Controls ConvAsmImplicitGemmGTCWrwXdlopsNHWC solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_DLOPS_NCHWC``
        | Controls ConvAsmImplicitGemmGTCFwdDlopsNCHWC solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_PK_ATOMIC_ADD_FP16``
        | Controls packed atomic add FP16 behavior for ASM implicit GEMM solutions.
      - | 0: Disable packed atomic add FP16
        | 1: Enable packed atomic add FP16

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_GROUP_BWD_XDLOPS``
        | Controls grouped convolution HIP implicit GEMM backward XDLOPS solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR``
        | Controls AI heuristics for grouped convolution HIP implicit GEMM backward XDLOPS.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_FWD_V4R4_XDLOPS_ADD_VECTOR_LOAD_GEMMN_TUNE_PARAM``
        | Controls vector load GEMM-N tuning parameters for implicit GEMM forward V4R4 XDLOPS.
      - | 0: Disable
        | 1: Enable

HIP implicit GEMM solution control
==================================

The HIP implicit GEMM solution control environment variables for MIOpen are collected in the
following table. For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1``
        | Controls ConvHipImplicitGemmV4R1Fwd solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4``
        | Controls ConvHipImplicitGemmV4R4Fwd solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1``
        | Controls ConvHipImplicitGemmBwdDataV1R1 solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1``
        | Controls ConvHipImplicitGemmBwdDataV4R1 solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R1``
        | Controls ConvHipImplicitGemmV4R1WrW solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4``
        | Controls ConvHipImplicitGemmV4R4WrW solution.
      - | 0: Disable
        | 1: Enable


    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_XDLOPS``
        | Controls ConvHipImplicitGemmForwardV4R4Xdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R5_XDLOPS``
        | Controls ConvHipImplicitGemmForwardV4R5Xdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS``
        | Controls ConvHipImplicitGemmBwdDataV1R1Xdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS``
        | Controls ConvHipImplicitGemmBwdDataV4R1Xdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4_XDLOPS``
        | Controls ConvHipImplicitGemmWrwV4R4Xdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_PADDED_GEMM_XDLOPS``
        | Controls ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_V4R4_PADDED_GEMM_XDLOPS``
        | Controls ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS``
        | Controls ConvHipImplicitGemmFwdXdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS``
        | Controls ConvHipImplicitGemmBwdXdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS``
        | Controls ConvHipImplicitGemmWrwXdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS``
        | Controls implicit GEMM XDLOPS solutions.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE``
        | Controls XDLOPS emulation for implicit GEMM solutions.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM``
        | Controls inline assembly for implicit GEMM XDLOPS solutions.
      - | 0: Disable
        | 1: Enable

3D implicit GEMM solution control
=================================

The 3D implicit GEMM solution control environment variables for MIOpen are collected in the
following table. For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS``
        | Controls 3D ConvHipImplicitGemmFwdXdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS``
        | Controls 3D ConvHipImplicitGemmBwdXdlops solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS``
        | Controls 3D ConvHipImplicitGemmWrwXdlops solution.
      - | 0: Disable
        | 1: Enable

GEMM backend control
====================

The GEMM backend control environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_GEMM_ENFORCE_BACKEND``
        | Overrides default GEMM backend (rocBLAS).
      - | 1: Use rocBLAS if enabled
        | 2: Reserved
        | 3: No GEMM is called
        | 4: Reserved
        | 5: Use hipBLASLt if enabled
        | Any other value: Use default behavior

    * - | ``ROCBLAS_LAYER``
        | Controls rocBLAS GEMM logging output.
      - | Unset: No logging
        | 1: Trace logging
        | 2: Bench logging
        | 3: Trace and bench logging

    * - | ``HIPBLASLT_LOG_LEVEL``
        | Controls hipBLASLt GEMM logging output.
      - | 0: Off (default)
        | 1: Error logging
        | 2: Trace (API calls with parameters)
        | 3: Hints (performance improvement suggestions)
        | 4: Info (general execution information)
        | 5: API trace (detailed API parameters)

Convolution attributes
======================

The convolution attribute environment variables for MIOpen are collected in the following table.
For more information, see :doc:`MI200 alternate implementation <../conceptual/MI200-alt-implementation>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL``
        | Controls the alternate ``FP16`` implementation that uses the ``BFloat16`` larger exponent
        | range for all convolution directions.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL``
        | Controls the alternate ``FP16`` implementation that uses the ``BFloat16`` larger exponent
        | range (alternative to the miopenSetConvolutionAttribute API).
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC``
        | Controls deterministic convolution behavior.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE``
        | Controls FP8 rounding mode for convolution attributes.
      - | Integer value specifying FP8 rounding mode

    * - | ``MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_SEED``
        | Controls FP8 rounding seed for convolution attributes.
      - | Integer value specifying FP8 rounding seed

Compilation control
===================

The compilation control environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_COMPILE_PARALLEL_LEVEL``
        | Controls parallel compilation thread count for ``*Find()`` calls.
      - | Integer value
        | Default: 1 when using ``COMGR``, otherwise half the number of available hardware threads
        | 1: Disable multi-threaded compilation

    * - | ``MIOPEN_DEBUG_COMPILE_ONLY``
        | Controls compile-only mode for debugging.
      - | 0: Disable
        | 1: Enable

Experimental controls
=====================

The experimental control environment variables for MIOpen are collected in the following table.
For more information, see :doc:`Logging and debugging <../how-to/debug-log>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE``
        | Controls Code Object (CO) version for GCN assembly kernels.
      - | 0 or unset: Auto-detect CO version (default)
        | 1: Always assemble v2 COs
        | 2: Behave as if both v2 and v3 COs supported
        | 3: Always assemble v3 COs

    * - | ``MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER``
        | Prefers older CO format when both v2 and v3 are supported.
      - | 1, "yes", "true", "enable", "enabled": Prefer v2 over v3
        | 0, "no", "false", "disable", "disabled": Use newer format

    * - | ``MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_VERSION``
        | Enforces CO format for OpenCL kernels (HIP backend only).
      - | Unset: Auto-detect CO version (default)
        | 2: Always build to v2 CO
        | 3: Always build to v3 CO
        | 4: Always build to v4 CO

RNN control
===========

The RNN control environment variables for MIOpen are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_RNNBWDMS_EXP``
        | Controls experimental RNN backward multi-stream behavior.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_RNNBWMS_EXP``
        | Controls experimental RNN backward multi-stream behavior.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_RNN_DYNAMIC_FORCE``
        | Forces dynamic RNN behavior.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_RNNFWD_EXP``
        | Controls experimental RNN forward behavior.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_RNNFWD_MS_DISPATCH``
        | Controls multi-stream dispatch for RNN forward operations.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_RNN_MS_STREAM_CNT``
        | Controls stream count for RNN multi-stream operations.
      - | Integer value specifying stream count

Composable Kernel (CK) solution control
=======================================

The Composable Kernel (CK) solution control environment variables for MIOpen are collected in the
following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW``
        | Controls CK implicit GEMM forward V6R1 DLOPS NCHW solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV``
        | Controls CK implicit GEMM forward bias activation fused solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_RES_ADD_ACTIV``
        | Controls CK implicit GEMM forward bias residual add activation fused solution.
      - | 0: Disable
        | 1: Enable

MLIR solution control
=====================

The MLIR solution control environment variables for MIOpen are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_CONV_MLIR_IGEMM_WRW_XDLOPS``
        | Controls MLIR implicit GEMM weight-gradient XDLOPS solution.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DEBUG_CONV_MLIR_IGEMM_BWD_XDLOPS``
        | Controls MLIR implicit GEMM backward XDLOPS solution.
      - | 0: Disable
        | 1: Enable

Attention and softmax control
=============================

The attention and softmax control environment variables for MIOpen are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DEBUG_ATTN_SOFTMAX``
        | Controls attention softmax solution.
      - | 0: Disable
        | 1: Enable

Driver and testing (Advanced)
=============================

The driver and testing environment variables for MIOpen are collected in the following table. These
variables are primarily intended for testing and driver purposes.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``MIOPEN_DRIVER_PAD_BUFFERS_2M``
        | Controls 2M buffer padding in MIOpen driver.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DRIVER_USE_GPU_REFERENCE``
        | Controls GPU reference usage in MIOpen driver.
      - | 0: Disable
        | 1: Enable

    * - | ``MIOPEN_DRIVER_SUBNORM_PERCENTAGE``
        | Controls subnormal percentage in MIOpen driver.
      - | Integer value specifying subnormal percentage
