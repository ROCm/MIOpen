// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.include "Conv_Winograd_v40_6_0_metadata.inc"

.if (.amdgcn.gfx_generation_number == 12)
    KERNEL_PROLOG fp32_f2x3_stride2

    .include "Conv_Winograd_v40_6_0_gfx12_fp32_f2x3_stride2.inc"

    KERNEL_EPILOG fp32_f2x3_stride2
.else
    .error "Unsupported gfx generation"
.endif
