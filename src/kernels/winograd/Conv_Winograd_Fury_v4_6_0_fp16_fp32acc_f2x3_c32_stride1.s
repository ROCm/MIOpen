// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.include "Conv_Winograd_Fury_v2_X_X_X_metadata.inc"

.if (.amdgcn.gfx_generation_number == 12)
    KERNEL_PROLOG 4_6_0, _1536vgprs_fp16_fp32acc_f2x3_c32_stride1
    .noaltmacro
    .include "Conv_Winograd_Fury_v4_6_0_gfx12_1536vgprs_fp16_fp32acc_f2x3_c32_stride1.inc"
    .altmacro
    KERNEL_EPILOG 4_6_0, _1536vgprs_fp16_fp32acc_f2x3_c32_stride1
.else
    .error "Unsupported gfx generation"
.endif
