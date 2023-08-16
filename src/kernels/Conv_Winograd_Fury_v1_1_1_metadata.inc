/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

.macro PROLOG_KERNEL_DESCRIPTOR kernel_name
.text
.globl \kernel_name
.p2align 8
.type \kernel_name,@function
\kernel_name:
.endm

.macro METADATA sc,wc,wg_x, kernel_name
.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: \kernel_name
    .symbol: \kernel_name\().kd
    .language: "OpenCL C"
    .language_version: [ 1, 2 ]
    .sgpr_count: \sc
    .vgpr_count: \wc
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_size: 208
    .kernarg_segment_align: 8
    .reqd_workgroup_size: [ \wg_x, 1, 1 ]
    .max_flat_workgroup_size: \wg_x
    .wavefront_size: 64
    .args:
    - { .size: 4, .offset:   0, .value_kind: by_value, .name: n_groups }
    - { .size: 2, .offset:   4, .value_kind: by_value, .name: out_W_e }
    - { .size: 2, .offset:   6, .value_kind: by_value, .name: out_H_s }
    - { .size: 2, .offset:   8, .value_kind: by_value, .name: d_W_window }
    - { .size: 2, .offset:  10, .value_kind: by_value, .name: d_H_window }
    - { .size: 2, .offset:  12, .value_kind: by_value, .name: d_W_clip_bot_neg }
    - { .size: 2, .offset:  14, .value_kind: by_value, .name: d_H_clip_bot_neg }
    - { .size: 4, .offset:  16, .value_kind: by_value, .name: d_N_stride }
    - { .size: 4, .offset:  20, .value_kind: by_value, .name: d_H_stride }
    - { .size: 4, .offset:  24, .value_kind: by_value, .name: d_C_stride }

    - { .size: 2, .offset:  28, .value_kind: by_value, .name: o_W_window }
    - { .size: 2, .offset:  30, .value_kind: by_value, .name: o_H_window }
    - { .size: 2, .offset:  32, .value_kind: by_value, .name: o_W_clip_bot_neg }
    - { .size: 2, .offset:  34, .value_kind: by_value, .name: o_H_clip_bot_neg }
    - { .size: 4, .offset:  36, .value_kind: by_value, .name: o_N_stride }
    - { .size: 4, .offset:  40, .value_kind: by_value, .name: o_H_stride }
    - { .size: 4, .offset:  44, .value_kind: by_value, .name: o_K_stride }

    - { .size: 8, .offset:  48, .value_kind: global_buffer, .name: data_addr,   .address_space: global, .is_const: true }
    - { .size: 8, .offset:  56, .value_kind: global_buffer, .name: output_addr, .address_space: global, .is_const: false }
    - { .size: 8, .offset:  64, .value_kind: global_buffer, .name: filter_addr, .address_space: global, .is_const: true }

    - { .size: 4, .offset:  72, .value_kind: by_value, .name: BATCH_SIZE }
    - { .size: 4, .offset:  76, .value_kind: by_value, .name: K }
    - { .size: 4, .offset:  80, .value_kind: by_value, .name: C }
    - { .size: 4, .offset:  84, .value_kind: by_value, .name: R }
    - { .size: 4, .offset:  88, .value_kind: by_value, .name: S }

    - { .size: 4, .offset:  92, .value_kind: by_value, .name: f_K_stride }
    - { .size: 4, .offset:  96, .value_kind: by_value, .name: f_C_stride }
    - { .size: 4, .offset: 100, .value_kind: by_value, .name: f_R_stride }
    - { .size: 4, .offset: 104, .value_kind: by_value, .name: f_S_stride }

    - { .size: 4, .offset: 108, .value_kind: by_value, .name: f_RS_offset }

    - { .size: 8, .offset: 112, .value_kind: hidden_none }
    - { .size: 8, .offset: 120, .value_kind: global_buffer, .name: bias_addr,   .address_space: global, .is_const: true }
    - { .size: 4, .offset: 128, .value_kind: by_value, .name: flags }

    - { .size: 1, .offset: 132, .value_kind: by_value, .name: activation_mode }
    - { .size: 1, .offset: 133, .value_kind: hidden_none }
    - { .size: 2, .offset: 134, .value_kind: hidden_none }

    - { .size: 4, .offset: 136, .value_kind: by_value, .name: alpha }
    - { .size: 4, .offset: 140, .value_kind: by_value, .name: beta }

    - { .size: 8, .offset: 144, .value_kind: by_value, .name: d_offset }
    - { .size: 8, .offset: 152, .value_kind: by_value, .name: o_offset }
    - { .size: 8, .offset: 160, .value_kind: by_value, .name: f_offset }
    - { .size: 8, .offset: 168, .value_kind: hidden_none }
    - { .size: 8, .offset: 176, .value_kind: by_value, .name: b_offset }

    - { .size: 8, .offset: 184, .value_kind: hidden_global_offset_x }
    - { .size: 8, .offset: 192, .value_kind: hidden_global_offset_y }
    - { .size: 8, .offset: 200, .value_kind: hidden_global_offset_z }
...
.end_amdgpu_metadata
.endm // METADATA

.altmacro
.macro METADATA_WRAPPER sc,wc,wg_x, kernel_name
    METADATA %\sc, %\wc, %\wg_x, \kernel_name
.endm

.macro kernel_end kernel_name
s_endpgm
.Lfunc_end0:
   .size \kernel_name, .Lfunc_end0 - \kernel_name
.endm

.macro EPILOG_KERNEL_DESCRIPTOR kernel_name

kernel_end \kernel_name

.if (.amdgcn.gfx_generation_number == 11)
    vgpr_size = 252
    workgroup_size_x = 384
.endif

.amdgcn.next_free_sgpr = 97
.amdgcn.next_free_vgpr = vgpr_size

//xnack disabled by default for asm kernels
__sgpr_reserve_vcc_default = 1
__sgpr_reserve_xnack_default = 0
__sgpr_reserve_flatscr_default = 0

__group_segment_fixed_size = 65536
__sgpr_dispatch_ptr = 1
__sgpr_kernarg_segment_ptr = 1
__sgpr_workgroup_id_x = 1
__sgpr_workgroup_id_y = 0
__sgpr_workgroup_id_z = 0
__vgpr_workitem_id = 0
__ieee_mode = 0
__dx10_clamp = 0

.rodata
.p2align 6
.if (.amdgcn.gfx_generation_number == 11)
.amdhsa_kernel \kernel_name
    .amdhsa_group_segment_fixed_size         __group_segment_fixed_size
    .amdhsa_user_sgpr_dispatch_ptr           __sgpr_dispatch_ptr // s[0:1]
    .amdhsa_user_sgpr_kernarg_segment_ptr    __sgpr_kernarg_segment_ptr // s[2:3]
    .amdhsa_system_sgpr_workgroup_id_x       __sgpr_workgroup_id_x
    .amdhsa_system_sgpr_workgroup_id_y       __sgpr_workgroup_id_y
    .amdhsa_system_sgpr_workgroup_id_z       __sgpr_workgroup_id_z
    .amdhsa_system_vgpr_workitem_id          __vgpr_workitem_id
    .amdhsa_next_free_vgpr                   .amdgcn.next_free_vgpr
    .amdhsa_next_free_sgpr                   .amdgcn.next_free_sgpr
    .amdhsa_reserve_vcc                      __sgpr_reserve_vcc_default
    .amdhsa_ieee_mode                        __ieee_mode
    .amdhsa_dx10_clamp                       __dx10_clamp
    .amdhsa_wavefront_size32                 0
.end_amdhsa_kernel
.endif

total_sgpr_count = .amdgcn.next_free_sgpr + 4 // vcc, xnack

METADATA_WRAPPER total_sgpr_count,.amdgcn.next_free_vgpr,workgroup_size_x, <\kernel_name>

.endm

.macro PROLOG_KERNEL_DESCRIPTOR_WRAPPER machine_version, kernel_name_postfix
    PROLOG_KERNEL_DESCRIPTOR miopenSp3AsmConv_fury_v1_1_1_gfx\machine_version\()_\kernel_name_postfix
.endm

.macro EPILOG_KERNEL_DESCRIPTOR_WRAPPER machine_version, kernel_name_postfix
    EPILOG_KERNEL_DESCRIPTOR miopenSp3AsmConv_fury_v1_1_1_gfx\machine_version\()_\kernel_name_postfix
.endm

.macro KERNEL_PROLOG kernel_name_postfix
	PROLOG_KERNEL_DESCRIPTOR_WRAPPER %.amdgcn.gfx_generation_number, \kernel_name_postfix
.endm

.macro KERNEL_EPILOG kernel_name_postfix
	EPILOG_KERNEL_DESCRIPTOR_WRAPPER %.amdgcn.gfx_generation_number, \kernel_name_postfix
.endm

.if (.amdgcn.gfx_generation_number != 11)
    .error "Unsupported gfx generation"
    .end
.endif
