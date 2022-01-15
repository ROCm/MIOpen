/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
    .kernarg_segment_size: 248
    .kernarg_segment_align: 8
    .reqd_workgroup_size: [ \wg_x, 1, 1 ]
    .max_flat_workgroup_size: \wg_x
    .wavefront_size: 64
    .args:
    - { .size: 4, .offset:   0, .value_kind: by_value, .value_type: i32, .name: C }
    - { .size: 4, .offset:   4, .value_kind: by_value, .value_type: i32, .name: K }
    - { .size: 4, .offset:   8, .value_kind: by_value, .value_type: i32, .name: n_groups }
    - { .size: 4, .offset:  12, .value_kind: by_value, .value_type: i32, .name: n_works }
    - { .size: 4, .offset:  16, .value_kind: by_value, .value_type: i32, .name: d_C_pitch }
    - { .size: 4, .offset:  20, .value_kind: by_value, .value_type: i32, .name: d_H_pitch }
    - { .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32, .name: d_step_1_pitch }
    - { .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32, .name: d_step_2_pitch }
    - { .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32, .name: o_K_pitch }
    - { .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32, .name: o_H_pitch }
    - { .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32, .name: o_step_1_pitch }
    - { .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32, .name: o_step_2_pitch }
    - { .size: 8, .offset:  48, .value_kind: global_buffer, .value_type: f32, .name: data_addr,     .address_space: global, .is_const: true  }
    - { .size: 8, .offset:  56, .value_kind: global_buffer, .value_type: f32, .name: output_addr,   .address_space: global, .is_const: false }
    - { .size: 8, .offset:  64, .value_kind: global_buffer, .value_type: f32, .name: control_addr,  .address_space: global, .is_const: true  }
    - { .size: 8, .offset:  72, .value_kind: global_buffer, .value_type: f32, .name: filter_addr,   .address_space: global, .is_const: true  }
    - { .size: 8, .offset:  80, .value_kind: global_buffer, .value_type: f32, .name: bias_addr,     .address_space: global, .is_const: true  }
    - { .size: 4, .offset:  88, .value_kind: by_value, .value_type: f32, .name: RELU_alpha }
    - { .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32, .name: flags }
    - { .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32, .name: R }
    - { .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32, .name: S }
    - { .size: 8, .offset: 104, .value_kind: by_value, .value_type: i64, .name: d_offset }
    - { .size: 8, .offset: 112, .value_kind: by_value, .value_type: i64, .name: o_offset }
    - { .size: 8, .offset: 120, .value_kind: by_value, .value_type: i64, .name: с_offset }
    - { .size: 8, .offset: 128, .value_kind: by_value, .value_type: i64, .name: f_offset }
    - { .size: 8, .offset: 136, .value_kind: by_value, .value_type: i64, .name: b_offset }
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

.macro const_data buf_name, buf_sz
    .section    .AMDGPU.csdata

    .protected  \buf_name
    .type       \buf_name,@object

    .section    .bss,#alloc,#write

    .globl      \buf_name
	.p2align    4
\buf_name:
	.zero       \buf_sz
	.size       \buf_name, \buf_sz
.endm

.macro EPILOG_KERNEL_DESCRIPTOR kernel_name

kernel_end \kernel_name

.ifndef control_buf_sz
    .error "control_buf_sz undefined"
.endif
const_data control_buf, control_buf_sz

.if (.amdgcn.gfx_generation_number == 9)
    vgpr_size = 128
    workgroup_size_x = 512
.elseif (.amdgcn.gfx_generation_number == 10)
    vgpr_size = 256
    workgroup_size_x = 256
.endif

.amdgcn.next_free_sgpr = 101
.amdgcn.next_free_vgpr = vgpr_size

//xnack disabled by default for asm kernels
__sgpr_reserve_vcc_default = 1
__sgpr_reserve_xnack_default = 0
__sgpr_reserve_flatscr_default = 0

__group_segment_fixed_size = 65536
__sgpr_private_segment_buffer = 1
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
.if (.amdgcn.gfx_generation_number >= 10)
.amdhsa_kernel \kernel_name
    .amdhsa_group_segment_fixed_size         __group_segment_fixed_size
    .amdhsa_user_sgpr_private_segment_buffer __sgpr_private_segment_buffer
    .amdhsa_user_sgpr_dispatch_ptr           __sgpr_dispatch_ptr
    .amdhsa_user_sgpr_kernarg_segment_ptr    __sgpr_kernarg_segment_ptr
    .amdhsa_system_sgpr_workgroup_id_x       __sgpr_workgroup_id_x
    .amdhsa_system_sgpr_workgroup_id_y       __sgpr_workgroup_id_y
    .amdhsa_system_sgpr_workgroup_id_z       __sgpr_workgroup_id_y
    .amdhsa_system_vgpr_workitem_id          __vgpr_workitem_id
    .amdhsa_next_free_vgpr                   .amdgcn.next_free_vgpr
    .amdhsa_next_free_sgpr                   .amdgcn.next_free_sgpr
    .amdhsa_reserve_vcc                      __sgpr_reserve_vcc_default
    .amdhsa_reserve_xnack_mask               __sgpr_reserve_xnack_default
    .amdhsa_reserve_flat_scratch             __sgpr_reserve_flatscr_default
    .amdhsa_ieee_mode                        __ieee_mode
    .amdhsa_dx10_clamp                       __dx10_clamp
    .amdhsa_wavefront_size32                 0
.end_amdhsa_kernel
.else
.amdhsa_kernel \kernel_name
    .amdhsa_group_segment_fixed_size         __group_segment_fixed_size
    .amdhsa_user_sgpr_private_segment_buffer __sgpr_private_segment_buffer
    .amdhsa_user_sgpr_dispatch_ptr           __sgpr_dispatch_ptr
    .amdhsa_user_sgpr_kernarg_segment_ptr    __sgpr_kernarg_segment_ptr
    .amdhsa_system_sgpr_workgroup_id_x       __sgpr_workgroup_id_x
    .amdhsa_system_sgpr_workgroup_id_y       __sgpr_workgroup_id_y
    .amdhsa_system_sgpr_workgroup_id_z       __sgpr_workgroup_id_y
    .amdhsa_system_vgpr_workitem_id          __vgpr_workitem_id
    .amdhsa_next_free_vgpr                   .amdgcn.next_free_vgpr
    .amdhsa_next_free_sgpr                   .amdgcn.next_free_sgpr
    .amdhsa_reserve_vcc                      __sgpr_reserve_vcc_default
    .amdhsa_reserve_xnack_mask               __sgpr_reserve_xnack_default
    .amdhsa_reserve_flat_scratch             __sgpr_reserve_flatscr_default
    .amdhsa_ieee_mode                        __ieee_mode
    .amdhsa_dx10_clamp                       __dx10_clamp
.end_amdhsa_kernel
.endif

total_sgpr_count = .amdgcn.next_free_sgpr + 4 // vcc, xnack

METADATA_WRAPPER total_sgpr_count,.amdgcn.next_free_vgpr,workgroup_size_x, <\kernel_name>

.endm

.macro PROLOG_KERNEL_DESCRIPTOR_WRAPPER machine_version, kernel_name_postfix
    PROLOG_KERNEL_DESCRIPTOR miopenSp3AsmConv_Ultra_v1_1_3_gfx\machine_version\()_\kernel_name_postfix
.endm

.macro EPILOG_KERNEL_DESCRIPTOR_WRAPPER machine_version, kernel_name_postfix
    EPILOG_KERNEL_DESCRIPTOR miopenSp3AsmConv_Ultra_v1_1_3_gfx\machine_version\()_\kernel_name_postfix
.endm

.macro KERNEL_PROLOG kernel_name_postfix
	PROLOG_KERNEL_DESCRIPTOR_WRAPPER %.amdgcn.gfx_generation_number, \kernel_name_postfix
.endm

.macro KERNEL_EPILOG kernel_name_postfix
	EPILOG_KERNEL_DESCRIPTOR_WRAPPER %.amdgcn.gfx_generation_number, \kernel_name_postfix
.endm

.if (.amdgcn.gfx_generation_number != 10)
    .error "Unsupported gfx generation"
.endif
