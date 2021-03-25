/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
    - { .size: 4, .offset:   0, .value_kind: by_value, .value_type: i32, .name: N }
    - { .size: 4, .offset:   4, .value_kind: by_value, .value_type: i32, .name: C }
    - { .size: 4, .offset:   8, .value_kind: by_value, .value_type: i32, .name: H }
    - { .size: 4, .offset:  12, .value_kind: by_value, .value_type: i32, .name: W }
    - { .size: 4, .offset:  16, .value_kind: by_value, .value_type: i32, .name: K }
    - { .size: 4, .offset:  20, .value_kind: by_value, .value_type: i32, .name: n_groups }
    - { .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32, .name: flags }
    - { .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32, .name: reserved  }
    - { .size: 8, .offset:  32, .value_kind: global_buffer, .value_type: f32, .name: in,   .address_space: global, .is_const: true }
    - { .size: 8, .offset:  40, .value_kind: global_buffer, .value_type: f32, .name: weights,    .address_space: global, .is_const: true }
    - { .size: 8, .offset:  48, .value_kind: global_buffer, .value_type: f32, .name: out, .address_space: global, .is_const: false }
    - { .size: 8, .offset:  56, .value_kind: global_buffer, .value_type: f32, .name: rsv_ptr,     .address_space: global, .is_const: false }
    - { .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32, .name: R }
    - { .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32, .name: S }
    - { .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32, .name: pad_h }
    - { .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32, .name: pad_w }
    - { .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32, .name: out_h }
    - { .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32, .name: out_w }
    - { .size: 8, .offset:  88, .value_kind: global_buffer, .value_type: f32, .name: bias_addr,    .address_space: global, .is_const: true }
    - { .size: 4, .offset:  96, .value_kind: by_value, .value_type: f32, .name: RELU_alpha }
    - { .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32, .name: reserved2 }
    - { .size: 8, .offset: 104, .value_kind: by_value, .value_type: i64, .name: d_offset }
    - { .size: 8, .offset: 112, .value_kind: by_value, .value_type: i64, .name: f_offset }
    - { .size: 8, .offset: 120, .value_kind: by_value, .value_type: i64, .name: o_offset }
    - { .size: 8, .offset: 128, .value_kind: by_value, .value_type: i64, .name: b_offset }
    - { .size: 4, .offset: 136, .value_kind: by_value, .value_type: i32, .name: d_N_stride }
    - { .size: 4, .offset: 140, .value_kind: by_value, .value_type: i32, .name: d_C_stride }
    - { .size: 4, .offset: 144, .value_kind: by_value, .value_type: i32, .name: d_H_stride }
    - { .size: 4, .offset: 148, .value_kind: by_value, .value_type: i32, .name: d_W_stride }
    - { .size: 4, .offset: 152, .value_kind: by_value, .value_type: i32, .name: f_K_stride }
    - { .size: 4, .offset: 156, .value_kind: by_value, .value_type: i32, .name: f_C_stride }
    - { .size: 4, .offset: 160, .value_kind: by_value, .value_type: i32, .name: f_R_stride }
    - { .size: 4, .offset: 164, .value_kind: by_value, .value_type: i32, .name: f_S_stride }
    - { .size: 4, .offset: 168, .value_kind: by_value, .value_type: i32, .name: o_N_stride }
    - { .size: 4, .offset: 172, .value_kind: by_value, .value_type: i32, .name: o_K_stride }
    - { .size: 4, .offset: 176, .value_kind: by_value, .value_type: i32, .name: o_H_stride }
    - { .size: 4, .offset: 180, .value_kind: by_value, .value_type: i32, .name: o_W_stride }
    - { .size: 4, .offset: 184, .value_kind: by_value, .value_type: i32, .name: G }
    - { .size: 4, .offset: 188, .value_kind: by_value, .value_type: i32, .name: d_G_stride }
    - { .size: 4, .offset: 192, .value_kind: by_value, .value_type: i32, .name: f_G_stride }
    - { .size: 4, .offset: 196, .value_kind: by_value, .value_type: i32, .name: o_G_stride }
    - { .size: 8, .offset: 200, .value_kind: hidden_global_offset_x, .value_type: i64 }
    - { .size: 8, .offset: 208, .value_kind: hidden_global_offset_y, .value_type: i64 }
    - { .size: 8, .offset: 216, .value_kind: hidden_global_offset_z, .value_type: i64 }
    - { .size: 8, .offset: 224, .value_kind: hidden_none,   .value_type: i8 }
    - { .size: 8, .offset: 232, .value_kind: hidden_none,   .value_type: i8 }
    - { .size: 8, .offset: 240, .value_kind: hidden_none,   .value_type: i8 }
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
.elseif (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_stepping == 10)
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
    .amdhsa_accum_offset                    .amdgcn.next_free_vgpr
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
	PROLOG_KERNEL_DESCRIPTOR miopenSp3AsmConv_v21_1_2_gfx\machine_version\()_\kernel_name_postfix
.endm

.macro EPILOG_KERNEL_DESCRIPTOR_WRAPPER machine_version, kernel_name_postfix
	EPILOG_KERNEL_DESCRIPTOR miopenSp3AsmConv_v21_1_2_gfx\machine_version\()_\kernel_name_postfix
.endm

.macro KERNEL_PROLOG kernel_name_postfix
	PROLOG_KERNEL_DESCRIPTOR_WRAPPER %.amdgcn.gfx_generation_number, \kernel_name_postfix
.endm

.macro KERNEL_EPILOG kernel_name_postfix
	EPILOG_KERNEL_DESCRIPTOR_WRAPPER %.amdgcn.gfx_generation_number, \kernel_name_postfix
.endm

.if (.amdgcn.gfx_generation_number != 9 && .amdgcn.gfx_generation_number != 10)
    .error "Unsupported gfx generation"
    .end
.endif
