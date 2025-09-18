/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

.macro PROLOG_KERNEL_DESCRIPTOR kernel_name:req
.text
.globl \kernel_name
.p2align 8
.type \kernel_name,@function
\kernel_name:
.endm

.macro METADATA sc:req, wc:req, wg_x:req, kernel_name:req
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: \kernel_name
    .symbol: \kernel_name\().kd
    .sgpr_count: \sc
    .vgpr_count: \wc
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_size: 232
    .kernarg_segment_align: 8
    .reqd_workgroup_size: [ \wg_x, 1, 1 ]
    .max_flat_workgroup_size: \wg_x
    .wavefront_size: 64
    .workgroup_processor_mode: 0
    .args:
    - { .size: 4, .offset:   0, .value_kind: by_value, .name: N_ }
    - { .size: 4, .offset:   4, .value_kind: by_value, .name: C }
    - { .size: 4, .offset:   8, .value_kind: by_value, .name: H }
    - { .size: 4, .offset:  12, .value_kind: by_value, .name: W }

    - { .size: 4, .offset:  16, .value_kind: by_value, .name: K }
    - { .size: 4, .offset:  20, .value_kind: by_value, .name: n_groups }

    - { .size: 8, .offset:  24, .value_kind: by_value, .name: flags64 }

    - { .size: 8, .offset:  32, .value_kind: global_buffer, .name: data_addr,   .address_space: global, .is_const: true }
    - { .size: 8, .offset:  40, .value_kind: global_buffer, .name: filter_addr, .address_space: global, .is_const: true }
    - { .size: 8, .offset:  48, .value_kind: global_buffer, .name: output_addr, .address_space: global, .is_const: false }
    - { .size: 8, .offset:  56, .value_kind: by_value }

    - { .size: 4, .offset:  64, .value_kind: by_value, .name: R }
    - { .size: 4, .offset:  68, .value_kind: by_value, .name: S }
    - { .size: 4, .offset:  72, .value_kind: by_value, .name: pad_h }
    - { .size: 4, .offset:  76, .value_kind: by_value, .name: pad_w }
    - { .size: 4, .offset:  80, .value_kind: by_value, .name: out_h }
    - { .size: 4, .offset:  84, .value_kind: by_value, .name: out_w }

    - { .size: 8, .offset:  88, .value_kind: global_buffer, .name: bias_addr, .address_space: global, .is_const: true }
    - { .size: 4, .offset:  96, .value_kind: by_value, .name: alpha }
    - { .size: 4, .offset: 100, .value_kind: by_value, .name: beta }

    - { .size: 8, .offset: 104, .value_kind: by_value, .name: d_offset }
    - { .size: 8, .offset: 112, .value_kind: by_value, .name: f_offset }
    - { .size: 8, .offset: 120, .value_kind: by_value, .name: o_offset }
    - { .size: 8, .offset: 128, .value_kind: by_value, .name: b_offset }

    - { .size: 4, .offset: 136, .value_kind: by_value, .name: d_N_stride }
    - { .size: 4, .offset: 140, .value_kind: by_value, .name: d_C_stride }
    - { .size: 4, .offset: 144, .value_kind: by_value, .name: d_H_stride }
    - { .size: 4, .offset: 148, .value_kind: by_value }

    - { .size: 4, .offset: 152, .value_kind: by_value, .name: f_K_stride }
    - { .size: 4, .offset: 156, .value_kind: by_value, .name: f_C_stride }
    - { .size: 4, .offset: 160, .value_kind: by_value, .name: f_R_stride }
    - { .size: 4, .offset: 164, .value_kind: by_value }

    - { .size: 4, .offset: 168, .value_kind: by_value, .name: o_N_stride }
    - { .size: 4, .offset: 172, .value_kind: by_value, .name: o_K_stride }
    - { .size: 4, .offset: 176, .value_kind: by_value, .name: o_H_stride }
    - { .size: 4, .offset: 180, .value_kind: by_value }

    - { .size: 4, .offset: 184, .value_kind: by_value, .name: G }
    - { .size: 4, .offset: 188, .value_kind: by_value, .name: d_G_stride }
    - { .size: 4, .offset: 192, .value_kind: by_value, .name: f_G_stride }
    - { .size: 4, .offset: 196, .value_kind: by_value, .name: o_G_stride }

    - { .size: 1, .offset: 200, .value_kind: by_value, .name: activation_mode }
    - { .size: 1, .offset: 201, .value_kind: by_value, .name: sync_limit }
    - { .size: 1, .offset: 202, .value_kind: by_value, .name: sync_period }
    - { .size: 1, .offset: 203, .value_kind: by_value }

    - { .size: 4, .offset: 204, .value_kind: by_value }
    - { .size: 8, .offset: 208, .value_kind: global_buffer, .name: sync_addr, .address_space: global, .is_const: false }

    - { .size: 8, .offset: 216, .value_kind: global_buffer, .name: acc_addr, .address_space: global, .is_const: false }
    - { .size: 8, .offset: 224, .value_kind: by_value, .name: a_offset }
...
.end_amdgpu_metadata
.endm // METADATA

.altmacro
.macro METADATA_WRAPPER sc:req, wc:req, wg_x:req, kernel_name:req
    METADATA %\sc, %\wc, %\wg_x, \kernel_name
.endm

.macro kernel_end kernel_name:req
.align 256
.Lfunc_end0:
   .size \kernel_name, .Lfunc_end0 - \kernel_name
.endm

.macro EPILOG_KERNEL_DESCRIPTOR kernel_name:req

kernel_end \kernel_name

.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor == 4 && .amdgcn.gfx_generation_stepping == 2)
    sgpr_cnt = 80
    vgpr_cnt = 164
    workgroup_size_x = 768
.endif

.amdgcn.next_free_sgpr = sgpr_cnt
.amdgcn.next_free_vgpr = vgpr_cnt

__group_segment_fixed_size = 65536
__sgpr_dispatch_ptr = 1
__sgpr_kernarg_segment_ptr = 1
__sgpr_workgroup_id_x = 1
__sgpr_workgroup_id_y = 0
__sgpr_workgroup_id_z = 0
__sgpr_workgroup_info = 0
__vgpr_workitem_id = 1
__reserve_vcc = 1
__float_round_mode_32 = 0
__float_round_mode_16_64 = 0
__float_denorm_mode_32 = 0
__float_denorm_mode_16_64 = 0
__dx10_clamp = 0
__ieee_mode = 0
__fp16_overflow = 0

.rodata
.p2align 6
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor == 4 && .amdgcn.gfx_generation_stepping == 2)
.amdhsa_kernel \kernel_name
    .amdhsa_group_segment_fixed_size         __group_segment_fixed_size
    .amdhsa_user_sgpr_dispatch_ptr           __sgpr_dispatch_ptr // s[0:1]
    .amdhsa_user_sgpr_kernarg_segment_ptr    __sgpr_kernarg_segment_ptr // s[2:3]
	.amdhsa_system_sgpr_workgroup_id_x       __sgpr_workgroup_id_x
	.amdhsa_system_sgpr_workgroup_id_y       __sgpr_workgroup_id_y
	.amdhsa_system_sgpr_workgroup_id_z       __sgpr_workgroup_id_z
	.amdhsa_system_sgpr_workgroup_info       __sgpr_workgroup_info
	.amdhsa_system_vgpr_workitem_id          __vgpr_workitem_id
	.amdhsa_reserve_vcc                      __reserve_vcc
    .amdhsa_next_free_sgpr                   .amdgcn.next_free_sgpr
    .amdhsa_next_free_vgpr                   .amdgcn.next_free_vgpr
	.amdhsa_accum_offset                     .amdgcn.next_free_vgpr
	.amdhsa_float_round_mode_32              __float_round_mode_32
	.amdhsa_float_round_mode_16_64           __float_round_mode_16_64
	.amdhsa_float_denorm_mode_32             __float_denorm_mode_32
	.amdhsa_float_denorm_mode_16_64          __float_denorm_mode_16_64
    .amdhsa_dx10_clamp                       __dx10_clamp
    .amdhsa_ieee_mode                        __ieee_mode
	.amdhsa_fp16_overflow                    __fp16_overflow
/* \todo add gfx12 support with these directives
    .amdhsa_wavefront_size32                 0
    .amdhsa_workgroup_processor_mode         0
*/
.end_amdhsa_kernel
.endif

METADATA_WRAPPER sgpr_cnt, vgpr_cnt, workgroup_size_x, <\kernel_name>

.endm

.macro PROLOG_KERNEL_DESCRIPTOR_WRAPPER machine_version:req, kernel_name_postfix:req
    PROLOG_KERNEL_DESCRIPTOR miopenSp3AsmConvRage_v4_6_0_gfx\machine_version\()\kernel_name_postfix
.endm

.macro EPILOG_KERNEL_DESCRIPTOR_WRAPPER machine_version:req, kernel_name_postfix:req
    EPILOG_KERNEL_DESCRIPTOR miopenSp3AsmConvRage_v4_6_0_gfx\machine_version\()\kernel_name_postfix
.endm

.macro KERNEL_PROLOG kernel_name_postfix:req
	PROLOG_KERNEL_DESCRIPTOR_WRAPPER %.amdgcn.gfx_generation_number, \kernel_name_postfix
.endm

.macro KERNEL_EPILOG kernel_name_postfix:req
	EPILOG_KERNEL_DESCRIPTOR_WRAPPER %.amdgcn.gfx_generation_number, \kernel_name_postfix
.endm

.if (.amdgcn.gfx_generation_number != 9 || .amdgcn.gfx_generation_minor != 4 && .amdgcn.gfx_generation_stepping != 2)
    .error "Unsupported gfx version"
    .end
.endif
