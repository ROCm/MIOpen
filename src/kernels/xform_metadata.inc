/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

.if ROCM_METADATA_VERSION == 5

.macro KERNEL_DESCRIPTOR_COV3 kernel_name
.rodata
.p2align 6
.amdhsa_kernel \kernel_name
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 0
        .amdhsa_system_sgpr_workgroup_id_z 0
        .amdhsa_system_vgpr_workitem_id 0
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
        .amdhsa_next_free_vgpr .AUTO_VGPR_COUNT
        .amdhsa_group_segment_fixed_size .AUTO_LDS_BYTE_SIZE
        .amdhsa_dx10_clamp 0
        .amdhsa_ieee_mode 0
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_reserve_flat_scratch __sgpr_reserve_flatscr
        .amdhsa_reserve_xnack_mask __sgpr_reserve_xnack
        .amdhsa_reserve_vcc __sgpr_reserve_vcc
.end_amdhsa_kernel
.endm

.altmacro
.macro METADATA sc, vc, wg_x, lds_size, kernarg_size, kernel_name
.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: \kernel_name
    .symbol: \kernel_name.kd
    .sgpr_count: \sc
    .vgpr_count: \vc
    .language: "OpenCL C"
    .language_version: [ 1, 2 ]
    .kernarg_segment_size: \kernarg_size
    .kernarg_segment_align: 8
    .group_segment_fixed_size: \lds_size
    .private_segment_fixed_size: 0
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
    - { .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32, .name: unused_1 }
    - { .size: 8, .offset:  32, .value_kind: global_buffer, .value_type: f32, .name: filter_ptr,   .address_space: global, .is_const: false }
    - { .size: 8, .offset:  40, .value_kind: global_buffer, .value_type: f32, .name: reserved2,    .address_space: global, .is_const: false }
    - { .size: 8, .offset:  48, .value_kind: global_buffer, .value_type: f32, .name: x_filter_ptr, .address_space: global, .is_const: false }
    - { .size: 8, .offset:  56, .value_kind: global_buffer, .value_type: f32, .name: ret_addr,     .address_space: global, .is_const: false }
    - { .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32, .name: R }
    - { .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32, .name: S }
    - { .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32, .name: pad_h }
    - { .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32, .name: pad_w }
    - { .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32, .name: out_h }
    - { .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32, .name: out_w }
    - { .size: 8, .offset:  88, .value_kind: global_buffer, .value_type: f32, .name: bias_addr,    .address_space: global, .is_const: true }
    - { .size: 4, .offset:  96, .value_kind: by_value, .value_type: f32, .name: RELU_alpha }
    - { .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32, .name: d_N_stride }
    - { .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32, .name: d_C_stride }
    - { .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32, .name: d_H_stride }
    - { .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32, .name: d_W_stride }
    - { .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32, .name: f_K_stride }
    - { .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32, .name: f_C_stride }
    - { .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32, .name: f_R_stride }
    - { .size: 4, .offset: 128, .value_kind: by_value, .value_type: i32, .name: f_S_stride }
    - { .size: 4, .offset: 132, .value_kind: by_value, .value_type: i32, .name: o_N_stride }
    - { .size: 4, .offset: 136, .value_kind: by_value, .value_type: i32, .name: o_K_stride }
    - { .size: 4, .offset: 140, .value_kind: by_value, .value_type: i32, .name: o_H_stride }
    - { .size: 4, .offset: 144, .value_kind: by_value, .value_type: i32, .name: o_W_stride }
...
.end_amdgpu_metadata
.endm // METADATA

.elseif ROCM_METADATA_VERSION == 4

.macro KERNEL_DESCRIPTOR_COV3 kernel_name
// V3 KD is not used here.
// V2 KD is included from "xform_kd_cov2.inc" where appropriate.
.endm

.macro METADATA sc, vc, wg_x, lds_size, kernarg_size, kernel_name
    .amd_amdgpu_hsa_metadata
        { Version: [ 1, 0 ],
            Kernels:
            - { Name: \kernel_name, SymbolName: \kernel_name@kd, Language: OpenCL C, LanguageVersion: [ 1, 2 ],
                Attrs:
                    { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
                CodeProps:
                    { KernargSegmentSize: \kernarg_size, GroupSegmentFixedSize: \lds_size, PrivateSegmentFixedSize: 0, KernargSegmentAlign: 8, WavefrontSize: 64, MaxFlatWorkGroupSize: \wg_x }
                Args:
                - { Name: N       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: C       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: H       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: W       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: K       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: n_groups, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: flags   , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: unused_1, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: filter_ptr      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default}
                - { Name: reserved2       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default}
                - { Name: x_filter_ptr    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default}
                - { Name: ret_addr        , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*'  , AddrSpaceQual: Global, AccQual: Default }
                - { Name: R    , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: S    , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: pad_h, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: pad_w, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: out_h, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: out_w, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: bias_addr       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
                - { Name: RELU_alpha      , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float', AccQual: Default, IsConst: true }
                - { Name: d_N_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: d_C_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: d_H_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: d_W_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: reserved5       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: reserved6       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: reserved7       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: reserved8       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: o_N_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: o_C_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: o_H_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                - { Name: o_W_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                }
        }
    .end_amd_amdgpu_hsa_metadata
.endm
.endif
