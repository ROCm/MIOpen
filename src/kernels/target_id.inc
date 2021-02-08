/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

////////////////////////////////
// THE FOLLOWING SYMBOLS MUST BE DEFINED PRIOR USING THE MACROS DECLARED BELOW:
// * amd_target_feature_xnack
// * amd_target_feature_sramecc
// Allowed values: 
// * 0 - The feature is OFF on the target device
// * 1 - The feature is ON
// * 2 - The target device does not have the feature OR the kernel does not depend on its state.
////////////////////////////////
/// Using:  1) include this file
///			2) Call METADATA_final macro
///					Arguments: 	1)kernel_name 
///								2)sgpr_count
///								3)vgpr_count
///								4)kernarg_size
///								4)lds_size
///								5)reqd_workgroup_size_x
///								6)reqd_workgroup_size_y
///								7)reqd_workgroup_size_z
///								8)max_flat_workgroup_size
///								9)wavefront_size=64
///								10)kernel_args
//macro for send_TARGET_ID_str generation 

.ifnotdef amd_target_feature_xnack
.error "Undefined amd_target_feature_xnack"
.end
.endif

.ifnotdef amd_target_feature_sramecc
.error "Undefined amd_target_feature_sramecc"
.end
.endif

.macro create_TARGET_ID_with_proc proc

.macro create_TARGET_ID_with_xnack xnack_str

.macro create_TARGET_ID_with_sramecc sramecc

.macro send_TARGET_ID_str next_m
    \next_m amdgcn-amd-amdhsa--\()\proc\xnack_str\sramecc
.endm //send_TARGET_ID_str

.endm //create_TARGET_ID_with_sramecc
.endm //create_TARGET_ID_with_xnack
.endm //create_TARGET_ID_with_proc

.macro create_TARGET_ID_xnack_status
    .if(amd_target_feature_xnack == 0)
        create_TARGET_ID_with_xnack ":xnack-"
    .elseif (amd_target_feature_xnack == 1)
        create_TARGET_ID_with_xnack ":xnack+"
    .else
        create_TARGET_ID_with_xnack ""
    .endif
.endm //create_TARGET_ID_xnack_status

.macro create_TARGET_ID_sramecc_status 
    .if(amd_target_feature_sramecc == 0)
        create_TARGET_ID_with_sramecc ":sramecc-"
    .elseif (amd_target_feature_sramecc == 1)
        create_TARGET_ID_with_sramecc ":sramecc+"
    .else
        create_TARGET_ID_with_sramecc ""
    .endif
.endm //create_TARGET_ID_sramecc_status

.macro create_TARGET_ID_proc major, minor, stepping
    create_TARGET_ID_with_proc "gfx\major\minor\stepping"
.endm //create_TARGET_ID_proc

//macro call
create_TARGET_ID_proc %.amdgcn.gfx_generation_number, %.amdgcn.gfx_generation_minor, %.amdgcn.gfx_generation_stepping

create_TARGET_ID_xnack_status

create_TARGET_ID_sramecc_status

//-- METADATA_final generator
.altmacro
.macro create_METADATA_generator_for_target target, version="[1,1]"

.macro METADATA_final kernel_name, sc, vc, kernarg_size, lds_size, wg_x, wg_y, wg_z, max_wg_f,wf_size=64, kernel_args
.amdgpu_metadata
---
amdhsa.version: \version
amdhsa.target: \target
amdhsa.kernels:
  - .name: \kernel_name
    .symbol: \kernel_name\\()().kd
    .sgpr_count: \sc
    .vgpr_count: \vc
    .language: "OpenCL C"
    .language_version: [ 1, 2 ]
    .kernarg_segment_size: \kernarg_size
    .kernarg_segment_align: 8
    .group_segment_fixed_size: \lds_size
    .private_segment_fixed_size: 0
    .reqd_workgroup_size: [ \wg_x, \wg_y, \wg_z ]
    .max_flat_workgroup_size: \max_wg_f
    .wavefront_size: \wf_size
    .args:
    \kernel_args
...
.end_amdgpu_metadata
.endm //METADATA_final
.endm //create_METADATA_generator_for_target
//--

send_TARGET_ID_str create_METADATA_generator_for_target

////////////////////////////////
/// macro for igemm
////////////////////////////////

.macro igemm_macro_generator_with_version_and_target target
.macro set_amdhsa_version_and_target_for_igemm next_m
.ifnotdef ROCM_METADATA_VERSION
    .error "Undefined ROCM_METADATA_VERSION"
    .end
.endif
.if(ROCM_METADATA_VERSION == 6)
\next_m "amdhsa.version: [1,1]
amdhsa.target: \target"
.elseif (ROCM_METADATA_VERSION == 5)
\next_m "amdhsa.version: [1,0]"
.endif
.endm // set_amdhsa_version_and_target_for_igemm
.endm // igemm_macro_generator_with_version_and_target 


send_TARGET_ID_str igemm_macro_generator_with_version_and_target

////////////////////////////////
