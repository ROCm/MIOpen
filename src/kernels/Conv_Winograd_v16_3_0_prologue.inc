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
.hsa_code_object_version 2, 1
.hsa_code_object_isa

.text
.globl sp3AsmConvRxSU
.p2align 8
.type sp3AsmConvRxSU,@function
.amdgpu_hsa_kernel sp3AsmConvRxSU

sp3AsmConvRxSU:
.amd_kernel_code_t
    is_ptr64 = 1
    enable_vgpr_workitem_id = 1 // v1 is not used at all for now, but what the heck.
    float_mode = 192
    enable_sgpr_private_segment_buffer = 1
    enable_sgpr_dispatch_ptr = 1
    enable_sgpr_kernarg_segment_ptr = 1
    granulated_wavefront_sgpr_count = 13
        // (arch <= gfx8)
        //      ? max(0, ceil(sgprs_used / 8) - 1)          // 105 -> 13
        //      : max(0, ceil(sgprs_used / 16) - 1) * 2     // 105 -> 12
        // Simplified:
        //      (wavefront_sgpr_count-1)/8                  // 105 -> 13
    granulated_workitem_vgpr_count = 31 // (workitem_vgpr_count-1)/4. Max=128 used, as per shader doc.
    compute_pgm_rsrc2_lds_size = 0
    compute_pgm_rsrc2_user_sgpr = 8
    compute_pgm_rsrc2_tidig_comp_cnt = 0
    compute_pgm_rsrc2_tgid_x_en = 1
    kernarg_segment_byte_size = 176 // From metadata provided after compilation of stub.
    workgroup_group_segment_byte_size = 65536
    wavefront_sgpr_count =  105 // 101 + 4 (vcc, xnack)
    workitem_vgpr_count = 128 // Max, as per shader doc.
    granulated_lds_size = 0
.end_amd_kernel_code_t

