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

.amd_kernel_code_t
    enable_sgpr_kernarg_segment_ptr = 1
    compute_pgm_rsrc2_tgid_x_en = 1
    is_ptr64 = 1
    compute_pgm_rsrc1_vgprs = .AUTO_VGPR_GRANULATED_COUNT
    compute_pgm_rsrc1_sgprs = .AUTO_SGPR_GRANULATED_COUNT
    compute_pgm_rsrc2_tidig_comp_cnt = 0
    compute_pgm_rsrc2_user_sgpr = 2
    kernarg_segment_byte_size = KERNEL_ARGUMENTS_SIZE
    wavefront_sgpr_count = .AUTO_SGPR_COUNT
    workitem_vgpr_count = .AUTO_VGPR_COUNT
    float_mode = 192
    workgroup_group_segment_byte_size = .AUTO_LDS_BYTE_SIZE
.end_amd_kernel_code_t
