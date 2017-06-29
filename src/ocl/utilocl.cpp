/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <cmath>
#include <miopen/kernel_cache.hpp>
#include <miopen/util.hpp>

namespace miopen {

float Im2ColGPU(Handle& handle,
                const int data_size,
                ConstData_t im,
                size_t im_offset,
                const int c,
                const int h,
                const int w,
                const int wei_h,
                const int wei_w,
                const int out_h,
                const int out_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                Data_t col)
{
    std::string program_name = "MIOpenUtilKernels.cl";
    std::string kernel_name  = "Im2Col";

    std::string params;
    int num_ch_per_wg;
    if((out_h <= 8 && out_w <= 8) && (stride_h == 1 && stride_w == 1) && (c % 4 == 0))
        num_ch_per_wg = 4;
    else
        num_ch_per_wg = 1;

    int tile_sz_x     = 32;
    int tile_sz_y     = 8;
    int num_blks_x    = std::ceil(static_cast<float>(out_w) / tile_sz_x);
    int num_blks      = num_blks_x * std::ceil(static_cast<float>(out_h) / tile_sz_y);
    int local_mem_sz  = (tile_sz_x * stride_w + wei_w) * (tile_sz_y * stride_h + wei_h);
    int data_size_off = data_size - im_offset;

    params += " -DNUM_CH_PER_WG=" + std::to_string(num_ch_per_wg);
    params += " -DNUM_IM_BLKS_X=" + std::to_string(num_blks_x);
    params += " -DNUM_IM_BLKS=" + std::to_string(num_blks);
    params += " -DLOCAL_MEM_SIZE=" + std::to_string(local_mem_sz);
    params += " -DSTRIDE_GT_1=" + std::to_string(stride_h * stride_w > 1);
    params += " -DTILE_SZ_X=" + std::to_string(tile_sz_x);
    params += " -DTILE_SZ_Y=" + std::to_string(tile_sz_y);
    params += " -DUSE_IM_OFF_GUARD=1";

    const std::vector<size_t> vld{256, 1, 1};
    size_t global_threads = 256 * std::max(1, (c / num_ch_per_wg)) * num_blks;
    const std::vector<size_t> vgd{global_threads, 1, 1};

    handle.GetKernel("miopenIm2Col", "", program_name, kernel_name, vld, vgd, params)(data_size_off,
                                                                                      im,
                                                                                      im_offset,
                                                                                      h,
                                                                                      w,
                                                                                      wei_h,
                                                                                      wei_w,
                                                                                      out_h,
                                                                                      out_w,
                                                                                      pad_h,
                                                                                      pad_w,
                                                                                      stride_h,
                                                                                      stride_w,
                                                                                      col);

    return handle.GetKernelTime();
}

float Col2ImGPU(Handle& handle,
                ConstData_t col,
                const int col_h,
                const int col_w,
                const int wei_h,
                const int wei_w,
                const int pad_h,
                const int pad_w,
                const int stride_h,
                const int stride_w,
                const int c,
                const int h,
                const int w,
                Data_t im,
                size_t im_offset)
{
    std::string program_name = "MIOpenUtilKernels2.cl";
    std::string kernel_name  = "Col2Im";

    std::string params;
    const std::vector<size_t> vld{256, 1, 1};
    size_t global_threads = c * h * w;
    const std::vector<size_t> vgd{global_threads, 1, 1};

    handle.GetKernel("miopenCol2Im", "", program_name, kernel_name, vld, vgd, params)(
        col, col_h, col_w, wei_h, wei_w, pad_h, pad_w, stride_h, stride_w, h, w, im, im_offset);

    return handle.GetKernelTime();
}

} // namespace miopen
