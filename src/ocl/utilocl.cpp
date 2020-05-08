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
#include <miopen/logger.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/datatype.hpp>

#include <boost/range/adaptors.hpp>

#define WG_SIZE 256
#define MAX_ACTIVE_THREADS (64 * 4 * 64)
#define MAX_LOCAL_MEM 65536

namespace miopen {

float Im2d2ColGPU(Handle& handle,
                  ConstData_t im,
                  const int im_offset,
                  const int c,
                  const int in_h,
                  const int in_w,
                  const int wei_h,
                  const int wei_w,
                  const int out_h,
                  const int out_w,
                  const int pad_h,
                  const int pad_w,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_h,
                  const int dilation_w,
                  Data_t col,
                  miopenDataType_t type)
{
    std::string program_name = "MIOpenIm2d2Col.cl";
    std::string kernel_name  = "Im2d2Col";

    // clang-format off
    std::string network_config =
        "c" + std::to_string(c) +
        "i" + std::to_string(in_h) +
        "_" + std::to_string(in_w) +
        "w" + std::to_string(wei_h) +
        "_" + std::to_string(wei_w) +
        "p" + std::to_string(pad_h) +
        "_" + std::to_string(pad_w) +
        "s" + std::to_string(stride_h) +
        "_" + std::to_string(stride_w) +
        "d" + std::to_string(dilation_h) +
        "_" + std::to_string(dilation_w) +
        "t" + std::to_string(type);
    // clang-format on

    auto&& kernels = handle.GetKernels("miopenIm2d2Col", network_config);

    int data_size_bound = c * in_h * in_w;

    int data_size_bound_pack = type == miopenInt8x4 ? data_size_bound * 4 : data_size_bound;
    int im_offset_pack       = type == miopenInt8x4 ? im_offset / 4 : im_offset;

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(data_size_bound_pack,
               im,
               im_offset_pack,
               in_h,
               in_w,
               wei_h,
               wei_w,
               out_h,
               out_w,
               pad_h,
               pad_w,
               stride_h,
               stride_w,
               dilation_h,
               dilation_w,
               col);
    }
    else
    {
        const int c_pack = type == miopenInt8x4 ? c / 4 : c;

        std::string params;
        int num_ch_per_wg;
        if((out_h <= 8 && out_w <= 8) && (stride_h == 1 && stride_w == 1) && (c_pack % 4 == 0))
            num_ch_per_wg = 4;
        else
            num_ch_per_wg = 1;

        int tile_sz_x  = 32;
        int tile_sz_y  = 8;
        int num_blks_x = std::ceil(static_cast<float>(out_w) / static_cast<float>(tile_sz_x));
        int num_blks =
            num_blks_x *
            static_cast<int>(std::ceil(static_cast<float>(out_h) / static_cast<float>(tile_sz_y)));
        int local_mem_sz;
        if(num_ch_per_wg == 1)
        {
            local_mem_sz = ((tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1) *
                           ((tile_sz_y - 1) * stride_h + (wei_h - 1) * dilation_h + 1);
        }
        else
        {
            auto uprdTileX = static_cast<int>(
                std::ceil(static_cast<float>(tile_sz_x) / static_cast<float>(num_ch_per_wg)) - 1);
            auto uprdTileY = static_cast<int>(
                std::ceil(static_cast<float>(tile_sz_y) / static_cast<float>(num_ch_per_wg)) - 1);
            auto memXsize = (num_ch_per_wg * uprdTileX * stride_w + (wei_w - 1) * dilation_w + 1) *
                            ((tile_sz_y - 1) * stride_h + (wei_h - 1) * dilation_h + 1);
            auto memYsize = num_ch_per_wg *
                            ((tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1) *
                            (uprdTileY * stride_h + (wei_h - 1) * dilation_h + 1);
            local_mem_sz = static_cast<int>(std::max(memXsize, memYsize));
        }

        // adjust mapping for large kernel
        int type_size    = 4; // Need to adjust for fp16, int8
        int extreme_case = num_ch_per_wg * ((wei_w - 1) * dilation_w + 1) *
                           ((wei_h - 1) * dilation_h + 1) * type_size;
        if(extreme_case > MAX_LOCAL_MEM)
        {
            params += " -DEXTREME_LARGE";
            params += " -DNUM_CH_TOTAL=" + std::to_string(c_pack);
        }
        else
        {
            while(local_mem_sz * type_size > MAX_LOCAL_MEM)
            {
                tile_sz_x  = tile_sz_x == 1 ? 1 : (tile_sz_y == 1 ? (tile_sz_x / 2) : tile_sz_x);
                tile_sz_y  = tile_sz_y == 1 ? 1 : (tile_sz_y / 2);
                num_blks_x = std::ceil(static_cast<float>(out_w) / static_cast<float>(tile_sz_x));
                num_blks   = num_blks_x * static_cast<int>(std::ceil(static_cast<float>(out_h) /
                                                                   static_cast<float>(tile_sz_y)));
                if(num_ch_per_wg == 1)
                {
                    local_mem_sz = ((tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1) *
                                   ((tile_sz_y - 1) * stride_h + (wei_h - 1) * dilation_h + 1);
                }
                else
                {
                    auto uprdTileX = static_cast<int>(std::ceil(static_cast<float>(tile_sz_x) /
                                                                static_cast<float>(num_ch_per_wg)) -
                                                      1);
                    auto uprdTileY = static_cast<int>(std::ceil(static_cast<float>(tile_sz_y) /
                                                                static_cast<float>(num_ch_per_wg)) -
                                                      1);
                    auto memXsize =
                        (num_ch_per_wg * uprdTileX * stride_w + (wei_w - 1) * dilation_w + 1) *
                        ((tile_sz_y - 1) * stride_h + (wei_h - 1) * dilation_h + 1);
                    auto memYsize = num_ch_per_wg *
                                    ((tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1) *
                                    (uprdTileY * stride_h + (wei_h - 1) * dilation_h + 1);
                    local_mem_sz = static_cast<int>(std::max(memXsize, memYsize));
                }
            }
        }

        params += " -DNUM_CH_PER_WG=" + std::to_string(num_ch_per_wg);
        params += " -DNUM_IM_BLKS_X=" + std::to_string(num_blks_x);
        params += " -DNUM_IM_BLKS=" + std::to_string(num_blks);
        params += " -DLOCAL_MEM_SIZE=" + std::to_string(local_mem_sz);
        params += " -DSTRIDE_GT_1=" + std::to_string(static_cast<int>(stride_h * stride_w > 1));
        params += " -DTILE_SZ_X=" + std::to_string(tile_sz_x);
        params += " -DTILE_SZ_Y=" + std::to_string(tile_sz_y);
        params += " -DUSE_IM_OFF_GUARD=1";

        params += GetDataTypeKernelParams(type);

        const std::vector<size_t> vld{256, 1, 1};
        size_t global_threads = 256 * std::max(1, (c_pack / num_ch_per_wg)) * num_blks;
        const std::vector<size_t> vgd{global_threads, 1, 1};
        handle.AddKernel(
            "miopenIm2Col", network_config, program_name, kernel_name, vld, vgd, params)(
            data_size_bound_pack,
            im,
            im_offset_pack,
            in_h,
            in_w,
            wei_h,
            wei_w,
            out_h,
            out_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            col);
    }

    return handle.GetKernelTime();
}

float Im3d2ColGPU(Handle& handle,
                  ConstData_t im,
                  const int im_offset,
                  const int im_c,
                  const int im_d,
                  const int im_h,
                  const int im_w,
                  const int wei_d,
                  const int wei_h,
                  const int wei_w,
                  const int out_d,
                  const int out_h,
                  const int out_w,
                  const int pad_d,
                  const int pad_h,
                  const int pad_w,
                  const int stride_d,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_d,
                  const int dilation_h,
                  const int dilation_w,
                  Data_t col,
                  miopenDataType_t type)
{
    std::string program_name = "MIOpenIm3d2Col.cl";
    std::string kernel_name  = "Im3d2Col";

    // clang-format off
    std::string network_config =
        "c" + std::to_string(im_c) +
        "i" + std::to_string(im_d) +
        "_" + std::to_string(im_h) +
        "_" + std::to_string(im_w) +
        "w" + std::to_string(wei_d) +
        "_" + std::to_string(wei_h) +
        "_" + std::to_string(wei_w) +
        "p" + std::to_string(pad_d) +
        "_" + std::to_string(pad_h) +
        "_" + std::to_string(pad_w) +
        "s" + std::to_string(stride_d) +
        "_" + std::to_string(stride_h) +
        "_" + std::to_string(stride_w) +
        "d" + std::to_string(dilation_d) +
        "_" + std::to_string(dilation_h) +
        "_" + std::to_string(dilation_w) +
        "t" + std::to_string(type);
    // clang-format on

    auto&& kernels = handle.GetKernels("miopenIm3d2Col", network_config);

    // int8x4 vectorize-c format
    int im_offset_pack = type == miopenInt8x4 ? im_offset / 4 : im_offset;
    int im_c_pack      = type == miopenInt8x4 ? im_c / 4 : im_c;

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(im,
               im_offset_pack,
               im_c_pack,
               im_d,
               im_h,
               im_w,
               wei_d,
               wei_h,
               wei_w,
               out_d,
               out_h,
               out_w,
               pad_d,
               pad_h,
               pad_w,
               stride_d,
               stride_h,
               stride_w,
               dilation_d,
               dilation_h,
               dilation_w,
               col);
    }
    else
    {
        std::string params = GetDataTypeKernelParams(type);

        const std::vector<size_t> vld{256, 1, 1};
        size_t global_threads = std::min(
            256 * static_cast<std::size_t>(out_d * out_h * out_w * im_c * wei_d * wei_h * wei_w) /
                8,
            static_cast<std::size_t>(256) * 1024);
        const std::vector<size_t> vgd{global_threads, 1, 1};

        handle.AddKernel(
            "miopenIm3d2Col", network_config, program_name, kernel_name, vld, vgd, params)(
            im,
            im_offset_pack,
            im_c_pack,
            im_d,
            im_h,
            im_w,
            wei_d,
            wei_h,
            wei_w,
            out_d,
            out_h,
            out_w,
            pad_d,
            pad_h,
            pad_w,
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
            dilation_h,
            dilation_w,
            col);
    }

    return handle.GetKernelTime();
}

float Col2Im2dGPU(Handle& handle,
                  ConstData_t col,
                  const int out_h,
                  const int out_w,
                  const int wei_h,
                  const int wei_w,
                  const int pad_h,
                  const int pad_w,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_h,
                  const int dilation_w,
                  const int in_c,
                  const int in_h,
                  const int in_w,
                  Data_t im,
                  int im_offset,
                  miopenDataType_t type)
{
    std::string program_name = "MIOpenCol2Im2d.cl";
    std::string kernel_name  = "Col2Im2d";

    // clang-format off
    std::string network_config =
        "c" + std::to_string(in_c) +
        "in_h" + std::to_string(in_h) +
        "in_w" + std::to_string(in_w) +
        "y" + std::to_string(wei_h) +
        "x" + std::to_string(wei_w) +
        "p" + std::to_string(pad_h) +
        "q" + std::to_string(pad_w) +
        "u" + std::to_string(stride_h) +
        "v" + std::to_string(stride_w) +
        "l" + std::to_string(dilation_h) +
        "j" + std::to_string(dilation_w) +
        "t" + std::to_string(type);
    // clang-format on

    auto&& kernels = handle.GetKernels("miopenCol2Im2d", network_config);

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(col,
               out_h,
               out_w,
               wei_h,
               wei_w,
               pad_h,
               pad_w,
               stride_h,
               stride_w,
               dilation_h,
               dilation_w,
               in_h,
               in_w,
               im,
               im_offset);
    }
    else
    {
        std::string params = GetDataTypeKernelParams(type);

        const std::vector<size_t> vld{256, 1, 1};
        size_t global_threads = in_c * in_h * in_w;
        const std::vector<size_t> vgd{global_threads, 1, 1};

        handle.AddKernel(
            "miopenCol2Im2d", network_config, program_name, kernel_name, vld, vgd, params)(
            col,
            out_h,
            out_w,
            wei_h,
            wei_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            in_h,
            in_w,
            im,
            im_offset);
    }
    return handle.GetKernelTime();
}

float Col2Im3dGPU(Handle& handle,
                  ConstData_t col,
                  const int out_d,
                  const int out_h,
                  const int out_w,
                  const int wei_d,
                  const int wei_h,
                  const int wei_w,
                  const int pad_d,
                  const int pad_h,
                  const int pad_w,
                  const int stride_d,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_d,
                  const int dilation_h,
                  const int dilation_w,
                  const int in_c,
                  const int in_d,
                  const int in_h,
                  const int in_w,
                  Data_t im,
                  int im_offset,
                  miopenDataType_t type)
{
    std::string program_name = "MIOpenCol2Im3d.cl";
    std::string kernel_name  = "Col2Im3d";

    // clang-format off
    std::string network_config =
        "c" + std::to_string(in_c) +
        "i" + std::to_string(in_d) +
        "_" + std::to_string(in_h) +
        "_" + std::to_string(in_w) +
        "w" + std::to_string(wei_d) +
        "_" + std::to_string(wei_h) +
        "_" + std::to_string(wei_w) +
        "p" + std::to_string(pad_d) +
        "_" + std::to_string(pad_h) +
        "_" + std::to_string(pad_w) +
        "s" + std::to_string(stride_d) +
        "_" + std::to_string(stride_h) +
        "_" + std::to_string(stride_w) +
        "d" + std::to_string(dilation_d) +
        "_" + std::to_string(dilation_h) +
        "_" + std::to_string(dilation_w) +
        "t" + std::to_string(type);
    // clang-format on

    auto&& kernels = handle.GetKernels("miopenCol2Im3d", network_config);

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(col,
               out_d,
               out_h,
               out_w,
               wei_d,
               wei_h,
               wei_w,
               pad_d,
               pad_h,
               pad_w,
               stride_d,
               stride_h,
               stride_w,
               dilation_d,
               dilation_h,
               dilation_w,
               in_d,
               in_h,
               in_w,
               im,
               im_offset);
    }
    else
    {
        std::string params = GetDataTypeKernelParams(type);

        const std::vector<size_t> vld{256, 1, 1};
        size_t global_threads = in_c * in_d * in_h * in_w;
        const std::vector<size_t> vgd{global_threads, 1, 1};

        handle.AddKernel(
            "miopenCol2Im3d", network_config, program_name, kernel_name, vld, vgd, params)(
            col,
            out_d,
            out_h,
            out_w,
            wei_d,
            wei_h,
            wei_w,
            pad_d,
            pad_h,
            pad_w,
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
            dilation_h,
            dilation_w,
            in_d,
            in_h,
            in_w,
            im,
            im_offset);
    }
    return handle.GetKernelTime();
}

float Im2ColGPU(
    Handle& handle,
    std::size_t spatial_dim,
    ConstData_t im,
    std::size_t im_offset,
    std::size_t in_c,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& in_spatial,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& wei_spatial,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& out_spatial,
    const std::vector<int>& pad_spatial,
    const std::vector<int>& stride_spatial,
    const std::vector<int>& dilation_spatial,
    Data_t col,
    miopenDataType_t type)
{
    switch(spatial_dim)
    {
    case 2:
    {
        return Im2d2ColGPU(handle,
                           im,
                           im_offset,
                           in_c,
                           in_spatial[0],
                           in_spatial[1],
                           wei_spatial[0],
                           wei_spatial[1],
                           out_spatial[0],
                           out_spatial[1],
                           pad_spatial[0],
                           pad_spatial[1],
                           stride_spatial[0],
                           stride_spatial[1],
                           dilation_spatial[0],
                           dilation_spatial[1],
                           col,
                           type);
    }
    case 3:
    {
        return Im3d2ColGPU(handle,
                           im,
                           im_offset,
                           in_c,
                           in_spatial[0],
                           in_spatial[1],
                           in_spatial[2],
                           wei_spatial[0],
                           wei_spatial[1],
                           wei_spatial[2],
                           out_spatial[0],
                           out_spatial[1],
                           out_spatial[2],
                           pad_spatial[0],
                           pad_spatial[1],
                           pad_spatial[2],
                           stride_spatial[0],
                           stride_spatial[1],
                           stride_spatial[2],
                           dilation_spatial[0],
                           dilation_spatial[1],
                           dilation_spatial[2],
                           col,
                           type);
    }
    default: { MIOPEN_THROW("unsupported convolution dimension");
    }
    }
}

float Col2ImGPU(
    Handle& handle,
    std::size_t spatial_dim,
    ConstData_t col,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& out_spatial,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& wei_spatial,
    const std::vector<int>& pad_spatial,
    const std::vector<int>& stride_spatial,
    const std::vector<int>& dilation_spatial,
    std::size_t in_c,
    const decltype(boost::adaptors::slice(std::vector<std::size_t>(), 0, 1))& in_spatial,
    Data_t im,
    std::size_t im_offset,
    miopenDataType_t type)
{
    switch(spatial_dim)
    {
    case 2:
    {
        return Col2Im2dGPU(handle,
                           col,
                           out_spatial[0],
                           out_spatial[1],
                           wei_spatial[0],
                           wei_spatial[1],
                           pad_spatial[0],
                           pad_spatial[1],
                           stride_spatial[0],
                           stride_spatial[1],
                           dilation_spatial[0],
                           dilation_spatial[1],
                           in_c,
                           in_spatial[0],
                           in_spatial[1],
                           im,
                           im_offset,
                           type);
    }
    case 3:
    {
        return Col2Im3dGPU(handle,
                           col,
                           out_spatial[0],
                           out_spatial[1],
                           out_spatial[2],
                           wei_spatial[0],
                           wei_spatial[1],
                           wei_spatial[2],
                           pad_spatial[0],
                           pad_spatial[1],
                           pad_spatial[2],
                           stride_spatial[0],
                           stride_spatial[1],
                           stride_spatial[2],
                           dilation_spatial[0],
                           dilation_spatial[1],
                           dilation_spatial[2],
                           in_c,
                           in_spatial[0],
                           in_spatial[1],
                           in_spatial[2],
                           im,
                           im_offset,
                           type);
    }
    default: { MIOPEN_THROW("unsupported convolution dimension");
    }
    }

    MIOPEN_THROW("unsupported convolution dimension");
}

float transpose_NCHW2CNHW(Handle& handle,
                          int n,
                          int c,
                          int h_in,
                          int w_in,
                          int h_out,
                          int w_out,
                          ConstData_t in,
                          Data_t out,
                          int in_offset,
                          int out_offset,
                          int h_stride,
                          int w_stride,
                          miopenDataType_t type)
{

    std::string program_name = "MIOpenUtilKernels4.cl";

    std::string network_config = "n" + std::to_string(n) + "c" + std::to_string(c) + "h" +
                                 std::to_string(h_in) + "w" + std::to_string(w_in) + "inoff" +
                                 std::to_string(in_offset) + "otoff" + std::to_string(out_offset) +
                                 "u" + std::to_string(h_stride) + "v" + std::to_string(w_stride) +
                                 "t" + std::to_string(type);

    std::string kernel_name = "transpose_NCHW2CNHW";

    if(h_stride == 1 && w_stride == 1 && type == miopenFloat)
        kernel_name += "_opt";

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(in, out);
    }
    else
    {
        std::string params = GetDataTypeKernelParams(type);

        if(type == miopenInt8x4)
        {
            c /= 4;
            in_offset /= 4;
            out_offset /= 4;
        }

        if(h_stride == 1 && w_stride == 1 && type == miopenFloat)
        {
            params +=
                " -DNC_TRANS_NCHW_OPT=1 -DNC_TRANS_CNHW_OPT=0 -DNC_TRANS_NCHW=0 -DNC_TRANS_CNHW=0";

            int RD_BLCK      = ((h_in * w_in) % 4 == 0) ? 4 : ((h_in * w_in) % 2 == 0) ? 2 : 1;
            int HW_RD        = (h_in * w_in) / RD_BLCK;
            size_t MAP_RD    = HW_RD * c;
            size_t lcl_size0 = WG_SIZE; //((MAP_RD + 63)/64 < 4) ? ((MAP_RD + 63)/64)*64 : 256;

            std::string READ_TYPE = (RD_BLCK == 1) ? "float" : "float" + std::to_string(RD_BLCK);

            params += " -DIN_OFF=" + std::to_string(in_offset);
            params += " -DOUT_OFF=" + std::to_string(out_offset);
            params += " -DH=" + std::to_string(h_in);
            params += " -DW=" + std::to_string(w_in);
            params += " -DN=" + std::to_string(n);
            params += " -DC=" + std::to_string(c);
            params += " -DRD_BLCK=" + std::to_string(RD_BLCK);
            params += " -DHW_RD=" + std::to_string(HW_RD);
            params += " -DMAP_RD=" + std::to_string(MAP_RD);
            params += " -DREAD_TYPE=" + READ_TYPE;

            const std::vector<size_t> vld{lcl_size0, 1, 1};
            std::vector<size_t> vgd{MAP_RD, 1, 1};

            if(MAP_RD < MAX_ACTIVE_THREADS)
            {
                vgd = {MAP_RD, static_cast<size_t>(n), 1};
                params += " -DIS_2D_WG=1";
            }
            else
            {
                params += " -DIS_2D_WG=0";
            }

            handle.AddKernel(
                kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(in, out);
        }
        else
        {
            params +=
                " -DNC_TRANS_NCHW_OPT=0 -DNC_TRANS_CNHW_OPT=0 -DNC_TRANS_NCHW=1 -DNC_TRANS_CNHW=0";

            params += " -DN=" + std::to_string(n);
            params += " -DC=" + std::to_string(c);
            params += " -DHW_IN=" + std::to_string(h_in * w_in);
            params += " -DHW_OUT=" + std::to_string(h_out * w_out);
            params += " -DW_IN=" + std::to_string(w_in);
            params += " -DW_OUT=" + std::to_string(w_out);
            params += " -DH_STRIDE=" + std::to_string(h_stride);
            params += " -DW_STRIDE=" + std::to_string(w_stride);
            params += " -DIN_OFF=" + std::to_string(in_offset);
            params += " -DOUT_OFF=" + std::to_string(out_offset);

            size_t ld0 = WG_SIZE;
            size_t gd0 = c * h_out * w_out;
            const std::vector<size_t> vld{ld0, 1, 1};
            std::vector<size_t> vgd{gd0, 1, 1};

            if(gd0 < MAX_ACTIVE_THREADS)
            {
                vgd = {gd0, static_cast<size_t>(n), 1};
                params += " -DIS_2D_WG=1";
            }
            else
            {
                params += " -DIS_2D_WG=0";
            }

            handle.AddKernel(
                kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(in, out);
        }
    }

    return handle.GetKernelTime();
}

float transpose_CNHW2NCHW(Handle& handle,
                          int n,
                          int c,
                          int h_out,
                          int w_out,
                          int h_in,
                          int w_in,
                          ConstData_t in,
                          Data_t out,
                          int in_offset,
                          int out_offset,
                          int h_stride,
                          int w_stride,
                          miopenDataType_t type)
{

    std::string program_name = "MIOpenUtilKernels4.cl";

    std::string network_config = "n" + std::to_string(n) + "c" + std::to_string(c) + "h" +
                                 std::to_string(h_in) + "w" + std::to_string(w_in) + "inoff" +
                                 std::to_string(in_offset) + "otoff" + std::to_string(out_offset) +
                                 "h_stride" + std::to_string(h_stride) + "w_stride" +
                                 std::to_string(w_stride) + "t" + std::to_string(type);

    std::string kernel_name = "transpose_CNHW2NCHW";

    if(h_stride == 1 && w_stride == 1 && type == miopenFloat)
        kernel_name += "_opt";

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(in, out);
    }
    else
    {
        std::string params = GetDataTypeKernelParams(type);

        if(type == miopenInt8x4)
        {
            c /= 4;
            in_offset /= 4;
            out_offset /= 4;
        }

        if(h_stride == 1 && w_stride == 1 && type == miopenFloat)
        {
            params +=
                " -DNC_TRANS_NCHW_OPT=0 -DNC_TRANS_CNHW_OPT=1 -DNC_TRANS_NCHW=0 -DNC_TRANS_CNHW=0";

            int RD_BLCK      = ((h_out * w_out) % 4 == 0) ? 4 : ((h_out * w_out) % 2 == 0) ? 2 : 1;
            int HW_RD        = (h_out * w_out) / RD_BLCK;
            size_t MAP_RD    = HW_RD * c;
            size_t lcl_size0 = WG_SIZE; //((MAP_RD + 63)/64 < 4) ? ((MAP_RD + 63)/64)*64 : 256;

            std::string READ_TYPE = (RD_BLCK == 1) ? "float" : "float" + std::to_string(RD_BLCK);

            params += " -DIN_OFF=" + std::to_string(in_offset);
            params += " -DOUT_OFF=" + std::to_string(out_offset);
            params += " -DH=" + std::to_string(h_out);
            params += " -DW=" + std::to_string(w_out);
            params += " -DN=" + std::to_string(n);
            params += " -DC=" + std::to_string(c);
            params += " -DRD_BLCK=" + std::to_string(RD_BLCK);
            params += " -DHW_RD=" + std::to_string(HW_RD);
            params += " -DMAP_RD=" + std::to_string(MAP_RD);
            params += " -DREAD_TYPE=" + READ_TYPE;

            const std::vector<size_t> vld{lcl_size0, 1, 1};
            std::vector<size_t> vgd{MAP_RD, 1, 1};

            if(MAP_RD < MAX_ACTIVE_THREADS)
            {
                vgd = {MAP_RD, static_cast<size_t>(n), 1};
                params += " -DIS_2D_WG=1";
            }
            else
            {
                params += " -DIS_2D_WG=0";
            }

            handle.AddKernel(
                kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(in, out);
        }
        else
        {
            params +=
                " -DNC_TRANS_NCHW_OPT=0 -DNC_TRANS_CNHW_OPT=0 -DNC_TRANS_NCHW=0 -DNC_TRANS_CNHW=1";

            params += " -DN=" + std::to_string(n);
            params += " -DC=" + std::to_string(c);
            params += " -DHW_IN=" + std::to_string(h_in * w_in);
            params += " -DHW_OUT=" + std::to_string(h_out * w_out);
            params += " -DW_IN=" + std::to_string(w_in);
            params += " -DW_OUT=" + std::to_string(w_out);
            params += " -DH_STRIDE=" + std::to_string(h_stride);
            params += " -DW_STRIDE=" + std::to_string(w_stride);
            params += " -DIN_OFF=" + std::to_string(in_offset);
            params += " -DOUT_OFF=" + std::to_string(out_offset);

            size_t ld0 = WG_SIZE;
            size_t gd0 = c * h_out * w_out;
            const std::vector<size_t> vld{ld0, 1, 1};
            std::vector<size_t> vgd{gd0, 1, 1};

            if(gd0 < MAX_ACTIVE_THREADS)
            {
                vgd = {gd0, static_cast<size_t>(n), 1};
                params += " -DIS_2D_WG=1";
            }
            else
            {
                params += " -DIS_2D_WG=0";
            }

            handle.AddKernel(
                kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(in, out);
        }
    }

    return handle.GetKernelTime();
}

// NCHW (or NCDHW) to NCHW_C4 (or NCDHW_C4)
float transpose_NCHW2Vec(Handle& handle,
                         const std::vector<std::size_t>& lens,
                         ConstData_t in,
                         Data_t out,
                         std::size_t vec_size,
                         bool trans,
                         bool forward,
                         const void* alpha,
                         const void* beta)
{
    std::string program_name = "MIOpenUtilKernels5.cl";

    if(!(vec_size == 2 || vec_size == 4))
    {
        MIOPEN_THROW("Only support type half and int8!");
    }

    const auto alpha_fp = *(static_cast<const float*>(alpha));
    const auto beta_fp  = *(static_cast<const float*>(beta));

    const auto n = lens[0];
    const auto c = lens[1];

    // "hw" is for any-D spatial data
    const auto hw = std::accumulate(
        lens.begin() + 2, lens.end(), std::size_t(1), std::multiplies<std::size_t>());

    // clang-format off
    std::string network_config =
        "n" + std::to_string(n) +
        "c" + std::to_string(c) +
        "hw" + std::to_string(hw) +
        "t" + std::to_string(static_cast<int>(trans)) +
        "v" + std::to_string(vec_size) +
        "f" + std::to_string(static_cast<int>(forward)) + "alp" + std::to_string(alpha_fp) + "beta" +
            std::to_string(beta_fp);
    // clang-format on

    std::string algo_name = "transpose_NCHWVecForward";

    auto&& kernels = handle.GetKernels(algo_name, network_config);

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(in, out);
    }
    else
    {
        auto n_vec = (trans && (n % vec_size != 0)) ? (n + (vec_size - n % vec_size)) : n;
        auto c_vec = (!trans && (c % vec_size != 0)) ? (c + (vec_size - c % vec_size)) : c;

        std::string kernel_name = "transpose_NCHW2Vec";

        const std::vector<size_t> vld{WG_SIZE, 1, 1};
        std::vector<size_t> vgd{1, 1, 1};

        int RD_BLCK = ((hw) % (vec_size * 2) == 0) ? static_cast<int>(vec_size) * 2
                                                   : static_cast<int>(vec_size);
        int HW_RD     = (static_cast<int>(hw) + RD_BLCK - 1) / RD_BLCK;
        size_t MAP_RD = HW_RD * (trans ? c : (c_vec / vec_size));

        std::string READ_TYPE =
            (RD_BLCK == vec_size) ? "uint" : "uint" + std::to_string(RD_BLCK / vec_size);
        int WR_BLCK            = RD_BLCK * static_cast<int>(vec_size);
        std::string WRITE_TYPE = "uint" + std::to_string(WR_BLCK / vec_size);

        std::string params;
        params += " -DFORWARD=" + std::to_string(static_cast<int>(forward));
        params += " -DN=" + std::to_string(n);
        params += " -DC=" + std::to_string(c);
        params += " -DHW=" + std::to_string(hw);
        params += " -DCHW=" + std::to_string(c * hw);
        params += " -DVEC_SIZE=" + std::to_string(vec_size);
        params += vec_size == 4 ? " -DDATA_TYPE=char" : " -DDATA_TYPE=ushort";

        params += " -DTRANS=" + std::to_string(static_cast<int>(trans));
        if(trans)
        {
            params += " -DNHW_OUT=" + std::to_string(n_vec * hw);
            params += " -DN_OUT=" + std::to_string(n_vec);
            params += " -DIS_N_ODD=" + std::to_string(static_cast<int>((n % vec_size) != 0));
        }
        else
        {
            params += " -DCHW_OUT=" + std::to_string(c_vec * hw);
            params += " -DIS_C_ODD=" + std::to_string(static_cast<int>((c % vec_size) != 0));
        }

        params += " -DIS_HW_ODD=" + std::to_string(static_cast<int>(((hw) % vec_size) != 0));
        params += " -DRD_BLCK=" + std::to_string(RD_BLCK);
        params += " -DWR_BLCK=" + std::to_string(WR_BLCK);
        params += " -DHW_RD=" + std::to_string(HW_RD);
        params += " -DMAP_RD=" + std::to_string(MAP_RD);
        params += " -DREAD_TYPE=" + READ_TYPE;
        params += " -DWRITE_TYPE=" + WRITE_TYPE;

        if(!float_equal(alpha_fp, 1.0))
            params += " -DUSE_ALPHA=1";

        if(!float_equal(beta_fp, 0))
            params += " -DUSE_BETA=1";

        vgd[0] = MAP_RD;

        uint gd1 = trans ? static_cast<size_t>(n_vec / vec_size) : static_cast<size_t>(n);

        /// disable iteration of n due to perf degrade
        /// \to-do fix the perf issue
        // if(vgd[0] < MAX_ACTIVE_THREADS)
        {
            vgd[1] = gd1;
            params += " -DIS_2D_WG=1";
        }
        // else
        //{
        // params += " -DIS_2D_WG=0";
        // params += " -DGD_1=" + std::to_string(gd1);
        //}

        handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, params)(
            in, out, alpha_fp, beta_fp);
    }

    return handle.GetKernelTime();
}

float transpose_packed_MN2NM(Handle& handle,
                             int m,
                             int n,
                             int in_offset,
                             int out_offset,
                             ConstData_t in,
                             Data_t out,
                             miopenDataType_t type)
{

    std::string program_name = "MIOpenUtilKernels4.cl";

    std::string network_config = "n" + std::to_string(n) + "m" + std::to_string(m) + "inoff" +
                                 std::to_string(in_offset) + "otoff" + std::to_string(out_offset) +
                                 "t" + std::to_string(type);

    std::string kernel_name = "transpose_packed_MN2NM";

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(in, out);
    }
    else
    {
        std::string params = GetDataTypeKernelParams(type);
        if(type == miopenInt8x4)
        {
            m /= 4;
            in_offset /= 4;
            out_offset /= 4;
        }

        if(!(type == miopenInt8x4 || type == miopenInt8))
        {
            MIOPEN_THROW("transpose_packed_MN2NM only meant for int8 variants.");
        }

        params += " -DNC_TRANS_MN2NM=1";

        params += " -DN=" + std::to_string(n);
        params += " -DM=" + std::to_string(m);
        params += " -DIN_OFF=" + std::to_string(in_offset);
        params += " -DOUT_OFF=" + std::to_string(out_offset);

        size_t ld0 = WG_SIZE;
        size_t gd0 = m * n;
        const std::vector<size_t> vld{ld0, 1, 1};
        std::vector<size_t> vgd{gd0, 1, 1};

        handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
            in, out);
    }

    return handle.GetKernelTime();
}
} // namespace miopen
