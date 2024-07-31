/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/handle.hpp>
#include <miopen/dropout.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>

#include <rocrand/rocrand_xorwow.h>

#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

#include "get_handle.hpp"
#include "random.hpp"
#include "driver.hpp"
#include "dropout_util.hpp"
#include "perf_helper.hpp"

#define PERF_ENABLE 0

// This code snippet was taken from src/ocl/dropooutocl.cpp
// Aligns the input and output tensor strides and lengths
// squashes contiguous dimensions and performs alignment of
// non-contiguous dimensions
template <typename T>
inline void SquashPairedTensor(const std::vector<T>& x_len,
                               const std::vector<T>& x_str,
                               const std::vector<T>& y_len,
                               const std::vector<T>& y_str,
                               std::vector<T>& in_len,
                               std::vector<T>& in_str,
                               std::vector<T>& out_len,
                               std::vector<T>& out_str)
{
    if(!std::equal(x_len.begin(), x_len.end(), y_len.begin()))
    {
        MIOPEN_THROW("Input/Output tensor lengths do not match");
    }

    in_len.back()  = x_len.back();
    in_str.back()  = x_str.back();
    out_len.back() = y_len.back();
    out_str.back() = y_str.back();

    int xl_idx = x_len.size() - 2;
    int yl_idx = y_len.size() - 2;
    int xs_idx = x_str.size() - 2;
    int ys_idx = y_str.size() - 2;

    int il_idx = in_len.size() - 1;
    int ol_idx = out_len.size() - 1;
    int is_idx = in_str.size() - 2;
    int os_idx = out_str.size() - 2;

    while(xl_idx >= 0 && x_str[xs_idx] == x_len[xl_idx + 1] * x_str[xs_idx + 1] &&
          y_str[ys_idx] == y_len[yl_idx + 1] * y_str[ys_idx + 1])
    {
        in_len[il_idx] *= x_len[xl_idx--];
        out_len[ol_idx] *= y_len[yl_idx--];

        xs_idx--;
        ys_idx--;
    }

    if(xl_idx < 0 && is_idx >= 0)
    {
        in_str[is_idx--]  = in_len[il_idx];
        out_str[os_idx--] = out_len[ol_idx];
    }
    else if(xl_idx >= 0)
    {
        il_idx--;
        ol_idx--;

        while(xl_idx >= 0 && il_idx >= 0)
        {
            in_len[il_idx--] = x_len[xl_idx--];
            in_str[is_idx--] = x_str[xs_idx--];
        }

        while(yl_idx >= 0 && ol_idx >= 0)
        {
            out_len[ol_idx--] = y_len[yl_idx--];
            out_str[os_idx--] = y_str[ys_idx--];
        }
    }

    while(is_idx >= 0)
        in_str[is_idx--] = in_str[is_idx + 1] * in_len[is_idx + 1];

    while(os_idx >= 0)
        out_str[os_idx--] = out_str[os_idx + 1] * out_len[os_idx + 1];

    if(!std::equal(in_len.begin(), in_len.end(), out_len.begin()))
    {
        MIOPEN_THROW("Input/Output tensor lengths do not match");
    }
}

void InitPRNGState(miopen::Handle& handle,
                   const miopen::DropoutDescriptor& DropoutDesc,
                   PerfHelper<float>& perf_helper,
                   bool use_hip)
{
    std::string program_name;
    std::string kernel_name;

    if(DropoutDesc.stateSizeInBytes > handle.GetMaxMemoryAllocSize())
    {
        MIOPEN_THROW("PRNG state size should not exceed system maximum memory allocation size.");
    }

    unsigned long long states_num = DropoutDesc.stateSizeInBytes / sizeof(rocrand_state_xorwow);
    size_t wk_grp_num =
        std::min(static_cast<unsigned long long>(MAX_PRNG_STATE / 256), (states_num + 255) / 256);

    std::string network_config = "initprngs-" + std::to_string(sizeof(rocrand_state_xorwow)) + "x" +
                                 std::to_string(DropoutDesc.rng_mode) + "x" +
                                 std::to_string(wk_grp_num);

    if(!use_hip)
    {
        program_name = "MIOpenDropout.cl";
        kernel_name  = "InitKernelState";
        network_config += "-ocl";
    }
    else
    {
        program_name = "MIOpenDropoutHIP.cpp";
        kernel_name  = "InitKernelStateHIP";
        network_config += "-hip";
    }

    const std::vector<size_t> vld{256, 1, 1};
    const std::vector<size_t> vgd{wk_grp_num * 256, 1, 1};

    std::string params;

    params += "-DRUN_FORWARD=0 -DRUN_INIT_PRNG=1";

    handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
        DropoutDesc.pstates, DropoutDesc.seed, states_num);

    if constexpr(PERF_ENABLE)
    {
        perf_helper.perfTest(handle,
                             kernel_name,
                             network_config,
                             use_hip,
                             DropoutDesc.pstates,
                             DropoutDesc.seed,
                             states_num);
    }
}

void DropoutForward(miopen::Handle& handle,
                    const miopen::TensorDescriptor& noise_shape,
                    const miopen::TensorDescriptor& xDesc,
                    ConstData_t x,
                    const miopen::TensorDescriptor& yDesc,
                    Data_t y,
                    Data_t reserveSpace,
                    size_t reserveSpaceSizeInBytes,
                    unsigned in_offset,
                    unsigned out_offset,
                    unsigned rsvsp_offset,
                    const miopen::DropoutDescriptor& DropoutDesc,
                    PerfHelper<float>& perf_helper,
                    bool use_hip)
{

    float dropout            = DropoutDesc.dropout;
    Data_t pstates           = DropoutDesc.pstates;
    size_t stateSizeInBytes  = DropoutDesc.stateSizeInBytes;
    unsigned long long seed  = DropoutDesc.seed;
    bool use_mask            = DropoutDesc.use_mask;
    bool state_evo           = DropoutDesc.state_evo;
    miopenRNGType_t rng_mode = DropoutDesc.rng_mode;

    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(xDesc.GetNumDims() != yDesc.GetNumDims())
    {
        MIOPEN_THROW("Input/Output dimension does not match");
    }

    if(xDesc.GetNumDims() > 5)
    {
        MIOPEN_THROW("Only support 1D to 5D tensors");
    }

    if(xDesc.GetElementSize() != yDesc.GetElementSize())
    {
        MIOPEN_THROW("Input/Output element size does not match");
    }

    if(xDesc.GetElementSize() != noise_shape.GetElementSize() ||
       xDesc.GetNumDims() != noise_shape.GetNumDims())
    {
        MIOPEN_THROW("Only support dropout with regular noise shape currently");
    }

    if(xDesc.GetType() != yDesc.GetType())
    {
        MIOPEN_THROW("Input/Output datatype does not match");
    }

    if(dropout < 0.0 || dropout > 1.0)
    {
        MIOPEN_THROW("Invalid dropout rate");
    }

    bool use_rsvsp = !(reserveSpace == nullptr);
    if(((use_rsvsp || use_mask) &&
        reserveSpaceSizeInBytes < xDesc.GetElementSize() * sizeof(bool)) ||
       (use_mask && reserveSpace == nullptr))
    {
        MIOPEN_THROW("Insufficient reservespace size");
    }

    if(stateSizeInBytes + reserveSpaceSizeInBytes +
           xDesc.GetElementSize() * miopen::GetTypeSize(xDesc.GetType()) +
           yDesc.GetElementSize() * miopen::GetTypeSize(yDesc.GetType()) >
       handle.GetGlobalMemorySize())
    {
        MIOPEN_THROW("Memory required by dropout forward configs exceeds GPU memory range.");
    }

    // support up to 5D tensor
    std::vector<size_t> in_len(5, 1);
    std::vector<size_t> in_str(5, 1);
    std::vector<size_t> out_len(5, 1);
    std::vector<size_t> out_str(5, 1);

    SquashPairedTensor(xDesc.GetLengths(),
                       xDesc.GetStrides(),
                       yDesc.GetLengths(),
                       yDesc.GetStrides(),
                       in_len,
                       in_str,
                       out_len,
                       out_str);

    size_t RD_BLCK = /* (in_len[4] % 4 == 0) ? 4 : */ (in_len[2] % 2 == 0) ? 2 : 1;

    unsigned int total_work = (in_len[4] / RD_BLCK) * in_len[3] * in_len[2] * in_len[1] * in_len[0];

    size_t max_wk_grp = use_mask ? size_t(MAX_WORKITEM_NUM)
                                 : std::min(size_t(MAX_PRNG_STATE), handle.GetImage3dMaxWidth());
    size_t wk_grp_num =
        std::min(max_wk_grp / 256,
                 ((in_len[4] * in_len[3] * in_len[2] * in_len[1] * in_len[0] + 255) / 256));

    size_t states_num = stateSizeInBytes / sizeof(rocrand_state_xorwow);
    if(states_num < wk_grp_num * 256 && !use_mask)
    {
        MIOPEN_THROW("Insufficient state size for parallel PRNG");
    }

    std::string network_config =
        "fwd-" + std::string(xDesc.GetType() == miopenHalf ? "fp16-" : "fp32-") + "-seed" +
        std::to_string(seed) + "-rng" + std::to_string(rng_mode) + "-rsvsp" +
        std::to_string(static_cast<int>(use_rsvsp)) + "-mask" +
        std::to_string(static_cast<int>(use_mask)) + "-evo" +
        std::to_string(static_cast<int>(state_evo)) + "-blk" + std::to_string(RD_BLCK) + "-wg" +
        std::to_string(wk_grp_num) /* + "-noise" + std::to_string(noise_shape.GetLengths()[0])*/;

    std::string program_name;
    std::string kernel_name;

    if(!use_hip)
    {
        program_name = "MIOpenDropout.cl";
        kernel_name  = "DropoutForward";
        network_config += "-ocl";
    }
    else
    {
        program_name = "MIOpenDropoutHIP.cpp";
        kernel_name  = "DropoutFW";
        network_config += "-hip";
    }

    // TODO: Add noise shape
    // for(int i = 1; i < noise_shape.GetNumDims(); i++)
    //    network_config += "x" + std::to_string(noise_shape.GetLengths()[i]);

    float amp_scale = miopen::float_equal(dropout, 1.0) ? 0 : 1 / (1 - dropout);

    std::string params;

    const std::string data_type = miopen::GetDataType(xDesc.GetType());
    const std::string READ_DAT_TYPE =
        RD_BLCK == 1 ? data_type : data_type + std::to_string(RD_BLCK);

    params += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DREAD_DAT_TYPE=" + READ_DAT_TYPE +
              " -DREAD_BOOL_TYPE=" +
              std::string(RD_BLCK == 4   ? "uint"
                          : RD_BLCK == 2 ? "ushort"
                                         : "uchar");

    if(xDesc.GetType() == miopenHalf)
        params += " -DMIOPEN_USE_FP16=1";
    else
        params += " -DMIOPEN_USE_FP32=1";

    params += " -DRUN_FORWARD=1";

    params += " -DUSE_RSVSP=" + std::to_string(static_cast<size_t>(use_rsvsp));
    params += " -DUSE_MASK=" + std::to_string(static_cast<size_t>(use_mask));

    const std::vector<size_t> vld{256, 1, 1};
    const std::vector<size_t> vgd{wk_grp_num * 256, 1, 1};

    handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
        pstates,
        dropout,
        amp_scale,
        static_cast<int>(in_len[1]),
        static_cast<int>(in_len[2]),
        static_cast<int>(in_len[3]),
        static_cast<int>(in_len[4]),
        y,
        static_cast<int>(out_str[0]),
        static_cast<int>(out_str[1]),
        static_cast<int>(out_str[2]),
        static_cast<int>(out_str[3]),
        x,
        static_cast<int>(in_str[0]),
        static_cast<int>(in_str[1]),
        static_cast<int>(in_str[2]),
        static_cast<int>(in_str[3]),
        reserveSpace,
        static_cast<unsigned>(total_work),
        static_cast<unsigned>(in_offset),
        static_cast<unsigned>(out_offset),
        static_cast<unsigned>(rsvsp_offset));

    if constexpr(PERF_ENABLE)
    {
        perf_helper.perfTest(handle,
                             kernel_name,
                             network_config,
                             use_hip,
                             pstates,
                             dropout,
                             amp_scale,
                             static_cast<int>(in_len[1]),
                             static_cast<int>(in_len[2]),
                             static_cast<int>(in_len[3]),
                             static_cast<int>(in_len[4]),
                             y,
                             static_cast<int>(out_str[0]),
                             static_cast<int>(out_str[1]),
                             static_cast<int>(out_str[2]),
                             static_cast<int>(out_str[3]),
                             x,
                             static_cast<int>(in_str[0]),
                             static_cast<int>(in_str[1]),
                             static_cast<int>(in_str[2]),
                             static_cast<int>(in_str[3]),
                             reserveSpace,
                             static_cast<unsigned>(total_work),
                             static_cast<unsigned>(in_offset),
                             static_cast<unsigned>(out_offset),
                             static_cast<unsigned>(rsvsp_offset));
    }
}

void DropoutBackward(miopen::Handle& handle,
                     const miopen::TensorDescriptor& noise_shape,
                     const miopen::TensorDescriptor& dyDesc,
                     ConstData_t dy,
                     const miopen::TensorDescriptor& dxDesc,
                     Data_t dx,
                     Data_t reserveSpace,
                     size_t reserveSpaceSizeInBytes,
                     unsigned in_offset,
                     unsigned out_offset,
                     unsigned rsvsp_offset,
                     const miopen::DropoutDescriptor& DropoutDesc,
                     PerfHelper<float>& perf_helper,
                     bool use_hip)
{

    float dropout            = DropoutDesc.dropout;
    Data_t pstates           = DropoutDesc.pstates;
    size_t stateSizeInBytes  = DropoutDesc.stateSizeInBytes;
    unsigned long long seed  = DropoutDesc.seed;
    bool use_mask            = DropoutDesc.use_mask;
    bool state_evo           = DropoutDesc.state_evo;
    miopenRNGType_t rng_mode = DropoutDesc.rng_mode;

    if(dx == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(dxDesc.GetNumDims() != dyDesc.GetNumDims())
    {
        MIOPEN_THROW("Input/Output dimension does not match");
    }

    if(dyDesc.GetNumDims() > 5)
    {
        MIOPEN_THROW("Only support 1D to 5D tensors");
    }

    if(dxDesc.GetElementSize() != dyDesc.GetElementSize())
    {
        MIOPEN_THROW("Input/Output element size does not match");
    }

    if(dxDesc.GetElementSize() != noise_shape.GetElementSize() ||
       dxDesc.GetNumDims() != noise_shape.GetNumDims())
    {
        MIOPEN_THROW("Only support dropout with regular noise shape currently");
    }

    if(dxDesc.GetType() != dyDesc.GetType())
    {
        MIOPEN_THROW("Input/Output datatype does not match");
    }

    if(dropout < 0.0 || dropout > 1.0)
    {
        MIOPEN_THROW("Invalid dropout rate");
    }

    bool use_prng = reserveSpace == nullptr;
    if(((!use_prng || use_mask) &&
        reserveSpaceSizeInBytes < dyDesc.GetElementSize() * sizeof(bool)) ||
       (use_mask && use_prng))
    {
        MIOPEN_THROW("Insufficient reservespace size");
    }

    if(reserveSpaceSizeInBytes + dxDesc.GetElementSize() * miopen::GetTypeSize(dxDesc.GetType()) +
           dyDesc.GetElementSize() * miopen::GetTypeSize(dyDesc.GetType()) >
       handle.GetGlobalMemorySize())
    {
        MIOPEN_THROW("Memory required by dropout backward configs exceeds GPU memory range.");
    }

    // support up to 5D tensor
    std::vector<size_t> in_len(5, 1);
    std::vector<size_t> in_str(5, 1);
    std::vector<size_t> out_len(5, 1);
    std::vector<size_t> out_str(5, 1);

    SquashPairedTensor(dxDesc.GetLengths(),
                       dxDesc.GetStrides(),
                       dyDesc.GetLengths(),
                       dyDesc.GetStrides(),
                       in_len,
                       in_str,
                       out_len,
                       out_str);

    size_t RD_BLCK          = /* (in_len[4] % 4 == 0) ? 4 : */ (in_len[2] % 2 == 0) ? 2 : 1;
    unsigned int total_work = (in_len[4] / RD_BLCK) * in_len[3] * in_len[2] * in_len[1] * in_len[0];

    size_t max_wk_grp = use_prng ? std::min(size_t(MAX_PRNG_STATE), handle.GetImage3dMaxWidth())
                                 : size_t(MAX_WORKITEM_NUM);
    size_t wk_grp_num =
        std::min(max_wk_grp / 256,
                 ((in_len[4] * in_len[3] * in_len[2] * in_len[1] * in_len[0] + 255) / 256));

    if(use_prng)
    {
        size_t states_num = stateSizeInBytes / sizeof(rocrand_state_xorwow);
        if(states_num < wk_grp_num * 256)
        {
            MIOPEN_THROW("Insufficient state size for parallel PRNG");
        }
    }

    std::string network_config =
        "bwd-" + std::string(dyDesc.GetType() == miopenHalf ? "fp16-" : "fp32-") + "-seed" +
        std::to_string(seed) + "-rng" + std::to_string(rng_mode) + "-prng" +
        std::to_string(static_cast<int>(use_prng)) + "-evo" +
        std::to_string(static_cast<int>(state_evo)) + "-blk" + std::to_string(RD_BLCK) + "-wg" +
        std::to_string(wk_grp_num) /* + "-noise" + std::to_string(noise_shape.GetLengths()[0]) */;

    std::string program_name;
    std::string kernel_name;

    if(!use_hip)
    {
        program_name = "MIOpenDropout.cl";
        kernel_name  = "DropoutBackward";
        network_config += "-ocl";
    }
    else
    {
        program_name = "MIOpenDropoutHIP.cpp";
        kernel_name  = "DropoutBW";
        network_config += "-hip";
    }

    // TODO: Add noise shape
    // for(int i = 1; i < noise_shape.GetNumDims(); i++)
    //    network_config += "x" + std::to_string(noise_shape.GetLengths()[i]);

    float amp_scale = miopen::float_equal(dropout, 1.0) ? 0 : 1 / (1 - dropout);

    std::string params;

    const std::string data_type = miopen::GetDataType(dyDesc.GetType());
    const std::string READ_DAT_TYPE =
        RD_BLCK == 1 ? data_type : data_type + std::to_string(RD_BLCK);

    params += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DREAD_DAT_TYPE=" + READ_DAT_TYPE +
              " -DREAD_BOOL_TYPE=" +
              std::string(RD_BLCK == 4   ? "uint"
                          : RD_BLCK == 2 ? "ushort"
                                         : "uchar");

    if(use_prng)
    {
        params += " -DUSE_PRNG=1";
    }

    if(dyDesc.GetType() == miopenHalf)
        params += " -DMIOPEN_USE_FP16=1";
    else
        params += " -DMIOPEN_USE_FP32=1";

    params += " -DRUN_FORWARD=0";

    const std::vector<size_t> vld{256, 1, 1};
    const std::vector<size_t> vgd{wk_grp_num * 256, 1, 1};

    handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
        pstates,
        dropout,
        amp_scale,
        static_cast<int>(in_len[1]),
        static_cast<int>(in_len[2]),
        static_cast<int>(in_len[3]),
        static_cast<int>(in_len[4]),
        dy,
        static_cast<int>(out_str[0]),
        static_cast<int>(out_str[1]),
        static_cast<int>(out_str[2]),
        static_cast<int>(out_str[3]),
        dx,
        static_cast<int>(in_str[0]),
        static_cast<int>(in_str[1]),
        static_cast<int>(in_str[2]),
        static_cast<int>(in_str[3]),
        reserveSpace,
        static_cast<unsigned>(total_work),
        static_cast<unsigned>(in_offset),
        static_cast<unsigned>(out_offset),
        static_cast<unsigned>(rsvsp_offset));

    if constexpr(PERF_ENABLE)
    {
        perf_helper.perfTest(handle,
                             kernel_name,
                             network_config,
                             use_hip,
                             pstates,
                             dropout,
                             amp_scale,
                             static_cast<int>(in_len[1]),
                             static_cast<int>(in_len[2]),
                             static_cast<int>(in_len[3]),
                             static_cast<int>(in_len[4]),
                             dy,
                             static_cast<int>(out_str[0]),
                             static_cast<int>(out_str[1]),
                             static_cast<int>(out_str[2]),
                             static_cast<int>(out_str[3]),
                             dx,
                             static_cast<int>(in_str[0]),
                             static_cast<int>(in_str[1]),
                             static_cast<int>(in_str[2]),
                             static_cast<int>(in_str[3]),
                             reserveSpace,
                             static_cast<unsigned>(total_work),
                             static_cast<unsigned>(in_offset),
                             static_cast<unsigned>(out_offset),
                             static_cast<unsigned>(rsvsp_offset));
    }
}

template <typename T>
tensor<T> FWDropGPU(const miopen::DropoutDescriptor& DropoutDesc,
                    const miopen::TensorDescriptor& NoiseShape,
                    const tensor<T>& input,
                    const tensor<T>& output,
                    std::vector<unsigned char>& rsvsp,
                    unsigned in_offset,
                    unsigned out_offset,
                    unsigned rsvsp_offset,
                    bool use_rsvsp,
                    PerfHelper<float>& perf_helper,
                    bool use_hip)
{

    auto&& handle  = get_handle();
    auto out_gpu   = output;
    auto rsvsp_dev = handle.Write(rsvsp);
    auto in_dev    = handle.Write(input.data);
    auto out_dev   = handle.Write(output.data);

    typename std::vector<unsigned char>::iterator rsvsp_ptr;

    rsvsp_ptr = rsvsp.begin();

    DropoutForward(handle,
                   input.desc,
                   input.desc,
                   in_dev.get(),
                   output.desc,
                   out_dev.get(),
                   use_rsvsp ? rsvsp_dev.get() : nullptr,
                   rsvsp.size(),
                   in_offset,
                   out_offset,
                   rsvsp_offset,
                   DropoutDesc,
                   perf_helper,
                   use_hip);

    out_gpu.data   = handle.Read<T>(out_dev, output.data.size());
    auto rsvsp_gpu = handle.Read<unsigned char>(rsvsp_dev, rsvsp.size());

    std::copy(rsvsp_gpu.begin(), rsvsp_gpu.end(), rsvsp_ptr);
    return out_gpu;
}

template <typename T>
tensor<T> BWDropGPU(const miopen::DropoutDescriptor& DropoutDesc,
                    const tensor<T>& din,
                    const tensor<T>& dout,
                    const std::vector<unsigned char>& rsvsp,
                    unsigned in_offset,
                    unsigned out_offset,
                    unsigned rsvsp_offset,
                    bool use_rsvsp,
                    PerfHelper<float>& perf_helper,
                    bool use_hip)
{

    auto&& handle = get_handle();
    auto din_gpu  = din;

    auto din_dev   = handle.Write(din.data);
    auto dout_dev  = handle.Write(dout.data);
    auto rsvsp_dev = handle.Write(rsvsp);

    DropoutBackward(handle,
                    din.desc,
                    dout.desc,
                    dout_dev.get(),
                    din.desc,
                    din_dev.get(),
                    use_rsvsp ? rsvsp_dev.get() : nullptr,
                    rsvsp.size(),
                    in_offset,
                    out_offset,
                    rsvsp_offset,
                    DropoutDesc,
                    perf_helper,
                    use_hip);

    din_gpu.data = handle.Read<T>(din_dev, din.data.size());
    return din_gpu;
}

struct DropoutTestCase
{
    size_t dim0;
    size_t dim1;
    size_t dim2;
    size_t dim3;
    size_t dim4;

    friend std::ostream& operator<<(std::ostream& os, const DropoutTestCase& tc)
    {
        return os << " dim0:" << tc.dim0 << " dim1:" << tc.dim1 << " dim2:" << tc.dim2
                  << " dim3:" << tc.dim3 << " dim4:" << tc.dim4;
    }

    std::vector<size_t> GetInput()
    {
        // check if the tensor is 1D, 2D, 3D, 4D or 5D
        if(dim4 > 1)
            return {dim0, dim1, dim2, dim3, dim4};
        else if(dim3 > 1)
            return {dim0, dim1, dim2, dim3};
        else if(dim2 > 1)
            return {dim0, dim1, dim2};
        else if(dim1 > 1)
            return {dim0, dim1};
        else
            return {dim0};
    }
};

template <typename T>
std::vector<DropoutTestCase> DropoutTestConfigs()
{

    if constexpr(PERF_ENABLE)
    {
        std::vector<DropoutTestCase> configs;
        const auto& handle = get_handle();
        size_t maxTotalSize;

        // Generate all NCHW tensors that are limited by L3 cache size
        // or 2xL2 cache size when L3 is not available
        if(miopen::StartsWith(handle.GetDeviceName(), "gfx90a") ||
           miopen::StartsWith(handle.GetDeviceName(), "gfx908"))
        {
            maxTotalSize = 16; // twice the 8MB L2
        }
        else if(miopen::StartsWith(handle.GetDeviceName(), "gfx803"))
        {
            maxTotalSize = 4; // twice the 2MB L2
        }
        else if(miopen::StartsWith(handle.GetDeviceName(), "gfx900") ||
                miopen::StartsWith(handle.GetDeviceName(), "gfx906"))
        {
            maxTotalSize = 8; // twice the 4MB L2
        }
        else if(miopen::StartsWith(handle.GetDeviceName(), "gfx942"))
        {
            maxTotalSize = 256; // 256MB L3
        }
        else if(miopen::StartsWith(handle.GetDeviceName(), "gfx103"))
        {
            maxTotalSize = 128; // 128MB L3
        }
        else
        {
            maxTotalSize = 4; // twice the 2MB L2, default case.
        }

        maxTotalSize = maxTotalSize * 1024ull * 1024ull / sizeof(T);

        for(size_t N = 1; N <= maxTotalSize; N *= 2)
        {
            for(size_t C = 1; C <= maxTotalSize / N; C *= 2)
            {
                for(size_t H = 1; H <= maxTotalSize / (N * C); H *= 2)
                {
                    for(size_t W = 1; W <= maxTotalSize / (N * C * H); W *= 2)
                    {
                        size_t totalSize = N * C * H * W;
                        // Ensure the total size does not exceed the maximum limit
                        if(totalSize <= maxTotalSize)
                        {
                            configs.push_back({N, C, H, W, 0});
                        }
                    }
                }
            }
        }

        return configs;
    }
    else
    {
        // clang-format off
        return {{16, 4,  8, 1, 4},
                { 2, 4,  8, 8, 4},
                {16, 4,  8, 4, 0},
                {13, 8,  4, 8, 0},
                { 3, 8,  7, 0, 0},
                {16, 4, 10, 0, 0},
                { 3, 8,  0, 0, 0},
                {16, 4,  0, 0, 0},
                { 4, 0,  0, 0, 0}};
        // clang-format on
    }
}

template <typename T>
struct DropoutTest : public ::testing::TestWithParam<std::tuple<DropoutTestCase, float, bool>>
{
protected:
    static const std::string sPerfTestFilename;

    void SetUp() override
    {
        auto&& handle = get_handle();

        std::tie(dropout_config, DropoutDesc.dropout, DropoutDesc.use_mask) = GetParam();

        std::vector<size_t> in_dim = dropout_config.GetInput();

        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        // Create tensors for the forward and backward dropout
        input_f  = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        output_b = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});

        // Allocate tensors for the forward and backward dropout on GPU
        output_f_ocl = tensor<T>{in_dim};
        output_f_hip = tensor<T>{in_dim};

        input_b_ocl = tensor<T>{in_dim};
        input_b_hip = tensor<T>{in_dim};

        size_t stateSizeInBytes = std::min(size_t(MAX_PRNG_STATE), handle.GetImage3dMaxWidth()) *
                                  sizeof(rocrand_state_xorwow);
        size_t reserveSpaceSizeInBytes = input_f.desc.GetElementSize() * sizeof(bool);
        size_t total_mem =
            2 * (2 * input_f.desc.GetNumBytes() + reserveSpaceSizeInBytes) + stateSizeInBytes;
        size_t device_mem = handle.GetGlobalMemorySize();

        if(total_mem >= device_mem)
        {
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
            return;
        }

        // Setup dropout descriptor
        DropoutDesc.stateSizeInBytes = stateSizeInBytes;
        DropoutDesc.seed             = 0;
        DropoutDesc.rng_mode         = MIOPEN_RNG_PSEUDO_XORWOW;

        // Allocate reserve space
        reserveSpace = std::vector<unsigned char>(input_f.desc.GetElementSize());
        if(DropoutDesc.use_mask)
        {
            for(size_t i = 0; i < input_f.desc.GetElementSize(); i++)
            {
                reserveSpace[i] =
                    static_cast<unsigned char>(prng::gen_canonical<float>() > DropoutDesc.dropout);
            }
        }
    }

    void RunDropoutOCL()
    {
        auto&& handle  = get_handle();
        auto state_buf = handle.Create<unsigned char>(
            DropoutDesc.stateSizeInBytes);     // Allocate GPU memory for PRNG states
        DropoutDesc.pstates = state_buf.get(); // Store GPU memory pointer to PRNG states
        // Initialize PRNG states
        InitPRNGState(handle, DropoutDesc, perf_helper, false); // Initialize PRNG states
        // forward pass OCL
        output_f_ocl = FWDropGPU<T>(DropoutDesc,
                                    noise_shape,
                                    input_f,
                                    output_f_ocl,
                                    reserveSpace,
                                    0,
                                    0,
                                    0,
                                    true,
                                    perf_helper,
                                    false);
        // backward pass OCL
        input_b_ocl = BWDropGPU<T>(
            DropoutDesc, input_b_ocl, output_b, reserveSpace, 0, 0, 0, true, perf_helper, false);

        if(!DropoutDesc.use_mask)
        {
            // forward pass OCL
            output_f_ocl = FWDropGPU<T>(DropoutDesc,
                                        noise_shape,
                                        input_f,
                                        output_f_ocl,
                                        reserveSpace,
                                        0,
                                        0,
                                        0,
                                        false,
                                        perf_helper,
                                        false);

            // backward pass OCL
            input_b_ocl = BWDropGPU<T>(DropoutDesc,
                                       input_b_ocl,
                                       output_b,
                                       reserveSpace,
                                       0,
                                       0,
                                       0,
                                       false,
                                       perf_helper,
                                       false);
        }
    }

    void RunDropoutHIP()
    {
        auto&& handle  = get_handle();
        auto state_buf = handle.Create<unsigned char>(
            DropoutDesc.stateSizeInBytes);     // Allocate GPU memory for PRNG states
        DropoutDesc.pstates = state_buf.get(); // Store GPU memory pointer to PRNG states
        InitPRNGState(handle, DropoutDesc, perf_helper, true);

        // forward pass HIP
        output_f_hip = FWDropGPU<T>(DropoutDesc,
                                    noise_shape,
                                    input_f,
                                    output_f_hip,
                                    reserveSpace,
                                    0,
                                    0,
                                    0,
                                    true,
                                    perf_helper,
                                    true);

        // backward pass HIP
        input_b_hip = BWDropGPU<T>(
            DropoutDesc, input_b_hip, output_b, reserveSpace, 0, 0, 0, true, perf_helper, true);

        if(!DropoutDesc.use_mask)
        {
            // forward pass HIP
            output_f_hip = FWDropGPU<T>(DropoutDesc,
                                        noise_shape,
                                        input_f,
                                        output_f_hip,
                                        reserveSpace,
                                        0,
                                        0,
                                        0,
                                        false,
                                        perf_helper,
                                        true);

            // backward pass HIP
            input_b_hip = BWDropGPU<T>(DropoutDesc,
                                       input_b_hip,
                                       output_b,
                                       reserveSpace,
                                       0,
                                       0,
                                       0,
                                       false,
                                       perf_helper,
                                       true);
        }
    }

    void VerifyGPU()
    {
        auto error_f = miopen::rms_range(output_f_ocl, output_f_hip);
        EXPECT_TRUE(miopen::range_distance(output_f_ocl) == miopen::range_distance(output_f_hip));
        EXPECT_TRUE(error_f == 0) << "GPU FW Outputs do not match each other. Error:" << error_f;

        auto error_b = miopen::rms_range(input_b_ocl, input_b_hip);
        EXPECT_TRUE(miopen::range_distance(input_b_ocl) == miopen::range_distance(input_b_hip));
        EXPECT_TRUE(error_b == 0) << "GPU BW Outputs do not match each other. Error:" << error_b;
    }

    void TearDown() override
    {
        if constexpr(PERF_ENABLE)
        {
            // get the input tensor size and store in a string with x in between
            std::string kernel_info =
                std::to_string(dropout_config.dim0) + "x" + std::to_string(dropout_config.dim1) +
                "x" + std::to_string(dropout_config.dim2) + "x" +
                std::to_string(dropout_config.dim3) + "x" + std::to_string(dropout_config.dim4);

            // Round off dropout rate to 2 decimal places
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << DropoutDesc.dropout;
            kernel_info += "_dropout" + ss.str();

            kernel_info += "_mask" + std::to_string(DropoutDesc.use_mask);

            perf_helper.writeStatsToCSV(
                sPerfTestFilename,
                "_" + kernel_info + "_" + (input_f.desc.GetType() == miopenHalf ? "FP16" : "FP32"));
        }
    }

    DropoutTestCase dropout_config;

    tensor<T> input_f;  // input tensor for dropout forward
    tensor<T> output_b; // output tensor for dropout backward

    // Create tensors for the forward and backward dropout OCL
    tensor<T> output_f_ocl;
    tensor<T> input_b_ocl;

    // Create tensors for the forward and backward dropout HIP
    tensor<T> output_f_hip;
    tensor<T> input_b_hip;

    // GetKernelTime returns time in float
    PerfHelper<float> perf_helper;

    miopen::DropoutDescriptor DropoutDesc;
    miopen::TensorDescriptor noise_shape;
    std::vector<unsigned char> reserveSpace;
};

template <typename T>
const std::string DropoutTest<T>::sPerfTestFilename = "DropoutPerf.csv";

namespace DropoutPort {

struct GPU_dropout_FP32 : DropoutTest<float>
{
};

struct GPU_dropout_FP16 : DropoutTest<half_float::half>
{
};

} // namespace DropoutPort
using namespace DropoutPort;

TEST_P(GPU_dropout_FP32, PortTest)
{
    RunDropoutOCL();
    RunDropoutHIP();
    VerifyGPU();
};

TEST_P(GPU_dropout_FP16, PortTest)
{
    RunDropoutOCL();
    RunDropoutHIP();
    VerifyGPU();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_dropout_FP32,
                         testing::Combine(testing::ValuesIn(DropoutTestConfigs<float>()),
                                          testing::Values(0, 0.25, 0.5, 0.75, 1),
                                          testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_dropout_FP16,
                         testing::Combine(testing::ValuesIn(DropoutTestConfigs<half_float::half>()),
                                          testing::Values(0, 0.25, 0.5, 0.75, 1),
                                          testing::Values(true, false)));
