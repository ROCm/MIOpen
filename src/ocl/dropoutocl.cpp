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
#include <miopen/config.h>
#include <miopen/dropout.hpp>
#include <miopen/env.hpp>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

#define DROPOUT_DEBUG 0

namespace miopen {

template <typename T>
inline void SquashPairedTensor(const std::vector<T> x_len,
                               const std::vector<T> x_str,
                               const std::vector<T> y_len,
                               const std::vector<T> y_str,
                               std::vector<T>& in_len,
                               std::vector<T>& in_str,
                               std::vector<T>& out_len,
                               std::vector<T>& out_str)
{

    auto itr_xl = x_len.end() - 1;
    auto itr_yl = y_len.end() - 1;
    auto itr_xs = x_str.end() - 1;
    auto itr_ys = y_str.end() - 1;

    auto itr_il = in_len.end() - 1;
    auto itr_ol = out_len.end() - 1;
    auto itr_is = in_str.end() - 1;
    auto itr_os = out_str.end() - 1;

    while((*(itr_xs - 1) == *itr_xl * *(itr_xs--) && *(itr_ys - 1) == *itr_yl * *(itr_ys--) &&
           itr_xl > x_len.begin()) ||
          itr_xl == x_len.begin())
    {
        *itr_il *= *(itr_xl--);
        *itr_ol *= *(itr_yl--);
    }

    while(itr_xl >= x_len.begin() && itr_il >= in_len.begin())
        *(itr_il--) = *(itr_xl--);

    while(itr_yl >= y_len.begin() && itr_ol >= out_len.begin())
        *(itr_ol--) = *(itr_yl--);

    itr_il = in_len.end() - 1;
    while((itr_is--) != in_str.begin())
        *itr_is = *(itr_is + 1) * *(itr_il--);

    itr_ol = out_len.end() - 1;
    while((itr_os--) != out_str.begin())
        *itr_os = *(itr_os + 1) * *(itr_ol--);

    if(!std::equal(in_len.begin(), in_len.end(), out_len.begin()))
    {
        MIOPEN_THROW("Input/Output tensor lengths do not match");
    }
}

void DropoutDescriptor::InitPRNGState(Handle& handle,
                                      Data_t prng_states,
                                      size_t prng_stateSizeInBytes,
                                      unsigned long long prng_seed) const
{
#if DROPOUT_DEBUG
    std::cout << "Check memory and threads info of dropout PRNG states in debug mode:" << std::endl;
#endif
    std::string program_name = "MIOpenDropout.cl";
    std::string kernel_name  = "InitKernelState";

    if(prng_stateSizeInBytes > handle.GetMaxMemoryAllocSize())
    {
        MIOPEN_THROW("PRNG state size should not exceed system maximum memory allocation size.");
    }

    size_t states_num = prng_stateSizeInBytes / sizeof(prngStates);

    std::string network_config = std::to_string(states_num) + "x" +
                                 std::to_string(sizeof(prngStates)) + "x" +
                                 std::to_string(rng_mode) + "x" + std::to_string(prng_seed);

    auto&& kernels = handle.GetKernels(kernel_name, network_config);
    if(!kernels.empty())
    {
        kernels.front()(prng_states);
    }
    else
    {
        size_t wk_grp_num = std::min(size_t(MAX_PRNG_STATE / 256), (states_num + 255) / 256);
        const std::vector<size_t> vld{256, 1, 1};
        const std::vector<size_t> vgd{wk_grp_num * 256, 1, 1};

        std::string params;
        params += " -DRUN_INIT_PRNG=1";
        params += " -DPRNG_SEED=" + std::to_string(prng_seed);
        params += " -DSTATES_NUM=" + std::to_string(states_num);
#if DROPOUT_DEBUG
        std::cout << "Threads allocated for PRNG states: " << vgd[0] << std::endl;
        std::cout << "Memory allocated for PRNG states: " << stateSizeInBytes << std::endl;
#endif
        handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
            prng_states);
#if DROPOUT_DEBUG
        std::cout << "Succeeded in launching InitPRNGState()." << stateSizeInBytes << std::endl;
#endif
    }
}

void DropoutDescriptor::DropoutForward(Handle& handle,
                                       const TensorDescriptor& noise_shape,
                                       const TensorDescriptor& xDesc,
                                       ConstData_t x,
                                       const TensorDescriptor& yDesc,
                                       Data_t y,
                                       Data_t reserveSpace,
                                       size_t reserveSpaceSizeInBytes,
                                       size_t in_offset,
                                       size_t out_offset,
                                       size_t rsvsp_offset) const
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(xDesc.GetSize() != yDesc.GetSize())
    {
        MIOPEN_THROW("Input/Output dimension does not match");
    }

    if(xDesc.GetSize() > 5)
    {
        MIOPEN_THROW("Only support 1D to 5D tensors");
    }

    if(xDesc.GetElementSize() != yDesc.GetElementSize())
    {
        MIOPEN_THROW("Input/Output element size does not match");
    }

    if(xDesc.GetElementSize() != noise_shape.GetElementSize() ||
       xDesc.GetSize() != noise_shape.GetSize())
    {
        MIOPEN_THROW("Only support dropout with regular noise shape currently");
    }

    if(xDesc.GetType() != yDesc.GetType())
    {
        MIOPEN_THROW("Input/Output datatype does not match");
    }

    if(dropout < 0.0 || dropout >= 1.0)
    {
        MIOPEN_THROW("Invalid dropout rate");
    }

    if(reserveSpaceSizeInBytes < xDesc.GetElementSize() * sizeof(bool))
    {
        MIOPEN_THROW("Insufficient reservespace size");
    }

    if(stateSizeInBytes + reserveSpaceSizeInBytes +
           xDesc.GetElementSize() * GetTypeSize(xDesc.GetType()) +
           yDesc.GetElementSize() * GetTypeSize(yDesc.GetType()) >
       handle.GetGlobalMemorySize())
    {
        MIOPEN_THROW("Memory required by dropout forward configs exceeds GPU memory range.");
    }

    if(miopen::CheckNumericsEnabled())
    {
        std::cout << "Dropout forward input numerics check at dropout rate " << dropout
                  << std::endl;
        miopen::checkNumericsInput(handle, xDesc, x);
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

    std::string program_name = "MIOpenDropout.cl";
    std::string kernel_name  = "DropoutForward";

    std::string network_config =
        "fwd-" + std::string(xDesc.GetType() == miopenHalf ? "fp16-" : "fp32-") + "dim" +
        std::to_string(in_len[0]) + "x" + std::to_string(in_len[1]) + "x" +
        std::to_string(in_len[2]) + "x" + std::to_string(in_len[3]) + "x" +
        std::to_string(in_len[4]) + "-xstr" + std::to_string(in_str[0]) + "x" +
        std::to_string(in_str[1]) + "x" + std::to_string(in_str[2]) + "x" +
        std::to_string(in_str[3]) + "x" + std::to_string(in_str[4]) + "-ystr" +
        std::to_string(out_str[0]) + "x" + std::to_string(out_str[1]) + "x" +
        std::to_string(out_str[2]) + "x" + std::to_string(out_str[3]) + "x" +
        std::to_string(out_str[4]) + "-dropout" + std::to_string(dropout) + "-seed" +
        std::to_string(seed) + "-rng" + std::to_string(rng_mode) + "-mask" +
        std::to_string(static_cast<int>(use_mask)) + "-evo" +
        std::to_string(static_cast<int>(state_evo)) + "-noise" +
        std::to_string(noise_shape.GetLengths()[0]);

    for(int i = 1; i < noise_shape.GetSize(); i++)
        network_config += "x" + std::to_string(noise_shape.GetLengths()[i]);

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    if(!kernels.empty())
    {
        kernels.front()(pstates,
                        dropout,
                        int(in_len[0]),
                        int(in_len[1]),
                        int(in_len[2]),
                        int(in_len[3]),
                        int(in_len[4]),
                        y,
                        int(out_str[0]),
                        int(out_str[1]),
                        int(out_str[2]),
                        int(out_str[3]),
                        int(out_str[4]),
                        x,
                        int(in_str[0]),
                        int(in_str[1]),
                        int(in_str[2]),
                        int(in_str[3]),
                        int(in_str[4]),
                        reserveSpace);
    }
    else
    {
        std::string params;

        size_t RD_BLCK              = /* (in_len[4] % 4 == 0) ? 4 : */ (in_len[2] % 2 == 0) ? 2 : 1;
        const std::string data_type = GetDataType(xDesc.GetType());
        const std::string READ_DAT_TYPE =
            RD_BLCK == 1 ? data_type : data_type + std::to_string(RD_BLCK);

        params += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DREAD_DAT_TYPE=" + READ_DAT_TYPE +
                  " -DREAD_BOOL_TYPE=" +
                  std::string(RD_BLCK == 4 ? "uint" : RD_BLCK == 2 ? "ushort" : "uchar");

        size_t max_wk_grp = use_mask ? MAX_WORKITEM_NUM : std::min(size_t(MAX_PRNG_STATE),
                                                                   handle.GetImage3dMaxWidth());
        size_t wk_grp_num =
            std::min(max_wk_grp / 256,
                     ((in_len[4] * in_len[3] * in_len[2] * in_len[1] * in_len[0] + 255) / 256));

        size_t states_num = stateSizeInBytes / sizeof(prngStates);
        if(states_num < wk_grp_num * 256 && !use_mask)
        {
            MIOPEN_THROW("Insufficient state size for parallel PRNG");
        }

        params +=
            " -DWK_GRP_NUM=" + std::to_string(wk_grp_num) + " -DTOTAL_WORK=" +
            std::to_string((in_len[4] / RD_BLCK) * in_len[3] * in_len[2] * in_len[1] * in_len[0]);

        params += " -DIN_OFFSET=" + std::to_string(in_offset) + " -DOUT_OFFSET=" +
                  std::to_string(out_offset) + " -DRSV_OFFSET=" + std::to_string(rsvsp_offset);

        if(xDesc.GetType() == miopenHalf)
            params += " -DMIOPEN_USE_FP16=1";
        else
            params += " -DMIOPEN_USE_FP32=1";

        params += " -DRUN_FORWARD=1";

        if(use_mask)
            params += " -DUSE_MASK=1";

        const std::vector<size_t> vld{256, 1, 1};
        const std::vector<size_t> vgd{wk_grp_num * 256, 1, 1};

        handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
            pstates,
            dropout,
            int(in_len[0]),
            int(in_len[1]),
            int(in_len[2]),
            int(in_len[3]),
            int(in_len[4]),
            y,
            int(out_str[0]),
            int(out_str[1]),
            int(out_str[2]),
            int(out_str[3]),
            int(out_str[4]),
            x,
            int(in_str[0]),
            int(in_str[1]),
            int(in_str[2]),
            int(in_str[3]),
            int(in_str[4]),
            reserveSpace);
    }

    if(miopen::CheckNumericsEnabled())
    {
        std::cout << "Dropout forward output numerics check at dropout rate " << dropout
                  << std::endl;
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
}

void DropoutDescriptor::DropoutBackward(Handle& handle,
                                        const TensorDescriptor& noise_shape,
                                        const TensorDescriptor& dyDesc,
                                        ConstData_t dy,
                                        const TensorDescriptor& dxDesc,
                                        Data_t dx,
                                        Data_t reserveSpace,
                                        size_t reserveSpaceSizeInBytes,
                                        size_t in_offset,
                                        size_t out_offset,
                                        size_t rsvsp_offset) const
{
    if(dx == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(dxDesc.GetSize() != dyDesc.GetSize())
    {
        MIOPEN_THROW("Input/Output dimension does not match");
    }

    if(dyDesc.GetSize() > 5)
    {
        MIOPEN_THROW("Only support 1D to 5D tensors");
    }

    if(dxDesc.GetElementSize() != dyDesc.GetElementSize())
    {
        MIOPEN_THROW("Input/Output element size does not match");
    }

    if(dxDesc.GetElementSize() != noise_shape.GetElementSize() ||
       dxDesc.GetSize() != noise_shape.GetSize())
    {
        MIOPEN_THROW("Only support dropout with regular noise shape currently");
    }

    if(dxDesc.GetType() != dyDesc.GetType())
    {
        MIOPEN_THROW("Input/Output datatype does not match");
    }

    if(dropout < 0.0 || dropout >= 1.0)
    {
        MIOPEN_THROW("Invalid dropout rate");
    }

    if(reserveSpaceSizeInBytes < dyDesc.GetElementSize() * sizeof(bool))
    {
        MIOPEN_THROW("Insufficient reservespace size");
    }

    if(reserveSpaceSizeInBytes + dxDesc.GetElementSize() * GetTypeSize(dxDesc.GetType()) +
           dyDesc.GetElementSize() * GetTypeSize(dyDesc.GetType()) >
       handle.GetGlobalMemorySize())
    {
        MIOPEN_THROW("Memory required by dropout backward configs exceeds GPU memory range.");
    }

    if(miopen::CheckNumericsEnabled())
    {
        std::cout << "Dropout backward input numerics check at dropout rate " << dropout
                  << std::endl;
        miopen::checkNumericsInput(handle, dyDesc, dy);
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

    std::string program_name = "MIOpenDropout.cl";
    std::string kernel_name  = "DropoutBackward";

    std::string network_config =
        "bwd-" + std::string(dyDesc.GetType() == miopenHalf ? "fp16-" : "fp32-") + "dim" +
        std::to_string(in_len[0]) + "x" + std::to_string(in_len[1]) + "x" +
        std::to_string(in_len[2]) + "x" + std::to_string(in_len[3]) + "x" +
        std::to_string(in_len[4]) + "-xstr" + std::to_string(in_str[0]) + "x" +
        std::to_string(in_str[1]) + "x" + std::to_string(in_str[2]) + "x" +
        std::to_string(in_str[3]) + "x" + std::to_string(in_str[4]) + "-ystr" +
        std::to_string(out_str[0]) + "x" + std::to_string(out_str[1]) + "x" +
        std::to_string(out_str[2]) + "x" + std::to_string(out_str[3]) + "x" +
        std::to_string(out_str[4]) + "-dropout" + std::to_string(dropout) + "-seed" +
        std::to_string(seed) + "-rng" + std::to_string(rng_mode) + "-mask" +
        std::to_string(static_cast<int>(use_mask)) + "-evo" +
        std::to_string(static_cast<int>(state_evo)) + "-noise" +
        std::to_string(noise_shape.GetLengths()[0]);

    for(int i = 1; i < noise_shape.GetSize(); i++)
        network_config += "x" + std::to_string(noise_shape.GetLengths()[i]);

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    if(!kernels.empty())
    {
        kernels.front()(dropout,
                        int(in_len[0]),
                        int(in_len[1]),
                        int(in_len[2]),
                        int(in_len[3]),
                        int(in_len[4]),
                        dy,
                        int(out_str[0]),
                        int(out_str[1]),
                        int(out_str[2]),
                        int(out_str[3]),
                        int(out_str[4]),
                        dx,
                        int(in_str[0]),
                        int(in_str[1]),
                        int(in_str[2]),
                        int(in_str[3]),
                        int(in_str[4]),
                        reserveSpace);
    }
    else
    {
        std::string params;

        size_t RD_BLCK              = /* (in_len[4] % 4 == 0) ? 4 : */ (in_len[2] % 2 == 0) ? 2 : 1;
        const std::string data_type = GetDataType(dyDesc.GetType());
        const std::string READ_DAT_TYPE =
            RD_BLCK == 1 ? data_type : data_type + std::to_string(RD_BLCK);

        params += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DREAD_DAT_TYPE=" + READ_DAT_TYPE +
                  " -DREAD_BOOL_TYPE=" +
                  std::string(RD_BLCK == 4 ? "uint" : RD_BLCK == 2 ? "ushort" : "uchar");

        size_t wk_grp_num =
            std::min(size_t(MAX_WORKITEM_NUM / 256),
                     ((in_len[4] * in_len[3] * in_len[2] * in_len[1] * in_len[0] + 255) / 256));

        params +=
            " -DWK_GRP_NUM=" + std::to_string(wk_grp_num) + " -DTOTAL_WORK=" +
            std::to_string((in_len[4] / RD_BLCK) * in_len[3] * in_len[2] * in_len[1] * in_len[0]);

        params += " -DIN_OFFSET=" + std::to_string(in_offset) + " -DOUT_OFFSET=" +
                  std::to_string(out_offset) + " -DRSV_OFFSET=" + std::to_string(rsvsp_offset);

        if(dyDesc.GetType() == miopenHalf)
            params += " -DMIOPEN_USE_FP16=1";
        else
            params += " -DMIOPEN_USE_FP32=1";

        params += " -DRUN_FORWARD=0";

        const std::vector<size_t> vld{256, 1, 1};
        const std::vector<size_t> vgd{wk_grp_num * 256, 1, 1};

        handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
            dropout,
            int(in_len[0]),
            int(in_len[1]),
            int(in_len[2]),
            int(in_len[3]),
            int(in_len[4]),
            dy,
            int(out_str[0]),
            int(out_str[1]),
            int(out_str[2]),
            int(out_str[3]),
            int(out_str[4]),
            dx,
            int(in_str[0]),
            int(in_str[1]),
            int(in_str[2]),
            int(in_str[3]),
            int(in_str[4]),
            reserveSpace);
    }

    if(miopen::CheckNumericsEnabled())
    {
        std::cout << "Dropout backward output numerics check at dropout rate " << dropout
                  << std::endl;
        miopen::checkNumericsOutput(handle, dxDesc, dx);
    }
}

} // namespace miopen
