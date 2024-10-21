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

#include "../../test/dropout_util.hpp"

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

void DropoutDescriptor::InitPRNGState(Handle& handle,
                                      Data_t prng_states,
                                      size_t prng_stateSizeInBytes,
                                      unsigned long long prng_seed) const
{
#if DROPOUT_DEBUG
    std::cout << "Check memory and threads info of dropout PRNG states in debug mode:" << std::endl;
#endif
    std::string program_name = "MIOpenDropoutHIP.cpp";
    std::string kernel_name  = "InitKernelStateHIP";

    if(prng_stateSizeInBytes > handle.GetMaxMemoryAllocSize())
    {
        MIOPEN_THROW("PRNG state size should not exceed system maximum memory allocation size.");
    }

    unsigned long long states_num = prng_stateSizeInBytes / sizeof(rocrand_state_xorwow);
    size_t wk_grp_num =
        std::min(static_cast<unsigned long long>(MAX_PRNG_STATE / 256), (states_num + 255) / 256);

    std::string network_config = "initprngs-" + std::to_string(sizeof(rocrand_state_xorwow)) + "x" +
                                 std::to_string(rng_mode) + "x" + std::to_string(wk_grp_num);

    auto&& kernels = handle.GetKernels(kernel_name, network_config);
    if(!kernels.empty())
    {
        kernels.front()(prng_states, prng_seed, states_num);
    }
    else
    {
        const std::vector<size_t> vld{256, 1, 1};
        const std::vector<size_t> vgd{wk_grp_num * 256, 1, 1};

        std::string params;
        params += "-DRUN_FORWARD=0  -DRUN_INIT_PRNG=1";
#if DROPOUT_DEBUG
        std::cout << "Threads allocated for PRNG states: " << vgd[0] << std::endl;
        std::cout << "Memory allocated for PRNG states: " << stateSizeInBytes << std::endl;
#endif
        handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
            prng_states, prng_seed, states_num);
#if DROPOUT_DEBUG
        std::cout << "Succeeded in launching InitPRNGState()." << stateSizeInBytes << std::endl;
#endif
    }
}

void DropoutDescriptor::Dropout(const Handle& handle,
                                const TensorDescriptor& noise_shape,
                                const TensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& yDesc,
                                Data_t y,
                                Data_t reserveSpace,
                                size_t reserveSpaceSizeInBytes,
                                size_t in_offset,
                                size_t out_offset,
                                size_t rsvsp_offset,
                                bool is_backward) const
{
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
    bool use_prng  = reserveSpace == nullptr;
    if(((use_rsvsp || use_mask) &&
        reserveSpaceSizeInBytes < xDesc.GetElementSize() * sizeof(bool)) ||
       (use_mask && reserveSpace == nullptr))
    {
        MIOPEN_THROW("Insufficient reservespace size");
    }

    if((is_backward ? 0 : stateSizeInBytes) + reserveSpaceSizeInBytes +
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

    size_t RD_BLCK    = /* (in_len[4] % 4 == 0) ? 4 : */ (in_len[2] % 2 == 0) ? 2 : 1;
    size_t total_work = (in_len[4] / RD_BLCK) * in_len[3] * in_len[2] * in_len[1] * in_len[0];

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

    std::string program_name   = "MIOpenDropoutHIP.cpp";
    std::string kernel_name    = "DropoutKernel";
    std::string network_config = "";
    if(is_backward)
    {
        network_config =
            "bwd-" + std::string(yDesc.GetType() == miopenHalf ? "fp16-" : "fp32-") + "-seed" +
            std::to_string(seed) + "-rng" + std::to_string(rng_mode) + "-prng" +
            std::to_string(static_cast<int>(use_prng)) + "-evo" +
            std::to_string(static_cast<int>(state_evo)) + "-blk" + std::to_string(RD_BLCK) + "-wg" +
            std::to_string(
                wk_grp_num) /* + "-noise" + std::to_string(noise_shape.GetLengths()[0]) */;
    }
    else
    {
        network_config =
            "fwd-" + std::string(xDesc.GetType() == miopenHalf ? "fp16-" : "fp32-") + "-seed" +
            std::to_string(seed) + "-rng" + std::to_string(rng_mode) + "-rsvsp" +
            std::to_string(static_cast<int>(use_rsvsp)) + "-mask" +
            std::to_string(static_cast<int>(use_mask)) + "-evo" +
            std::to_string(static_cast<int>(state_evo)) + "-blk" + std::to_string(RD_BLCK) + "-wg" +
            std::to_string(
                wk_grp_num) /* + "-noise" + std::to_string(noise_shape.GetLengths()[0])*/;
    }

    // TODO: Add noise shape
    // for(int i = 1; i < noise_shape.GetNumDims(); i++)
    //    network_config += "x" + std::to_string(noise_shape.GetLengths()[i]);

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    float amp_scale = float_equal(dropout, 1.0) ? 0 : 1 / (1 - dropout);
    if(!kernels.empty())
    {
        kernels.front()(pstates,
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
    else
    {
        std::string params;

        const std::string data_type = GetDataType(xDesc.GetType());
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

        // params += " -DRUN_FORWARD=1";

        if(is_backward)
        {
            params +=
                " -DUSE_MASK=" + std::to_string(static_cast<size_t>(!use_prng)) + " -DUSE_RSVSP=0";
        }
        else
        {
            params += " -DUSE_RSVSP=" + std::to_string(static_cast<size_t>(use_rsvsp));
            params += " -DUSE_MASK=" + std::to_string(static_cast<size_t>(use_mask));
        }

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
    }

    if(miopen::CheckNumericsEnabled())
    {
        std::cout << "Dropout forward output numerics check at dropout rate " << dropout
                  << std::endl;
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
}

} // namespace miopen
