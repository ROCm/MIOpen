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

#include <miopen/conv/invokers/gcn_asm_wino.hpp>

#include <miopen/conv_algo_name.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/handle.hpp>

namespace miopen {
namespace conv {

InvokerFactory MakeGcnAsmWinoV40InvokerFactory(const WinoShaderArgsV40& args,
                                               Direction direction,
                                               std::size_t sync_buffer_size)
{
    const bool is_backWrW = (direction == Direction::BackwardWeights);

    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto k = handle.Run(kernels[0]);

            const auto data_addr   = !is_backWrW
                                         ? primitive_params.CastTo<DataInvokeParams>().tensors.in
                                         : primitive_params.CastTo<WrWInvokeParams>().tensors.x;
            const auto filter_addr = !is_backWrW
                                         ? primitive_params.CastTo<DataInvokeParams>().tensors.w
                                         : primitive_params.CastTo<WrWInvokeParams>().tensors.dy;
            const auto output_addr = !is_backWrW
                                         ? primitive_params.CastTo<DataInvokeParams>().tensors.out
                                         : primitive_params.CastTo<WrWInvokeParams>().tensors.dw;
            const auto sync_addr   = !is_backWrW
                                         ? primitive_params.CastTo<DataInvokeParams>().workSpace
                                         : primitive_params.CastTo<WrWInvokeParams>().workSpace;

            uint64_t bias_addr = 0;

            uint64_t d_offset = 0;
            uint64_t f_offset = 0;
            uint64_t o_offset = 0;
            uint64_t b_offset = 0;

            // clang-format off
            MIOPEN_LOG_I2(" N=" << args.N << " C=" << args.C << " H=" << args.H << " W=" << args.W
                << " K=" << args.K << " R=" << args.R << " S=" << args.S
                << " pad_H=" << args.pad_h << " pad_W=" << args.pad_w
                << " out_H=" << args.out_h << " out_W=" << args.out_w
                << " G=" << args.G
                << " alpha=" << args.alpha << " beta=" << args.beta << " act_mode=" << args.activation_mode
                << " d_offset=" << d_offset << " f_offset=" << f_offset
                << " o_offset=" << o_offset << " b_offset=" << b_offset
                << " d_N_stride=" << args.d_N_stride << " d_C_stride=" << args.d_C_stride
                << " d_H_stride=" << args.d_H_stride << " d_G_stride=" << args.d_G_stride
                << " f_K_stride=" << args.f_K_stride << " f_C_stride=" << args.f_C_stride
                << " f_R_stride=" << args.f_R_stride << " f_G_stride=" << args.f_G_stride
                << " o_N_stride=" << args.o_N_stride << " o_K_stride=" << args.o_K_stride
                << " o_H_stride=" << args.o_H_stride << " o_G_stride=" << args.o_G_stride
                << " n_groups=" << args.n_groups << " flags=" << args.flags
                << " sync_limit=" << args.sync_limit << " sync_period=" << args.sync_period);
            // clang-format on

            // 2KB sync buffer that has to be zeroed before each shader dispatch
            hipMemsetAsync(sync_addr, 0, sync_buffer_size, handle.GetStream());

            // clang-format off
            // Any reserved fields should be set to 0
            k(args.N,                     // uint32_t,    batch size
                args.C,                   // uint32_t,    number of input channels in each filter group
                args.H,                   // uint32_t,    input height
                args.W,                   // uint32_t,    input width
                args.K,                   // uint32_t,    number of output channels in each filter group
                args.n_groups,            // uint32_t,    number of shader groups
                args.flags,               // uint64_t,    shader flags
                data_addr,                // uint64_t,    address of input tensor
                filter_addr,              // uint64_t,    address of filter tensor
                output_addr,              // uint64_t,    address of output tensor
                static_cast<uint64_t>(0), // uint64_t,    not used, for backward compatibility only
                args.R,                   // uint32_t,    filter height
                args.S,                   // uint32_t,    filter width
                args.pad_h,               // int32_t,     padding in h dimension
                args.pad_w,               // int32_t,     padding in w dimension
                args.out_h,               // uint32_t,    output height
                args.out_w,               // uint32_t,    output width
                bias_addr,                // uint64_t,    address of bias buffer
                args.alpha,               // fp32,        activation parameter alpha
                args.beta,                // fp32,        activation parameter beta
                d_offset,                 // uint64_t,    byte offset for buffer referenced by data_addr
                f_offset,                 // uint64_t,    byte offset for buffer referenced by filter_addr
                o_offset,                 // uint64_t,    byte offset for buffer referenced by output_addr
                b_offset,                 // uint64_t,    byte offset for buffer referenced by bias_addr
                args.d_N_stride,          // uint32_t,    stride in number of elements of the N dimension of the input data buffer
                args.d_C_stride,          // uint32_t,    stride in number of elements of the C dimension of the input data buffer
                args.d_H_stride,          // uint32_t,    stride in number of elements of the H dimension of the input data buffer
                static_cast<uint32_t>(0), // uint32_t,    reserved
                args.f_K_stride,          // uint32_t,    stride in number of elements of the K dimension of the filter buffer
                args.f_C_stride,          // uint32_t,    stride in number of elements of the C dimension of the filter buffer
                args.f_R_stride,          // uint32_t,    stride in number of elements of the R dimension of the filter buffer
                static_cast<uint32_t>(0), // uint32_t,    reserved
                args.o_N_stride,          // uint32_t,    stride in number of elements of the N dimension of the output buffer
                args.o_K_stride,          // uint32_t,    stride in number of elements of the K dimension of the output buffer
                args.o_H_stride,          // uint32_t,    stride in number of elements of the H dimension of the output buffer
                static_cast<uint32_t>(0), // uint32_t,    reserved
                args.G,                   // uint32_t,    number of filter groups
                args.d_G_stride,          // uint32_t,    stride in number of elements of the G dimension of the input data buffer
                args.f_G_stride,          // uint32_t,    stride in number of elements of the G dimension of the filter buffer
                args.o_G_stride,          // uint32_t,    stride in number of elements of the G dimension of the output buffer
                args.activation_mode,     // uint8_t,     activation mode
                args.sync_limit,          // uint8_t,     maximum number of sync attempts
                args.sync_period,         // uint8_t,     synchronization period
                static_cast<uint8_t>(0),  // uint8_t,     reserved
                static_cast<uint32_t>(0), // uint32_t,    reserved
                sync_addr);               // uint64_t,    address of sync buffer
            // clang-format on
        };
    };
}

} // namespace conv
} // namespace miopen
