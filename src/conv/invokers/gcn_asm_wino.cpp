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
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/handle.hpp>

namespace miopen {

InvokerFactory MakeGcnAsmWinoV2InvokerFactory(const WinoShaderArgsV2& args,
                                              conv::Direction direction,
                                              std::size_t sync_buffer_size,
                                              bool fused)
{
    const bool is_backWrW  = (direction == conv::Direction::BackwardWeights);
    const bool coop_launch = (args.sync_period != 0);
    const bool do_bias =
        ((args.flags64 & WinoShaderFlagsV2::F_BIAS) != static_cast<WinoShaderFlagsV2>(0));

    if(fused && (direction != conv::Direction::Forward))
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto k = handle.Run(kernels[0], coop_launch);

            // pointers
            ConstData_t data_addr;
            ConstData_t filter_addr;
            Data_t output_addr;
            ConstData_t bias_addr = nullptr;
            Data_t acc_addr       = nullptr;
            Data_t sync_addr      = nullptr;

            if(fused)
            {
                const auto& invoke_ctx = primitive_params.CastTo<fusion::FusionInvokeParams>();
                const auto& conv_params =
                    dynamic_cast<fusion::ConvolutionOpInvokeParam&>(*invoke_ctx.op_args.params[0]);

                data_addr   = invoke_ctx.in;
                filter_addr = conv_params.weights;
                output_addr = invoke_ctx.out;

                if(do_bias)
                {
                    const auto& bias_params =
                        dynamic_cast<fusion::BiasOpInvokeParam&>(*invoke_ctx.op_args.params[1]);
                    bias_addr = bias_params.bdata;
                }

                if(coop_launch)
                    sync_addr = invoke_ctx.GetWorkspace();
            }
            else if(!is_backWrW)
            {
                const auto& invoke_ctx = primitive_params.CastTo<conv::DataInvokeParams>();

                data_addr   = invoke_ctx.tensors.in;
                filter_addr = invoke_ctx.tensors.w;
                output_addr = invoke_ctx.tensors.out;

                if(coop_launch)
                    sync_addr = invoke_ctx.GetWorkspace();
            }
            else
            {
                const auto& invoke_ctx = primitive_params.CastTo<conv::WrWInvokeParams>();

                data_addr   = invoke_ctx.tensors.x;
                filter_addr = invoke_ctx.tensors.dy;
                output_addr = invoke_ctx.tensors.dw;

                if(coop_launch)
                    sync_addr = invoke_ctx.GetWorkspace();
            }

            // offsets
            uint64_t d_offset = 0;
            uint64_t f_offset = 0;
            uint64_t o_offset = 0;

            uint64_t b_offset = 0;
            uint64_t a_offset = 0;

            // activation parameters
            float alpha = 0.0f;
            float beta  = 0.0f;

            if(fused && (args.activation_mode != WinoShaderActivationModeV2_t::IDENTITY))
            {
                const auto& invoke_ctx = primitive_params.CastTo<fusion::FusionInvokeParams>();
                const int idx          = do_bias ? 2 : 1;
                const auto& activ_args =
                    dynamic_cast<fusion::ActivationOpInvokeParam&>(*invoke_ctx.op_args.params[idx]);
                if(args.activation_mode == WinoShaderActivationModeV2_t::SCALED_TANH)
                {
                    // The kernel uses a different expression in which alpha and beta are swapped
                    alpha = activ_args.activBeta;
                    beta  = activ_args.activAlpha;
                }
                else
                {
                    alpha = activ_args.activAlpha;
                    beta  = activ_args.activBeta;
                }
            }

            // clang-format off
            MIOPEN_LOG_I2(" N=" << args.N << " C=" << args.Cg << " H=" << args.H << " W=" << args.W
                << " K=" << args.Kg << " R=" << args.R << " S=" << args.S
                << " pad_H=" << args.pad_h << " pad_W=" << args.pad_w
                << " out_H=" << args.out_h << " out_W=" << args.out_w
                << " G=" << args.G
                << " alpha=" << alpha << " beta=" << beta << " act_mode=" << args.activation_mode
                << " d_offset=" << d_offset << " f_offset=" << f_offset
                << " o_offset=" << o_offset << " b_offset=" << b_offset
                << " d_N_stride=" << args.d_N_stride << " d_C_stride=" << args.d_C_stride
                << " d_H_stride=" << args.d_H_stride << " d_G_stride=" << args.d_G_stride
                << " f_K_stride=" << args.f_K_stride << " f_C_stride=" << args.f_C_stride
                << " f_R_stride=" << args.f_R_stride << " f_G_stride=" << args.f_G_stride
                << " o_N_stride=" << args.o_N_stride << " o_K_stride=" << args.o_K_stride
                << " o_H_stride=" << args.o_H_stride << " o_G_stride=" << args.o_G_stride
                << " n_groups=" << args.n_groups << " flags64=" << args.flags64
                << " sync_limit=" << static_cast<unsigned>(args.sync_limit)
                << " sync_period=" << static_cast<unsigned>(args.sync_period));
            // clang-format on

            if(coop_launch)
            {
                // Sync buffer that has to be zeroed before each shader dispatch
#if MIOPEN_BACKEND_HIP
                auto status = hipMemsetAsync(sync_addr, 0, sync_buffer_size, handle.GetStream());
                if(status != hipSuccess)
                    MIOPEN_THROW_HIP_STATUS(status, "hipMemsetAsync() failed");
#else
#error "Unsupported backend"
#endif
            }

            // clang-format off
            // Any reserved fields should be set to 0
            k(args.N,                     // uint32_t,    batch size
                args.Cg,                  // uint32_t,    number of input channels in each filter group
                args.H,                   // uint32_t,    input height
                args.W,                   // uint32_t,    input width
                args.Kg,                  // uint32_t,    number of output channels in each filter group
                args.n_groups,            // uint32_t,    number of shader groups
                args.flags64,             // uint64_t,    shader flags
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
                alpha,                    // fp32,        activation parameter alpha
                beta,                     // fp32,        activation parameter beta
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
                sync_addr,                // uint64_t,    address of sync buffer
                acc_addr,                 // uint64_t,    address of accumulation buffer
                a_offset);                // uint64_t,    byte offset for buffer referenced by acc_addr
            // clang-format on
        };
    };
}

} // namespace miopen
