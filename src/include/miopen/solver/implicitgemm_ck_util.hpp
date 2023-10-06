/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/batched_transpose_sol.hpp>

namespace miopen {
namespace solver {

template <typename ConvPtrsType>
typename ConvPtrsType::iterator FindConvPtrByID(ConvPtrsType& conv_ptrs,
                                                const std::string& kernel_id)
{
    return std::find_if(conv_ptrs.begin(), conv_ptrs.end(), [&kernel_id](const auto& ptr) {
        return ptr->GetTypeString() == kernel_id;
    });
}

template <typename DeviceOpType, typename CKArgsType>
std::vector<std::string> FillValidKernelsIDs(const ProblemDescription& problem)
{
    const auto args      = CKArgsType{problem};
    const auto conv_ptrs = DeviceOpType::GetInstances();
    assert(!conv_ptrs.empty());

    std::vector<std::string> valid_kernels;
    valid_kernels.reserve(conv_ptrs.size());
    for(size_t idx = 0; idx < conv_ptrs.size(); ++idx)
    {
        if(args.IsSupportedBy(conv_ptrs[idx]))
            valid_kernels.emplace_back(std::move(conv_ptrs[idx]->GetTypeString()));
    }
    assert(!valid_kernels.empty());
    return valid_kernels;
}

template <typename DeviceOpType, typename CKArgsType>
bool IsCKArgsSupported(const ProblemDescription& problem, const std::string& kernel_id)
{
    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    return (ptr_iter != conv_ptrs.end()) && CKArgsType{problem}.IsSupportedBy(*ptr_iter);
}

template <typename DeviceOpType, typename CKArgsType>
bool IsCKApplicable(const ProblemDescription& problem)
{
    const auto args = CKArgsType{problem};
    if(!std::all_of(args.strides.begin(), args.strides.end(), [](auto x) { return x == 1; }))
        return false;

    const auto ptrs = DeviceOpType::GetInstances();
    return std::any_of(
        ptrs.begin(), ptrs.end(), [&args](auto& ptr) { return args.IsSupportedBy(ptr); });
}

template <typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactory(const ProblemDescription& problem, const std::string& kernel_id)
{
    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    if(ptr_iter == conv_ptrs.end())
        MIOPEN_THROW("PerformanceConfig kernel '" + kernel_id + "' does not exist");

    ConvSolution result;
    result.invoker_factory =
        [ck_args     = CKArgsType{problem},
         sh_conv_ptr = std::shared_ptr{std::move(*ptr_iter)}](const std::vector<Kernel>&) mutable {
            return [ck_args = std::move(ck_args), sh_conv_ptr = std::move(sh_conv_ptr)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& data_ctx = primitive_parameters.CastTo<CastType>();
                auto argument_ptr    = ck_args.MakeArgPtr(sh_conv_ptr, data_ctx.tensors);
                auto invoker_ptr     = sh_conv_ptr->MakeInvokerPointer();

                const auto enable_profiling = handle.IsProfilingEnabled();
                float elapsed_time =
                    invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
                if(enable_profiling)
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed_time);
                }
            };
        };
    return result;
}

struct TransposeInstance {
  size_t tensor_sz = 0;
  std::vector<OpKernelArg> kern_args{};
  int index = -1;
  size_t buf_offset = 0;

  template <typename TransSolnType>
  TransposeInstance(const TransSolnType& trans_sol, int idx, const MultiBufferWorkspaceTraits& wt) {
    tensor_sz = trans_sol.GetOutputTensorSize();
    kern_args = trans_sol.GetKernelArg();
    index = idx;
    assert(index >= 0);
    buf_offset = wt.GetOffset(index);
  }

  shared<Data_t> AllocBuffer(const Handle& handle, Data_t workSpace) const {
    auto buf_handle = handle.CreateSubBuffer(workSpace, buf_offset, tensor_sz);
    assert(buf_handle.get());
    return buf_handle;
  }

  void Run(const Handle& handle, const std::vector<Kernel>& kernels, Data_t out_ptr, ConstData_t in_ptr) {
    assert(out_ptr);
    assert(in_ptr);
    assert(kernels.size() > index);

    kern_args[0] = out_ptr;
    kern_args[1] = in_ptr;
    MIOPEN_LOG_I("Running kernel for transpose: " << kernels[index].name);
    MIOPEN_LOG_I("out_ptr = " << out_ptr << ", in_ptr = " << in_ptr);

    for (int i = 2; i < kern_args.size(); ++i) {
      MIOPEN_LOG_I("arg #" << i << " = " << *(reinterpret_cast<uint32_t*>(kern_args[i].buffer.data())));
    }

    handle.Run(kernels[index])(kern_args);
    if (handle.IsProfilingEnabled()) {
      handle.AccumKernelTime(handle.GetKernelTime());
    }
  }

  TransposeInstance() = delete;
  TransposeInstance(const TransposeInstance&) = default;
  TransposeInstance(TransposeInstance&&) = default;
  ~TransposeInstance() = default;
};

template <typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryNCHW(const ExecutionContext& ctx, const ProblemDescription& problem, const std::string& kernel_id)
{

    assert(problem.IsLayoutDefault());

    ConvSolution result;
    auto ck_args = CKArgsType{problem};

    TransposeSolutionDefault2Ndhwc tr_in(
        ctx,
        problem.GetInDataType(),
        ck_args.N,
        ck_args.C1,
        ck_args.Di,
        ck_args.Hi,
        ck_args.Wi);

    TransposeSolutionDefault2Ndhwc tr_wei(
        ctx,
        problem.GetWeightsDataType(),
        ck_args.K1,
        ck_args.C,
        ck_args.Z,
        ck_args.Y,
        ck_args.X);

    TransposeSolutionNdhwc2Default tr_out(
        ctx,
        problem.GetOutDataType(),
        ck_args.N,
        ck_args.K1,
        ck_args.Do,
        ck_args.Ho,
        ck_args.Wo);

    result.construction_params.insert(
        result.construction_params.end(),
        {tr_in.GetKernelInfo(), 
         tr_wei.GetKernelInfo(),
         tr_out.GetKernelInfo()});


    constexpr size_t buf_alignment = 256;
    MultiBufferWorkspaceTraits wt(
      {
        tr_in.GetOutputTensorSize(),
        tr_wei.GetOutputTensorSize(),
        tr_out.GetOutputTensorSize()
      },
      buf_alignment);


    TransposeInstance tr_inst_in(tr_in, 0, wt);
    TransposeInstance tr_inst_wei(tr_wei, 1, wt);
    TransposeInstance tr_inst_out(tr_out, 2, wt);


    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    if(ptr_iter == conv_ptrs.end())
        MIOPEN_THROW("PerformanceConfig kernel '" + kernel_id + "' does not exist");

    result.invoker_factory =
        [ck_args     = std::move(ck_args),
         sh_conv_ptr = std::shared_ptr{std::move(*ptr_iter)},
         tr_inst_in = std::move(tr_inst_in),
         tr_inst_wei = std::move(tr_inst_wei),
         tr_inst_out = std::move(tr_inst_out)]
           (const std::vector<Kernel>& kernels) mutable {
            return [ck_args = std::move(ck_args), 
              kernels = kernels,
              sh_conv_ptr = std::move(sh_conv_ptr),
              tr_inst_in = std::move(tr_inst_in),
              tr_inst_wei = std::move(tr_inst_wei),
              tr_inst_out = std::move(tr_inst_out)] (const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable 
            {
                handle.ResetKernelTime();

                const auto& data_ctx = primitive_parameters.CastTo<CastType>();

                auto tmp_buf_in = tr_inst_in.AllocBuffer(handle, data_ctx.workSpace);
                auto tmp_buf_wei = tr_inst_wei.AllocBuffer(handle, data_ctx.workSpace);
                auto tmp_buf_out = tr_inst_out.AllocBuffer(handle, data_ctx.workSpace);

                MIOPEN_LOG_I("Running input transpose");
                tr_inst_in.Run(handle, kernels, tmp_buf_in.get(), data_ctx.tensors.in);
                handle.Finish();
                MIOPEN_LOG_I("Running weight transpose");
                tr_inst_wei.Run(handle, kernels, tmp_buf_wei.get(), data_ctx.tensors.w);
                handle.Finish();

                MIOPEN_LOG_I("Running CK convolution");
                auto argument_ptr    = ck_args.MakeArgPtr(sh_conv_ptr, 
                     tmp_buf_in.get(),
                    tmp_buf_wei.get(),
                    tmp_buf_out.get());

                auto invoker_ptr     = sh_conv_ptr->MakeInvokerPointer();
                float elapsed_time =
                    invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), handle.IsProfilingEnabled()});

                if(handle.IsProfilingEnabled())
                {
                    handle.AccumKernelTime(elapsed_time);
                }

                handle.Finish();
                MIOPEN_LOG_I("Running output transpose");
                tr_inst_out.Run(handle, kernels, data_ctx.tensors.out, tmp_buf_out.get());
                
                handle.Finish();
                MIOPEN_LOG_I("Inovker finished executing");


            };
        };
    return result;
}

} // namespace solver
} // namespace miopen
