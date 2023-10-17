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

template <typename DeviceOpType,
          typename CKArgsType,
          typename ProblemDescriptionType = ProblemDescription>
std::vector<std::string> FillValidKernelsIDs(const ProblemDescriptionType& problem)
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

template <typename DeviceOpType,
          typename CKArgsType,
          typename ProblemDescriptionType = ProblemDescription>
bool IsCKArgsSupported(const ProblemDescriptionType& problem, const std::string& kernel_id)
{
    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    return (ptr_iter != conv_ptrs.end()) && CKArgsType{problem}.IsSupportedBy(*ptr_iter);
}

template <typename DeviceOpType,
          typename CKArgsType,
          typename ProblemDescriptionType = ProblemDescription>
bool IsCKApplicable(const ProblemDescriptionType& problem)
{
    const auto args = CKArgsType{problem};

    const auto ptrs = DeviceOpType::GetInstances();
    return std::any_of(
        ptrs.begin(), ptrs.end(), [&args](auto& ptr) { return args.IsSupportedBy(ptr); });
}

template <typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename ProblemDescriptionType = ProblemDescription>
ConvSolution InitInvokerFactoryNHWC([[maybe_unused]] const ExecutionContext& ctx,
                                    const ProblemDescriptionType& problem,
                                    const std::string& kernel_id)
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

template <typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename ProblemDescriptionType = ProblemDescription>
ConvSolution InitAnyInvokerFactory(const ProblemDescriptionType& problem,
                                   const std::string& kernel_id)
{
    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    if(ptr_iter == conv_ptrs.end())
        return {miopenStatusInvalidValue};

    ConvSolution result;
    result.invoker_factory =
        [ck_args     = CKArgsType{problem},
         sh_conv_ptr = std::shared_ptr{std::move(*ptr_iter)}](const std::vector<Kernel>&) mutable {
            return [ck_args = std::move(ck_args), sh_conv_ptr = std::move(sh_conv_ptr)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& data_ctx = primitive_parameters.CastTo<CastType>();
                auto argument_ptr    = ck_args.MakeArgPtr(sh_conv_ptr, data_ctx);
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

struct TransposeInstance
{
    size_t tensor_sz = 0;
    std::vector<OpKernelArg> kern_args{};
    int index         = -1;
    size_t buf_offset = 0;
    shared<Data_t> buf_handle{};

    template <typename TransSolnType>
    TransposeInstance(const TransSolnType& trans_sol, int idx, const MultiBufferWorkspaceTraits& wt)
    {
        tensor_sz = trans_sol.GetOutputTensorSize();
        kern_args = trans_sol.GetKernelArg();
        index     = idx;
        assert(index >= 0);
        buf_offset = wt.GetOffset(index);
    }

    void AllocBuffer(const Handle& handle, Data_t workSpace)
    {
        buf_handle = handle.CreateSubBuffer(workSpace, buf_offset, tensor_sz);
        assert(buf_handle.get());
    }

    Data_t GetBufferPtr() const { return buf_handle.get(); }

    void ConvertFrom(const Handle& handle, const std::vector<Kernel>& kernels, ConstData_t in_ptr)
    {
        Run(handle, kernels, buf_handle.get(), in_ptr);
    }

    void ConvertTo(const Handle& handle, const std::vector<Kernel>& kernels, Data_t out_ptr)
    {
        Run(handle, kernels, out_ptr, buf_handle.get());
    }

    TransposeInstance()                         = delete;
    TransposeInstance(const TransposeInstance&) = default;
    TransposeInstance(TransposeInstance&&)      = default;
    ~TransposeInstance()                        = default;

private:
    void Run(const Handle& handle,
             const std::vector<Kernel>& kernels,
             Data_t out_ptr,
             ConstData_t in_ptr)
    {
        assert(out_ptr);
        assert(in_ptr);
        assert(kernels.size() > index);

        kern_args[0] = out_ptr;
        kern_args[1] = in_ptr;

        handle.Run(kernels[index])(kern_args);
        if(handle.IsProfilingEnabled())
        {
            handle.AccumKernelTime(handle.GetKernelTime());
        }
    }
};

template <typename InTransSolType, typename WeiTransSolType, typename OutTransSolType>
struct TransposeInstanceMaker2D
{

    template <typename CKArgsType>
    std::tuple<TransposeInstance, TransposeInstance, TransposeInstance>
    operator()(ConvSolution& result,
               const ExecutionContext& ctx,
               const ProblemDescription& problem,
               const CKArgsType& ck_args) const
    {

        InTransSolType tr_in(
            ctx, problem.GetInDataType(), ck_args.N, ck_args.C1, ck_args.Hi, ck_args.Wi);

        WeiTransSolType tr_wei(
            ctx, problem.GetWeightsDataType(), ck_args.K1, ck_args.C, ck_args.Y, ck_args.X);

        OutTransSolType tr_out(
            ctx, problem.GetOutDataType(), ck_args.N, ck_args.K1, ck_args.Ho, ck_args.Wo);

        result.construction_params.insert(
            result.construction_params.end(),
            {tr_in.GetKernelInfo(), tr_wei.GetKernelInfo(), tr_out.GetKernelInfo()});

        constexpr size_t buf_alignment = 256;
        MultiBufferWorkspaceTraits wt({tr_in.GetOutputTensorSize(),
                                       tr_wei.GetOutputTensorSize(),
                                       tr_out.GetOutputTensorSize()},
                                      buf_alignment);

        return std::make_tuple(TransposeInstance(tr_in, 0, wt),
                               TransposeInstance(tr_wei, 1, wt),
                               TransposeInstance(tr_out, 2, wt));
    }
};

template <typename InTransSolType, typename WeiTransSolType, typename OutTransSolType>
struct TransposeInstanceMaker3D
{

    template <typename CKArgsType>
    std::tuple<TransposeInstance, TransposeInstance, TransposeInstance>
    operator()(ConvSolution& result,
               const ExecutionContext& ctx,
               const ProblemDescription& problem,
               const CKArgsType& ck_args) const
    {

        InTransSolType tr_in(ctx,
                             problem.GetInDataType(),
                             ck_args.N,
                             ck_args.C1,
                             ck_args.Di,
                             ck_args.Hi,
                             ck_args.Wi);

        WeiTransSolType tr_wei(ctx,
                               problem.GetWeightsDataType(),
                               ck_args.K1,
                               ck_args.C,
                               ck_args.Z,
                               ck_args.Y,
                               ck_args.X);

        OutTransSolType tr_out(ctx,
                               problem.GetOutDataType(),
                               ck_args.N,
                               ck_args.K1,
                               ck_args.Do,
                               ck_args.Ho,
                               ck_args.Wo);

        result.construction_params.insert(
            result.construction_params.end(),
            {tr_in.GetKernelInfo(), tr_wei.GetKernelInfo(), tr_out.GetKernelInfo()});

        constexpr size_t buf_alignment = 256;
        MultiBufferWorkspaceTraits wt({tr_in.GetOutputTensorSize(),
                                       tr_wei.GetOutputTensorSize(),
                                       tr_out.GetOutputTensorSize()},
                                      buf_alignment);

        return std::make_tuple(TransposeInstance(tr_in, 0, wt),
                               TransposeInstance(tr_wei, 1, wt),
                               TransposeInstance(tr_out, 2, wt));
    }
};

#if 1

namespace internal {

enum class ConvOperandType: int {
  Input = 0,
  Weights,
  Output
};

enum class  TransposeType: int {
  NHWC_TO_NCHW = 0,
  NCHW_TO_NHWC
};

template <unsigned ND, TransposeType TPOSE_TYPE>
struct ChooseTransposeSolver;

template <unsigned ND> 
ChooseTransposeSolver<ND, NHWC_TO_NCHW> {
  static_assert(ND == 2 or ND == 3);
  using type = std::conditional_t<ND == 2,
        TransposeSolutionNhwc2Default,
        TransposeSolutionNdhwc2Default>;
};

template <unsigned ND> 
ChooseTransposeSolver<ND, NCHW_TO_NHWC> {
  static_assert(ND == 2 or ND == 3);
  using type = std::conditional_t<ND == 2,
        TransposeSolutionDefault2Nhwc,
        TransposeSolutionDefault2Ndhwc>;
};


template <ConvOperandType CONV_OP>
class TransposeInstanceTagged: public TransposeInstance {

public:

  template <typename TransSolnType>
  TransposeInstanceTagged(
      const TransSolnType& sol, 
      int idx,
      const MultiBufferWorkspaceTraits& wt): 
    TransposeInstance(sol, idx, wt) {}

  constexpr ConvOperandType GetConvOperandType() const noexcept {
    return  CONV_OP;
  }

    void ConvertFrom(const Handle& handle, const std::vector<Kernel>& kernels, const ConvTensors& tensors)
    {
      ConvertFrom(handle, kernels, pickTensorPtr(tensors));
    }

    void ConvertTo(const Handle& handle, const std::vector<Kernel>& kernels, const ConvTensors& tensors)    
    {
      ConvertTo(handle, kernels, pickTensorPtr(tensors));
    }

private:

    Data_t pickTensorPtr(const ConvTensors& tensors) {
      std::array<Data_t, 3> data_ptrs = {
        const_cast<Data_t>(tensors.x),
        const_cast<Data_t>(tensors.w),
        tensors.y};

      using IntType = std::underlying_type_t<ConvOperandType>;
      return data_ptrs[static_cast<IntType>(CONV_OP)];
    }

};

template <unsigned ND, TransposeType TPOSE_TYPE, ConvOperandType CONV_OP>
struct TransposeOperand;

template <unsigned ND, TransposeType TPOSE_TYPE>
struct TransposeOperand<ND, TPOSE_TYPE, ConvOperandType::Input> {
  using TransposeSolver = typename ChooseTransposeSolver<ND, TPOSE_TYPE>::type;

  template <typename CKArgsType>
  TransposeSolver MakeTransposeSolver(const ExecutionContext& ctx,
      const ProblemDescription& problem,
      const CKArgsType& ck_args) const {

    if constexpr(ND == 2) {
      return TransposeSolver {
        ctx,
        problem.GetInDataType(),
        ck_args.N,
        ck_args.C1,
        ck_args.Di,
        ck_args.Hi,
        ck_args.Wi};
    } else {
      return TransposeSolver {
        ctx,
        problem.GetInDataType(),
        ck_args.N,
        ck_args.C1,
        ck_args.Hi,
        ck_args.Wi};

    }
  }

  TransposeInstanceTagged<ConvOperandType::Input>


};


void
sortByEnumOrder(std::array<TransposeInstanceTagged*>, 3>& tagged_ptrs) {

  std::sort(tagged_ptrs.begin(), tagged_ptrs.end(),
      [] (const auto& left, const auto& right) {
        using IntType = std::underlying_type_t<ConvOperandType>;
        return static_cast<IntType>(left->GetConvOperandType()) < static_cast<IntType>(right->GetConvOperandType());
      });
}


} // end namespace internal

template <typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename TpostInstMaker,
          typename PreTposeFn,
          typename PostTposeFn>
ConvSolution InitInvokerFactoryNCHW(const ExecutionContext& ctx,
                                    const ProblemDescription& problem,
                                    const std::string& kernel_id,
                                    TpostInstMaker&& tr_inst_maker,
                                    PreTposeFn&& pre_tpose_fn,
                                    PostTposeFn&& post_tpose_fn)
{

    assert(problem.IsLayoutDefault());

    ConvSolution result;
    auto ck_args = CKArgsType{problem};

    auto input1_solv = input1.MakeTransposeSolver(ctx, problem, ck_args);
    auto input2_solv = input2.MakeTransposeSolver(ctx, problem, ck_args);
    auto output1_solv = output1.MakeTransposeSolver(ctx, problem, ck_args);

    auto combined_initialization = [&] (auto... tpose_ops) {
      result.construction_params.insert(
          result.construction_params.end(),
          { tpose_ops.GetKernelInfo(), ...});

      constexpr size_t buf_alignment = 256ull;
      MultiBufferWorkspaceTraits wt(
          { tpose_ops.GetOutputTensorSize(), ...},
          buf_alignment);

      int i = 0;
      size_t[] unused = { tpose_ops.InitBufferOffset(i++, wt),...};
      assert(size_t(i) == sizeof...(tpose_ops));
    };


    { 
      { // inside invoker 
        
        
            input1.AllocBuffer(handle, data_ctx.workSpace);
            input2.AllocBuffer(handle, data_ctx.workSpace);
            output1.AllocBuffer(handle, data_ctx.workSpace);



        input1.ConvertFrom(handle, kernels, data_ctx.tensors);

        input2.ConvertFrom(handle, kernels, data_ctx.tensors);


        std::array<TransposeInstanceTagged*, 3> tagged_ptrs = 
                          { &input1_ti, &input2_ti, output1_ti};
        sortByEnumOrder(tagged_ptrs);


            auto invoker_ptr   = sh_conv_ptr->MakeInvokerPointer();
            auto argument_ptr  = ck_args.MakeArgPtr(sh_conv_ptr,
                                                   tagged_ptrs[0]->GetBufferPtr(),
                                                   tagged_ptrs[1]->GetBufferPtr(),
                                                   tagged_ptrs[2]->GetBufferPtr());


         output1.ConvertFrom(handle, kernels, data_ctx.tensors);


      }
    }


    auto [_tr_inst_in, _tr_inst_wei, _tr_inst_out] = tr_inst_maker(result, ctx, problem, ck_args);

    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    if(ptr_iter == conv_ptrs.end())
        MIOPEN_THROW("PerformanceConfig kernel '" + kernel_id + "' does not exist");

    result.invoker_factory = [pre_tpose_fn,
                              post_tpose_fn,
                              ck_args     = std::move(ck_args),
                              sh_conv_ptr = std::shared_ptr{std::move(*ptr_iter)},
                              tr_inst_in  = std::move(_tr_inst_in),
                              tr_inst_wei = std::move(_tr_inst_wei),
                              tr_inst_out = std::move(_tr_inst_out)](
                                 const std::vector<Kernel>& kernels) mutable {
        return [pre_tpose_fn,
                post_tpose_fn,
                kernels,
                ck_args     = std::move(ck_args),
                sh_conv_ptr = std::move(sh_conv_ptr),
                tr_inst_in  = std::move(tr_inst_in),
                tr_inst_wei = std::move(tr_inst_wei),
                tr_inst_out = std::move(tr_inst_out)](
                   const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            handle.ResetKernelTime();

            const auto& data_ctx = primitive_parameters.CastTo<CastType>();

            tr_inst_in.AllocBuffer(handle, data_ctx.workSpace);
            tr_inst_wei.AllocBuffer(handle, data_ctx.workSpace);
            tr_inst_out.AllocBuffer(handle, data_ctx.workSpace);

            pre_tpose_fn(handle, kernels, data_ctx.tensors, tr_inst_in, tr_inst_wei, tr_inst_out);

            auto invoker_ptr   = sh_conv_ptr->MakeInvokerPointer();
            auto argument_ptr  = ck_args.MakeArgPtr(sh_conv_ptr,
                                                   tr_inst_in.GetBufferPtr(),
                                                   tr_inst_wei.GetBufferPtr(),
                                                   tr_inst_out.GetBufferPtr());
            float elapsed_time = invoker_ptr->Run(
                argument_ptr.get(), {handle.GetStream(), handle.IsProfilingEnabled()});
            if(handle.IsProfilingEnabled())
            {
                handle.AccumKernelTime(elapsed_time);
            }

            post_tpose_fn(handle, kernels, data_ctx.tensors, tr_inst_in, tr_inst_wei, tr_inst_out);
        };
    };
    return result;
}

#else 

template <typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename TpostInstMaker,
          typename PreTposeFn,
          typename PostTposeFn>
ConvSolution InitInvokerFactoryNCHW(const ExecutionContext& ctx,
                                    const ProblemDescription& problem,
                                    const std::string& kernel_id,
                                    TpostInstMaker&& tr_inst_maker,
                                    PreTposeFn&& pre_tpose_fn,
                                    PostTposeFn&& post_tpose_fn)
{

    assert(problem.IsLayoutDefault());

    ConvSolution result;
    auto ck_args = CKArgsType{problem};

    auto [_tr_inst_in, _tr_inst_wei, _tr_inst_out] = tr_inst_maker(result, ctx, problem, ck_args);

    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    if(ptr_iter == conv_ptrs.end())
        MIOPEN_THROW("PerformanceConfig kernel '" + kernel_id + "' does not exist");

    result.invoker_factory = [pre_tpose_fn,
                              post_tpose_fn,
                              ck_args     = std::move(ck_args),
                              sh_conv_ptr = std::shared_ptr{std::move(*ptr_iter)},
                              tr_inst_in  = std::move(_tr_inst_in),
                              tr_inst_wei = std::move(_tr_inst_wei),
                              tr_inst_out = std::move(_tr_inst_out)](
                                 const std::vector<Kernel>& kernels) mutable {
        return [pre_tpose_fn,
                post_tpose_fn,
                kernels,
                ck_args     = std::move(ck_args),
                sh_conv_ptr = std::move(sh_conv_ptr),
                tr_inst_in  = std::move(tr_inst_in),
                tr_inst_wei = std::move(tr_inst_wei),
                tr_inst_out = std::move(tr_inst_out)](
                   const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            handle.ResetKernelTime();

            const auto& data_ctx = primitive_parameters.CastTo<CastType>();

            tr_inst_in.AllocBuffer(handle, data_ctx.workSpace);
            tr_inst_wei.AllocBuffer(handle, data_ctx.workSpace);
            tr_inst_out.AllocBuffer(handle, data_ctx.workSpace);

            pre_tpose_fn(handle, kernels, data_ctx.tensors, tr_inst_in, tr_inst_wei, tr_inst_out);

            auto invoker_ptr   = sh_conv_ptr->MakeInvokerPointer();
            auto argument_ptr  = ck_args.MakeArgPtr(sh_conv_ptr,
                                                   tr_inst_in.GetBufferPtr(),
                                                   tr_inst_wei.GetBufferPtr(),
                                                   tr_inst_out.GetBufferPtr());
            float elapsed_time = invoker_ptr->Run(
                argument_ptr.get(), {handle.GetStream(), handle.IsProfilingEnabled()});
            if(handle.IsProfilingEnabled())
            {
                handle.AccumKernelTime(elapsed_time);
            }

            post_tpose_fn(handle, kernels, data_ctx.tensors, tr_inst_in, tr_inst_wei, tr_inst_out);
        };
    };
    return result;
}

#endif//1

#if 1
template <int ND, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryFwdNCHW(const ExecutionContext& ctx,
                                       const ProblemDescription& problem,
                                       const std::string& kernel_id)
{

    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using Input1 = TransposeInput<ND, internal::TransposeType::NHWC_TO_NCHW, internal::ConvOperandType::Input>;
    using Input2 = TransposeInput<ND, internal::TransposeType::NHWC_TO_NCHW, internal::ConvOperandType::Weights>;
    using Output = TransposeOutput<ND, internal::TransposeType::NCHW_TO_NHWC, internal::ConvOperandType::Output>;




    using TrInstMaker =
        std::conditional_t<ND == 2,
                           TransposeInstanceMaker2D<TransposeSolutionDefault2Nhwc,
                                                    TransposeSolutionDefault2Nhwc,
                                                    TransposeSolutionNhwc2Default>,
                           TransposeInstanceMaker3D<TransposeSolutionDefault2Ndhwc,
                                                    TransposeSolutionDefault2Ndhwc,
                                                    TransposeSolutionNdhwc2Default>>;

    TrInstMaker tr_inst_maker{};

    auto pre_tpose_fn = [](const Handle& handle,
                           const std::vector<Kernel>& kernels,
                           const ConvTensors& tensors,
                           TransposeInstance& tr_inst_in,
                           TransposeInstance& tr_inst_wei,
                           [[maybe_unused]] TransposeInstance& tr_inst_out) {
        tr_inst_in.ConvertFrom(handle, kernels, tensors.x);
        tr_inst_wei.ConvertFrom(handle, kernels, tensors.w);
    };

    auto post_tpose_fn = [](const Handle& handle,
                            const std::vector<Kernel>& kernels,
                            const ConvTensors& tensors,
                            [[maybe_unused]] TransposeInstance& tr_inst_in,
                            [[maybe_unused]] TransposeInstance& tr_inst_wei,
                            TransposeInstance& tr_inst_out) {
        tr_inst_out.ConvertTo(handle, kernels, const_cast<Data_t>(tensors.y));
    };

    return InitInvokerFactoryNCHW<DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, tr_inst_maker, pre_tpose_fn, post_tpose_fn);
}

#else 

template <int ND, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryFwdNCHW(const ExecutionContext& ctx,
                                       const ProblemDescription& problem,
                                       const std::string& kernel_id)
{
    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using TrInstMaker =
        std::conditional_t<ND == 2,
                           TransposeInstanceMaker2D<TransposeSolutionDefault2Nhwc,
                                                    TransposeSolutionDefault2Nhwc,
                                                    TransposeSolutionNhwc2Default>,
                           TransposeInstanceMaker3D<TransposeSolutionDefault2Ndhwc,
                                                    TransposeSolutionDefault2Ndhwc,
                                                    TransposeSolutionNdhwc2Default>>;

    TrInstMaker tr_inst_maker{};

    auto pre_tpose_fn = [](const Handle& handle,
                           const std::vector<Kernel>& kernels,
                           const ConvTensors& tensors,
                           TransposeInstance& tr_inst_in,
                           TransposeInstance& tr_inst_wei,
                           [[maybe_unused]] TransposeInstance& tr_inst_out) {
        tr_inst_in.ConvertFrom(handle, kernels, tensors.x);
        tr_inst_wei.ConvertFrom(handle, kernels, tensors.w);
    };

    auto post_tpose_fn = [](const Handle& handle,
                            const std::vector<Kernel>& kernels,
                            const ConvTensors& tensors,
                            [[maybe_unused]] TransposeInstance& tr_inst_in,
                            [[maybe_unused]] TransposeInstance& tr_inst_wei,
                            TransposeInstance& tr_inst_out) {
        tr_inst_out.ConvertTo(handle, kernels, const_cast<Data_t>(tensors.y));
    };

    return InitInvokerFactoryNCHW<DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, tr_inst_maker, pre_tpose_fn, post_tpose_fn);
}
#endif

template <int ND, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryBwdNCHW(const ExecutionContext& ctx,
                                       const ProblemDescription& problem,
                                       const std::string& kernel_id)
{

    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using TrInstMaker =
        std::conditional_t<ND == 2,
                           TransposeInstanceMaker2D<TransposeSolutionNhwc2Default,
                                                    TransposeSolutionDefault2Nhwc,
                                                    TransposeSolutionDefault2Nhwc>,
                           TransposeInstanceMaker3D<TransposeSolutionNdhwc2Default,
                                                    TransposeSolutionDefault2Ndhwc,
                                                    TransposeSolutionDefault2Ndhwc>>;

    TrInstMaker tr_inst_maker{};

    auto pre_tpose_fn = [](const Handle& handle,
                           const std::vector<Kernel>& kernels,
                           const ConvTensors& tensors,
                           [[maybe_unused]] TransposeInstance& tr_inst_in,
                           TransposeInstance& tr_inst_wei,
                           TransposeInstance& tr_inst_out) {
        tr_inst_out.ConvertFrom(handle, kernels, tensors.y);
        tr_inst_wei.ConvertFrom(handle, kernels, tensors.w);
    };

    auto post_tpose_fn = [](const Handle& handle,
                            const std::vector<Kernel>& kernels,
                            const ConvTensors& tensors,
                            TransposeInstance& tr_inst_in,
                            [[maybe_unused]] TransposeInstance& tr_inst_wei,
                            [[maybe_unused]] TransposeInstance& tr_inst_out) {
        tr_inst_in.ConvertTo(handle, kernels, const_cast<Data_t>(tensors.x));
    };

    return InitInvokerFactoryNCHW<DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, tr_inst_maker, pre_tpose_fn, post_tpose_fn);
}

template <int ND, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryWrwNCHW(const ExecutionContext& ctx,
                                       const ProblemDescription& problem,
                                       const std::string& kernel_id)
{
    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using TrInstMaker =
        std::conditional_t<ND == 2,
                           TransposeInstanceMaker2D<TransposeSolutionDefault2Nhwc,
                                                    TransposeSolutionNhwc2Default,
                                                    TransposeSolutionDefault2Nhwc>,
                           TransposeInstanceMaker3D<TransposeSolutionDefault2Ndhwc,
                                                    TransposeSolutionNdhwc2Default,
                                                    TransposeSolutionDefault2Ndhwc>>;

    TrInstMaker tr_inst_maker{};

    auto pre_tpose_fn = [](const Handle& handle,
                           const std::vector<Kernel>& kernels,
                           const ConvTensors& tensors,
                           TransposeInstance& tr_inst_in,
                           [[maybe_unused]] TransposeInstance& tr_inst_wei,
                           TransposeInstance& tr_inst_out) {
        tr_inst_in.ConvertFrom(handle, kernels, tensors.x);
        tr_inst_out.ConvertFrom(handle, kernels, tensors.y);
    };

    auto post_tpose_fn = [](const Handle& handle,
                            const std::vector<Kernel>& kernels,
                            const ConvTensors& tensors,
                            [[maybe_unused]] TransposeInstance& tr_inst_in,
                            TransposeInstance& tr_inst_wei,
                            [[maybe_unused]] TransposeInstance& tr_inst_out) {
        tr_inst_wei.ConvertTo(handle, kernels, const_cast<Data_t>(tensors.w));
    };

    return InitInvokerFactoryNCHW<DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, tr_inst_maker, pre_tpose_fn, post_tpose_fn);
}

template <typename InvokerFactoryMakerNCHW, typename InvokerFactoryMakerNHWC>
ConvSolution
MakeSolutionGroupConvImplicitGemmXdlops(const ProblemDescription& problem,
                                        InvokerFactoryMakerNCHW&& invoker_factory_maker_ncdhw,
                                        InvokerFactoryMakerNHWC&& invoker_factory_maker_ndhwc)
{

    if(problem.IsLayoutDefault())
    {
        switch(problem.GetInDataType())
        {
        case miopenInt8: return invoker_factory_maker_ncdhw(int8_t{});
        case miopenHalf: return invoker_factory_maker_ncdhw(ck::half_t{});
        case miopenFloat: return invoker_factory_maker_ncdhw(float{});
        case miopenInt32:
        case miopenInt8x4: // Support discontinued
        case miopenBFloat16:
        case miopenDouble:
        case miopenFloat8:
        case miopenBFloat8:
        default:
            MIOPEN_THROW(miopenStatusInternalError,
                         "3DGroupConvolutionImplicitGemmXdlops operation not implemented for this "
                         "data type");
        }
    }
    else if(problem.IsLayoutNHWC())
    {
        switch(problem.GetInDataType())
        {
        case miopenInt8: return invoker_factory_maker_ndhwc(int8_t{});
        case miopenHalf: return invoker_factory_maker_ndhwc(ck::half_t{});
        case miopenFloat: return invoker_factory_maker_ndhwc(float{});
        case miopenInt32:
        case miopenInt8x4: // Support discontinued
        case miopenBFloat16:
        case miopenDouble:
        case miopenFloat8:
        case miopenBFloat8:
        default:
            MIOPEN_THROW(miopenStatusInternalError,
                         "3DGroupConvolutionImplicitGemmXdlops operation not implemented for this "
                         "data type");
        }
    }
    else
    {
        MIOPEN_THROW(
            miopenStatusInternalError,
            "3DGroupConvolutionImplicitGemmXdlops operation not implemented for this data type");
    }
    return {};
}

} // namespace solver
} // namespace miopen
