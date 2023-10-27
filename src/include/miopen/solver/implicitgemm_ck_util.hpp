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
    {
        MIOPEN_LOG_E("PerformanceConfig kernel '" + kernel_id + "' does not exist.");
        return {miopenStatusInvalidValue};
    }

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

namespace internal {

enum class ConvOperandTag : int
{
    Input = 0,
    Weights,
    Output
};

enum class TransposeType : int
{
    NHWC_TO_NCHW = 0,
    NCHW_TO_NHWC
};

template <int ND, TransposeType TPOSE_TYPE>
struct ChooseTransposeSolver;

template <int ND>
struct ChooseTransposeSolver<ND, TransposeType::NHWC_TO_NCHW>
{
    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");
    using type = std::conditional_t<ND == 2,
                                    miopen::TransposeSolutionNhwc2Default,
                                    miopen::TransposeSolutionNdhwc2Default>;
};

template <int ND>
struct ChooseTransposeSolver<ND, TransposeType::NCHW_TO_NHWC>
{
    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");
    using type = std::conditional_t<ND == 2,
                                    miopen::TransposeSolutionDefault2Nhwc,
                                    miopen::TransposeSolutionDefault2Ndhwc>;
};

template <int ND, TransposeType TPOSE_TYPE, ConvOperandTag CONV_OP>
struct TransposeOperandBase
{
    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");
    // using TransposeSolver = typename ChooseTransposeSolver<ND, TPOSE_TYPE>::type;
    constexpr static ConvOperandTag CONV_OP_TAG = CONV_OP;
};

template <int ND, TransposeType TPOSE_TYPE, ConvOperandTag CONV_OP>
struct TransposeOperand;

template <int ND, TransposeType TPOSE_TYPE>
struct TransposeOperand<ND, TPOSE_TYPE, ConvOperandTag::Input>
    : public TransposeOperandBase<ND, TPOSE_TYPE, ConvOperandTag::Input>
{

    using TransposeSolver = typename ChooseTransposeSolver<ND, TPOSE_TYPE>::type;

    template <typename CKArgsType>
    TransposeSolver MakeTransposeSolver(const miopen::ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem,
                                        const CKArgsType& ck_args) const
    {

        if constexpr(ND == 3)
        {
            return TransposeSolver{ctx,
                                   problem.GetInDataType(),
                                   static_cast<uint32_t>(ck_args.N),
                                   static_cast<uint32_t>(ck_args.C1),
                                   static_cast<uint32_t>(ck_args.Di),
                                   static_cast<uint32_t>(ck_args.Hi),
                                   static_cast<uint32_t>(ck_args.Wi)};
        }
        else
        {
            return TransposeSolver{ctx,
                                   problem.GetInDataType(),
                                   static_cast<uint32_t>(ck_args.N),
                                   static_cast<uint32_t>(ck_args.C1),
                                   static_cast<uint32_t>(ck_args.Hi),
                                   static_cast<uint32_t>(ck_args.Wi)};
        }
    }
};

template <int ND, TransposeType TPOSE_TYPE>
struct TransposeOperand<ND, TPOSE_TYPE, ConvOperandTag::Weights>
    : public TransposeOperandBase<ND, TPOSE_TYPE, ConvOperandTag::Weights>
{

    using TransposeSolver = typename ChooseTransposeSolver<ND, TPOSE_TYPE>::type;

    template <typename CKArgsType>
    TransposeSolver MakeTransposeSolver(const miopen::ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem,
                                        const CKArgsType& ck_args) const
    {

        if constexpr(ND == 3)
        {
            return TransposeSolver{ctx,
                                   problem.GetWeightsDataType(),
                                   static_cast<uint32_t>(ck_args.K1),
                                   static_cast<uint32_t>(ck_args.C),
                                   static_cast<uint32_t>(ck_args.Z),
                                   static_cast<uint32_t>(ck_args.Y),
                                   static_cast<uint32_t>(ck_args.X)};
        }
        else
        {
            return TransposeSolver{ctx,
                                   problem.GetWeightsDataType(),
                                   static_cast<uint32_t>(ck_args.K1),
                                   static_cast<uint32_t>(ck_args.C),
                                   static_cast<uint32_t>(ck_args.Y),
                                   static_cast<uint32_t>(ck_args.X)};
        }
    }
};

template <int ND, TransposeType TPOSE_TYPE>
struct TransposeOperand<ND, TPOSE_TYPE, ConvOperandTag::Output>
    : public TransposeOperandBase<ND, TPOSE_TYPE, ConvOperandTag::Output>
{

    using TransposeSolver = typename ChooseTransposeSolver<ND, TPOSE_TYPE>::type;

    template <typename CKArgsType>
    TransposeSolver MakeTransposeSolver(const miopen::ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem,
                                        const CKArgsType& ck_args) const
    {

        if constexpr(ND == 3)
        {
            return TransposeSolver{ctx,
                                   problem.GetOutDataType(),
                                   static_cast<uint32_t>(ck_args.N),
                                   static_cast<uint32_t>(ck_args.K1),
                                   static_cast<uint32_t>(ck_args.Do),
                                   static_cast<uint32_t>(ck_args.Ho),
                                   static_cast<uint32_t>(ck_args.Wo)};
        }
        else
        {
            return TransposeSolver{ctx,
                                   problem.GetOutDataType(),
                                   static_cast<uint32_t>(ck_args.N),
                                   static_cast<uint32_t>(ck_args.K1),
                                   static_cast<uint32_t>(ck_args.Ho),
                                   static_cast<uint32_t>(ck_args.Wo)};
        }
    }
};

// Shorthand aliases for CK assuming CK always expects and generates NHWC/NDHWC layouts
template <int ND, ConvOperandTag CONV_OP>
using CKTransposeInputOp = TransposeOperand<ND, TransposeType::NCHW_TO_NHWC, CONV_OP>;

template <int ND, ConvOperandTag CONV_OP>
using CKTransposeOutputOp = TransposeOperand<ND, TransposeType::NHWC_TO_NCHW, CONV_OP>;

class TransposeInstance
{
    size_t tensor_sz = 0;
    std::vector<OpKernelArg> kern_args{};
    int index         = -1;
    size_t buf_offset = 0;
    shared<Data_t> buf_handle{};

public:
    template <typename TransSolnType>
    TransposeInstance(const TransSolnType& trans_sol, int idx, const MultiBufferWorkspaceTraits& wt)
    {
        tensor_sz = trans_sol.GetOutputTensorSize();
        kern_args = trans_sol.GetKernelArg();
        index     = idx;
        assert(index >= 0);
        buf_offset = wt.GetOffset(index);
    }

    void AssignBuffer(const Handle& handle, Data_t workSpace)
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

class TransposeInstanceTagged : public TransposeInstance
{

    ConvOperandTag conv_op_tag_;

public:
    template <typename TransSolnType>
    TransposeInstanceTagged(const TransSolnType& sol,
                            int idx,
                            const MultiBufferWorkspaceTraits& wt,
                            ConvOperandTag conv_op_tag)
        : TransposeInstance(sol, idx, wt), conv_op_tag_(conv_op_tag)
    {
    }

    ConvOperandTag GetConvOperandTag() const noexcept { return conv_op_tag_; }

    std::underlying_type_t<ConvOperandTag> GetConvOperandTagAsInt() const noexcept
    {
        using IntType = std::underlying_type_t<ConvOperandTag>;
        return static_cast<IntType>(GetConvOperandTag());
    }

    void ConvertFrom(const Handle& handle,
                     const std::vector<Kernel>& kernels,
                     const ConvTensors& tensors)
    {
        TransposeInstance::ConvertFrom(handle, kernels, pickTensorPtr(tensors));
    }

    void
    ConvertTo(const Handle& handle, const std::vector<Kernel>& kernels, const ConvTensors& tensors)
    {
        TransposeInstance::ConvertTo(handle, kernels, pickTensorPtr(tensors));
    }

    TransposeInstanceTagged()                               = delete;
    TransposeInstanceTagged(const TransposeInstanceTagged&) = default;
    TransposeInstanceTagged(TransposeInstanceTagged&&)      = default;
    ~TransposeInstanceTagged()                              = default;

private:
    Data_t pickTensorPtr(const ConvTensors& tensors) const
    {
        std::array<Data_t, 3> data_ptrs = {const_cast<Data_t>(tensors.x),
                                           const_cast<Data_t>(tensors.w),
                                           const_cast<Data_t>(tensors.y)};

        return data_ptrs[GetConvOperandTagAsInt()];
    }
};

template <typename CKArgsType, typename... TransposeOps, size_t... indices>
auto MakeTaggedTransposeInstancesHelper(ConvSolution& result,
                                        const ExecutionContext& ctx,
                                        const ProblemDescription& problem,
                                        const CKArgsType& ck_args,
                                        const std::tuple<TransposeOps...>& transpose_ops,
                                        std::index_sequence<indices...>)
{

    auto solvers = std::make_tuple(
        std::get<indices>(transpose_ops).MakeTransposeSolver(ctx, problem, ck_args)...);

    result.construction_params.insert(result.construction_params.end(),
                                      {std::get<indices>(solvers).GetKernelInfo()...});

    constexpr size_t buf_alignment = 256ull;
    MultiBufferWorkspaceTraits wt({std::get<indices>(solvers).GetOutputTensorSize()...},
                                  buf_alignment);

    return std::make_tuple(TransposeInstanceTagged{
        std::get<indices>(solvers), indices, wt, TransposeOps::CONV_OP_TAG}...);
}

template <typename CKArgsType, typename... TransposeOps>
auto MakeTaggedTransposeInstances(ConvSolution& result,
                                  const ExecutionContext& ctx,
                                  const ProblemDescription& problem,
                                  const CKArgsType& ck_args,
                                  const TransposeOps&... transpose_ops)
{
    return MakeTaggedTransposeInstancesHelper<CKArgsType>(
        result,
        ctx,
        problem,
        ck_args,
        std::make_tuple(transpose_ops...),
        std::index_sequence_for<TransposeOps...>{});
}

} // end namespace internal

template <typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename Input1TposeOp,
          typename Input2TposeOp,
          typename OutputTposeOp>
ConvSolution InitInvokerFactoryNCHW(const ExecutionContext& ctx,
                                    const ProblemDescription& problem,
                                    const std::string& kernel_id,
                                    const Input1TposeOp& input1_op,
                                    const Input2TposeOp& input2_op,
                                    const OutputTposeOp& output_op)
{

    assert(problem.IsLayoutDefault());

    ConvSolution result;
    auto ck_args = CKArgsType{problem};

    auto [_input1_tr_inst, _input2_tr_inst, _output_tr_inst] =
        internal::MakeTaggedTransposeInstances<CKArgsType>(
            result, ctx, problem, ck_args, input1_op, input2_op, output_op);

    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    if(ptr_iter == conv_ptrs.end())
    {
        MIOPEN_LOG_E("PerformanceConfig kernel '" + kernel_id + "' does not exist.");
        return {miopenStatusInvalidValue};
    }

    result.invoker_factory =
        [ck_args        = std::move(ck_args),
         sh_conv_ptr    = std::shared_ptr{std::move(*ptr_iter)},
         input1_tr_inst = std::move(_input1_tr_inst),
         input2_tr_inst = std::move(_input2_tr_inst),
         output_tr_inst = std::move(_output_tr_inst)](const std::vector<Kernel>& kernels) mutable {
            return [kernels,
                    ck_args        = std::move(ck_args),
                    sh_conv_ptr    = std::move(sh_conv_ptr),
                    input1_tr_inst = std::move(input1_tr_inst),
                    input2_tr_inst = std::move(input2_tr_inst),
                    output_tr_inst = std::move(output_tr_inst)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
                handle.ResetKernelTime();

                const auto& data_ctx = primitive_parameters.CastTo<CastType>();

                input1_tr_inst.AssignBuffer(handle, data_ctx.workSpace);
                input2_tr_inst.AssignBuffer(handle, data_ctx.workSpace);
                output_tr_inst.AssignBuffer(handle, data_ctx.workSpace);

                input1_tr_inst.ConvertFrom(handle, kernels, data_ctx.tensors);

                input2_tr_inst.ConvertFrom(handle, kernels, data_ctx.tensors);

                std::array<internal::TransposeInstanceTagged*, 3> tr_ptrs = {
                    &input1_tr_inst, &input2_tr_inst, &output_tr_inst};

                // sort by tag in order: Input, Weights, Output
                std::sort(tr_ptrs.begin(), tr_ptrs.end(), [](const auto& left, const auto& right) {
                    return left->GetConvOperandTagAsInt() < right->GetConvOperandTagAsInt();
                });

                auto invoker_ptr  = sh_conv_ptr->MakeInvokerPointer();
                auto argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                                       tr_ptrs[0]->GetBufferPtr(),
                                                       tr_ptrs[1]->GetBufferPtr(),
                                                       tr_ptrs[2]->GetBufferPtr());

                float elapsed_time = invoker_ptr->Run(
                    argument_ptr.get(), {handle.GetStream(), handle.IsProfilingEnabled()});
                if(handle.IsProfilingEnabled())
                {
                    handle.AccumKernelTime(elapsed_time);
                }

                output_tr_inst.ConvertTo(handle, kernels, data_ctx.tensors);
            };
        };
    return result;
}

template <int ND, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryFwdNCHW(const ExecutionContext& ctx,
                                       const ProblemDescription& problem,
                                       const std::string& kernel_id)
{

    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using Input1 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Input>;
    using Input2 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Weights>;
    using Output = internal::CKTransposeOutputOp<ND, internal::ConvOperandTag::Output>;

    return InitInvokerFactoryNCHW<DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, Input1{}, Input2{}, Output{});
}

template <int ND, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryBwdNCHW(const ExecutionContext& ctx,
                                       const ProblemDescription& problem,
                                       const std::string& kernel_id)
{

    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using Input1 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Output>;
    using Input2 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Weights>;
    using Output = internal::CKTransposeOutputOp<ND, internal::ConvOperandTag::Input>;

    return InitInvokerFactoryNCHW<DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, Input1{}, Input2{}, Output{});
}

template <int ND, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryWrwNCHW(const ExecutionContext& ctx,
                                       const ProblemDescription& problem,
                                       const std::string& kernel_id)
{
    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using Input1 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Input>;
    using Input2 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Output>;
    using Output = internal::CKTransposeOutputOp<ND, internal::ConvOperandTag::Weights>;

    return InitInvokerFactoryNCHW<DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, Input1{}, Input2{}, Output{});
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
