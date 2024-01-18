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

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/utility/data_type.hpp>
#endif // MIOPEN_USE_COMPOSABLEKERNEL

namespace miopen {

namespace conv {
struct ProblemDescription;
} // namespace conv

namespace solver {

struct ConvSolution;

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
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
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
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
bool IsCKArgsSupported(const ProblemDescriptionType& problem, const std::string& kernel_id)
{
    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    return (ptr_iter != conv_ptrs.end()) && CKArgsType{problem}.IsSupportedBy(*ptr_iter);
}

template <typename DeviceOpType,
          typename CKArgsType,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
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
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
ConvSolution InitInvokerFactoryNHWC(const ExecutionContext&,
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
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
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

        std::printf("IN TPOSE_TYPE=%d, (N=%d,C=%d,Hi=%d,Wi=%d)\n",
                    int(TPOSE_TYPE),
                    ck_args.N,
                    ck_args.C1,
                    ck_args.Hi,
                    ck_args.Wi);
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

        std::printf("WEI TPOSE_TYPE=%d, (K=%d,C=%d,Y=%d,X=%d)\n",
                    int(TPOSE_TYPE),
                    ck_args.K1,
                    ck_args.C,
                    ck_args.Y,
                    ck_args.X);
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
        std::printf("OUT TPOSE_TYPE=%d, (N=%d,K=%d,Ho=%d,Wo=%d)\n",
                    int(TPOSE_TYPE),
                    ck_args.N,
                    ck_args.K1,
                    ck_args.Ho,
                    ck_args.Wo);

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
    size_t index      = std::numeric_limits<size_t>::max();
    size_t buf_offset = 0;
    shared<Data_t> buf_handle{};

public:
    template <typename TransSolnType>
    TransposeInstance(const TransSolnType& trans_sol,
                      size_t idx,
                      const MultiBufferWorkspaceTraits& wt)
    {
        tensor_sz  = trans_sol.GetOutputTensorSize();
        kern_args  = trans_sol.GetKernelArg();
        index      = idx;
        buf_offset = wt.GetOffset(index);
    }

    void AssignBuffer(const Handle& handle, Data_t workSpace)
    {
        // TODO(amber): remove
        void* p = reinterpret_cast<char*>(workSpace) + buf_offset;
        MIOPEN_LOG_I("buffer start = " << p << ", tensor_sz = " << tensor_sz);
        buf_handle = handle.CreateSubBuffer(workSpace, buf_offset, tensor_sz);
        assert(buf_handle.get());
    }

    Data_t GetBufferPtr() const { return buf_handle.get(); }

    void ConvertFrom(const Handle& handle, const std::vector<Kernel>& kernels, ConstData_t in_ptr)
    {
        MIOPEN_LOG_I("ConvertFrom src ptr = " << in_ptr);
        Run(handle, kernels, buf_handle.get(), in_ptr);
    }

    void ConvertTo(const Handle& handle, const std::vector<Kernel>& kernels, Data_t out_ptr)
    {
        MIOPEN_LOG_I("ConvertTo dst ptr = " << out_ptr);
        Run(handle, kernels, out_ptr, buf_handle.get());
    }

    void ZeroOutBuffer()
    {
        [[maybe_unused]] auto status = hipMemset(buf_handle.get(), 0, tensor_sz);
        assert(status == hipSuccess);
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
        std::array<Data_t, 3> data_ptrs = {
            const_cast<Data_t>(tensors.x), // NOLINT (cppcoreguidelines-pro-type-const-cast)
            const_cast<Data_t>(tensors.w), // NOLINT (cppcoreguidelines-pro-type-const-cast)
            const_cast<Data_t>(tensors.y)  // NOLINT (cppcoreguidelines-pro-type-const-cast)
        };

        return data_ptrs[GetConvOperandTagAsInt()];
    }
};

template <typename CKArgsType, typename... TransposeOps, size_t... indices>
auto MakeTaggedTransposeInstancesHelper(ConvSolution& result,
                                        const ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem,
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
                                  const miopen::conv::ProblemDescription& problem,
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

/// \todo move to a cpp file
inline size_t GetWorkspaceSizeLayoutTransformConv(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsLayoutNHWC())
    {
        return 0;
    }

    assert(problem.IsLayoutDefault());
    // packed size in bytes
    auto GetPackedSize = [](const TensorDescriptor& td) {
        auto sz                         = td.GetElementSize() * GetTypeSize(td.GetType());
        constexpr size_t alignment      = 256u;
        constexpr size_t alignment_mask = alignment - 1;
        static_assert(alignment_mask > 0);
        static_assert((alignment & alignment_mask) == 0);
        return (sz + alignment_mask) & ~(alignment_mask);
    };

    auto w_sz = GetPackedSize(problem.GetIn()) + GetPackedSize(problem.GetWeights()) +
                GetPackedSize(problem.GetOut());

    return w_sz;
}

template <typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename Input1TposeOp,
          typename Input2TposeOp,
          typename OutputTposeOp>
ConvSolution InitInvokerFactoryNCHW(const ExecutionContext& ctx,
                                    const miopen::conv::ProblemDescription& problem,
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

                // TODO(amber): remove
                void* wb = data_ctx.workSpace;
                void* we = reinterpret_cast<char*>(data_ctx.workSpace) + data_ctx.workSpaceSize;
                MIOPEN_LOG_I("Workspace beg ptr = " << wb << " end ptr = " << we);
                if(!data_ctx.workSpace)
                {
                    MIOPEN_THROW(miopenStatusInvalidValue, "workspace pointer is null");
                }

                input1_tr_inst.AssignBuffer(handle, data_ctx.workSpace);
                input2_tr_inst.AssignBuffer(handle, data_ctx.workSpace);
                output_tr_inst.AssignBuffer(handle, data_ctx.workSpace);

                // conversion operator applied here to convert to ConvTensors
                auto conv_tensors = ConvTensors(data_ctx.tensors);

                std::printf("Invoker inputs, x=%p, w=%p, y=%p\n",
                            conv_tensors.x,
                            conv_tensors.w,
                            conv_tensors.y);

                auto print_vec = [](const char* name, const auto& vec) {
                    std::cout << name << " = [ ";
                    for(const auto& v : vec)
                    {
                        std::cout << v << ", ";
                    }
                    std::cout << "]\n";
                };
#define PRINT_VEC(x) print_vec(#x, x);

                PRINT_VEC(conv_tensors.xDesc.GetLengths());
                PRINT_VEC(conv_tensors.wDesc.GetLengths());
                PRINT_VEC(conv_tensors.yDesc.GetLengths());

                // PRINT_VEC(ck_args.input);
                // PRINT_VEC(ck_args.in_strides);
                // PRINT_VEC(ck_args.weight);
                // PRINT_VEC(ck_args.wei_strides);
                // PRINT_VEC(ck_args.output);
                // PRINT_VEC(ck_args.out_strides);

                // TODO(amber): remove this when DataInvokeParams stops swapping
                // "in" and "out" tensors for backward pass
                if(output_tr_inst.GetConvOperandTag() == internal::ConvOperandTag::Input)
                {
                    // this is backward pass, swap back input and output
                    std::swap(conv_tensors.x, conv_tensors.y);
                    std::swap(conv_tensors.xDesc, conv_tensors.yDesc);
                    std::printf("Invoker inputs after swap, x=%p, w=%p, y=%p\n",
                                conv_tensors.x,
                                conv_tensors.w,
                                conv_tensors.y);
                }

                if(output_tr_inst.GetConvOperandTag() == internal::ConvOperandTag::Weights)
                {
                    // TODO(amber): remove
                    handle.Finish();
                    MIOPEN_LOG_I("calling ZeroOutBuffer");
                    output_tr_inst.ZeroOutBuffer();
                }

                float tot_time = 0;

                // TODO(amber): remove
                handle.Finish();
                MIOPEN_LOG_I("calling ConvertFrom");
                input1_tr_inst.ConvertFrom(handle, kernels, conv_tensors);
                tot_time += handle.GetKernelTime();

                // TODO(amber): remove
                handle.Finish();
                MIOPEN_LOG_I("calling 2nd ConvertFrom");
                input2_tr_inst.ConvertFrom(handle, kernels, conv_tensors);
                tot_time += handle.GetKernelTime();

                std::array<internal::TransposeInstanceTagged*, 3> tr_ptrs = {
                    &input1_tr_inst, &input2_tr_inst, &output_tr_inst};

                // sort by tag in order: Input, Weights, Output
                std::sort(tr_ptrs.begin(), tr_ptrs.end(), [](const auto& left, const auto& right) {
                    return left->GetConvOperandTagAsInt() < right->GetConvOperandTagAsInt();
                });

                // TODO(amber): remove
                MIOPEN_LOG_I("Inputs for CK conv kernel (x,w,y): "
                             << tr_ptrs[0]->GetBufferPtr() << ", " << tr_ptrs[1]->GetBufferPtr()
                             << ", " << tr_ptrs[2]->GetBufferPtr() << ", ");

                auto invoker_ptr  = sh_conv_ptr->MakeInvokerPointer();
                auto argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                                       tr_ptrs[0]->GetBufferPtr(),
                                                       tr_ptrs[1]->GetBufferPtr(),
                                                       tr_ptrs[2]->GetBufferPtr());
                // TODO(amber): remove
                handle.Finish();
                MIOPEN_LOG_I("calling CK convolution");
                tot_time += invoker_ptr->Run(argument_ptr.get(),
                                             {handle.GetStream(), handle.IsProfilingEnabled()});

                // TODO(amber): remove
                handle.Finish();
                MIOPEN_LOG_I("calling ConvertTo");
                output_tr_inst.ConvertTo(handle, kernels, conv_tensors);
                tot_time += handle.GetKernelTime();

                // TODO(amber): remove
                handle.Finish();
                MIOPEN_LOG_I("done calling ConvertTo");

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(tot_time);
                }
            };
        };

    result.workspace_sz = GetWorkspaceSizeLayoutTransformConv(problem);

    return result;
}

template <int ND, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryFwdNCHW(const ExecutionContext& ctx,
                                       const miopen::conv::ProblemDescription& problem,
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
                                       const miopen::conv::ProblemDescription& problem,
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
                                       const miopen::conv::ProblemDescription& problem,
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
MakeSolutionGroupConvImplicitGemmXdlops(const miopen::conv::ProblemDescription& problem,
                                        InvokerFactoryMakerNCHW&& invoker_factory_maker_ncdhw,
                                        InvokerFactoryMakerNHWC&& invoker_factory_maker_ndhwc)
{

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
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
#else
    return {};
#endif
}

} // namespace solver
} // namespace miopen
