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
#include <miopen/buffer_info.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/miopen_internal.h>

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/utility/data_type.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp>
#endif // MIOPEN_USE_COMPOSABLEKERNEL

namespace miopen {

namespace conv {
struct ProblemDescription;
} // namespace conv

namespace solver {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
namespace conv {
template <typename DataType>
using DeviceOpGWrw = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::tensor_layout::convolution::NHWGK,
    DataType,
    DataType,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;
template <typename DataType>
using DeviceOpGWrwPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOpGWrw<DataType>>;
} // namespace conv
#endif

inline bool IsLinear(int L, int H, const int v)
{
    assert(L <= H);
    return L <= v && v <= H;
}

inline bool NextLinear(int L, int H, int& v)
{
    assert((IsLinear(L, H, v)));
    if(H == v)
    {
        v = L;
        return true;
    }
    ++v;
    return false;
}

struct ConvSolution;

struct CKBWDWeightBufferDescriptor
{
    size_t ck_size;
    size_t ck_offset;

    CKBWDWeightBufferDescriptor(size_t _ck_size, size_t _ck_offset)
        : ck_size(_ck_size), ck_offset(_ck_offset)
    {
    }
};

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
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(!kernel_id.empty())
    {
        auto conv_ptrs = DeviceOpType::GetInstances();
        if constexpr(std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<ck::half_t>> ||
                     std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<float>> ||
                     std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<int8_t>> ||
                     std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<ck::bhalf_t>>)
        {
            auto pos      = kernel_id.find_last_of('+');
            int split_k   = std::stoi(kernel_id.substr(pos + 1));
            auto ptr_iter = FindConvPtrByID(conv_ptrs, kernel_id.substr(0, pos));
            return (ptr_iter != conv_ptrs.end()) &&
                   CKArgsType{problem}.IsSupportedBySplitK(*ptr_iter, split_k);
        }
        else
        {
            auto ptr_iter = FindConvPtrByID(conv_ptrs, kernel_id);
            return (ptr_iter != conv_ptrs.end()) && CKArgsType{problem}.IsSupportedBy(*ptr_iter);
        }
    }
#endif
    return false;
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

#define WORKAROUND_CK_ISSUE_1184 1
#if WORKAROUND_CK_ISSUE_1184
using WorkAroundHipEventProfiler = HipEventProfiler;
#endif

inline bool isDataTypeHalfAndChannelsEven(const miopen::conv::ProblemDescription& problem)
{
    return (problem.GetOutDataType() == miopenHalf) &&
           ((problem.GetInChannels() & 1) != 0 ||
            (problem.GetOutChannels() & 1) != 0 /* Test if odd*/);
}

inline bool ShouldAllocateWorkSpaceBufferForWRW(const miopen::conv::ProblemDescription& problem)
{
    return (problem.GetAlphaBetaCase() != DEFAULT || isDataTypeHalfAndChannelsEven(problem));
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

enum class TranposeKind : int
{
    NHWC_TO_NCHW = 0,
    NCHW_TO_NHWC
};

template <int ND, TranposeKind TPOSE_KIND, ConvOperandTag CONV_OP>
struct TransposeOperand
{
    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");
    constexpr static int NDIM                    = ND;
    constexpr static ConvOperandTag CONV_OP_TAG  = CONV_OP;
    constexpr static TranposeKind TRANSPOSE_KIND = TPOSE_KIND;

    using SolverType =
        std::conditional_t<TPOSE_KIND == TranposeKind::NHWC_TO_NCHW,
                           // NHWC_TO_NCHW
                           std::conditional_t<ND == 2,
                                              miopen::TransposeSolutionNhwc2Default,
                                              miopen::TransposeSolutionNdhwc2Default>,
                           // NCHW_TO_NHWC
                           std::conditional_t<ND == 2,
                                              miopen::TransposeSolutionDefault2Nhwc,
                                              miopen::TransposeSolutionDefault2Ndhwc>>;

    template <typename CKArgsType>
    SolverType MakeTransposeSolver(const miopen::ExecutionContext& ctx,
                                   const miopen::conv::ProblemDescription& problem,
                                   const CKArgsType& ck_args) const
    {

        if constexpr(CONV_OP_TAG == ConvOperandTag::Input)
        {
            if constexpr(ND == 3)
            {

                return SolverType{ctx,
                                  problem.GetInDataType(),
                                  static_cast<uint32_t>(ck_args.N),
                                  static_cast<uint32_t>(ck_args.C1),
                                  static_cast<uint32_t>(ck_args.Di),
                                  static_cast<uint32_t>(ck_args.Hi),
                                  static_cast<uint32_t>(ck_args.Wi)};
            }
            else
            {
                return SolverType{ctx,
                                  problem.GetInDataType(),
                                  static_cast<uint32_t>(ck_args.N),
                                  static_cast<uint32_t>(ck_args.C1),
                                  static_cast<uint32_t>(ck_args.Hi),
                                  static_cast<uint32_t>(ck_args.Wi)};
            }
        }
        else if constexpr(CONV_OP_TAG == ConvOperandTag::Weights)
        {
            if constexpr(ND == 3)
            {
                return SolverType{ctx,
                                  problem.GetWeightsDataType(),
                                  static_cast<uint32_t>(ck_args.K1),
                                  static_cast<uint32_t>(ck_args.C),
                                  static_cast<uint32_t>(ck_args.Z),
                                  static_cast<uint32_t>(ck_args.Y),
                                  static_cast<uint32_t>(ck_args.X)};
            }
            else
            {
                return SolverType{ctx,
                                  problem.GetWeightsDataType(),
                                  static_cast<uint32_t>(ck_args.K1),
                                  static_cast<uint32_t>(ck_args.C),
                                  static_cast<uint32_t>(ck_args.Y),
                                  static_cast<uint32_t>(ck_args.X)};
            }
        }
        else
        {
            static_assert(CONV_OP_TAG == ConvOperandTag::Output);
            if constexpr(ND == 3)
            {
                return SolverType{ctx,
                                  problem.GetOutDataType(),
                                  static_cast<uint32_t>(ck_args.N),
                                  static_cast<uint32_t>(ck_args.K1),
                                  static_cast<uint32_t>(ck_args.Do),
                                  static_cast<uint32_t>(ck_args.Ho),
                                  static_cast<uint32_t>(ck_args.Wo)};
            }
            else
            {
                return SolverType{ctx,
                                  problem.GetOutDataType(),
                                  static_cast<uint32_t>(ck_args.N),
                                  static_cast<uint32_t>(ck_args.K1),
                                  static_cast<uint32_t>(ck_args.Ho),
                                  static_cast<uint32_t>(ck_args.Wo)};
            }
        }
    }
};

// Shorthand aliases for CK assuming CK always expects and generates NHWC/NDHWC layouts
template <int ND, ConvOperandTag CONV_OP>
using CKTransposeInputOp = TransposeOperand<ND, TranposeKind::NCHW_TO_NHWC, CONV_OP>;

template <int ND, ConvOperandTag CONV_OP>
using CKTransposeOutputOp = TransposeOperand<ND, TranposeKind::NHWC_TO_NCHW, CONV_OP>;

class TransposeInstance
{
    size_t tensor_sz = 0;
    std::vector<OpKernelArg> kern_args{};
    size_t kern_idx   = std::numeric_limits<size_t>::max();
    size_t buf_offset = 0;
    shared<Data_t> buf_handle{};

public:
    template <typename TransSolnType>
    TransposeInstance(const TransSolnType& trans_sol,
                      size_t k_idx,
                      const MultiBufferWorkspaceTraits& wt,
                      size_t wspace_index)
        : tensor_sz(trans_sol.GetOutputTensorSize()),
          kern_args(trans_sol.GetKernelArg()),
          kern_idx(k_idx),
          buf_offset(wt.GetOffset(wspace_index))
    {
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

    void ZeroOutBuffer(const Handle& handle)
    {
        [[maybe_unused]] auto status =
            hipMemsetAsync(buf_handle.get(), 0, tensor_sz, handle.GetStream());
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
        assert(kernels.size() > kern_idx);

        kern_args[0] = out_ptr;
        kern_args[1] = in_ptr;

        auto save = handle.IsProfilingEnabled() ? 0.0f : handle.GetKernelTime();
        handle.Run(kernels[kern_idx])(kern_args);
        if(handle.IsProfilingEnabled())
        {
            handle.AccumKernelTime(save);
        }
    }
};

class TransposeInstanceTagged : public TransposeInstance
{

    ConvOperandTag conv_op_tag_;

public:
    template <typename TransSolnType>
    TransposeInstanceTagged(const TransSolnType& sol,
                            size_t k_idx,
                            const MultiBufferWorkspaceTraits& wt,
                            size_t wspace_index,
                            ConvOperandTag conv_op_tag)
        : TransposeInstance(sol, k_idx, wt, wspace_index), conv_op_tag_(conv_op_tag)
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

template <typename CKArgsType,
          typename Input1TposeOp,
          typename Input2TposeOp,
          typename OutputTposeOp>
auto MakeTaggedTransposeInstances(ConvSolution& result,
                                  const ExecutionContext& ctx,
                                  const miopen::conv::ProblemDescription& problem,
                                  const CKArgsType& ck_args,
                                  const Input1TposeOp& input1_op,
                                  const Input2TposeOp& input2_op,
                                  const OutputTposeOp& output_op,
                                  std::optional<CKBWDWeightBufferDescriptor>& ck_buff_des)
{

    auto input1_solver = input1_op.MakeTransposeSolver(ctx, problem, ck_args);
    auto input2_solver = input2_op.MakeTransposeSolver(ctx, problem, ck_args);
    auto output_solver = output_op.MakeTransposeSolver(ctx, problem, ck_args);

    // NOTE: In cases where the convolution updates only a subset of output
    // indices, we need to first initialize the workspace buffer for
    // output with the real tensor for the output and then apply the convolution.
    // This is achieved by creating an input transpose op for the output workspace
    // bufffer.

    using OutputInitOp = CKTransposeInputOp<OutputTposeOp::NDIM, OutputTposeOp::CONV_OP_TAG>;

    auto output_init_solver = OutputInitOp{}.MakeTransposeSolver(ctx, problem, ck_args);

    result.construction_params.insert(result.construction_params.end(),
                                      {input1_solver.GetKernelInfo(),
                                       input2_solver.GetKernelInfo(),
                                       output_solver.GetKernelInfo(),
                                       output_init_solver.GetKernelInfo()});

    if(ck_buff_des.has_value())
    {
        MultiBufferWorkspaceTraits wt({input1_solver.GetOutputTensorSize(),
                                       input2_solver.GetOutputTensorSize(),
                                       output_solver.GetOutputTensorSize(),
                                       ck_buff_des->ck_size});
        ck_buff_des->ck_offset = wt.GetOffset(3);
        return std::make_tuple(
            TransposeInstanceTagged{input1_solver, 0, wt, 0, Input1TposeOp::CONV_OP_TAG},
            TransposeInstanceTagged{input2_solver, 1, wt, 1, Input2TposeOp::CONV_OP_TAG},
            TransposeInstanceTagged{output_solver, 2, wt, 2, OutputTposeOp::CONV_OP_TAG},
            TransposeInstanceTagged{output_init_solver, 3, wt, 2, OutputTposeOp::CONV_OP_TAG});
    }

    MultiBufferWorkspaceTraits wt({input1_solver.GetOutputTensorSize(),
                                   input2_solver.GetOutputTensorSize(),
                                   output_solver.GetOutputTensorSize()});
    return std::make_tuple(
        TransposeInstanceTagged{input1_solver, 0, wt, 0, Input1TposeOp::CONV_OP_TAG},
        TransposeInstanceTagged{input2_solver, 1, wt, 1, Input2TposeOp::CONV_OP_TAG},
        TransposeInstanceTagged{output_solver, 2, wt, 2, OutputTposeOp::CONV_OP_TAG},
        TransposeInstanceTagged{output_init_solver, 3, wt, 2, OutputTposeOp::CONV_OP_TAG});
}

#ifndef NDEBUG // disable for release builds, enable for debug builds

template <typename V>
void DebugPrintVec(const char* name, const V& vec)
{
    std::ostringstream oss;
    oss << name << " = [ ";
    for(const auto& v : vec)
    {
        oss << v << ", ";
    }
    oss << "]";
    MIOPEN_LOG_I(oss.str());
}

#define DEBUG_PRINT_VEC(x) DebugPrintVec(#x, x);

template <typename CKArgsType, typename ConvPtr>
void DebugPrintCKArgPtrs(
    const CKArgsType& ck_args, const ConvPtr& conv_ptr, ConstData_t x, ConstData_t w, ConstData_t y)
{

    MIOPEN_LOG_I("CK Instance: " << conv_ptr->GetTypeString());
    MIOPEN_LOG_I("in ptr = " << x);
    MIOPEN_LOG_I("w ptr = " << w);
    MIOPEN_LOG_I("out ptr = " << y);

    DEBUG_PRINT_VEC(ck_args.input);
    DEBUG_PRINT_VEC(ck_args.in_strides);
    DEBUG_PRINT_VEC(ck_args.weight);
    DEBUG_PRINT_VEC(ck_args.wei_strides);
    DEBUG_PRINT_VEC(ck_args.output);
    DEBUG_PRINT_VEC(ck_args.out_strides);
}

inline void DebugPrintConvTensors(const ConvTensors& conv_tensors)
{
    MIOPEN_LOG_I("in ptr = " << conv_tensors.x);
    MIOPEN_LOG_I("w ptr = " << conv_tensors.w);
    MIOPEN_LOG_I("out ptr = " << conv_tensors.y);

    DEBUG_PRINT_VEC(conv_tensors.xDesc.GetLengths());
    DEBUG_PRINT_VEC(conv_tensors.wDesc.GetLengths());
    DEBUG_PRINT_VEC(conv_tensors.yDesc.GetLengths());
}

#undef DEBUG_PRINT_VEC

#endif // NDEBUG
} // end namespace internal

// packed size in bytes
inline size_t GetPackedSize(const TensorDescriptor& td)
{
    return td.GetElementSize() * GetTypeSize(td.GetType());
}

inline size_t GetCKAlphaBetaWorkspace(const miopen::conv::ProblemDescription& problem)
{
    std::size_t buff_size;

    TensorDescriptor input          = problem.GetIn();
    TensorDescriptor output         = problem.GetOut();
    ConvolutionDescriptor conv_desc = problem.GetConv();

    miopenConvolutionABBackwardWeightsGetWorkSpaceSize(
        problem.GetAlphaBetaCase(), &input, &output, &conv_desc, &buff_size);
    return buff_size;
}

inline bool CKWrwRequireWorkspace(
    size_t G, size_t C, size_t K, miopenDataType_t data_type, miopenAlphaBetaCase_t alpha_beta_case)
{
    auto is_odd        = [](size_t num) { return num % 2 != 0; };
    size_t C_per_group = C / G;
    size_t K_per_group = K / G;

    return (alpha_beta_case == BILINEAR || alpha_beta_case == SCALE) ||
           (data_type == miopenHalf && (is_odd(C_per_group) || is_odd(K_per_group)));
}

/// \todo move to a cpp file
inline size_t GetWorkspaceSizeLayoutTransformConv(const miopen::conv::ProblemDescription& problem)
{
    if(problem.IsLayoutNHWC())
    {
        if(problem.GetDirection() == ::miopen::conv::Direction::BackwardWeights)
        {
            return GetCKAlphaBetaWorkspace(problem);
        }
        return 0;
    }

    assert(problem.IsLayoutDefault());

    if(problem.GetDirection() == ::miopen::conv::Direction::BackwardWeights)
    {
        MultiBufferWorkspaceTraits wt({GetPackedSize(problem.GetIn()),
                                       GetPackedSize(problem.GetWeights()),
                                       GetPackedSize(problem.GetOut()),
                                       GetCKAlphaBetaWorkspace(problem)});
        return wt.GetSize();
    }

    MultiBufferWorkspaceTraits wt({GetPackedSize(problem.GetIn()),
                                   GetPackedSize(problem.GetWeights()),
                                   GetPackedSize(problem.GetOut())});
    return wt.GetSize();
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
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    auto ck_args = CKArgsType{problem};

    auto conv_ptrs = DeviceOpType::GetInstances();

    std::optional<int> split_k = std::nullopt;
    std::string id_string      = kernel_id;
    auto pos                   = kernel_id.find_last_of('+');
    if(pos != std::string::npos)
    {
        split_k   = std::stoi(kernel_id.substr(pos + 1));
        id_string = kernel_id.substr(0, pos);
    }

    std::optional<CKBWDWeightBufferDescriptor> _ck_buff_des;

    if(problem.IsDirectionBackwardWrW())
    {
        _ck_buff_des.emplace(GetCKAlphaBetaWorkspace(problem), 0);
    }

    auto ptr_iter = FindConvPtrByID(conv_ptrs, id_string);
    if(ptr_iter == conv_ptrs.end())
    {
        MIOPEN_LOG_E("PerformanceConfig kernel '" + kernel_id + "' does not exist.");
        return {miopenStatusInvalidValue};
    }

    auto [_input1_tr_inst, _input2_tr_inst, _output_tr_inst, _output_init_tr_inst] =
        internal::MakeTaggedTransposeInstances<CKArgsType>(
            result, ctx, problem, ck_args, input1_op, input2_op, output_op, _ck_buff_des);

    result.invoker_factory = [split_k             = split_k,
                              ck_args             = std::move(ck_args),
                              sh_conv_ptr         = std::shared_ptr{std::move(*ptr_iter)},
                              input1_tr_inst      = std::move(_input1_tr_inst),
                              input2_tr_inst      = std::move(_input2_tr_inst),
                              output_tr_inst      = std::move(_output_tr_inst),
                              output_init_tr_inst = std::move(_output_init_tr_inst),
                              ck_buff_des =
                                  _ck_buff_des](const std::vector<Kernel>& kernels) mutable {
        return [split_k = split_k,
                kernels,
                ck_args             = std::move(ck_args),
                sh_conv_ptr         = std::move(sh_conv_ptr),
                input1_tr_inst      = std::move(input1_tr_inst),
                input2_tr_inst      = std::move(input2_tr_inst),
                output_tr_inst      = std::move(output_tr_inst),
                output_init_tr_inst = std::move(output_init_tr_inst),
                ck_buff_des         = ck_buff_des](const Handle& handle,
                                           const AnyInvokeParams& primitive_parameters) mutable {
            handle.ResetKernelTime();

            const auto& data_ctx = primitive_parameters.CastTo<CastType>();

            if(!data_ctx.workSpace)
            {
                MIOPEN_THROW(miopenStatusInvalidValue, "workspace pointer is null");
            }

            input1_tr_inst.AssignBuffer(handle, data_ctx.workSpace);
            input2_tr_inst.AssignBuffer(handle, data_ctx.workSpace);
            output_tr_inst.AssignBuffer(handle, data_ctx.workSpace);
            output_init_tr_inst.AssignBuffer(handle, data_ctx.workSpace);

            // conversion operator applied here to convert to ConvTensors
            auto conv_tensors = ConvTensors(data_ctx.tensors);

            /// \todo remove this when DataInvokeParams stops swapping
            // "in" and "out" tensors for backward pass
            if(output_tr_inst.GetConvOperandTag() == internal::ConvOperandTag::Input)
            {
                // this is backward pass, swap back input and output
                std::swap(conv_tensors.x, conv_tensors.y);
                std::swap(conv_tensors.xDesc, conv_tensors.yDesc);
            }
            WorkAroundHipEventProfiler pfr(handle);
            input1_tr_inst.ConvertFrom(handle, kernels, conv_tensors);

            input2_tr_inst.ConvertFrom(handle, kernels, conv_tensors);

            output_init_tr_inst.ConvertFrom(handle, kernels, conv_tensors);

            /// \todo: Will need SetTensor() to properly zero out non-packed tensors
            if(output_tr_inst.GetConvOperandTag() == internal::ConvOperandTag::Weights)
            {
                output_tr_inst.ZeroOutBuffer(handle);
            }

            std::array<internal::TransposeInstanceTagged*, 3> tr_ptrs = {
                &input1_tr_inst, &input2_tr_inst, &output_tr_inst};

            // sort by tag in order: Input, Weights, Output
            std::sort(tr_ptrs.begin(), tr_ptrs.end(), [](const auto& left, const auto& right) {
                return left->GetConvOperandTagAsInt() < right->GetConvOperandTagAsInt();
            });

            auto invoker_ptr = sh_conv_ptr->MakeInvokerPointer();
            std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr;
            if constexpr(std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<ck::half_t>> ||
                         std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<float>> ||
                         std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<int8_t>> ||
                         std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<ck::bhalf_t>>)
            {
                if(split_k.has_value())
                {
                    argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                                      tr_ptrs[0]->GetBufferPtr(),
                                                      tr_ptrs[1]->GetBufferPtr(),
                                                      tr_ptrs[2]->GetBufferPtr(),
                                                      data_ctx.alpha.GetAsFloat(),
                                                      data_ctx.beta.GetAsFloat(),
                                                      split_k.value());
                }
            }
            else
            {
                std::ignore  = split_k;
                argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                                  tr_ptrs[0]->GetBufferPtr(),
                                                  tr_ptrs[1]->GetBufferPtr(),
                                                  tr_ptrs[2]->GetBufferPtr(),
                                                  data_ctx.alpha.GetAsFloat(),
                                                  data_ctx.beta.GetAsFloat());
            }

            if(ck_buff_des.has_value() && ck_buff_des->ck_size)
            {
                auto buf_handle =
                    handle.CreateSubBuffer(data_ctx.workSpace, ck_buff_des->ck_offset, 0);
                assert(buf_handle.get());
                sh_conv_ptr->SetWorkSpacePointer(argument_ptr.get(), buf_handle.get());
            }
            invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), false});
            output_tr_inst.ConvertTo(handle, kernels, conv_tensors);
        };
    };

    result.workspace_sz = GetWorkspaceSizeLayoutTransformConv(problem);
#endif
    return result;
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

    std::optional<int> split_k = std::nullopt;
    std::string id_string      = kernel_id;
    auto pos                   = kernel_id.find_last_of('+');
    if(pos != std::string::npos)
    {
        split_k   = std::stoi(kernel_id.substr(pos + 1));
        id_string = kernel_id.substr(0, pos);
    }

    auto ptr_iter = FindConvPtrByID(conv_ptrs, id_string);

    if(ptr_iter == conv_ptrs.end())
    {
        MIOPEN_LOG_E("PerformanceConfig kernel '" + kernel_id + "' does not exist.");
        return {miopenStatusInvalidValue};
    }

    if constexpr(std::is_same_v<CastType, miopen::conv::WrWInvokeParams>)
    {
        ConvSolution result;
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        miopenAlphaBetaCase_t alpha_beta_case = problem.GetAlphaBetaCase();
        [[maybe_unused]] bool should_allocated_wrw_buffer =
            ShouldAllocateWorkSpaceBufferForWRW(problem);

        result.invoker_factory = [split_k                     = split_k,
                                  ck_args                     = CKArgsType{problem},
                                  alpha_beta_case             = alpha_beta_case,
                                  should_allocated_wrw_buffer = should_allocated_wrw_buffer,
                                  sh_conv_ptr = std::shared_ptr{std::move(*ptr_iter)}](
                                     const std::vector<Kernel>&) mutable {
            return [split_k                     = split_k,
                    ck_args                     = std::move(ck_args),
                    alpha_beta_case             = alpha_beta_case,
                    should_allocated_wrw_buffer = should_allocated_wrw_buffer,
                    sh_conv_ptr                 = std::move(sh_conv_ptr)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& data_ctx = primitive_parameters.CastTo<CastType>();
                std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr;
                if constexpr(std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<ck::half_t>> ||
                             std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<float>> ||
                             std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<int8_t>> ||
                             std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<ck::bhalf_t>>)
                {
                    argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                                      data_ctx.tensors,
                                                      data_ctx.alpha.GetAsFloat(),
                                                      data_ctx.beta.GetAsFloat(),
                                                      split_k.value());
                }
                else
                {
                    std::ignore  = split_k;
                    argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                                      data_ctx.tensors,
                                                      data_ctx.alpha.GetAsFloat(),
                                                      data_ctx.beta.GetAsFloat());
                }

                auto invoker_ptr = sh_conv_ptr->MakeInvokerPointer();
                HipEventProfiler pfr(handle);

                if(alpha_beta_case == DEFAULT)
                {
                    auto zero           = 0.0f;
                    const auto& tensors = data_ctx.tensors;
                    SetTensor(handle, tensors.dwDesc, tensors.dw, &zero);
                }
                // use captured value, other wise getting warning
                // "lambda capture is not used" since this variable is only used in assert.
                (void)should_allocated_wrw_buffer;
                assert((should_allocated_wrw_buffer && data_ctx.workSpace != nullptr) ||
                       !(should_allocated_wrw_buffer && data_ctx.workSpace == nullptr));
                if(data_ctx.workSpace)
                {
                    sh_conv_ptr->SetWorkSpacePointer(argument_ptr.get(), data_ctx.workSpace);
                }
                invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), false});
            };
        };
        result.workspace_sz = GetWorkspaceSizeLayoutTransformConv(problem);
#endif
        return result;
    }
    else
    {
        ConvSolution result;
        result.invoker_factory = [ck_args     = CKArgsType{problem},
                                  sh_conv_ptr = std::shared_ptr{std::move(*ptr_iter)}](
                                     const std::vector<Kernel>&) mutable {
            return [ck_args = std::move(ck_args), sh_conv_ptr = std::move(sh_conv_ptr)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& data_ctx = primitive_parameters.CastTo<CastType>();
                auto argument_ptr    = ck_args.MakeArgPtr(sh_conv_ptr,
                                                       data_ctx.tensors,
                                                       data_ctx.alpha.GetAsFloat(),
                                                       data_ctx.beta.GetAsFloat());
                auto invoker_ptr     = sh_conv_ptr->MakeInvokerPointer();
                HipEventProfiler pfr(handle);
                invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), false});
            };
        };
        return result;
    }
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
        case miopenBFloat16: return invoker_factory_maker_ncdhw(ck::bhalf_t{});
        case miopenInt64:
        case miopenInt32:
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
        case miopenBFloat16: return invoker_factory_maker_ndhwc(ck::bhalf_t{});
        case miopenInt64:
        case miopenInt32:
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
