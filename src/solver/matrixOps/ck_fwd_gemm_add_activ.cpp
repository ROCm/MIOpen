
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

#include <vector>
#include <cstdint>

// are we doing numerics check?
// #include <miopen/check_numerics.hpp>

#include <miopen/fusion/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/matrixOps/invoke_params.hpp>
#include <miopen/matrixOps/problem_description.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_add_fastgelu.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CK_GEMM_ADD_FASTGELU)

namespace miopen {
namespace solver {
namespace fusion {

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using Row          = ck::tensor_layout::gemm::RowMajor;
using Col          = ck::tensor_layout::gemm::ColumnMajor;
using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::AddFastGelu;

const auto a_element_op   = AElementOp{};
const auto b_element_op   = BElementOp{};
const auto cde_element_op = CDEElementOp{};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename D0DataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename ELayout>
using DeviceOpGemmAddActivPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        ck::tensor_operation::device::DeviceGemmMultipleD<ALayout,
                                                          BLayout,
                                                          ck::Tuple<D0Layout>,
                                                          ELayout,
                                                          ADataType,
                                                          BDataType,
                                                          ck::Tuple<D0DataType>,
                                                          EDataType,
                                                          AElementOp,
                                                          BElementOp,
                                                          CDEElementOp>>;

struct CKArgsGemmAddActiv
{
    CKArgsGemmAddActiv(const GemmAddProblemDescription& problems)
    {
        const miopen::gemm::ProblemDescription& gemm_problem(std::get<0>(problems));
        const miopen::matrixAdd::ProblemDescription& mat_add_problem(std::get<1>(problems));

        M = gemm_problem.GetGemmDescriptor().GetM();
        N = gemm_problem.GetGemmDescriptor().GetN();
        K = gemm_problem.GetGemmDescriptor().GetK();

        stride_a  = gemm_problem.GetGemmDescriptor().GetStrideA();
        stride_b  = gemm_problem.GetGemmDescriptor().GetStrideB();
        stride_d0 = mat_add_problem.GetMatrixAddDescriptor().GetStrideD();
        stride_e  = mat_add_problem.GetMatrixAddDescriptor().GetStrideE();
    }

    CKArgsGemmAddActiv(const CKArgsGemmAddActiv&) = default;
    CKArgsGemmAddActiv(CKArgsGemmAddActiv&&)      = default;
    CKArgsGemmAddActiv& operator=(const CKArgsGemmAddActiv&) = default;

    template <typename InvokerPtr, typename InvokerParams>
    auto MakeArgPtr(const InvokerPtr& invoker_ptr, const InvokerParams& invoke_ctx) const
    {
        return invoker_ptr->MakeArgumentPointer(invoke_ctx.a_buff,
                                                invoke_ctx.b_buff,
                                                std::array<const void*, 1>{invoke_ctx.d0_buff},
                                                invoke_ctx.e_buff,
                                                M,
                                                N,
                                                K,
                                                stride_a,
                                                stride_b,
                                                std::array<ck::index_t, 1>{stride_d0},
                                                stride_e,
                                                a_element_op,
                                                b_element_op,
                                                cde_element_op);
    }

    template <typename GemmAddActivPtr>
    bool IsSupportedBy(const GemmAddActivPtr& invoker_ptr) const
    {
        auto arg_ptr = MakeArgPtr(invoker_ptr, miopen::matrixOps::GemmAddActiv{});
        return invoker_ptr->IsSupportedArgument(arg_ptr.get());
    }

    long long int M;
    long long int N;
    long long int K;

    int stride_a;
    int stride_b;
    int stride_d0;
    int stride_e;
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename D0DataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename ELayout>
void PerformanceConfigCKGEMMAddActiv::Init(const GemmAddProblemDescription& problem)
{
    valid_kernels = FillValidKernelsIDs<DeviceOpGemmAddActivPtrs<ADataType,
                                                                 BDataType,
                                                                 AccDataType,
                                                                 D0DataType,
                                                                 EDataType,
                                                                 ALayout,
                                                                 BLayout,
                                                                 D0Layout,
                                                                 ELayout>,
                                        CKArgsGemmAddActiv,
                                        GemmAddProblemDescription>(problem);
    index         = 0;
    kernel_id     = valid_kernels[index];
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename D0DataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename ELayout>
bool PerformanceConfigCKGEMMAddActiv::CheckIsSupportCKArgs(
    const GemmAddProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpGemmAddActivPtrs<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      D0DataType,
                                                      EDataType,
                                                      ALayout,
                                                      BLayout,
                                                      D0Layout,
                                                      ELayout>,
                             CKArgsGemmAddActiv>(problem, kernel_id);
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename D0DataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename ELayout>
bool CKGEMMAddActiv::CheckCKApplicability(const GemmAddProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpGemmAddActivPtrs<ADataType,
                                                   BDataType,
                                                   AccDataType,
                                                   D0DataType,
                                                   EDataType,
                                                   ALayout,
                                                   BLayout,
                                                   D0Layout,
                                                   ELayout>,
                          CKArgsGemmAddActiv>(problem);
}

#endif

void PerformanceConfigCKGEMMAddActiv::HeuristicInit(
    [[maybe_unused]] const FusionDescription& fdesc_problem)
{
    this->index     = 0;
    this->kernel_id = "";

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    auto gemm_problem = fdesc_problem.GetGemmProblem(0, miopen::gemm::Direction::ForwardInference);
    auto mat_add_problem =
        fdesc_problem.GetMatrixAddProblem(1, miopen::matrixAdd::Direction::ForwardInference);
    GemmAddProblemDescription gemm_add_problem(gemm_problem, mat_add_problem);
    switch(gemm_problem.GetADataType())
    {
    case miopenInt8:
    case miopenHalf:
    case miopenFloat:
        Init<float, float, float, float, float, Row, Row, Row, Row>(gemm_add_problem);
        break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt32:
    case miopenBFloat16:
    case miopenDouble: break;
    }
#endif
}

bool PerformanceConfigCKGEMMAddActiv::SetNextValue(const FusionDescription& fdesc_problem)
{
    if(valid_kernels.empty())
    {
        HeuristicInit(fdesc_problem);
        assert(!valid_kernels.empty());
        return true;
    }
    if((index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
        return false;
}

bool PerformanceConfigCKGEMMAddActiv::IsValidValue() const { return index < valid_kernels.size(); }

bool PerformanceConfigCKGEMMAddActiv::IsValid(
    [[maybe_unused]] const FusionContext&,
    [[maybe_unused]] const FusionDescription& fdesc_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    auto gemm_problem = fdesc_problem.GetGemmProblem(0, miopen::gemm::Direction::ForwardInference);
    auto mat_add_problem =
        fdesc_problem.GetMatrixAddProblem(1, miopen::matrixAdd::Direction::ForwardInference);
    GemmAddProblemDescription gemm_add_problem(gemm_problem, mat_add_problem);
    switch(gemm_problem.GetADataType())
    {
    case miopenHalf:
    case miopenFloat:
        return CheckIsSupportCKArgs<float, float, float, float, float, Row, Row, Row, Row>(
            gemm_add_problem);
    case miopenInt8:
    case miopenInt32:
    case miopenBFloat16:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble: break;
    }
#endif
    return false;
}

bool PerformanceConfigCKGEMMAddActiv::operator==(const PerformanceConfigCKGEMMAddActiv& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigCKGEMMAddActiv
CKGEMMAddActiv::GetDefaultPerformanceConfig(const FusionContext&,
                                            const FusionDescription& fdesc_problem) const
{
    PerformanceConfigCKGEMMAddActiv pp;
    pp.HeuristicInit(fdesc_problem);
    return pp;
}

PerformanceConfigCKGEMMAddActiv CKGEMMAddActiv::Search(const FusionContext& ctx,
                                                       const FusionDescription& fdesc_problem,
                                                       const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, fdesc_problem, invoke_ctx);
}

bool CKGEMMAddActiv::IsValidPerformanceConfig(const FusionContext& ctx,
                                              const FusionDescription& fdesc_problem,
                                              const PerformanceConfigCKGEMMAddActiv& config) const
{
    return config.IsValid(ctx, fdesc_problem);
}

bool CKGEMMAddActiv::IsApplicable([[maybe_unused]] const FusionContext& ctx,
                                  [[maybe_unused]] const FusionDescription& fdesc_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = fdesc_problem;
    return false;
#else
    if(miopen::IsDisabled(MIOPEN_DEBUG_CK_GEMM_ADD_FASTGELU{}))
        return false;
    const auto& fp_desc = fdesc_problem.fusion_plan_desc;
    if(fp_desc->op_map[0]->kind() != miopenFusionOpGEMM)
    {
        return false;
    }
    if(fp_desc->op_map[1]->kind() != miopenFusionOpMatricxAdd)
    {
        return false;
    }

    const auto& gemm_problem =
        fdesc_problem.GetGemmProblem(0, miopen::gemm::Direction::ForwardInference);
    const auto& matrix_problem =
        fdesc_problem.GetMatrixAddProblem(1, miopen::matrixAdd::Direction::ForwardInference);
    if(gemm_problem.GetADataType() != gemm_problem.GetBDataType() ||
       gemm_problem.GetADataType() != gemm_problem.GetCDataType())
    {
        return false;
    }
    const std::string arch = ctx.GetStream().GetDeviceName();
    if(!ck_utility::is_ck_whitelist(arch))
    {
        return false;
    }

    switch(gemm_problem.GetADataType())
    {
    case miopenHalf:
    case miopenFloat:
        return CheckCKApplicability<float, float, float, float, float, Row, Row, Row, Row>(
            {gemm_problem, matrix_problem});
    case miopenInt32:
    case miopenInt8:
    case miopenBFloat16:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8: break;
    }
    return false;
#endif
}

ConvSolution
CKGEMMAddActiv::GetSolution([[maybe_unused]] const FusionContext&,
                            [[maybe_unused]] const FusionDescription& fdesc_problem,
                            [[maybe_unused]] const PerformanceConfigCKGEMMAddActiv& config) const
{
    const auto& gemm_problem =
        fdesc_problem.GetGemmProblem(0, miopen::gemm::Direction::ForwardInference);
    const auto& matrix_problem =
        fdesc_problem.GetMatrixAddProblem(1, miopen::matrixAdd::Direction::ForwardInference);
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(gemm_problem.GetADataType())
    {
    case miopenHalf:
    case miopenFloat:
        // return MakeInvokerFactory<
        //         DeviceOpGemmAddActivPtrs<float, float, float, float, float, Row, Row, Row, Row>,
        //         CKArgsGemmAddActiv, conv::DataInvokeParams, GemmAddProblemDescription>(
        //         {gemm_problem, matrix_problem}, config.kernel_id);
    case miopenInt32:
    case miopenInt8:
    case miopenBFloat16:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "ConvHipImplicitGemmFwdXdlops operation not implemented for this data type");
    }
#endif
    return {};
}

} // namespace fusion
} // namespace solver
} // namespace miopen
