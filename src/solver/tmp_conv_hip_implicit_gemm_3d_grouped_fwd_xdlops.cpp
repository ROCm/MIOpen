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

#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using InLayout                             = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                            = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                            = ck::tensor_layout::convolution::NDHWGK;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using Bilinear                             = ck::tensor_operation::element_wise::Bilinear;
using Scale                                = ck::tensor_operation::element_wise::Scale;
static constexpr ck::index_t NumDimSpatial = 2;

const std::string conv_compile_check = R"__ck__(
#include <${include}>

${template};

)__ck__";

namespace {

std::string epilogue = R"(
struct Epilogue
{
    __host__ __device__ Epilogue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename D>
    __host__ __device__ constexpr void operator()(E& e, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>(ck::half_t& e,
                                                                          const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * e + beta_ * ck::type_convert<float>(d));
    }

    float alpha_;
    float beta_;
};
)";
std::string prologue = "";

// template <typename DataType>
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {

        ck::host::conv::Problem_Conv_Fwd prob;
        prob.NumDim = NumDimSpatial;
        prob.G      = ProblemInterpreter::GetGroupCountG(problem);
        prob.N      = ProblemInterpreter::GetBatchN(problem);
        int K1      = ProblemInterpreter::GetOutputChannelK(problem);
        int C1      = ProblemInterpreter::GetInputChannelC(problem);
        prob.C      = C1 / prob.G; // Number of input Channel per group
        prob.K      = K1 / prob.G; // Number of output Channel per group
        prob.Y      = ProblemInterpreter::GetFilterHeightY(problem);
        prob.X      = ProblemInterpreter::GetFilterWidthX(problem);
        prob.Hi     = ProblemInterpreter::GetInputHeightHi(problem);
        prob.Wi     = ProblemInterpreter::GetInputWidthWi(problem);
        prob.Ho     = ProblemInterpreter::GetOutputHeightHo(problem);
        prob.Wo     = ProblemInterpreter::GetOutputWidthWo(problem);

        in_lengths  = {prob.G, prob.N, prob.C, prob.Hi, prob.Wi};
        out_lengths = {prob.G, prob.N, prob.K, prob.Ho, prob.Wo};
        wei_lengths = {prob.G, prob.K, prob.C, prob.Y, prob.X};

        in_strides       = {prob.C,
                      prob.Hi * prob.Wi * prob.G * prob.C,
                      1,
                      prob.Wi * prob.G * prob.C,
                      prob.G * prob.C};
        out_strides      = {prob.K,
                       prob.Ho * prob.Wo * prob.G * prob.K,
                       1,
                       prob.Wo * prob.G * prob.K,
                       prob.G * prob.K};
        wei_strides      = {prob.K * prob.Y * prob.X * prob.C,
                       prob.Y * prob.X * prob.C,
                       1,
                       prob.X * prob.C,
                       prob.C};
        filter_strides   = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                          ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilations = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                            ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding         = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding         = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    /**CKArgs(const CKArgs&)     = default;
    CKArgs(CKArgs&&) noexcept = default;
    CKArgs& operator=(const CKArgs&) = default;

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Di;
    int Ho;
    int Wo;
    int Do;
    int Y;
    int X;
    int Z;**/
    std::array<ck::index_t, 5> in_lengths;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> out_lengths;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> wei_lengths;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> filter_strides;
    std::array<ck::index_t, 2> filter_dilations;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
    // miopenAlphaBetaCase_t alpha_beta_case;
};

} // namespace

#endif

size_t
ConvHipImplicitGemmGroupFwdXdlopsCodegen::GetWorkspaceSize(const ExecutionContext&,
                                                           const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

/**PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
ConvHipImplicitGemm3DGroupFwdXdlops::Search(const ExecutionContext& ctx,
                                            const ProblemDescription& problem,
                                            const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}**/

bool ConvHipImplicitGemmGroupFwdXdlopsCodegen::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS))
        return false;
    // check if type float else return false
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(!problem.Is3d())
        return false;
    if(!(problem.IsLayoutNHWC() || problem.IsLayoutDefault()))
        return false;
    // needed because layout transpose kernel does not support non-packed tensors
    if(problem.IsLayoutDefault() && problem.HasNonPackedTensors())
        return false;
    if(!ck_utility::is_ck_whitelist(ctx.GetStream().GetDeviceName()))
        return false;
        /**switch(problem.GetInDataType())
        {
        case miopenHalf: return CheckCKApplicability<ck::half_t>(problem);
        case miopenFloat: return CheckCKApplicability<float>(problem);
        case miopenInt8: return CheckCKApplicability<int8_t>(problem);
        case miopenBFloat16: return CheckCKApplicability<ck::bhalf_t>(problem);
        case miopenInt64:
        case miopenInt32:
        case miopenFloat8:
        case miopenBFloat8:
        case miopenDouble: break;
        }**/
#endif
    return false;
}

ConvSolution ConvHipImplicitGemmGroupFwdXdlopsCodegen::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
    auto prob = CKArgs(problem);
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    auto in_dev  = to_gpu(generate_buffer<ck::half_t, ck::Array<ck::index_t, 5>>(
        prob.in_lengths, prob.in_strides, 0));
    auto wei_dev = to_gpu(generate_buffer<ck::half_t, ck::Array<ck::index_t, 5>>(
        prob.wei_lengths, prob.wei_strides, 1));
    auto out_dev = to_gpu(generate_buffer<ck::half_t, ck::Array<ck::index_t, 5>>(
        prob.out_lengths, prob.out_strides, 2));

    auto solution : prob.GetSolutions("gfx908", prologue, epilogue);
    // substitute instance values into the template
    auto src = ck::host::InterpolateString(
        conv_compile_check,
        {{"include", prob.GetIncludeHeader()}, {"template", solution[0].ToTemplateString()}});

    auto srcs = get_headers_for_test();
    srcs.push_back({"main.cpp", src});
    rtc::compile_options options;
    auto name           = solution[0].GetTemplateParameter<std::string>("name");
    options.kernel_name = "run_" + name;
    auto k              = rtc::compile_kernel(srcs, options);

    // Grid size calculation
    auto block_size = solution[0].GetTemplateParameter<ck::index_t>("BlockSize");

    auto tmp = get_launch_params(solution[0], prob.out_lengths, prob.out_strides);

    auto grid_size = tmp * prob.in_lengths[1];

    // launch the kernel with arguments needed for the argument pointer
    k.launch(nullptr, grid_size * block_size, block_size)(in_dev.data(),
                                                          wei_dev.data(),
                                                          out_dev.data(),
                                                          prob.in_lengths,
                                                          prob.in_strides,
                                                          prob.wei_lengths,
                                                          prob.wei_strides,
                                                          prob.out_lengths,
                                                          prob.out_strides,
                                                          prob.filter_strides,
                                                          prob.filter_dilations,
                                                          prob.lPadding,
                                                          prob.rPadding);

#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
