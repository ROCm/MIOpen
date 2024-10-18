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

#include <miopen/conv/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/hipoc_program.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_op.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_problem.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/tensor_operation/gpu/device/helper.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
//#include "common.hpp"
#include <fstream>
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

// TODO: temporarily have these two here due to build issues with ck_rtc, remove once resolved
struct src_file
{
    std::filesystem::path path;
    std::string_view content;
};
std::vector<src_file> get_headers_for_test()
{
    std::vector<src_file> result;
    auto hs = ck::host::GetHeaders();
    std::transform(
        hs.begin(), hs.end(), std::back_inserter(result), [&](const auto& p) -> src_file {
            return {p.first, p.second};
        });
    return result;
}

struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {

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

    CKArgs(const CKArgs&)     = default;
    CKArgs(CKArgs&&) noexcept = default;
    CKArgs& operator=(const CKArgs&) = default;

    /**int G;
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
    ck::host::conv::Problem_Conv_Fwd prob;
    ck::Array<ck::index_t, 5> in_lengths;
    ck::Array<ck::index_t, 5> in_strides;
    ck::Array<ck::index_t, 5> out_lengths;
    ck::Array<ck::index_t, 5> out_strides;
    ck::Array<ck::index_t, 5> wei_lengths;
    ck::Array<ck::index_t, 5> wei_strides;
    ck::Array<ck::index_t, 2> filter_strides;
    ck::Array<ck::index_t, 2> filter_dilations;
    ck::Array<ck::index_t, 2> lPadding;
    ck::Array<ck::index_t, 2> rPadding;
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

bool ConvHipImplicitGemmGroupFwdXdlopsCodegen::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
    // FIXME: rewrite this function
    return true;
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
#endif
    return false;
}

ConvSolution ConvHipImplicitGemmGroupFwdXdlopsCodegen::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
    auto x = CKArgs(problem);
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    /**decltype(auto) conv = problem.GetConv();
    decltype(auto) in   = problem.GetIn();
    decltype(auto) wei  = problem.GetWeights();
    decltype(auto) out  = problem.GetOut();**/

    const auto workspace_req = GetWorkspaceSize(ctx, problem);
    std::cout << "workspace: " << workspace_req << std::endl;

    auto soln         = ConvSolution{miopenStatusSuccess};
    soln.workspace_sz = workspace_req;

    auto solution = x.prob.GetSolutions("gfx908", prologue, epilogue);
    // substitute instance values into the template
    auto src = ck::host::InterpolateString(
        conv_compile_check,
        {{"include", x.prob.GetIncludeHeader()}, {"template", solution[0].ToTemplateString()}});
    auto srcs = get_headers_for_test();
    srcs.push_back({"main.cpp", src});
    auto name = solution[0].GetTemplateParameter<std::string>("name");

    auto kernel        = KernelInfo{};
    kernel.kernel_file = srcs[srcs.size() - 1].path.filename().string();
    kernel.kernel_name = "run_" + name;
    // rtc::compile_options options;
    // auto name           = solution[0].GetTemplateParameter<std::string>("name");
    // options.kernel_name = "run_" + name;
    // TODO: MIOpen has it's own handlers for compilation
    // auto k = rtc::compile_kernel(srcs, options);

    /**auto pImpl     = std::make_shared<HIPOCProgramImpl>();
    pImpl->program = program_name;
    pImpl->target  = this->GetTargetProperties();
    auto p           = HIPOCProgram{};
    p.impl           = pImpl;
    pImpl->BuildCodeObject(params, src);**/

    // Grid size calculation
    auto block_size = solution[0].GetTemplateParameter<ck::index_t>("BlockSize");

    auto tmp = get_launch_params(solution[0], x.out_lengths, x.out_strides);

    auto grid_size = tmp * x.in_lengths[1];

    kernel.l_wk = {block_size, 1, 1};
    kernel.g_wk = {block_size * grid_size, 1, 1};

    bool bfp16parm = true;
    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)}};
    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    kernel.comp_options += " -DCK_DONT_USE_HIP_RUNTIME_HEADERS";
    kernel.comp_options += " -DCK_CODE_GEN_RTC";
    std::cout << "comp options: " << kernel.comp_options << std::endl;

    soln.construction_params.push_back(kernel);

    soln.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::conv::DataInvokeParams>();

            kernel(params.tensors.in,
                   params.tensors.w,
                   params.tensors.out,
                   x.in_lengths,
                   x.in_strides,
                   x.wei_lengths,
                   x.wei_strides,
                   x.out_lengths,
                   x.out_strides,
                   x.filter_strides,
                   x.filter_dilations,
                   x.lPadding,
                   x.rPadding);
        };
    };
    // TODO: remove this, replace with lambda. MIOpen has it's own invoker to launch the kernel
    // launch the kernel with arguments needed for the argument pointer
    /**k.launch(nullptr, grid_size * block_size, block_size)(in_dev.data(),
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
                                                          prob.rPadding);**/

    return soln;
#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
