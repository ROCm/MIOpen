/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/batchnorm/common_spatial.hpp>
#include <miopen/batchnorm/solvers.hpp>

#include <miopen/generic_search.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>

namespace miopen {

namespace solver {

namespace batchnorm {

// Spatial multiple needs space for 4 fp32 elements
// per each x thread (including the last workgroup)
// to stash intermediate mean and variance
const unsigned int stash_values_bwd = 4;

bool PerformanceConfigBnBwdBackward::IsValid(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem) const
{
    if(this->kernel_id.empty())
    {
        return false;
    }

    // if default config is variant 2, check if it can be applied
    // (based on variant 2 restrictions)
    size_t vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements;
    int variant;
    GetVariantFromKernelId(
        this->kernel_id, variant, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements);
    if(variant == 2)
    {
        unsigned int stash_values = !problem.UseSaved() ? stash_values_bwd : stash_values_bwd / 2;
        return IsSpatialMultipleApplicable(
            problem, vectorsize, stash_values, ylocalsize, zlocalsize, nelements);
    }
    return true;
}

void PerformanceConfigBnBwdBackward::HeuristicInit(
    const miopen::batchnorm::ProblemDescription& problem)
{
    unsigned int stash_values = !problem.UseSaved() ? stash_values_bwd : stash_values_bwd / 2;
    // Define default configuration based on heuristics and
    // add all other valid configurations for the given problem
    if(UseMultiple(problem))
    {
        DefaultConfigSpatialMultiple(problem, stash_values, this->valid_kernels);
        // if more than 2 instances are present, it means that variant 1 will be slower
        if((this->valid_kernels.size() < 2 && problem.IsLayoutNHWC()) || !problem.IsLayoutNHWC())
        {
            DefaultConfigSpatialSingle(problem, this->valid_kernels);
        }
    }
    else
    {
        DefaultConfigSpatialSingle(problem, this->valid_kernels);
        // if valid_kernels is 2, it means that variant 0 or variant 3 were added and in
        // this case it doesn't make sense to add instances for variant 2 because it is
        // very unlikely that they will be faster than those variants
        if(this->valid_kernels.size() < 2)
        {
            DefaultConfigSpatialMultiple(problem, stash_values, this->valid_kernels);
        }
    }

    // Set index and kernel_id to default value
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

bool PerformanceConfigBnBwdBackward::SetNextValue(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
    // In case the valid_kernel list is empty, we fill it with
    // default value as first one and all other valid ones will follow
    if(this->valid_kernels.empty())
    {
        this->HeuristicInit(problem_desc);
        return true;
    }
    // Get next valid configuration
    if((this->index + 1) < valid_kernels.size())
    {
        ++this->index;
        this->kernel_id = this->valid_kernels[index];
        return true;
    }
    else
    {
        return false;
    }
}

bool PerformanceConfigBnBwdBackward::operator==(const PerformanceConfigBnBwdBackward& other) const
{
    return this->kernel_id == other.kernel_id;
}

bool PerformanceConfigBnBwdBackward::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool BnBwdTrainingSpatial::IsApplicable(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& bn_problem) const
{
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::Backward ||
       bn_problem.GetMode() != miopenBNSpatial)
        return false;

    if(!bn_problem.Is2D())
        return false;

#if WORKAROUND_ISSUE_1549_FP16_BUILD_ERROR
    if(bn_problem.GetXDesc().GetType() == miopenHalf &&
       bn_problem.GetBnScale().GetType() == miopenHalf)
    {
        // bfp16parm = true;
        // Unsupported kernel mode, error in kernel code
        // MIOpenBatchNormBwdSpatial.cl:526 issue#1549
        return false;
    }
#endif
    if(!IsOCLBwdTypeValid(bn_problem))
        return false;

    int activ_mode = bn_problem.GetActivationDesc().GetMode();
    if(activ_mode != miopenActivationPASTHRU && activ_mode != miopenActivationRELU &&
       activ_mode != miopenActivationCLIPPEDRELU && activ_mode != miopenActivationCLAMP)
    {
        return false;
    }

    return true;
}

PerformanceConfigBnBwdBackward BnBwdTrainingSpatial::GetDefaultPerformanceConfig(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
    PerformanceConfigBnBwdBackward pp;
    pp.HeuristicInit(problem_desc);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool BnBwdTrainingSpatial::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const miopen::batchnorm::ProblemDescription& problem_desc,
    const PerformanceConfigBnBwdBackward& config) const
{
    return config.IsValid(ctx, problem_desc);
}

PerformanceConfigBnBwdBackward
BnBwdTrainingSpatial::Search(const ExecutionContext& ctx,
                             const miopen::batchnorm::ProblemDescription& problem,
                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution BnBwdTrainingSpatial::GetSolution(const ExecutionContext& context,
                                               const miopen::batchnorm::ProblemDescription& problem,
                                               const PerformanceConfigBnBwdBackward& config) const
{
    const auto& handle      = context.GetStream();
    const unsigned wavesize = (miopen::StartsWith(handle.GetDeviceName(), "gfx10") ? 32 : 64);

    bool bfpmixparm   = false;
    bool bbfpmixparam = false;
    bool bfp16parm    = false;
    bool bfp32parm    = true;

    if(problem.GetXDesc().GetType() == miopenHalf && problem.GetBnScale().GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
    }
    else if(problem.GetXDesc().GetType() == miopenHalf &&
            problem.GetBnScale().GetType() == miopenFloat)
    {
        bfpmixparm = true;
        bfp32parm  = false;
    }
    else if(problem.GetXDesc().GetType() == miopenBFloat16 &&
            problem.GetBnScale().GetType() == miopenFloat)
    {
        bbfpmixparam = true;
        bfp32parm    = false;
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nstride = c * in_cstride;
    unsigned int in_nhw     = n * in_cstride;
    unsigned int in_nchw    = n * in_nstride;

    auto inhw = float(1.0 / in_nhw);

    int variant       = -1;
    size_t vectorsize = 1;
    size_t xlocalsize, xgridsize;
    size_t ylocalsize = 1, ygridsize = 1;
    size_t zlocalsize = 1, zgridsize = 1;
    unsigned int ldsgcn, ldsnogcn;
    int stash_method = 0;
    size_t nelements;

    GetVariantFromKernelId(
        config.kernel_id, variant, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements);

    size_t xlocalsize_final = xlocalsize, ylocalsize_final = ylocalsize,
           zlocalsize_final = zlocalsize;
    if(variant != 2)
    {
        xlocalsize = 1024;
        xgridsize  = static_cast<size_t>(1024) * c;
        ldsgcn     = xlocalsize / wavesize;
        ldsnogcn   = xlocalsize;
    }
    else
    {
        // Compute grid size
        if(problem.IsLayoutNHWC())
        {
            xgridsize = xlocalsize * ((c / vectorsize + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
        }
        else
        {
            xgridsize = xlocalsize * ((c + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((in_cstride / vectorsize + ylocalsize - 1) / ylocalsize);
        }
        zgridsize = zlocalsize * ((n / nelements + zlocalsize - 1) / zlocalsize);

        unsigned int stash_values = !problem.UseSaved() ? stash_values_bwd : stash_values_bwd / 2;
        // Get the stash method based on problem size and WG size
        stash_method = GetStashMethod(problem.IsLayoutNHWC(),
                                      problem.GetXDesc().GetType(),
                                      stash_values,
                                      c,
                                      n,
                                      in_cstride,
                                      ylocalsize,
                                      zlocalsize,
                                      nelements);

        // WG size for Final kernels (NHWC)
        if(problem.IsLayoutNHWC() && c % 2 == 0 && xlocalsize % 2 == 0)
        {
            // increase number of blocks (xgridsize does not change for final kernels)
            // 2 is the lower bound because of stashing
            xlocalsize_final = 2;
            // increase the number of threads in the y and z direction to decrease the number of
            // loads/stores for each thread
            zlocalsize_final = zgridsize / zlocalsize * zlocalsize;
            ylocalsize_final =
                (xlocalsize * ylocalsize * zlocalsize) / xlocalsize_final / zlocalsize_final;
        }
        ldsnogcn = xlocalsize * ylocalsize * zlocalsize;
        ldsgcn   = xlocalsize * ylocalsize * zlocalsize / wavesize;
    }

    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto kernel = KernelInfo{};

        auto build_params =
            KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
                                  {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
                                  {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
                                  {"MIOPEN_USE_BFPMIX", static_cast<int>(bbfpmixparam)},
                                  {"MIO_BN_USESAVED", static_cast<int>(problem.UseSaved())},
                                  {"MIO_BN_N", static_cast<int>(n)},
                                  {"MIO_BN_C", static_cast<int>(c)},
                                  {"MIO_BN_HW", static_cast<int>(in_cstride)},
                                  {"MIO_BN_NHW", static_cast<int>(in_nhw)},
                                  {"MIO_BN_CHW", in_nstride},
                                  {"MIO_BN_NCHW", in_nchw},
                                  {"MIO_BN_NGRPS", ygridsize / ylocalsize},
                                  {"MIO_BN_NGRPS2", zgridsize / zlocalsize},
                                  {"MIO_BN_N_ELEMENTS", nelements},
                                  {"MIO_BN_LDS_SIZE", ldsnogcn},
                                  {"MIO_BN_LDSGCN_SIZE", ldsgcn},
                                  {"MIO_BN_VARIANT", variant},
                                  {"MIO_WAVESIZE", wavesize},
                                  {"MIO_BN_GRP0", xlocalsize},
                                  {"MIO_BN_GRP1", ylocalsize},
                                  {"MIO_BN_GRP2", zlocalsize},
                                  {"MIO_BN_GRP0_FINAL", xlocalsize_final},
                                  {"MIO_BN_GRP1_FINAL", ylocalsize_final},
                                  {"MIO_BN_GRP2_FINAL", zlocalsize_final},
                                  {"MIO_LAYOUT_NHWC", static_cast<int>(problem.IsLayoutNHWC())},
                                  {"MIO_BN_VECTORIZE", static_cast<int>(vectorsize > 1)},
                                  {"MIO_BN_VEC_SIZE", vectorsize},
                                  {"MIO_BN_STASH_METHOD", stash_method},
                                  {"MIOPEN_NRN_OP_ID", problem.GetActivationDesc().GetMode()}};

        {
            // OpenCL kernels for variant 0-4
            kernel.kernel_file      = "MIOpenBatchNormBwdSpatial.cl";
            std::string kernel_name = "MIOpenBatchNormBwdSpatial";

            build_params << KernelBuildParameters{
                {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
                {"MIO_BN_GFX110X", (StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
                {"MIO_BN_GFX120X", (StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
                {"MIO_BN_GFX115X", (StartsWith(handle.GetDeviceName(), "gfx115") ? "1" : "0")},
            };

            kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(ylocalsize);
            kernel.l_wk.push_back(zlocalsize);

            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(ygridsize);
            kernel.g_wk.push_back(zgridsize);

            if(variant != 2)
            {
                kernel.kernel_name = kernel_name;
                result.construction_params.push_back(kernel);
            }
            else
            {
                auto single_yzgroup_kernel = kernel;

                single_yzgroup_kernel.l_wk[0] = xlocalsize_final;
                single_yzgroup_kernel.l_wk[1] = ylocalsize_final;
                single_yzgroup_kernel.l_wk[2] = zlocalsize_final;
                single_yzgroup_kernel.g_wk[1] = ylocalsize_final;
                single_yzgroup_kernel.g_wk[2] = zlocalsize_final;

                if(!problem.UseSaved())
                {
                    kernel.kernel_name = kernel_name + "MeanVariance";
                    result.construction_params.push_back(kernel);

                    single_yzgroup_kernel.kernel_name = kernel_name + "FinalMeanVariance";
                    result.construction_params.push_back(single_yzgroup_kernel);
                }

                kernel.kernel_name = kernel_name + "DScaleDBias";
                result.construction_params.push_back(kernel);

                single_yzgroup_kernel.kernel_name = kernel_name + "FinalDScaleDBias";
                result.construction_params.push_back(single_yzgroup_kernel);

                kernel.kernel_name = kernel_name + "DX";
                result.construction_params.push_back(kernel);
            }
        }
    }

    const auto dtype    = problem.GetBnScale().GetType();
    const auto useSaved = problem.UseSaved();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::BwdInvokeParams>();

            float alpha_activ = problem.GetActivationDesc().GetAlpha();
            float beta_activ  = problem.GetActivationDesc().GetBeta();
            float ctime       = 0.;
            visit_float(dtype, [&](auto as_float) {
                if(variant != 2)
                {
                    decltype(auto) kernel = handle_.Run(kernels.front());
                    if(useSaved)
                    {
                        kernel(params.x,
                               params.dy,
                               params.dx,
                               params.bnScale,
                               params.bnBias,
                               params.resultBnScaleDiff,
                               params.resultBnBiasDiff,
                               params.savedMean,
                               params.savedInvVariance,
                               as_float(inhw),
                               alpha_activ,
                               beta_activ);
                    }
                    else
                    {
                        kernel(params.x,
                               params.dy,
                               params.dx,
                               params.bnScale,
                               params.bnBias,
                               params.resultBnScaleDiff,
                               params.resultBnBiasDiff,
                               params.epsilon,
                               inhw,
                               alpha_activ,
                               beta_activ);
                    }
                }
                else
                {
                    if(useSaved)
                    {
                        handle_.Run(kernels[0])(params.x,
                                                params.dy,
                                                params.dx,
                                                params.bnScale,
                                                params.bnBias,
                                                params.savedMean,
                                                params.savedInvVariance,
                                                alpha_activ,
                                                beta_activ);
                        profileSequence(handle_, 0, &ctime);

                        handle_.Run(kernels[1])(
                            params.dx, params.resultBnScaleDiff, params.resultBnBiasDiff);
                        profileSequence(handle_, 1, &ctime);

                        handle_.Run(kernels[2])(params.x,
                                                params.dy,
                                                params.dx,
                                                params.bnScale,
                                                params.bnBias,
                                                params.resultBnScaleDiff,
                                                params.resultBnBiasDiff,
                                                params.savedMean,
                                                params.savedInvVariance,
                                                as_float(inhw),
                                                alpha_activ,
                                                beta_activ);
                        profileSequence(handle_, 2, &ctime);
                    }
                    else
                    {
                        handle_.Run(kernels[0])(params.x, params.dx); // mean variance
                        profileSequence(handle_, 0, &ctime);

                        handle_.Run(kernels[1])(
                            params.dx, as_float(inhw), params.epsilon); // final mean variance
                        profileSequence(handle_, 1, &ctime);

                        handle_.Run(kernels[2])(params.x,
                                                params.dy,
                                                params.dx, // dscale dbias
                                                params.bnScale,
                                                params.bnBias,
                                                alpha_activ,
                                                beta_activ);
                        profileSequence(handle_, 1, &ctime);

                        handle_.Run(kernels[3])(params.dx,
                                                params.resultBnScaleDiff,
                                                params.resultBnBiasDiff); // final dscale dbias
                        profileSequence(handle_, 1, &ctime);

                        handle_.Run(kernels[4])(params.x,
                                                params.dy,
                                                params.dx,
                                                params.bnScale,
                                                params.bnBias,
                                                params.resultBnScaleDiff,
                                                params.resultBnBiasDiff,
                                                as_float(inhw),
                                                alpha_activ,
                                                beta_activ);
                        profileSequence(handle_, 2, &ctime);
                    }
                }
            });
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
