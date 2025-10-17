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

// Spatial multiple needs space for 2 fp32 elements
// per each x thread (including the last workgroup)
// to stash intermediate mean and variance
const unsigned int stash_values_fwd = 2;

bool PerformanceConfigBnFwdTraining::IsValid(
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
        return IsSpatialMultipleApplicable(
            problem, vectorsize, stash_values_fwd, ylocalsize, zlocalsize, nelements);
    }
    return true;
}

void PerformanceConfigBnFwdTraining::HeuristicInit(
    const miopen::batchnorm::ProblemDescription& problem)
{
    // Define default configuration based on heuristics and
    // add all other valid configurations for the given problem
    if(UseMultiple(problem))
    {
        DefaultConfigSpatialMultiple(problem, stash_values_fwd, this->valid_kernels);
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
            DefaultConfigSpatialMultiple(problem, stash_values_fwd, this->valid_kernels);
        }
    }

    // Set index and kernel_id to default value
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

bool PerformanceConfigBnFwdTraining::SetNextValue(
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

bool PerformanceConfigBnFwdTraining::operator==(const PerformanceConfigBnFwdTraining& other) const
{
    return this->kernel_id == other.kernel_id;
}

bool PerformanceConfigBnFwdTraining::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool BnFwdTrainingSpatial::IsApplicable(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& bn_problem) const
{
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::ForwardTraining ||
       bn_problem.GetMode() != miopenBNSpatial)
        return false;

    if(!bn_problem.Is2D())
        return false;

    if(!IsOCLFwdTrainTypeValid(bn_problem))
        return false;

    int activ_mode = bn_problem.GetActivationDesc().GetMode();
    if(activ_mode != miopenActivationPASTHRU && activ_mode != miopenActivationRELU &&
       activ_mode != miopenActivationCLIPPEDRELU && activ_mode != miopenActivationCLAMP)
    {
        return false;
    }

    return true;
}

PerformanceConfigBnFwdTraining BnFwdTrainingSpatial::GetDefaultPerformanceConfig(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
    PerformanceConfigBnFwdTraining pp;
    pp.HeuristicInit(problem_desc);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool BnFwdTrainingSpatial::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const miopen::batchnorm::ProblemDescription& problem_desc,
    const PerformanceConfigBnFwdTraining& config) const
{
    bool valid = config.IsValid(ctx, problem_desc);
    return valid;
}

PerformanceConfigBnFwdTraining
BnFwdTrainingSpatial::Search(const ExecutionContext& ctx,
                             const miopen::batchnorm::ProblemDescription& problem,
                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution BnFwdTrainingSpatial::GetSolution(const ExecutionContext& context,
                                               const miopen::batchnorm::ProblemDescription& problem,
                                               const PerformanceConfigBnFwdTraining& config) const
{
    const auto& handle = context.GetStream();
    // Only one can be true
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
    auto inhw               = float(1.0 / in_nhw);

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
        if(((in_cstride < 256) && (n < 256)) || ((in_cstride < 100) && (n <= 256)))
        {
            xlocalsize = 256;
        }
        xgridsize = c * xlocalsize;
        ldsgcn    = xlocalsize / 64;
        ldsnogcn  = xlocalsize;
#if(WORKAROUND_SWDEV_253606 == 0)
        if(variant == 4)
        {
            xlocalsize = 256;
            xgridsize  = c * xlocalsize;
            ylocalsize = 1;
            ygridsize  = 1;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }
#endif
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

        // Get the stash method based on problem size and WG size
        stash_method = GetStashMethod(problem.IsLayoutNHWC(),
                                      problem.GetXDesc().GetType(),
                                      stash_values_fwd,
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
        ldsgcn   = xlocalsize * ylocalsize * zlocalsize / 64;
    }

    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto use_hip = variant == 1;
        auto kernel  = KernelInfo{};

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
            {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
            {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
            {"MIOPEN_USE_BFPMIX", static_cast<int>(bbfpmixparam)},
            {"MIO_SAVE_MEAN_VARIANCE", static_cast<int>(problem.GetResultSave())},
            {"MIO_RUNNING_RESULT",
             context.is_for_generic_search ? static_cast<int>(0)
                                           : static_cast<int>(problem.GetResultRunning())},
            {"MIO_BN_VARIANT", variant},
            {"MIO_BN_LDS_SIZE", ldsnogcn},
            {"MIO_BN_LDSGCN_SIZE", std::to_string(ldsgcn)},
            {"MIO_BN_N", n},
            {"MIO_BN_NGRPS", ygridsize / ylocalsize},
            {"MIO_BN_NGRPS2", zgridsize / zlocalsize},
            {"MIO_BN_N_ELEMENTS", nelements},
            {"MIO_BN_GRP0", xlocalsize},
            {"MIO_BN_GRP1", ylocalsize},
            {"MIO_BN_GRP2", zlocalsize},
            {"MIO_BN_GRP0_FINAL", xlocalsize_final},
            {"MIO_BN_GRP1_FINAL", ylocalsize_final},
            {"MIO_BN_GRP2_FINAL", zlocalsize_final},
            {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
            {"MIO_BN_GFX110X", (StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
            {"MIO_BN_GFX120X", (StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
            {"MIO_BN_GFX115X", (StartsWith(handle.GetDeviceName(), "gfx115") ? "1" : "0")},
            {"MIO_LAYOUT_NHWC", static_cast<int>(problem.IsLayoutNHWC())},
            {"MIO_BN_VECTORIZE", static_cast<int>(vectorsize > 1)},
            {"MIO_BN_VEC_SIZE", vectorsize},
            {"MIO_BN_STASH_METHOD", stash_method},
            {"MIOPEN_NRN_OP_ID", problem.GetActivationDesc().GetMode()}};

        if(variant != 4)
        {
            build_params.Define("MIO_BN_C", c);
            build_params.Define("MIO_BN_HW", in_cstride);
            build_params.Define("MIO_BN_NHW", in_nhw);
            build_params.Define("MIO_BN_CHW", in_nstride);
            build_params.Define("MIO_BN_NCHW", in_nchw);
        }

        kernel.kernel_file =
            use_hip ? "MIOpenBatchNormFwdTrainSpatialHIP.cpp" : "MIOpenBatchNormFwdTrainSpatial.cl";
        std::string kernel_name = "MIOpenBatchNormFwdTrainSpatial";
        if(use_hip)
        {
            kernel_name += "HIP";
            kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
        }
        else
        {
            kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});
        }

        kernel.kernel_name = kernel_name;

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        if(variant != 2)
        {
            result.construction_params.push_back(kernel);
        }
        else
        {
            auto single_yzgroup_kernel    = kernel;
            single_yzgroup_kernel.l_wk[0] = xlocalsize_final;
            single_yzgroup_kernel.l_wk[1] = ylocalsize_final;
            single_yzgroup_kernel.l_wk[2] = zlocalsize_final;
            single_yzgroup_kernel.g_wk[1] = ylocalsize_final;
            single_yzgroup_kernel.g_wk[2] = zlocalsize_final;

            kernel.kernel_name = kernel_name + "MeanVariance";
            result.construction_params.push_back(kernel);

            single_yzgroup_kernel.kernel_name = kernel_name + "FinalMeanVariance";
            result.construction_params.push_back(single_yzgroup_kernel);

            kernel.kernel_name = kernel_name + "Norm";
            result.construction_params.push_back(kernel);
        }
    }

    const auto dtype = problem.GetBnScale().GetType();
    const auto vn4   = (variant != 4);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::FwdTrainInvokeParams>();
            const auto resultsave =
                params.resultSaveMean != nullptr && params.resultSaveInvVariance != nullptr;
            const auto resultrunning = params.resultRunningMean != nullptr &&
                                       params.resultRunningVariance != nullptr &&
                                       !context.is_for_generic_search;

            float alpha_activ = problem.GetActivationDesc().GetAlpha();
            float beta_activ  = problem.GetActivationDesc().GetBeta();

            float ctime = 0.;
            visit_float(dtype, [&](auto as_float) {
                if(variant != 2)
                {
                    decltype(auto) kernel = handle_.Run(kernels.front());
                    if(resultsave && resultrunning)
                    {
                        if(vn4)
                        {
                            kernel(params.x,
                                   params.y,
                                   params.bnScale,
                                   params.bnBias,
                                   as_float(inhw),
                                   params.expAvgFactor,
                                   params.resultRunningMean,
                                   params.resultRunningVariance,
                                   params.epsilon,
                                   params.resultSaveMean,
                                   params.resultSaveInvVariance,
                                   alpha_activ,
                                   beta_activ);
                        }
                        else
                        {
                            kernel(params.x,
                                   params.y,
                                   params.bnScale,
                                   params.bnBias,
                                   as_float(inhw),
                                   params.expAvgFactor,
                                   params.resultRunningMean,
                                   params.resultRunningVariance,
                                   params.epsilon,
                                   params.resultSaveMean,
                                   params.resultSaveInvVariance,
                                   in_cstride,
                                   in_nstride,
                                   alpha_activ,
                                   beta_activ);
                        }
                    }
                    else if(resultsave)
                    {
                        if(vn4)
                        {
                            kernel(params.x,
                                   params.y,
                                   params.bnScale,
                                   params.bnBias,
                                   as_float(inhw),
                                   params.epsilon,
                                   params.resultSaveMean,
                                   params.resultSaveInvVariance,
                                   alpha_activ,
                                   beta_activ);
                        }
                        else
                        {
                            kernel(params.x,
                                   params.y,
                                   params.bnScale,
                                   params.bnBias,
                                   as_float(inhw),
                                   params.epsilon,
                                   params.resultSaveMean,
                                   params.resultSaveInvVariance,
                                   in_cstride,
                                   in_nstride,
                                   alpha_activ,
                                   beta_activ);
                        }
                    }
                    else if(resultrunning)
                    {
                        if(vn4)
                        {
                            kernel(params.x,
                                   params.y,
                                   params.bnScale,
                                   params.bnBias,
                                   as_float(inhw),
                                   params.expAvgFactor,
                                   params.resultRunningMean,
                                   params.resultRunningVariance,
                                   params.epsilon,
                                   alpha_activ,
                                   beta_activ);
                        }
                        else
                        {
                            kernel(params.x,
                                   params.y,
                                   params.bnScale,
                                   params.bnBias,
                                   as_float(inhw),
                                   params.expAvgFactor,
                                   params.resultRunningMean,
                                   params.resultRunningVariance,
                                   params.epsilon,
                                   in_cstride,
                                   in_nstride,
                                   alpha_activ,
                                   beta_activ);
                        }
                    }
                    else
                    {
                        if(vn4)
                        {
                            kernel(params.x,
                                   params.y,
                                   params.bnScale,
                                   params.bnBias,
                                   as_float(inhw),
                                   params.epsilon,
                                   alpha_activ,
                                   beta_activ);
                        }
                        else
                        {
                            kernel(params.x,
                                   params.y,
                                   params.bnScale,
                                   params.bnBias,
                                   as_float(inhw),
                                   params.epsilon,
                                   in_cstride,
                                   in_nstride,
                                   alpha_activ,
                                   beta_activ);
                        }
                    }
                }
                else
                {
                    handle_.Run(kernels[0])(params.x, params.y);
                    profileSequence(handle_, 0, &ctime);

                    if(resultsave && resultrunning)
                    {
                        handle_.Run(kernels[1])(params.y,
                                                as_float(inhw),
                                                params.expAvgFactor,
                                                params.resultRunningMean,
                                                params.resultRunningVariance,
                                                params.epsilon,
                                                params.resultSaveMean,
                                                params.resultSaveInvVariance);
                    }
                    else if(resultsave)
                    {
                        handle_.Run(kernels[1])(params.y,
                                                as_float(inhw),
                                                params.epsilon,
                                                params.resultSaveMean,
                                                params.resultSaveInvVariance);
                    }
                    else if(resultrunning)
                    {
                        handle_.Run(kernels[1])(params.y,
                                                as_float(inhw),
                                                params.expAvgFactor,
                                                params.resultRunningMean,
                                                params.resultRunningVariance,
                                                params.epsilon);
                    }
                    else
                    {
                        handle_.Run(kernels[1])(params.y, as_float(inhw), params.epsilon);
                    }

                    profileSequence(handle_, 1, &ctime);

                    handle_.Run(kernels[2])(
                        params.x, params.y, params.bnScale, params.bnBias, alpha_activ, beta_activ);
                    profileSequence(handle_, 2, &ctime);
                }
            });
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
