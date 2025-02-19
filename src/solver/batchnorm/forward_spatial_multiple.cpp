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

#include <miopen/batchnorm/solvers.hpp>

#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>

#define WORKAROUND_SWDEV_253606 1

namespace miopen {

namespace solver {

namespace batchnorm {

bool BnFwdTrainingSpatialMultiple::IsApplicable(
    const ExecutionContext& context, const miopen::batchnorm::ProblemDescription& problem) const
{
    if(problem.GetDirection() != miopen::batchnorm::Direction::ForwardTraining ||
       problem.GetMode() != miopenBNSpatial)
        return false;

    if(!IsOCLFwdTrainTypeValid(problem))
        return false;

    return !BnFwdTrainingSpatialSingle{}.IsApplicable(context, problem);
}

ConvSolution BnFwdTrainingSpatialMultiple::GetSolution(
    const ExecutionContext& context, const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& handle                 = context.GetStream();
    const auto& xDesc                  = problem.GetXDesc();
    const auto& bnScaleBiasMeanVarDesc = problem.GetBnScale();

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nstride = c * in_cstride;
    unsigned int in_nhw     = n * in_cstride;
    unsigned int in_nchw    = n * in_nstride;
    auto inhw               = float(1.0 / in_nhw);

    size_t xlocalsize = 1024;
    if(((in_cstride < 256) && (n < 256)) || ((in_cstride < 100) && (n <= 256)))
        xlocalsize = 256;

    size_t ylocalsize = 1;

    size_t xgridsize = c * xlocalsize;
    size_t ygridsize = 1;

    bool bfpmixparm   = false;
    bool bbfpmixparam = false;
    bool bfp16parm    = false;
    bool bfp32parm    = true;
    if(xDesc.GetType() == miopenHalf && bnScaleBiasMeanVarDesc.GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
    }
    else if(xDesc.GetType() == miopenHalf && bnScaleBiasMeanVarDesc.GetType() == miopenFloat)
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

    int variant           = 1;
    unsigned int ldsgcn   = xlocalsize / 64;
    unsigned int ldsnogcn = xlocalsize;

    if(!problem.IsLayoutNHWC())
    {
#if(WORKAROUND_SWDEV_253606 == 0)
        if(n < 3)
        {
            variant    = 4;
            xlocalsize = 256;
            xgridsize  = c * xlocalsize;
            ylocalsize = 1;
            ygridsize  = 1;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }
        else
#endif

            // clang-format off
        if((in_nhw < 33554432 && in_cstride > 1024) ||
            ((n >= 256) && (in_cstride > 60) && bfpmixparm) ||
            ((in_cstride > 512) && bfpmixparm))
        {
            variant = 1;
        }
        else if(in_cstride <= 512)
        {
            variant = 0;
        }
        else
        {
            variant      = 2;
            xlocalsize   = 1;
            ylocalsize   = 1024;
            auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize    = c;
            ygridsize    = segment * ylocalsize;
            ldsgcn       = ylocalsize / 64;
            ldsnogcn     = ylocalsize;
        }
        // clang-format on

        if((n > 768) && (in_cstride > 150) && bfp32parm)
        {
            variant      = 2;
            xlocalsize   = 1;
            ylocalsize   = 1024;
            auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize    = c;
            ygridsize    = segment * ylocalsize;
            ldsgcn       = ylocalsize / 64;
            ldsnogcn     = ylocalsize;
        }
    }

    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto kernel = KernelInfo{};

        kernel.kernel_name = "MIOpenBatchNormFwdTrainSpatial";
        kernel.kernel_file = "MIOpenBatchNormFwdTrainSpatial.cl";

        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
            {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
            {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
            {"MIOPEN_USE_BFPMIX", static_cast<int>(bbfpmixparam)},
            {"MIO_SAVE_MEAN_VARIANCE", static_cast<int>(problem.GetResultSave())},
            {"MIO_RUNNING_RESULT", static_cast<int>(problem.GetResultRunning())},
            {"MIO_BN_N", n},
            {"MIO_BN_C", c},
            {"MIO_BN_HW", in_cstride},
            {"MIO_BN_NHW", in_nhw},
            {"MIO_BN_CHW", in_nstride},
            {"MIO_BN_NCHW", in_nchw},
            {"MIO_BN_NGRPS", int(std::ceil(float(ygridsize) / ylocalsize))},
            {"MIO_BN_LDS_SIZE", ldsnogcn},
            {"MIO_BN_LDSGCN_SIZE", ldsgcn},
            {"MIO_BN_VARIANT", variant},
            {"MIO_BN_GRP0", xlocalsize},
            {"MIO_BN_GRP1", ylocalsize},
            {"MIO_BN_GRP2", zlocalsize},
            {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
            {"MIO_BN_GFX110X", (StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
            {"MIO_BN_GFX120X", (StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
            {"MIO_LAYOUT_NHWC", static_cast<int>(problem.IsLayoutNHWC())},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        auto copy        = kernel;
        copy.kernel_name = kernel.kernel_name + "MeanVariance";
        result.construction_params.push_back(copy);

        copy.kernel_name = kernel.kernel_name + "FinalMeanVariance";
        result.construction_params.push_back(copy);

        copy.kernel_name = kernel.kernel_name + "Norm";
        result.construction_params.push_back(copy);
    }

    const auto dtype = bnScaleBiasMeanVarDesc.GetType();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::FwdTrainInvokeParams>();
            const auto resultsave =
                params.resultSaveMean != nullptr && params.resultSaveInvVariance != nullptr;
            const auto resultrunning =
                params.resultRunningMean != nullptr && params.resultRunningVariance != nullptr;

            float ctime = 0.;
            visit_float(dtype, [&](auto as_float) {
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

                handle_.Run(kernels[2])(params.x, params.y, params.bnScale, params.bnBias);
                profileSequence(handle_, 2, &ctime);
            });
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
