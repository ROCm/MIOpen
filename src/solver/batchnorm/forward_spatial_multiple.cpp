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

namespace miopen {

namespace solver {

namespace batchnorm {

bool BnFwdTrainingSpatialMultiple::IsApplicable(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem) const
{
    if(problem.GetDirection() != miopen::batchnorm::Direction::ForwardTraining ||
       problem.GetMode() != miopenBNSpatial)
        return false;

    if(!IsOCLFwdTrainTypeValid(problem))
        return false;

    size_t n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nhw     = n * in_cstride;

    // Variant 2 needs space for 2 fp32 elements per each x thread (including the last workgroup)
    // to stash intermediate mean and variance
    unsigned int stash_values = 2;
    if(problem.IsLayoutNHWC())
    {
        // TODO: For now enable variant 2 for NHWC because other variants are slower.
        // Remove when other variants are optimized
        unsigned int xlocalsize = std::min(size_t{1 << int(std::ceil(std::log2(c)))}, size_t{64});
        unsigned int ylocalsize = 1024 / xlocalsize;
        unsigned int last_ylocalsize =
            in_cstride % ylocalsize == 0 ? ylocalsize : in_cstride % ylocalsize;
        if(problem.GetXDesc().GetType() == miopenFloat)
        {
            if(last_ylocalsize < stash_values)
                return false;
        }
        else
        {
            // Even threads use 2 values at even rows, odd threads - at odd rows.
            if(c % 2 != 0 || last_ylocalsize < stash_values * 2)
                return false;
        }
    }
    else
    {
        bool bfpmixparm = false;

        if((problem.GetXDesc().GetType() == miopenHalf ||
            problem.GetXDesc().GetType() == miopenBFloat16) &&
           problem.GetBnScale().GetType() == miopenFloat)
        {
            bfpmixparm = true;
        }
        if(!((n >= 3 && in_cstride > 512 && (in_nhw >= 33554432 || in_cstride <= 1024) &&
              ((n < 256) || (in_cstride <= 60) || !bfpmixparm) &&
              (!bfpmixparm || in_cstride <= 512)) ||
             ((n > 768) && (in_cstride > 150))))
        {
            return false;
        }

        unsigned int ylocalsize = 1024;
        unsigned int last_ylocalsize =
            in_cstride % ylocalsize == 0 ? ylocalsize : in_cstride % ylocalsize;
        if(last_ylocalsize < stash_values * (problem.GetXDesc().GetType() == miopenFloat ? 1 : 2))
            return false;
    }
    return true;
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

    size_t xlocalsize;
    size_t ylocalsize;
    size_t xgridsize;
    size_t ygridsize;

    size_t max_localsize = 1024;
    if(((in_cstride < 256) && (n < 256)) || ((in_cstride < 100) && (n <= 256)))
        max_localsize = 256;
    int variant           = 2;
    unsigned int ldsgcn   = max_localsize / 64;
    unsigned int ldsnogcn = max_localsize;
    if(problem.IsLayoutNHWC())
    {
        xlocalsize = std::min(size_t{1 << int(std::ceil(std::log2(c)))}, size_t{64});
        xgridsize  = xlocalsize * ((c + xlocalsize - 1) / xlocalsize);
        ylocalsize = max_localsize / xlocalsize;
        ygridsize  = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
    }
    else
    {
        xlocalsize = 1;
        xgridsize  = c;
        ylocalsize = max_localsize;
        ygridsize  = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
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
            {"MIO_BN_NGRPS", ygridsize / ylocalsize},
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
        copy.g_wk[1]     = kernel.l_wk[1];
        result.construction_params.push_back(copy);

        copy.kernel_name = kernel.kernel_name + "Norm";
        copy.g_wk[1]     = kernel.g_wk[1];
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
