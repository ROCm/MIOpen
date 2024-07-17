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

bool BnFwdTrainingPerActivation::IsApplicable(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem) const
{
    return problem.GetDirection() == miopen::batchnorm::Direction::ForwardTraining ||
           problem.GetMode() == miopenBNPerActivation;
}

ConvSolution
BnFwdTrainingPerActivation::GetSolution(const ExecutionContext& context,
                                        const miopen::batchnorm::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    decltype(auto) xDesc = problem.GetXDesc();

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nstride = c * in_cstride;

    {
        decltype(auto) handle                 = context.GetStream();
        decltype(auto) bnScaleBiasMeanVarDesc = problem.GetBnScaleBiasMeanVarDesc();

        unsigned int in_nhw  = n * in_cstride;
        unsigned int in_nchw = n * in_nstride;

        bool bfpmixparm = false;
        bool bfp16parm  = false;
        bool bfp32parm  = true;
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

        std::size_t xlocalsize = 1;
        std::size_t ylocalsize = 256;
        std::size_t zlocalsize = 1;
        std::size_t segment    = (in_cstride + ylocalsize - 1) / ylocalsize;
        std::size_t xgridsize  = c;
        std::size_t ygridsize  = segment * ylocalsize;
        std::size_t zgridsize  = 1;

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
            {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
            {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
            {"MIO_SAVE_MEAN_VARIANCE", static_cast<int>(problem.GetResultSave())},
            {"MIO_RUNNING_RESULT", static_cast<int>(problem.GetResultRunning())},
            {"MIO_BN_N", n},
            {"MIO_BN_C", c},
            {"MIO_BN_HW", in_cstride},
            {"MIO_BN_NHW", in_nhw},
            {"MIO_BN_CHW", in_nstride},
            {"MIO_BN_NCHW", in_nchw},
            {"MIO_BN_LDS_SIZE", ylocalsize},
            {"MIO_BN_GRP0", xlocalsize},
            {"MIO_BN_GRP1", ylocalsize},
            {"MIO_BN_GRP2", zlocalsize},
            {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
            {"MIO_BN_GFX110X", (StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
            {"MIO_BN_GFX120X", (StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
        };

        auto kernel = KernelInfo{};

        kernel.kernel_name = "MIOpenBatchNormFwdTrainPerActivation";
        kernel.kernel_file = "MIOpenBatchNormFwdTrainPerAct.cl";

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::InvokeParams>();
            const auto resultsave =
                params.resultSaveMean != nullptr && params.resultSaveInvVariance != nullptr;
            const auto resultrunning =
                params.resultRunningMean != nullptr && params.resultRunningVariance != nullptr;

            if(resultsave && resultrunning)
            {
                kernel(params.x,
                       in_nstride,
                       in_cstride,
                       params.y,
                       params.bnScale,
                       params.bnBias,
                       params.expAvgFactor,
                       params.resultRunningMean,
                       params.resultRunningVariance,
                       params.epsilon,
                       params.resultSaveMean,
                       params.resultSaveInvVariance);
            }
            else if(resultsave)
            {
                kernel(params.x,
                       in_nstride,
                       in_cstride,
                       params.y,
                       params.bnScale,
                       params.bnBias,
                       params.epsilon,
                       params.resultSaveMean,
                       params.resultSaveInvVariance);
            }
            else if(resultrunning)
            {
                kernel(params.x,
                       in_nstride,
                       in_cstride,
                       params.y,
                       params.bnScale,
                       params.bnBias,
                       params.expAvgFactor,
                       params.resultRunningMean,
                       params.resultRunningVariance,
                       params.epsilon);
            }
            else
            {
                kernel(params.x,
                       in_nstride,
                       in_cstride,
                       params.y,
                       params.bnScale,
                       params.bnBias,
                       params.epsilon);
            }
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
