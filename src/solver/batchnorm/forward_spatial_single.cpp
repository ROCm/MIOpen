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
#include <miopen/stringutils.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>

#define WORKAROUND_SWDEV_253606 1

namespace miopen {

namespace solver {

namespace batchnorm {

bool BnFwdTrainingSpatialSingle::IsApplicable(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& bn_problem) const
{
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::ForwardTraining ||
       bn_problem.GetMode() != miopenBNSpatial)
        return false;

    if(!IsOCLFwdTrainTypeValid(bn_problem))
        return false;

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(bn_problem.GetXDesc().GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nhw     = n * in_cstride;

    bool bfpmixparm = false;
    bool bfp32parm  = true;

    if(bn_problem.GetXDesc().GetType() == miopenHalf &&
       bn_problem.GetBnScale().GetType() == miopenHalf)
    {
        bfp32parm = false;
    }
    else if((bn_problem.GetXDesc().GetType() == miopenHalf ||
             bn_problem.GetXDesc().GetType() == miopenBFloat16) &&
            bn_problem.GetBnScale().GetType() == miopenFloat)
    {
        bfpmixparm = true;
        bfp32parm  = false;
    }

    // clang-format off
    if(!(WORKAROUND_SWDEV_253606 == 0 && n < 3) &&
        !((in_nhw < 33554432 && in_cstride > 1024) ||
          ((n >= 256) && (in_cstride > 60) && bfpmixparm) ||
          ((in_cstride > 512) && bfpmixparm) ||
          in_cstride <= 512))
        return false;
    // clang-format on

    if((n > 768) && (in_cstride > 150) && bfp32parm)
    {
        return false;
    }

    return true;
}

ConvSolution
BnFwdTrainingSpatialSingle::GetSolution(const ExecutionContext& context,
                                        const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& handle = context.GetStream();

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

    size_t xlocalsize = 1024;
    if(((in_cstride < 256) && (n < 256)) || ((in_cstride < 100) && (n <= 256)))
        xlocalsize = 256;

    size_t ylocalsize = 1;

    size_t xgridsize = c * xlocalsize;
    size_t ygridsize = 1;

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
        {
            // clang-format off
            if( (in_nhw < 33554432 && in_cstride > 1024) ||
                    ((n >= 256) && (in_cstride > 60) && (bfpmixparm || bbfpmixparam)) ||
                    ((in_cstride > 512) && (bfpmixparm || bbfpmixparam)))
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
    }

    auto result = ConvSolution{miopenStatusSuccess};

    {
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_name = "MIOpenBatchNormFwdTrainSpatial";
        kernel.kernel_file = "MIOpenBatchNormFwdTrainSpatial.cl";

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
            {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
            {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
            {"MIOPEN_USE_BFPMIX", static_cast<int>(bbfpmixparam)},
            {"MIO_SAVE_MEAN_VARIANCE", static_cast<int>(problem.GetResultSave())},
            {"MIO_RUNNING_RESULT", static_cast<int>(problem.GetResultRunning())},
            {"MIO_BN_VARIANT", variant},
            {"MIO_BN_LDS_SIZE", ldsnogcn},
            {"MIO_BN_LDSGCN_SIZE", std::to_string(ldsgcn)},
            {"MIO_BN_N", n},
            {"MIO_BN_GRP0", xlocalsize},
            {"MIO_BN_GRP1", ylocalsize},
            {"MIO_BN_GRP2", zlocalsize},
            {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
            {"MIO_BN_GFX110X", (StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
            {"MIO_BN_GFX120X", (StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
            {"MIO_LAYOUT_NHWC", static_cast<int>(problem.IsLayoutNHWC())},
        };

        if(variant != 4)
        {
            build_params.Define("MIO_BN_C", c);
            build_params.Define("MIO_BN_HW", in_cstride);
            build_params.Define("MIO_BN_NHW", in_nhw);
            build_params.Define("MIO_BN_CHW", in_nstride);
            build_params.Define("MIO_BN_NCHW", in_nchw);
        }

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    const auto dtype = problem.GetBnScale().GetType();
    const auto vn4   = (variant != 4);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::FwdTrainInvokeParams>();
            const auto resultsave =
                params.resultSaveMean != nullptr && params.resultSaveInvVariance != nullptr;
            const auto resultrunning =
                params.resultRunningMean != nullptr && params.resultRunningVariance != nullptr;

            visit_float(dtype, [&](auto as_float) {
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
                               params.resultSaveInvVariance);
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
                               in_nstride);
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
                               params.resultSaveInvVariance);
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
                               in_nstride);
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
                               params.epsilon);
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
                               in_nstride);
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
                               params.epsilon);
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
                               in_nstride);
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
