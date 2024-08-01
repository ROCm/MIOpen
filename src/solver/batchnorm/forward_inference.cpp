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

bool BnFwdInference::IsApplicable(const ExecutionContext&,
                                  const miopen::batchnorm::ProblemDescription& problem) const
{
    if(problem.IsLayoutNHWC())
        return false;
    if(problem.GetDirection() != miopen::batchnorm::Direction::ForwardInference)
        return false;
    if(!(problem.IsFp32() or problem.IsFp16()))
        return false;
    if(!problem.Is2D())
        return false;
    return true;
}

ConvSolution BnFwdInference::GetSolution(const ExecutionContext& context,
                                         const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& handle = context.GetStream();

    bool bfpmixparm = false;
    bool bfp16parm  = false;
    bool bfp32parm  = true;
    if(problem.GetXDesc().GetType() == miopenHalf &&
       problem.GetBnScaleBiasMeanVarDesc().GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
    }
    else if(problem.GetXDesc().GetType() == miopenHalf &&
            problem.GetBnScaleBiasMeanVarDesc().GetType() == miopenFloat)
    {
        bfpmixparm = true;
        bfp32parm  = false;
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    unsigned int in_cstride = h * w;

    auto result = ConvSolution{miopenStatusSuccess};

    {
        size_t xlocalsize = 1;
        auto xgridsize    = c;
        size_t ylocalsize = 256;
        size_t ygridsize  = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenBatchNormFwdInfer"; // build this up
        kernel.kernel_name = "MIOpenBatchNormFwdInfer";
        if(problem.GetMode() == miopenBNSpatial)
        { // SPATIAL kernels
            kernel.kernel_file += "Spatial.cl";
            kernel.kernel_name += "SpatialEst";
        }
        else
        { // PER ACTIVATION
            kernel.kernel_file += "PerAct.cl";
            kernel.kernel_name += "PerActivationEst";
        }

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
            {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
            {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
            {"MIO_BN_GRP0", xlocalsize},
            {"MIO_BN_GRP1", ylocalsize},
            {"MIO_BN_GRP2", zlocalsize},
            {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
            {"MIO_BN_GFX110X", (StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
            {"MIO_BN_GFX120X", (StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::InfInvokeParams>();

            int n_, c_, h_, w_;
            std::tie(n_, c_, h_, w_) = tien<4>(params.xDesc->GetLengths());

            unsigned int in_nstride_ = c_ * h_ * w_;
            unsigned int in_cstride_ = h_ * w_;

            kernel(params.x,
                   params.y,
                   params.estimatedMean,
                   params.estimatedVariance,
                   params.bnScale,
                   params.bnBias,
                   params.epsilon,
                   n_,
                   in_cstride_,
                   in_nstride_);
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
