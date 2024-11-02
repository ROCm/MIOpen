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

bool BnBwdTrainingPerActivation::IsApplicable(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& problem) const
{
    if(!problem.Is2D())
        return false;
    if(problem.GetDirection() != miopen::batchnorm::Direction::Backward &&
       problem.GetMode() != miopenBNPerActivation)
        return false;
    if(!::miopen::batchnorm::IsOCLBwdTypeValid(problem))
        return false;
    return true;
}

ConvSolution
BnBwdTrainingPerActivation::GetSolution(const ExecutionContext& context,
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

    auto result = ConvSolution{miopenStatusSuccess};

    {
        unsigned int in_nhw  = n * in_cstride;
        unsigned int in_nchw = n * in_nstride;

        size_t xlocalsize = 1;
        size_t ylocalsize = (64 >= in_cstride) ? 64 : 256;
        size_t zlocalsize = 1;

        unsigned int segment = std::ceil(double(in_cstride) / double(ylocalsize));

        size_t xgridsize = c;
        size_t ygridsize = segment * ylocalsize;
        size_t zgridsize = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenBatchNormBwdPerAct.cl";
        kernel.kernel_name = "MIOpenBatchNormBwdPerActivation";

        if(problem.UseSaved())
            kernel.kernel_name += "Saved";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
            {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
            {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
            {"MIOPEN_USE_BFPMIX", static_cast<int>(bbfpmixparam)},
            {"MIO_BN_N", static_cast<int>(n)},
            {"MIO_BN_C", static_cast<int>(c)},
            {"MIO_BN_HW", static_cast<int>(in_cstride)},
            {"MIO_BN_NHW", static_cast<int>(in_nhw)},
            {"MIO_BN_CHW", in_nstride},
            {"MIO_BN_NCHW", in_nchw},
            {"MIO_BN_NGRPS", int(std::ceil(float(ygridsize) / ylocalsize))},
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

    const auto useSaved = problem.UseSaved();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::BwdInvokeParams>();

            if(useSaved)
            {
                kernel(params.x,
                       params.dy,
                       n,
                       in_nstride,
                       in_cstride,
                       params.dx,
                       params.bnScale,
                       params.resultBnScaleDiff,
                       params.resultBnBiasDiff,
                       params.savedMean,
                       params.savedInvVariance);
            }
            else
            {
                kernel(params.x,
                       params.dy,
                       n,
                       in_nstride,
                       in_cstride,
                       params.dx,
                       params.bnScale,
                       params.resultBnScaleDiff,
                       params.resultBnBiasDiff,
                       params.epsilon);
            }
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
