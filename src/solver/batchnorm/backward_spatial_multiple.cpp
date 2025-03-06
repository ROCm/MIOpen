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

bool BnBwdTrainingSpatialMultiple::IsApplicable(
    const ExecutionContext& context, const miopen::batchnorm::ProblemDescription& problem) const
{
    if(problem.GetDirection() != miopen::batchnorm::Direction::Backward ||
       problem.GetMode() != miopenBNSpatial)
        return false;
    if(!problem.Is2D())
    {
        return false;
    }
    if(!IsOCLBwdTypeValid(problem))
        return false;

#if WORKAROUND_ISSUE_1549_FP16_BUILD_ERROR
    if(problem.GetXDesc().GetType() == miopenHalf && problem.GetBnScale().GetType() == miopenHalf)
    {
        // bfp16parm = true;
        // Unsupported kernel mode, error in kernel code
        // MIOpenBatchNormBwdSpatial.cl:526 issue#1549
        return false;
    }
#endif

    return !BnBwdTrainingSpatialSingle{}.IsApplicable(context, problem);
}

ConvSolution BnBwdTrainingSpatialMultiple::GetSolution(
    const ExecutionContext& context, const miopen::batchnorm::ProblemDescription& problem) const
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

    auto inhw = float(1.0 / in_nhw);

    size_t xlocalsize = 1;
    size_t ylocalsize = 1;

    size_t xgridsize = 1;
    size_t ygridsize = 1;

    unsigned int ldsgcn   = 0;
    unsigned int ldsnogcn = 0;
    int variant           = 1;

    if(problem.IsLayoutNHWC())
    {
        xlocalsize = 1024;
        xgridsize  = c * xlocalsize;
        ldsgcn     = xlocalsize / 64;
        ldsnogcn   = xlocalsize;
    }
    else
    {
        //*************************************************************************************************
        // N*H*W < 32M and H*W > 1024, use batchnorm variant#1 implementation which parallelize
        // work groups over channels and loop through NHW.
        //*************************************************************************************************
        if((in_nhw < (32 * 1024 * 1024) && in_cstride > 1024))
        {
            variant    = 1;
            xlocalsize = 1024;
            xgridsize  = c * xlocalsize;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }
        //*************************************************************************************************
        // N*H*W < 32M and H*W > 512  use batchnorm variant#1 or variant#3 implementation which
        // parallelize
        // work groups over channels and loop through N.
        //*************************************************************************************************
        else if(in_nhw < (32 * 1024 * 1024) && in_cstride > 512)
        {
            variant    = (n >= 32) ? 1 : 3;
            xlocalsize = std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned int>(1024));
            xgridsize  = c * xlocalsize;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }
        //*************************************************************************************************
        // H*W < 512  use batchnorm variant#0 or variant#3 implementation based on batch size and
        // H*W
        //*************************************************************************************************
        else if(in_cstride <= 512)
        {
            if((n > 64) && (in_cstride > 160))
            {
                variant = 3;
                xlocalsize =
                    std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned int>(1024));
                xgridsize = c * xlocalsize;
                ldsgcn    = xlocalsize / 64;
                ldsnogcn  = xlocalsize;
            }
            else
            {
                variant = 0;
                if(bfp32parm)
                {
                    xlocalsize = 1024;
                    xgridsize  = static_cast<size_t>(1024) * c;
                }
                else
                {
                    xlocalsize = 256;
                    xgridsize  = static_cast<size_t>(256) * c;
                }
                ldsgcn   = xlocalsize / 64;
                ldsnogcn = xlocalsize;
            }
        }
        //*************************************************************************************************
        // N*H*W > 32M, use batchnorm variant#2 implementation which parallelize
        // work groups over channels and data segments.
        //*************************************************************************************************
        else
        {
            variant      = 2;
            ylocalsize   = 1024;
            auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize    = c;
            ygridsize    = segment * ylocalsize;
            ldsgcn       = ylocalsize / 64;
            ldsnogcn     = ylocalsize;
        }
        if((in_cstride < 200) && (in_cstride > 60) && bfpmixparm)
        {
            variant    = 1;
            xlocalsize = 1024;
            xgridsize  = c * xlocalsize;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }
    }

    auto result = ConvSolution{miopenStatusSuccess};

    {
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenBatchNormBwdSpatial.cl";
        kernel.kernel_name = "MIOpenBatchNormBwdSpatial";

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
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

        if(problem.UseSaved())
        {
            auto copy = kernel;

            copy.kernel_name = kernel.kernel_name + "DScaleDBias";
            result.construction_params.push_back(copy);

            copy.kernel_name = kernel.kernel_name + "FinalDScaleDBias";
            result.construction_params.push_back(copy);

            copy.kernel_name = kernel.kernel_name + "DX";
            result.construction_params.push_back(copy);
        }
        else
        {
            auto copy = kernel;

            copy.kernel_name = kernel.kernel_name + "MeanVariance";
            result.construction_params.push_back(copy);

            copy.kernel_name = kernel.kernel_name + "FinalMeanVariance";
            result.construction_params.push_back(copy);

            copy.kernel_name = kernel.kernel_name + "DScaleDBias";
            result.construction_params.push_back(copy);

            copy.kernel_name = kernel.kernel_name + "FinalDScaleDBias";
            result.construction_params.push_back(copy);

            copy.kernel_name = kernel.kernel_name + "DX";
            result.construction_params.push_back(copy);
        }
    }

    const auto dtype    = problem.GetBnScale().GetType();
    const auto useSaved = problem.UseSaved();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::BwdInvokeParams>();

            float ctime = 0.;
            visit_float(dtype, [&](auto as_float) {
                if(useSaved)
                {
                    handle_.Run(kernels[0])(
                        params.x, params.dy, params.dx, params.savedMean, params.savedInvVariance);
                    profileSequence(handle_, 0, &ctime);

                    handle_.Run(kernels[1])(
                        params.dx, params.resultBnScaleDiff, params.resultBnBiasDiff);
                    profileSequence(handle_, 1, &ctime);

                    handle_.Run(kernels[2])(params.x,
                                            params.dy,
                                            params.dx,
                                            params.bnScale,
                                            params.resultBnScaleDiff,
                                            params.resultBnBiasDiff,
                                            params.savedMean,
                                            params.savedInvVariance,
                                            as_float(inhw));
                    profileSequence(handle_, 2, &ctime);
                }
                else
                {
                    handle_.Run(kernels[0])(params.x, params.dx); // mean variance
                    profileSequence(handle_, 0, &ctime);

                    handle_.Run(kernels[1])(
                        params.dx, as_float(inhw), params.epsilon); // final mean variance
                    profileSequence(handle_, 1, &ctime);

                    handle_.Run(kernels[2])(params.x, params.dy, params.dx); // dscale dbias
                    profileSequence(handle_, 1, &ctime);

                    handle_.Run(kernels[3])(params.dx,
                                            params.resultBnScaleDiff,
                                            params.resultBnBiasDiff); // final dscale dbias
                    profileSequence(handle_, 1, &ctime);

                    handle_.Run(kernels[4])(params.x,
                                            params.dy,
                                            params.dx,
                                            params.bnScale,
                                            params.resultBnScaleDiff,
                                            params.resultBnBiasDiff,
                                            as_float(inhw)); // dx
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
