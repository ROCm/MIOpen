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

#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/names.hpp>

#include <cmath>
#include <sstream>

#define WORKAROUND_SWDEV_253606 1

namespace miopen {

namespace batchnorm {

bool is_fp16_or_bfp16(miopenDataType_t type)
{
    return ((type == miopenHalf) || (type == miopenBFloat16));
}

bool is_fp32_or_fp64(miopenDataType_t type)
{
    return ((type == miopenFloat) || (type == miopenDouble));
}

bool is_fp32(miopenDataType_t type) { return (type == miopenFloat); }

bool IsOCLInferTypeValid(const ProblemDescription& bn_problem)
{
    // case 1 : mix type
    return (
        (is_fp16_or_bfp16(bn_problem.GetXDesc().GetType()) &&
         is_fp16_or_bfp16(bn_problem.GetYDesc().GetType()) &&
         is_fp32(bn_problem.GetBnScale().GetType()) && is_fp32(bn_problem.GetBnBias().GetType())) ||
        // case 2 : float type
        (is_fp32(bn_problem.GetXDesc().GetType()) && is_fp32(bn_problem.GetYDesc().GetType()) &&
         is_fp32(bn_problem.GetBnScale().GetType()) && is_fp32(bn_problem.GetBnBias().GetType())));
}

bool IsCKInferTypeValid(const ProblemDescription& bn_problem)
{
    // case 1 : mix type
    return ((is_fp16_or_bfp16(bn_problem.GetXDesc().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetYDesc().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetBnScale().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetBnBias().GetType()) &&
             is_fp32(bn_problem.GetBnSMean().GetType()) &&
             is_fp32(bn_problem.GetBnSVar().GetType())) ||
            // case 2 : fp32 or fp64
            (is_fp32_or_fp64(bn_problem.GetXDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetYDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnScale().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnBias().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSMean().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSVar().GetType())));
}

bool IsOCLFwdTrainTypeValid(const ProblemDescription& bn_problem)
{
    // case 1 : mix type
    return (
        (is_fp16_or_bfp16(bn_problem.GetXDesc().GetType()) &&
         is_fp16_or_bfp16(bn_problem.GetYDesc().GetType()) &&
         is_fp32(bn_problem.GetBnScale().GetType()) && is_fp32(bn_problem.GetBnBias().GetType())) ||
        // case 2 : float type
        (is_fp32(bn_problem.GetXDesc().GetType()) && is_fp32(bn_problem.GetYDesc().GetType()) &&
         is_fp32(bn_problem.GetBnScale().GetType()) && is_fp32(bn_problem.GetBnBias().GetType())));
}

bool IsCKFwdTrainTypeValid(const ProblemDescription& bn_problem)
{
    // case 1 : mix type
    return ((is_fp16_or_bfp16(bn_problem.GetXDesc().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetYDesc().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetBnScale().GetType()) &&
             is_fp16_or_bfp16(bn_problem.GetBnBias().GetType()) &&
             is_fp32(bn_problem.GetBnSMean().GetType()) &&
             is_fp32(bn_problem.GetBnSVar().GetType())) ||
            // case 2 : fp32 or fp64
            (is_fp32_or_fp64(bn_problem.GetXDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetYDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnScale().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnBias().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSMean().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSVar().GetType())));
}

bool IsOCLBwdTypeValid(const ProblemDescription& bn_problem)
{
    return (
        (is_fp16_or_bfp16(bn_problem.GetXDesc().GetType()) &&
         is_fp16_or_bfp16(bn_problem.GetDXDesc().GetType()) &&
         is_fp16_or_bfp16(bn_problem.GetDYDesc().GetType()) &&
         is_fp32(bn_problem.GetBnScale().GetType()) && is_fp32(bn_problem.GetBnSMean().GetType()) &&
         is_fp32(bn_problem.GetBnSVar().GetType())) ||
        // case 1 : fp32
        (is_fp32(bn_problem.GetXDesc().GetType()) && is_fp32(bn_problem.GetDXDesc().GetType()) &&
         is_fp32(bn_problem.GetBnScale().GetType()) && is_fp32(bn_problem.GetBnBias().GetType()) &&
         is_fp32(bn_problem.GetBnSMean().GetType()) && is_fp32(bn_problem.GetBnSVar().GetType())));
}

bool IsCKBwdTypeValid(const ProblemDescription& bn_problem)
{
    return ((is_fp16_or_bfp16(bn_problem.GetXDesc().GetType()) &&
             bn_problem.GetDXDesc().GetType() == miopenFloat &&
             is_fp16_or_bfp16(bn_problem.GetBnScale().GetType()) &&
             bn_problem.GetDYDesc().GetType() == miopenFloat &&
             bn_problem.GetBnSMean().GetType() == miopenFloat &&
             bn_problem.GetBnSVar().GetType() == miopenFloat) ||
            // case 1 : fp32 or fp64
            (is_fp32_or_fp64(bn_problem.GetXDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetDXDesc().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnScale().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnBias().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSMean().GetType()) &&
             is_fp32_or_fp64(bn_problem.GetBnSVar().GetType())));
}

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    switch(direction)
    {
    case Direction::ForwardTraining: return MakeForwardTrainingNetworkConfig();
    case Direction::ForwardInference: return MakeForwardInferenceNetworkConfig();
    case Direction::Backward: return MakeBackwardNetworkConfig();
    default: MIOPEN_THROW(miopenStatusInternalError);
    }
}

NetworkConfig ProblemDescription::MakeForwardTrainingNetworkConfig() const
{
    std::ostringstream ss;

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    const unsigned int in_cstride = h * w;
    const unsigned int in_nhw     = n * in_cstride;

    size_t xlocalsize = 1024;
    if(((in_cstride < 256) && (n < 256)) || ((in_cstride < 100) && (n <= 256)))
        xlocalsize = 256;

    size_t ylocalsize = 1;

    size_t xgridsize = c * xlocalsize;
    size_t ygridsize = 1;

    bool bfpmixparm = false;
    if(IsMix())
    {
        bfpmixparm = true;
    }

    if(bn_mode == miopenBNSpatial)
    {
        bool single         = true;
        int variant         = 1;
        unsigned int ldsgcn = xlocalsize / 64;

#if(WORKAROUND_SWDEV_253606 == 0)
        if(n < 3)
        {
            variant    = 4;
            xlocalsize = 256;
            xgridsize  = c * xlocalsize;
            ylocalsize = 1;
            ygridsize  = 1;
            ldsgcn     = xlocalsize / 64;
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
            const auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize    = c;
            ygridsize    = segment * ylocalsize;
            single       = false;
            ldsgcn       = ylocalsize / 64;
        }
        // clang-format on

        if((n > 768) && (in_cstride > 150) && IsFp32())
        {
            variant            = 2;
            xlocalsize         = 1;
            ylocalsize         = 1024;
            const auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize          = c;
            ygridsize          = segment * ylocalsize;
            single             = false;
            ldsgcn             = ylocalsize / 64;
        }

        ss << "variant" << variant;

#if(WORKAROUND_SWDEV_253606 == 0)
        if(variant == 4)
        {
            ss << "rs" << static_cast<int>(resultsave);
            ss << "rr" << static_cast<int>(resultrunning);
            ss << "fp16" << static_cast<int>(IsFp16());
            ss << "fp32" << static_cast<int>(IsFp32());
            ss << "fp64" << static_cast<int>(IsFp64());
            ss << "fbf16" << static_cast<int>(IsBFp16());
            ss << "fmix" << static_cast<int>(IsMix());
            ss << "c" << c;
        }
        else
#endif
        {
            ss << "gx" << xgridsize;
            ss << "gy" << ygridsize;
            ss << "xl" << xlocalsize;
            ss << "yl" << ylocalsize;
            ss << "ldsgcn" << ldsgcn;
            ss << "rs" << static_cast<int>(resultsave);
            ss << "rr" << static_cast<int>(resultrunning);
            ss << "fp16" << static_cast<int>(IsFp16());
            ss << "fp32" << static_cast<int>(IsFp32());
            ss << "fp64" << static_cast<int>(IsFp64());
            ss << "fbf16" << static_cast<int>(IsBFp16());
            ss << "fmix" << static_cast<int>(IsMix());
            ss << "single" << static_cast<int>(single);
            ss << "n" << n;
            ss << "c" << c;
            ss << "hw" << in_cstride;
        }
    }
    else
    {
        xlocalsize                = 1;
        ylocalsize                = 256;
        const std::size_t segment = (in_cstride + ylocalsize - 1) / ylocalsize;
        xgridsize                 = c;
        ygridsize                 = segment * ylocalsize;

        ss << "fp16" << static_cast<int>(IsFp16());
        ss << "fp32" << static_cast<int>(IsFp32());
        ss << "fp64" << static_cast<int>(IsFp64());
        ss << "fbf16" << static_cast<int>(IsBFp16());
        ss << "fmix" << static_cast<int>(IsMix());
        ss << "gx" << xgridsize;
        ss << "gy" << ygridsize;
        ss << "lx" << xlocalsize;
        ss << "ly" << ylocalsize;
        ss << "rs" << static_cast<int>(resultsave);
        ss << "rr" << static_cast<int>(resultrunning);
        ss << "segment" << segment;
        ss << "n" << n;
        ss << "c" << c;
        ss << "hw" << in_cstride;
    }
    ss << "layout" << in_layout;
    ss << "scaleType" << static_cast<int>(IsScaleFp16());
    ss << "scaleType" << static_cast<int>(IsScaleFp32());

    return NetworkConfig{ss.str()};
}

NetworkConfig ProblemDescription::MakeForwardInferenceNetworkConfig() const
{
    std::ostringstream ss;

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    const unsigned int in_cstride = h * w;

    ss << "fp16" << static_cast<int>(IsFp16());
    ss << "fp32" << static_cast<int>(IsFp32());
    ss << "fp64" << static_cast<int>(IsFp64());
    ss << "fbf16" << static_cast<int>(IsBFp16());
    ss << "fmix" << static_cast<int>(IsMix());
    ss << "mode" << bn_mode;
    ss << "HWdims" << in_cstride;
    ss << "C" << c;
    ss << "layout" << in_layout;
    ss << "scaleType" << static_cast<int>(IsScaleFp16());
    ss << "scaleType" << static_cast<int>(IsScaleFp32());

    return NetworkConfig{ss.str()};
}

NetworkConfig ProblemDescription::MakeBackwardNetworkConfig() const
{
    std::ostringstream ss;

    bool bfpmixparm = false;
    if(xDesc.GetType() == miopenHalf && GetBnScale().GetType() == miopenFloat)
    {
        bfpmixparm = true;
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    const unsigned int in_cstride = h * w;
    const unsigned int in_nhw     = n * in_cstride;

    size_t xlocalsize = 1;
    size_t ylocalsize = 1;

    size_t xgridsize = 1;
    size_t ygridsize = 1;

    if(bn_mode == miopenBNSpatial)
    {
        unsigned int ldsgcn = 0;
        bool single         = true;
        int variant         = 1;

        if((in_nhw < (32 * 1024 * 1024) && in_cstride > 1024))
        {
            variant    = 1;
            xlocalsize = 1024;
            xgridsize  = c * xlocalsize;
            ldsgcn     = xlocalsize / 64;
        }
        else if(in_nhw < (32 * 1024 * 1024) && in_cstride > 512)
        {
            variant    = (n >= 32) ? 1 : 3;
            xlocalsize = std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned int>(1024));
            xgridsize  = c * xlocalsize;
            ldsgcn     = xlocalsize / 64;
        }
        else if(in_cstride <= 512)
        {
            if((n > 64) && (in_cstride > 160))
            {
                variant = 3;
                xlocalsize =
                    std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned int>(1024));
                xgridsize = c * xlocalsize;
                ldsgcn    = xlocalsize / 64;
            }
            else
            {
                variant = 0;
                if(IsFp32())
                {
                    xlocalsize = 1024;
                    xgridsize  = 1024 * static_cast<size_t>(c);
                }
                else
                {
                    xlocalsize = 256;
                    xgridsize  = 256 * static_cast<size_t>(c);
                }
                ldsgcn = xlocalsize / 64;
            }
        }
        else
        {
            variant      = 2;
            ylocalsize   = 1024;
            auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize    = c;
            ygridsize    = segment * ylocalsize;
            single       = false;
            ldsgcn       = ylocalsize / 64;
        }
        if((in_cstride < 200) && (in_cstride > 60) && bfpmixparm)
        {
            variant    = 1;
            xlocalsize = 1024;
            xgridsize  = c * xlocalsize;
            ldsgcn     = xlocalsize / 64;
        }

        ss << "variant" << variant;
        ss << "gx" << xgridsize;
        ss << "n" << n;
        ss << "c" << c;
        ss << "hw" << in_cstride;
        ss << "gy" << ygridsize;
        ss << "lx" << xlocalsize;
        ss << "ly" << ylocalsize;
        ss << "us" << static_cast<int>(useSaved);
        ss << "fp16" << static_cast<int>(IsFp16());
        ss << "fp32" << static_cast<int>(IsFp32());
        ss << "fp64" << static_cast<int>(IsFp64());
        ss << "fbf16" << static_cast<int>(IsBFp16());
        ss << "fmix" << static_cast<int>(IsMix());
        ss << "single" << static_cast<int>(single);
        ss << "gcn" << ldsgcn;
    }
    else
    {
        ylocalsize                 = (64 >= in_cstride) ? 64 : 256;
        const unsigned int segment = std::ceil(double(in_cstride) / double(ylocalsize));
        xgridsize                  = c;
        ygridsize                  = segment * ylocalsize;

        ss << "gx" << xgridsize;
        ss << "gy" << ygridsize;
        ss << "lx" << xlocalsize;
        ss << "ly" << ylocalsize;
        ss << "n" << n;
        ss << "c" << c;
        ss << "hw" << in_cstride;
        ss << "u" << static_cast<int>(useSaved);
        ss << "fp16" << static_cast<int>(IsFp16());
        ss << "fp32" << static_cast<int>(IsFp32());
        ss << "fp64" << static_cast<int>(IsFp64());
        ss << "fbf16" << static_cast<int>(IsBFp16());
        ss << "fmix" << static_cast<int>(IsMix());
        ss << "nhw" << in_nhw;
    }
    ss << "layout" << in_layout;
    ss << "scaleType" << static_cast<int>(IsScaleFp16());
    ss << "scaleType" << static_cast<int>(IsScaleFp32());

    return NetworkConfig{ss.str()};
}

} // namespace batchnorm

} // namespace miopen
