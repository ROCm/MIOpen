/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/batchnorm/problem_description.hpp>

#define WORKAROUND_SWDEV_253606 1

namespace miopen {

namespace solver {

namespace batchnorm {

inline void GetWGSizeNHWC(size_t c,
                          size_t h,
                          size_t w,
                          size_t min_workgroups,
                          bool bfp32parm,
                          size_t vectorsize,
                          size_t& xlocalsize,
                          size_t& ylocalsize)
{
    unsigned int xlocalsize_limit = vectorsize > 1 ? (bfp32parm ? 16 : 32) : 64;
    // shared memory size per workgroup is fixed
    unsigned int max_localsize = 1024 / vectorsize;

    size_t nworkgroups = 0;
    xlocalsize         = 0;
    // decrease max_localsize until the number of workgroups is greater than 80%
    // of the available CUs
    while(nworkgroups < min_workgroups && max_localsize >= xlocalsize_limit)
    {
        // xlocalsize must be power of 2 as reductions in the kernels rely on it, here c is rounded
        // up to next power of 2.
        xlocalsize  = std::min(size_t{1 << int(std::ceil(std::log2(c / vectorsize)))},
                              size_t{xlocalsize_limit});
        ylocalsize  = max_localsize / xlocalsize;
        nworkgroups = ((c / vectorsize + xlocalsize - 1) / xlocalsize) *
                      ((h * w + ylocalsize - 1) / ylocalsize);
        max_localsize >>= 1;
    }
}

inline int GetStashMethod(bool IsLayoutNHWC,
                          miopenDataType_t problem_type,
                          unsigned int stash_values,
                          size_t c,
                          size_t n,
                          size_t in_cstride,
                          unsigned int ylocalsize)
{
    // See `batchnorm_functions.hpp` for stash implementation of different methods
    int stash_method = 0;
    stash_values *= (problem_type == miopenFloat ? 1 : 2);
    unsigned int last_ylocalsize =
        (in_cstride) % ylocalsize == 0 ? ylocalsize : (in_cstride) % ylocalsize;
    if(last_ylocalsize < stash_values && n >= (size_t)stash_values)
    {
        stash_method = 1;
    }
    if(IsLayoutNHWC && !(problem_type == miopenFloat) && (c % 2 != 0) && (n >= stash_values))
    {
        stash_method = 2;
    }
    return stash_method;
}

// Returns true if spatial multiple is applicable and fill NHWC configuration
// (xlocalsize, ylocalsize).
// First workgroup size is computed given a problem and vectorsize, then it checks
// if the computed workgroup is applicable (spatial multiple restrictions)
inline bool GetLocalConfigNHWC(const miopen::batchnorm::ProblemDescription& problem,
                               unsigned int stash_values,
                               size_t vectorsize,
                               size_t& xlocalsize,
                               size_t& ylocalsize)
{
    bool bfp32parm =
        problem.GetXDesc().GetType() == miopenHalf || problem.GetXDesc().GetType() == miopenBFloat16
            ? false
            : true;

    size_t n, c, h, w = 0;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());
    assert((n != 0) && "n cannot be 0");
    assert((c != 0) && "c cannot be 0");
    assert((h != 0) && "h cannot be 0");
    assert((w != 0) && "w cannot be 0");

    GetWGSizeNHWC(
        c, h, w, problem.GetMinWorkgroups(), bfp32parm, vectorsize, xlocalsize, ylocalsize);
    assert((xlocalsize != 0) && "xlocalsize cannot be 0");
    assert((ylocalsize != 0) && "ylocalsize cannot be 0");
    if(ylocalsize == 0)
    {
        ylocalsize = 1;
    }
    stash_values *= (bfp32parm ? 1 : 2);
    unsigned int last_ylocalsize = (h * w) % ylocalsize == 0 ? ylocalsize : (h * w) % ylocalsize;
    // FP32:
    //  - last block must have enough space to stash intermediate results in HW dimension
    //  - if last block doesn't fit, intermediate results are stored in N dimension which must
    //    be large enough
    // Mix precision:
    //  - last block must have enough space to stash intermediate results in HW dimension
    //  - if last block doesn't fit, intermediate results are stored in N dimension which must
    //    be large enough
    //  - if C is not multiple of 2, intermediate results are stored in N dimension splitting
    //    float values in group of 2 bytes. N must be large enough
    if((!bfp32parm && (c % 2 != 0 && n < (size_t)stash_values)) ||
       ((last_ylocalsize < stash_values) && (n < (size_t)stash_values)))
    {
        return false;
    }

    return true;
}

inline bool IsSpatialMultipleApplicable(const miopen::batchnorm::ProblemDescription& problem,
                                        size_t vectorsize,
                                        unsigned int stash_values)
{
    int n, c, h, w = 0;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());
    assert((n != 0) && "n cannot be 0");
    assert((c != 0) && "c cannot be 0");
    assert((h != 0) && "h cannot be 0");
    assert((w != 0) && "w cannot be 0");

    unsigned int in_cstride = h * w;

    if(problem.IsLayoutNHWC())
    {
        // check if the provided vectorsize can be used
        if(c % vectorsize != 0)
        {
            return false;
        }
        // Variant 2 is the primary choice for NHWC
        size_t xlocalsize, ylocalsize = 0;

        // The configuration is ignored at this point, it was just computed to check
        // if spatial multiple could be applied.
        return GetLocalConfigNHWC(problem, stash_values, vectorsize, xlocalsize, ylocalsize);
    }
    else
    {
        // check if the provided vectorsize can be used
        if(in_cstride % vectorsize != 0)
        {
            return false;
        }

        unsigned int ylocalsize = 1024;
        unsigned int last_ylocalsize =
            in_cstride % ylocalsize == 0 ? ylocalsize : in_cstride % ylocalsize;
        // Restrictions:
        //  - last block must have enough space to stash intermediate results in HW dimension
        //  - if last block doesn't fit, intermediate results are stored in N dimension which must
        //    be large enough
        stash_values *= (problem.GetXDesc().GetType() == miopenFloat ? 1 : 2);
        if(last_ylocalsize < stash_values && n < (size_t)stash_values)
        {
            return false;
        }
    }
    return true;
}

inline void GetSpatialMultipleConfig(const miopen::batchnorm::ProblemDescription& problem,
                                     unsigned int stash_values,
                                     size_t vectorsize,
                                     size_t& xlocalsize,
                                     size_t& ylocalsize,
                                     size_t& xgridsize,
                                     size_t& ygridsize,
                                     int& stash_method)
{
    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    if(problem.IsLayoutNHWC())
    {
        // The function returns if the method is valid but we can ignore it
        // at this point
        GetLocalConfigNHWC(problem, stash_values, vectorsize, xlocalsize, ylocalsize);

        xgridsize = xlocalsize * ((c / vectorsize + xlocalsize - 1) / xlocalsize);
        ygridsize = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
    }
    else
    {
        xlocalsize = 1;
        xgridsize  = c;
        ylocalsize = 1024;
        if(ylocalsize > in_cstride / vectorsize)
        {
            // No need to use workgroups larger than the HW dimension
            ylocalsize = std::max(size_t{64},
                                  size_t{1 << int(std::ceil(std::log2(in_cstride / vectorsize)))});
        }
        ygridsize = ylocalsize * ((in_cstride / vectorsize + ylocalsize - 1) / ylocalsize);
    }
    stash_method = GetStashMethod(problem.IsLayoutNHWC(),
                                  problem.GetXDesc().GetType(),
                                  stash_values,
                                  c,
                                  n,
                                  in_cstride,
                                  ylocalsize);
}

inline void GetVariantFromKernelId(const std::string& kernel_id, int& variant, size_t& vectorsize)
{
    // kernel_id has the following standard:
    // Variant<variant>-<vectorsize>
    size_t pos = kernel_id.find("Variant");
    if(pos != std::string::npos)
    {
        variant    = kernel_id[pos + 7] - '0';
        vectorsize = kernel_id[pos + 9] - '0';
    }
}

inline std::string GetKernelIdFromVariant(int variant, size_t vectorsize)
{
    std::stringstream stream;
    stream << "Variant" << variant << "-" << vectorsize;
    return stream.str();
}

inline bool UseMultiple(const miopen::batchnorm::ProblemDescription& problem)
{
    size_t n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    bool bfpmixparm = (problem.GetXDesc().GetType() == miopenHalf ||
                       problem.GetXDesc().GetType() == miopenBFloat16) &&
                              problem.GetBnScale().GetType() == miopenFloat
                          ? true
                          : false;

    unsigned int in_cstride = h * w;
    unsigned int in_nhw     = n * in_cstride;
    // Check heuristics (used to choose between spatial single and multiple for performance)
    // TODO: review these conditions (variant 2 was optimized and vectorization was added,
    // so we need a set of benchmarks to check that these conditions are still correct)
    if(!problem.IsLayoutNHWC() &&
       problem.GetDirection() == miopen::batchnorm::Direction::Backward &&
       (!((in_nhw >= static_cast<size_t>(32 * 1024 * 1024) || in_cstride <= 1024) &&
          (in_nhw >= static_cast<size_t>(32 * 1024 * 1024) || in_cstride <= 512) &&
          in_cstride > 512)))
    {
        return false;
    }

    if(!problem.IsLayoutNHWC() &&
       problem.GetDirection() == miopen::batchnorm::Direction::ForwardTraining &&
       (!((n >= 3 && in_cstride > 512 && (in_nhw >= 33554432 || in_cstride <= 1024) &&
           ((n < 256) || (in_cstride <= 60) || !bfpmixparm) &&
           (!bfpmixparm || in_cstride <= 512)) ||
          ((n > 768) && (in_cstride > 150)))))
    {
        return false;
    }

    return true;
}

inline void DefaultConfigSpatialSingle(const miopen::batchnorm::ProblemDescription& problem,
                                       std::vector<std::string>& valid_kernels)
{
    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nhw     = n * in_cstride;

    bool bfpmixparm =
        problem.GetXDesc().GetType() == miopenHalf && problem.GetBnScale().GetType() == miopenFloat
            ? true
            : false;

    bool bbfpmixparam = problem.GetXDesc().GetType() == miopenBFloat16 &&
                                problem.GetBnScale().GetType() == miopenFloat
                            ? true
                            : false;

    // NCHW supports also variants 0 and 3 which can be much faster than
    // variant 1 but have more restrictions. Here we decide if we use variant
    // 0, 1, 3
    // In case variant 0 or 3 are selected, we add also variant 1 for tuning.
    // Almost always variant 0 and 3 will be faster than variant 1 but
    // we add the latter for tuning to be sure and because it is cheap
    if(!problem.IsLayoutNHWC())
    {
        if(problem.GetDirection() == miopen::batchnorm::Direction::Backward)
        {
            if((in_cstride < 200) && (in_cstride > 60) && bfpmixparm)
            {
                valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                return;
            }

            // N*H*W < 32M and H*W > 1024
            // use batchnorm variant#1 implementation which parallelize
            // work groups over channels and loop through NHW.
            if((in_nhw < (32 * 1024 * 1024) && in_cstride > 1024))
            {
                valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                return;
            }
            // N*H*W < 32M and H*W > 512
            // use batchnorm variant#1 or variant#3 implementation which
            // parallelize work groups over channels and loop through N.
            else if(in_nhw < (32 * 1024 * 1024) && in_cstride > 512)
            {
                if(n >= 32)
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }
                else
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(3, 1));
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }
            }
            // H*W < 512  use batchnorm variant#0 or variant#3 implementation
            // based on batch size and H*W
            else if(in_cstride <= 512)
            {
                if((n > 64) && (in_cstride > 160))
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(3, 1));
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }
                else
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(0, 1));
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }
            }
        }
        else
        {
#if(WORKAROUND_SWDEV_253606 == 0)
            if(n < 3)
            {
                valid_kernels.push_back(GetKernelIdFromVariant(4, 1));
                valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                return;
            }
            else
#endif
            {
                // clang-format off
                if(in_cstride > 512 && in_cstride <= 1024 && n < 32)
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(3, 1));
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }

                if( (in_nhw < 33554432 && in_cstride > 1024) ||
                ((n >= 256) && (in_cstride > 60) && (bfpmixparm || bbfpmixparam)) ||
                ((in_cstride > 512) && (bfpmixparm || bbfpmixparam)))
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    if(in_cstride <= 512)
                    {
                        valid_kernels.push_back(GetKernelIdFromVariant(0, 1));
                    }
                    return;
                }
                else if(in_cstride <= 512)
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(0, 1));
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }
                // clang-format on
            }
        }
        valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
    }
}

inline void DefaultConfigSpatialMultiple(const miopen::batchnorm::ProblemDescription& problem,
                                         unsigned int stash_values,
                                         std::vector<std::string>& valid_kernels)
{
    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    // Default configuration for spatial multiple tries to use vectorization
    // for both NCHW or NHWC
    size_t vectorsize =
        problem.IsLayoutNHWC() ? (c % 4 == 0 ? 4 : 1) : (in_cstride % 4 == 0 ? 4 : 1);
    if(IsSpatialMultipleApplicable(problem, vectorsize, stash_values))
    {
        valid_kernels.push_back(GetKernelIdFromVariant(2, vectorsize));
        // if vectorized version is applicable, then the non vectorized version
        // is also added to the list of configurations
        if(vectorsize > 1)
        {
            valid_kernels.push_back(GetKernelIdFromVariant(2, 1));
        }
        return;
    }

    // If spatial multiple with vectorization can not be used, try without vectorization
    if(vectorsize > 1 && IsSpatialMultipleApplicable(problem, 1, stash_values))
    {
        valid_kernels.push_back(GetKernelIdFromVariant(2, 1));
    }
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
