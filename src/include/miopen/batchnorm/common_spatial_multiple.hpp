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

namespace miopen {

namespace solver {

namespace batchnorm {

inline void GetWGSizeNHWC(size_t c,
                          size_t h,
                          size_t w,
                          size_t maxCUs,
                          bool bfp32parm,
                          size_t vectorsize,
                          size_t& xlocalsize,
                          size_t& ylocalsize)
{
    unsigned int xlocalsize_limit = vectorsize > 1 ? (bfp32parm ? 16 : 32) : 64;
    // shared memory size per workgroup is fixed
    unsigned int max_localsize = 1024 / vectorsize;

    size_t nworkgroups = 0;
    // decrease max_localsize until the number of workgroups is greater than 80%
    // of the available CUs
    while((float)nworkgroups < 0.8f * maxCUs && max_localsize >= xlocalsize_limit)
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
                               size_t maxCUs,
                               unsigned int stash_values,
                               size_t vectorsize,
                               size_t& xlocalsize,
                               size_t& ylocalsize)
{
    bool bfp32parm =
        problem.GetXDesc().GetType() == miopenHalf || problem.GetXDesc().GetType() == miopenBFloat16
            ? false
            : true;

    size_t n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    GetWGSizeNHWC(c, h, w, maxCUs, bfp32parm, vectorsize, xlocalsize, ylocalsize);

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

// Returns true if spatial multiple is applicable and fill NHWC configuration
// (xlocalsize, ylocalsize, vectorsize).
// Internally, it tries to use vectorization if possible. If vectorization can be
// used but spatial multiple is not applicable, it tries to see if spatial multiple
// is applicable without vectorization as a fallback.
inline bool GetConfigNHWC(const miopen::batchnorm::ProblemDescription& problem,
                          size_t maxCUs,
                          unsigned int stash_values,
                          size_t& xlocalsize,
                          size_t& ylocalsize,
                          size_t& vectorsize)
{
    size_t c = problem.GetXDesc().GetLengths()[1];
    // Apply vectorization if possible, given the size of C
    vectorsize = c % 4 == 0 ? 4 : 1;

    // Check if variant 2 is applicable (with possible vectorization)
    bool valid =
        GetLocalConfigNHWC(problem, maxCUs, stash_values, vectorsize, xlocalsize, ylocalsize);

    // If vectorization is used but variant 2 is not applicable,
    // check if it's applicable without vectorization
    if(!valid && vectorsize > 1)
    {
        vectorsize = 1;
        valid =
            GetLocalConfigNHWC(problem, maxCUs, stash_values, vectorsize, xlocalsize, ylocalsize);
    }
    return valid;
}

inline void GetSpatialMultipleConfig(const miopen::batchnorm::ProblemDescription& problem,
                                     size_t maxCUs,
                                     unsigned int stash_values,
                                     size_t& xlocalsize,
                                     size_t& ylocalsize,
                                     size_t& xgridsize,
                                     size_t& ygridsize,
                                     size_t& vectorsize,
                                     int& stash_method)
{
    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    if(problem.IsLayoutNHWC())
    {
        // The function returns if the method is valid but we can ignore it
        // at this point
        GetConfigNHWC(problem, maxCUs, stash_values, xlocalsize, ylocalsize, vectorsize);

        xgridsize = xlocalsize * ((c / vectorsize + xlocalsize - 1) / xlocalsize);
        ygridsize = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
    }
    else
    {
        vectorsize = in_cstride % 4 == 0 ? 4 : 1;
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

inline bool IsSpatialMultipleApplicable(const miopen::batchnorm::ProblemDescription& problem,
                                        size_t maxCUs,
                                        unsigned int stash_values)
{
    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    if(problem.IsLayoutNHWC())
    {
        // Variant 2 is the primary choice for NHWC
        size_t xlocalsize, ylocalsize, vectorsize;

        // The configuration is ignored at this point, it was just computed to check
        // if spatial multiple could be applied.
        return GetConfigNHWC(problem, maxCUs, stash_values, xlocalsize, ylocalsize, vectorsize);
    }
    else
    {
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

} // namespace batchnorm

} // namespace solver

} // namespace miopen
