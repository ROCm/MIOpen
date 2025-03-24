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

// Compute workgroup size configuration given a problem (NHWC) and a vectorsize
// It supports only 2D workgroups
inline void GetLocalConfigNHWC(const miopen::batchnorm::ProblemDescription& problem,
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

    // Compute workgroup size
    unsigned int xlocalsize_limit = vectorsize > 1 ? (bfp32parm ? 16 : 32) : 64;
    // shared memory size per workgroup is fixed
    unsigned int max_localsize = 1024 / vectorsize;

    size_t nworkgroups = 0;
    // decrease max_localsize until the number of workgroups is greater than 80%
    // of the available CUs
    while(nworkgroups < problem.GetMinWorkgroups() && max_localsize >= xlocalsize_limit &&
          max_localsize > 64)
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

// Provide workgroup sizes for spatial multiple configuration.
// It returns the preferred spatial multiple configuration, which is used without tuning.
// If tuning is enabled, this configuration is also added to the group of instances.
inline void GetSpatialMultipleConfig(const miopen::batchnorm::ProblemDescription& problem,
                                     size_t vectorsize,
                                     size_t& xlocalsize,
                                     size_t& ylocalsize)
{
    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    if(problem.IsLayoutNHWC())
    {
        GetLocalConfigNHWC(problem, vectorsize, xlocalsize, ylocalsize);
    }
    else
    {
        xlocalsize = 1;
        ylocalsize = 1024;
        if(ylocalsize > in_cstride / vectorsize)
        {
            // No need to use workgroups larger than the HW dimension
            ylocalsize = std::max(size_t{64},
                                  size_t{1 << int(std::ceil(std::log2(in_cstride / vectorsize)))});
        }
    }
}

// Return true if spatial multiple is the preferred method to be used.
// The function is based on heuristics and it returns always true for NHWC.
inline bool UseMultiple(const miopen::batchnorm::ProblemDescription& problem)
{
    size_t n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nhw     = n * in_cstride;
    // Check heuristics (used to choose between spatial single and multiple for performance)
    if(!problem.IsLayoutNHWC() &&
       problem.GetDirection() == miopen::batchnorm::Direction::Backward &&
       (!((in_nhw >= static_cast<size_t>(32 * 1024 * 1024) || in_cstride <= 1024) &&
          in_cstride > 512)))
    {
        return false;
    }

    if(!problem.IsLayoutNHWC() &&
       problem.GetDirection() == miopen::batchnorm::Direction::ForwardTraining &&
       (!(n >= 3 && ((in_nhw >= static_cast<size_t>(32 * 1024 * 1024) || in_cstride <= 1024) &&
                     in_cstride > 512))))
    {
        return false;
    }

    return true;
}

// Provide the stash method to use for spatial multiple implementation
inline int GetStashMethod(bool IsLayoutNHWC,
                          miopenDataType_t problem_type,
                          unsigned int stash_values,
                          size_t c,
                          size_t n,
                          size_t in_cstride,
                          size_t ylocalsize,
                          size_t zlocalsize,
                          size_t nelements)
{
    // See `batchnorm_functions.hpp` for stash implementation of different methods
    int stash_method = 0;
    stash_values *= (problem_type == miopenFloat ? 1 : 2);
    unsigned int last_ylocalsize =
        (in_cstride) % ylocalsize == 0 ? ylocalsize : (in_cstride) % ylocalsize;
    unsigned int last_zlocalsize =
        n % (zlocalsize * nelements) == 0 ? (zlocalsize * nelements) : n % (zlocalsize * nelements);
    if(last_ylocalsize < stash_values && last_zlocalsize >= (size_t)stash_values)
    {
        stash_method = 1;
    }
    if(IsLayoutNHWC && !(problem_type == miopenFloat) && (c % 2 != 0) &&
       (last_zlocalsize >= stash_values))
    {
        stash_method = 2;
    }
    return stash_method;
}

// Spatial single
// Variant<variant>-<vectorsize>
inline std::string GetKernelIdFromVariant(int variant, size_t vectorsize)
{
    std::stringstream stream;
    stream << "Variant" << variant << "-" << vectorsize;
    return stream.str();
}

// Spatial multiple
// Variant<variant>-<vectorsize>-<xlocalsize>-<ylocalsize>-<zlocalsize>-<nelements>
inline std::string GetKernelIdFromVariant(int variant,
                                          size_t vectorsize,
                                          size_t xlocalsize,
                                          size_t ylocalsize,
                                          size_t zlocalsize,
                                          size_t nelements)
{
    std::stringstream stream;
    stream << "Variant" << variant << "-" << vectorsize << "-" << xlocalsize << "-" << ylocalsize
           << "-" << zlocalsize << "-" << nelements;
    return stream.str();
}

// Return tuning parameters from kernel_id string
// In case of variant != 2 (spatial single), only variant and vectorsize are meaningful
inline void GetVariantFromKernelId(const std::string& kernel_id,
                                   int& variant,
                                   size_t& vectorsize,
                                   size_t& xlocalsize,
                                   size_t& ylocalsize,
                                   size_t& zlocalsize,
                                   size_t& nelements)
{
    std::stringstream iss(&kernel_id[7]);
    std::string segment;
    std::vector<std::string> seglist;

    while(std::getline(iss, segment, '-'))
    {
        seglist.push_back(segment);
    }
    variant    = std::stoi(seglist[0]);
    vectorsize = std::stoi(seglist[1]);
    if(variant != 2)
    {
        return;
    }
    xlocalsize = std::stoi(seglist[2]);
    ylocalsize = std::stoi(seglist[3]);
    zlocalsize = std::stoi(seglist[4]);
    nelements  = std::stoi(seglist[5]);
}

// Add spatial single instances for given problem
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

    // NCHW supports also variants 0 and 3 which can be much faster than
    // variant 1 but have more restrictions. Here we decide if we use variant
    // 0, 1, 3
    // In case variant 0 or 3 are selected, we add also variant 1 for tuning.
    // Almost always variant 0 and 3 will be faster than variant 1 but
    // we add the latter for tuning to be sure and because it is cheap
    if(!problem.IsLayoutNHWC())
    {

#if(WORKAROUND_SWDEV_253606 == 0)
        if(n < 3 && problem.GetDirection() == miopen::batchnorm::Direction::ForwardTraining)
        {
            valid_kernels.push_back(GetKernelIdFromVariant(4, 1));
            valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
            return;
        }
#endif

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
    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
}

// Check if spatial multiple implementation can be used for a given problem
// and workgroup configuration.
inline bool IsSpatialMultipleApplicable(const miopen::batchnorm::ProblemDescription& problem,
                                        size_t vectorsize,
                                        unsigned int stash_values,
                                        size_t ylocalsize,
                                        size_t zlocalsize,
                                        size_t nelements)
{
    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    if(problem.IsLayoutNHWC())
    {
        // check if the provided vectorsize can be used
        if(c % vectorsize != 0)
        {
            return false;
        }

        bool bfp32parm = problem.GetXDesc().GetType() == miopenHalf ||
                                 problem.GetXDesc().GetType() == miopenBFloat16
                             ? false
                             : true;

        stash_values *= (bfp32parm ? 1 : 2);
        unsigned int last_ylocalsize =
            in_cstride % ylocalsize == 0 ? ylocalsize : in_cstride % ylocalsize;

        unsigned int last_zlocalsize = n % (zlocalsize * nelements) == 0
                                           ? (zlocalsize * nelements)
                                           : n % (zlocalsize * nelements);

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
        if((!bfp32parm && (c % 2 != 0 && last_zlocalsize < (size_t)stash_values)) ||
           ((last_ylocalsize < stash_values) && (last_zlocalsize < (size_t)stash_values)))
        {
            return false;
        }
    }
    else
    {
        // check if the provided vectorsize can be used
        if(in_cstride % vectorsize != 0)
        {
            return false;
        }

        unsigned int last_ylocalsize =
            in_cstride % ylocalsize == 0 ? ylocalsize : in_cstride % ylocalsize;

        unsigned int last_zlocalsize = n % (zlocalsize * nelements) == 0
                                           ? (zlocalsize * nelements)
                                           : n % (zlocalsize * nelements);
        // Restrictions:
        //  - last block must have enough space to stash intermediate results in HW dimension
        //  - if last block doesn't fit, intermediate results are stored in N dimension which must
        //    be large enough
        stash_values *= (problem.GetXDesc().GetType() == miopenFloat ? 1 : 2);
        if(last_ylocalsize < stash_values && last_zlocalsize < (size_t)stash_values)
        {
            return false;
        }
    }
    return true;
}

// Add spatial multiple instances for given problem
// The first instance added is based on heuristics.
// No more instances are added in case of NCHW with n <= 64 because on average performance
// uplift is very limited. With large batch sizes the full parameter space is added.
// For NHWC the full parameter space is always added.
inline void DefaultConfigSpatialMultiple(const miopen::batchnorm::ProblemDescription& problem,
                                         unsigned int stash_values,
                                         std::vector<std::string>& valid_kernels)
{
    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    // Largest supported vector size for this problem
    size_t vectorsize_limit = problem.IsLayoutNHWC()
                                  ? (c % 4 == 0 ? 4 : (c % 2 == 0 ? 2 : 1))
                                  : (in_cstride % 4 == 0 ? 4 : (in_cstride % 2 == 0 ? 2 : 1));

    // First add the default config (heuristics).
    // Try to create a configuration with the largest vector size (vectorsize_limit).
    // If that's not applicable, fall back to configuration without vectorization.
    {
        size_t xlocalsize, ylocalsize;
        size_t vectorsize = vectorsize_limit;
        size_t zlocalsize = 1;
        size_t nelements  = n;
        GetSpatialMultipleConfig(problem, vectorsize, xlocalsize, ylocalsize);

        if(IsSpatialMultipleApplicable(
               problem, vectorsize, stash_values, ylocalsize, zlocalsize, nelements))
        {
            valid_kernels.push_back(GetKernelIdFromVariant(
                2, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements));
        }
        else
        {
            if(vectorsize > 1)
            {
                GetSpatialMultipleConfig(problem, 1, xlocalsize, ylocalsize);

                if(IsSpatialMultipleApplicable(
                       problem, 1, stash_values, ylocalsize, zlocalsize, nelements))
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(
                        2, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements));
                }
            }
        }
    }

    // Add the full parameter space
    if(problem.IsLayoutNHWC())
    {
        // All vector sizes less or equal to the supported vector size limit
        for(size_t vectorsize = vectorsize_limit; vectorsize > 0; vectorsize >>= 1)
        {
            size_t xlocalsize_limit_high = vectorsize > 1 ? 32 : 64;
            size_t xlocalsize_limit_low  = vectorsize > 1 ? 16 : 32;
            // this local size seems to always be the best one, so there is no need to check
            // other ones
            size_t max_localsize = 1024 / vectorsize;
            // xlocalsize = 32, 16 with vectorization
            // xlocalsize = 64, 32 without vectorization
            for(size_t xlocalsize_limit = xlocalsize_limit_high;
                xlocalsize_limit >= xlocalsize_limit_low;
                xlocalsize_limit >>= 1)
            {
                size_t xlocalsize = std::min(size_t{1 << int(std::ceil(std::log2(c / vectorsize)))},
                                             xlocalsize_limit);
                // zlocalsize = 1, 2, 4
                for(size_t zlocalsize = 1; zlocalsize <= 4; zlocalsize <<= 1)
                {
                    // 1 zblock: nelements = n / zlocalsize
                    // 2 zblock: nelements = n / (2 * zlocalsize)
                    for(size_t i = 1; i <= 2; ++i)
                    {
                        size_t nelements = n / (i * zlocalsize);
                        if(nelements == 0)
                        {
                            continue;
                        }
                        // Currently only this case is supported
                        if(n % nelements != 0)
                        {
                            continue;
                        }
                        size_t ylocalsize = max_localsize / xlocalsize / zlocalsize;
                        // Check if the computed instance is applicable and add it
                        if(IsSpatialMultipleApplicable(problem,
                                                       vectorsize,
                                                       stash_values,
                                                       ylocalsize,
                                                       zlocalsize,
                                                       nelements))
                        {
                            valid_kernels.push_back(GetKernelIdFromVariant(
                                2, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements));
                        }
                    }
                }
            }
        }
    }
    else
    {
        // Do not add full parameter space with small batch sizes
        if(n < 64)
        {
            return;
        }
        // All vector sizes less equal than the supported vector size limit
        for(size_t vectorsize = vectorsize_limit; vectorsize > 0; vectorsize >>= 1)
        {
            size_t xlocalsize       = 1;
            size_t ylocalsize_limit = 1024;
            if(ylocalsize_limit > in_cstride / vectorsize)
            {
                // No need to use workgroups larger than the HW dimension
                ylocalsize_limit = std::max(
                    size_t{64}, size_t{1 << int(std::ceil(std::log2(in_cstride / vectorsize)))});
            }
            // Workgroup sizes = ylocalsize_limit, ylocalsize_limit / 2 and ylocalsize_limit / 4
            // It was observed than smaller workgroup size can be beneficial but could not
            // generalize it, so all cases are considered.
            for(size_t localsize_limit = ylocalsize_limit; localsize_limit >= ylocalsize_limit / 4;
                localsize_limit >>= 1)
            {
                // zlocalsize = 1, 2, 4
                for(size_t zlocalsize = 1; zlocalsize <= 4; zlocalsize <<= 1)
                {
                    size_t ylocalsize = localsize_limit / zlocalsize;
                    // Only include case with 1 zblock
                    size_t nelements = n / zlocalsize;
                    // Currently only this case is supported
                    if(n % nelements != 0)
                    {
                        continue;
                    }
                    // Condition necessary for running / saved mean and variance correctness
                    if(ylocalsize <= 64)
                    {
                        continue;
                    }
                    // Check if the computed instance is applicable and add it
                    if(IsSpatialMultipleApplicable(
                           problem, vectorsize, stash_values, ylocalsize, zlocalsize, nelements))
                    {
                        valid_kernels.push_back(GetKernelIdFromVariant(
                            2, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements));
                    }
                }
            }
        }
    }
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
