/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_REDUCE_UTILS_HPP_
#define MIOPEN_REDUCE_UTILS_HPP_

#include <miopen/reduce/solvers.hpp>

namespace miopen {
namespace solver {
namespace reduce {

#define LOCAL_SIZE 256

inline size_t get_reqd_work_item_cnt(const ExecutionContext& context)
{
    // At least 4 WGs per one CU
    return static_cast<size_t>(LOCAL_SIZE * context.GetStream().GetMaxComputeUnits() * 4);
}

inline size_t get_reqd_work_item_cnt(const Handle& handle)
{
    // At least 4 WGs per one CU
    return static_cast<size_t>(LOCAL_SIZE * handle.GetMaxComputeUnits() * 4);
}

inline size_t
get_parallelism_size(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size)
{
    size_t parallelism_size = 1ULL;
    while(parallelism_size * output_numel < reqd_work_item_cnt &&
          parallelism_size < std::sqrt(reduce_size))
    {
        parallelism_size *= 2ULL;
    }
    return parallelism_size;
}

inline bool is_parallelism(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size)
{
    return !(output_numel > reqd_work_item_cnt) &&
           (output_numel * reduce_size > reqd_work_item_cnt);
}

inline bool IsImprovementOverROCm(const ExecutionContext& context,
                                  const miopen::reduce::ProblemDescriptionCalculation& problem)
{
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();
    auto dim   = problem.GetDim();

    auto reduce_size = xdims[dim];
    auto output_numel =
        std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context);

    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        // It's large enough to parallelization, but calling the kernel twice is overhead.
        // For cases smaller than this, ROCm pytorch performed better.
        bool is_improvement_ROCm = (output_numel * reduce_size < reqd_work_item_cnt * 64);
        // But the reduce size is small, MIOpen HIP performed better.
        bool is_reduce_large = (reduce_size > 64);

        if(is_improvement_ROCm && is_reduce_large)
            return false;
    }

    return true;
}

} // namespace reduce
} // namespace solver
} // namespace miopen

#endif // _MIOPEN_REDUCE_UTILS_HPP_
