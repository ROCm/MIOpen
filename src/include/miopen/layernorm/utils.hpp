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
#ifndef MIOPEN_LAYERNORM_UTILS_HPP_
#define MIOPEN_LAYERNORM_UTILS_HPP_

#include <miopen/layernorm/solvers.hpp>

namespace miopen {
namespace solver {
namespace layernorm {

#define LOCAL_SIZE 256

inline std::size_t sizeof_kernel_FLOAT(const miopen::layernorm::ProblemDescription& problem)
{
    const auto datatype = problem.GetXDesc().GetType();
    return get_data_size(datatype);
}

inline std::size_t sizeof_local_memory(const miopen::layernorm::ProblemDescription& problem)
{
    std::size_t rv = 0;
    rv += LOCAL_SIZE * sizeof_kernel_FLOAT(problem) * 2;
    return rv;
}

inline std::size_t sizeof_local_memory_t5(const miopen::layernorm::ProblemDescription& problem)
{
    std::size_t rv = 0;
    rv += LOCAL_SIZE * sizeof_kernel_FLOAT(problem);
    return rv;
}

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

inline size_t get_parallelism_size(size_t reqd_work_item_cnt, size_t inner_size, size_t outer_size)
{
    size_t parallelism_size = 1ULL;
    while(parallelism_size * inner_size < reqd_work_item_cnt &&
          parallelism_size < std::sqrt(outer_size))
    {
        parallelism_size *= 2ULL;
    }
    return parallelism_size;
}

inline bool is_parallelism(size_t reqd_work_item_cnt, size_t inner_size, size_t outer_size)
{
    return !(inner_size > reqd_work_item_cnt) && (inner_size * outer_size > reqd_work_item_cnt);
}

} // namespace layernorm
} // namespace solver
} // namespace miopen

#endif // _MIOPEN_LAYERNORM_UTILS_HPP_
