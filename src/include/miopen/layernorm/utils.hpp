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
#pragma once

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

} // namespace layernorm
} // namespace solver
} // namespace miopen
