/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/db_path.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/gemm/problem_description.hpp>
#include <miopen/miopen.h>

#include <string>

namespace miopen {
struct GemmNewDescriptor;
struct Handle;
struct TensorDescriptor;

/// A leftover of the legacy design, houses problem config,
/// environmental context (e.g. HW/SW platform) and solver-specific state.
///
/// TODO: These three entities should be made separate.
struct GemmContext : ExecutionContext
{
    // Solution-specific
    std::string general_compile_options;

    GemmContext() = default;
    GemmContext(const GemmNewDescriptor& gemm_desc,
                       const TensorDescriptor& a_desc,
                       const TensorDescriptor& b_desc,
                       const TensorDescriptor& c_desc)
        : problem(gemm_desc, a_desc, b_desc, c_desc)
    {
    }
    explicit GemmContext(const gemm::ProblemDescription& problem_) : problem(problem_) {}
    GemmContext(const gemm::ProblemDescription& problem_, const ExecutionContext& ctx)
        : ExecutionContext(ctx), problem(problem_)
    {
    }

    void SetupFloats();

public:
    bool is_for_generic_search = false;

    gemm::ProblemDescription problem;
};

} // namespace miopen
