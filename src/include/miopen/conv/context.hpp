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
#include <miopen/miopen.h>

#include <string>

namespace miopen {
struct ConvolutionDescriptor;
struct Handle;
struct TensorDescriptor;

/// A leftover of the legacy design, houses problem config,
/// environmental context (e.g. HW/SW platform) and solver-specific state.
///
/// TODO: These three entities should be made separate.
struct ConvolutionContext : ExecutionContext
{
    ConvolutionContext() = default;
    ConvolutionContext(conv::Direction dir) : problem(dir) {}
    ConvolutionContext(const TensorDescriptor& in,
                       const TensorDescriptor& weights,
                       const TensorDescriptor& out,
                       const ConvolutionDescriptor& conv,
                       conv::Direction dir,
                       int bias_ = 0)
        : problem(in, weights, out, conv, dir, bias_)
    {
    }
    ConvolutionContext(const ExecutionContext& exec_ctx, const ProblemDescription& problem_)
        : ExecutionContext(exec_ctx), problem(problem_)
    {
    }
    explicit ConvolutionContext(const ProblemDescription& problem_) : problem(problem_) {}
    ConvolutionContext(const conv::ProblemDescription& problem_, const ExecutionContext& ctx)
        : ExecutionContext(ctx), problem(problem_)
    {
    }

    ConvolutionContext& SetupFloats();

public:
    bool is_for_generic_search = false;

    ProblemDescription problem;
};

} // namespace miopen
