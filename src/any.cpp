/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

// // #include "miopen/any.hpp"
// // #include "miopen/miopen.h"
// // #include <miopen/>
// #include <miopen/any.hpp>
// // #include <miopen/any/invoke_params.hpp>
// // #include <miopen/any/solvers.hpp>
// #include <miopen/check_numerics.hpp>
// #include <miopen/datatype.hpp>
// #include <miopen/find_solution.hpp>
// #include <miopen/float_equal.hpp>
// #include <miopen/kernel_cache.hpp>
// #include <miopen/tensor.hpp>

#include "miopen/any/problem_description.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/miopen.h"
#include "miopen/names.hpp"
#include <miopen/any.hpp>
#include <miopen/any/invoke_params.hpp>
#include <miopen/any/solvers.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

std::size_t GetAnyWorkspaceSize(Handle& handle,
                                const TensorDescriptor& inputDesc,
                                TensorDescriptor& outputDesc,
                                const int32_t dim,
                                const bool keepdim)
{
    if(dim != -1)
    {
        return 0;
    }

    auto ctx           = ExecutionContext{&handle};
    const auto problem = any::ProblemDescription{inputDesc, outputDesc, dim, keepdim};

    // NOTE(anhduong): Check later: algorithm name is kernel function's name or solver's name?
    const auto algo    = AlgorithmName("AnyForward");
    const auto solvers = solver::SolverContainer<solver::any::AnyForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t AnyForward(Handle& handle,
                          Data_t workspace,
                          size_t workspaceSizeInBytes,
                          const TensorDescriptor& inputDesc,
                          ConstData_t input,
                          const TensorDescriptor& outputDesc,
                          Data_t output,
                          const int32_t dim,
                          const bool keepdim)
{
    const auto problem = any::ProblemDescription{inputDesc, outputDesc, dim, keepdim};

    const auto invoke_params = [&]() {
        auto tmp           = any::InvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.inputDesc      = &inputDesc;
        tmp.outputDesc     = &outputDesc;
        tmp.input          = input;
        tmp.output         = output;
        tmp.dim            = dim;
        tmp.keepdim        = keepdim;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"AnyForward"};
    const auto solvers = solver::SolverContainer<solver::any::AnyForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen