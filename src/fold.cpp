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

#include "miopen/miopen.h"
#include "miopen/fold/problem_description.hpp"
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/fold/invoke_params.hpp>
#include <miopen/fold/solvers.hpp>
#include <miopen/fold.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t UnfoldForward(Handle& handle,
                             const TensorDescriptor& inputDesc,
                             ConstData_t input,
                             const TensorDescriptor& outputDesc,
                             Data_t output,
                             const int32_t* kernel_size,
                             const int kernel_size_size,
                             const int32_t* stride,
                             const int stride_size,
                             const int32_t* padding,
                             const int padding_size,
                             const int32_t* dilation,
                             const int dilation_size)
{
    const auto problem = fold::UnfoldFwdProblemDescription{inputDesc,
                                                           outputDesc,
                                                           kernel_size,
                                                           kernel_size_size,
                                                           stride,
                                                           stride_size,
                                                           padding,
                                                           padding_size,
                                                           dilation,
                                                           dilation_size};

    const auto invoke_params = [&]() {
        auto tmp             = fold::InvokeParams{};
        tmp.type             = InvokeType::Run;
        tmp.inputDesc        = &inputDesc;
        tmp.outputDesc       = &outputDesc;
        tmp.input            = input;
        tmp.output           = output;
        tmp.kernel_size      = kernel_size;
        tmp.stride           = stride;
        tmp.padding          = padding;
        tmp.dilation         = dilation;
        tmp.kernel_size_size = kernel_size_size;
        tmp.stride_size      = stride_size;
        tmp.padding_size     = padding_size;
        tmp.dilation_size    = dilation_size;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"UnfoldFwd"};
    const auto solvers = solver::SolverContainer<solver::fold::UnfoldFwd>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
