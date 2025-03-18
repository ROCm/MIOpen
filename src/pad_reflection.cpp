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

#include <miopen/find_solution.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/pad_reflection.hpp>
#include <miopen/pad_reflection/invoke_params.hpp>
#include <miopen/pad_reflection/solvers.hpp>
#include <miopen/pad_reflection/problem_description.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace pad_reflection {

miopenStatus_t PadReflectionFwd(Handle& handle,
                                const TensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& yDesc,
                                Data_t y,
                                const int64_t* padding,
                                const size_t num_padding)
{
    const auto problem =
        pad_reflection::PadReflectionFwdProblemDescription{xDesc, yDesc, padding, num_padding};

    const auto invoke_params = [&]() {
        auto tmp        = pad_reflection::FwdInvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.xDesc       = &xDesc;
        tmp.yDesc       = &yDesc;
        tmp.x           = x;
        tmp.y           = y;
        tmp.padding     = padding;
        tmp.num_padding = num_padding;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"PadReflectionFwd"};
    const auto solvers = solver::SolverContainer<solver::pad_reflection::PadReflectionFwd>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t PadReflectionBwd(Handle& handle,
                                const TensorDescriptor& dxDesc,
                                Data_t dx,
                                const TensorDescriptor& dyDesc,
                                ConstData_t dy,
                                const int64_t* padding,
                                const size_t num_padding)
{
    const auto problem =
        pad_reflection::PadReflectionBwdProblemDescription{dxDesc, dyDesc, padding, num_padding};

    const auto invoke_params = [&]() {
        auto tmp        = pad_reflection::BwdInvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.dxDesc      = &dxDesc;
        tmp.dyDesc      = &dyDesc;
        tmp.dx          = dx;
        tmp.dy          = dy;
        tmp.padding     = padding;
        tmp.num_padding = num_padding;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"PadReflectionBwd"};
    const auto solvers = solver::SolverContainer<solver::pad_reflection::PadReflectionBwd>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace pad_reflection
} // namespace miopen
