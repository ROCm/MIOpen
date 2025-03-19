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

#include <miopen/common.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>
#include <miopen/pad_constant.hpp>
#include <miopen/pad_constant/problem_description.hpp>
#include <miopen/pad_constant/invoke_params.hpp>
#include <miopen/pad_constant/solvers.hpp>

namespace miopen {
namespace pad_constant {

miopenStatus_t PadConstantForward(Handle& handle,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& yDesc,
                                  ConstData_t x,
                                  Data_t y,
                                  const int64_t* padding,
                                  const int padding_size,
                                  float value)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = pad_constant::FwdProblemDescription{xDesc, yDesc, padding, padding_size};

    const auto invoke_params = [&]() {
        auto tmp          = pad_constant::FwdInvokeParams{};
        tmp.xDesc         = &xDesc;
        tmp.yDesc         = &yDesc;
        tmp.x             = x;
        tmp.y             = y;
        tmp.padding       = padding;
        tmp.padding_size  = padding_size;
        tmp.padding_value = value;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"PadConstantFwd"};
    const auto solvers = solver::SolverContainer<solver::pad_constant::PadConstantFwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t PadConstantBackward(Handle& handle,
                                   const TensorDescriptor& dxDesc,
                                   const TensorDescriptor& dyDesc,
                                   Data_t dx,
                                   ConstData_t dy,
                                   const int64_t* padding,
                                   const int padding_size)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = pad_constant::BwdProblemDescription{dxDesc, dyDesc, padding, padding_size};

    const auto invoke_params = [&]() {
        auto tmp         = pad_constant::BwdInvokeParams{};
        tmp.dxDesc       = &dxDesc;
        tmp.dyDesc       = &dyDesc;
        tmp.dx           = dx;
        tmp.dy           = dy;
        tmp.padding      = padding;
        tmp.padding_size = padding_size;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"PadConstantBwd"};
    const auto solvers = solver::SolverContainer<solver::pad_constant::PadConstantBwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace pad_constant
} // namespace miopen
