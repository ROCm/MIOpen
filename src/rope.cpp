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

#include <miopen/rope.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/rope/invoke_params.hpp>
#include <miopen/rope/solvers.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t RoPEForward(Handle& handle,
                           const TensorDescriptor& xDesc,
                           ConstData_t x,
                           const TensorDescriptor& cosDesc,
                           ConstData_t cos,
                           const TensorDescriptor& sinDesc,
                           ConstData_t sin,
                           const TensorDescriptor& yDesc,
                           Data_t y)
{
    const auto problem = rope::ProblemDescriptionFwd{xDesc, cosDesc, sinDesc, yDesc};

    const auto invoke_params = [&]() {
        auto tmp    = rope::FwdInvokeParams{};
        tmp.type    = InvokeType::Run;
        tmp.xDesc   = &xDesc;
        tmp.cosDesc = &cosDesc;
        tmp.sinDesc = &sinDesc;
        tmp.yDesc   = &yDesc;
        tmp.x       = x;
        tmp.cos     = cos;
        tmp.sin     = sin;
        tmp.y       = y;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"RoPEForward"};
    const auto solvers = solver::SolverContainer<solver::rope::RoPEForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t RoPEBackward(Handle& handle,
                            const TensorDescriptor& dyDesc,
                            ConstData_t dy,
                            const TensorDescriptor& cosDesc,
                            ConstData_t cos,
                            const TensorDescriptor& sinDesc,
                            ConstData_t sin,
                            const TensorDescriptor& dxDesc,
                            Data_t dx)
{
    const auto problem = rope::ProblemDescriptionBwd{dyDesc, cosDesc, sinDesc, dxDesc};

    const auto invoke_params = [&]() {
        auto tmp    = rope::BwdInvokeParams{};
        tmp.type    = InvokeType::Run;
        tmp.dyDesc  = &dyDesc;
        tmp.cosDesc = &cosDesc;
        tmp.sinDesc = &sinDesc;
        tmp.dxDesc  = &dxDesc;
        tmp.dy      = dy;
        tmp.cos     = cos;
        tmp.sin     = sin;
        tmp.dx      = dx;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"RoPEBackward"};
    const auto solvers = solver::SolverContainer<solver::rope::RoPEBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
