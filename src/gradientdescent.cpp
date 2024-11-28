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
#include <miopen/gradientdescent.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/gradientdescent/invoke_params.hpp>
#include <miopen/gradientdescent/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace GradientDescent {

miopenStatus_t GradientDescent(Handle& handle,
                               const TensorDescriptor& varInDesc,
                               ConstData_t var_in,
                               const TensorDescriptor& varOutDesc,
                               Data_t var_out,
                               const TensorDescriptor& alphaInDesc,
                               ConstData_t alpha_in,
                               const TensorDescriptor& deltaInDesc,
                               ConstData_t delta_in)
{
    const auto problem =
        GradientDescent::ProblemDescription{varInDesc, varOutDesc, alphaInDesc, deltaInDesc};

    const auto invoke_params = [&]() {
        auto tmp        = GradientDescent::InvokeParams{};
        tmp.varInDesc   = &varInDesc;
        tmp.varOutDesc  = &varOutDesc;
        tmp.alphaInDesc = &alphaInDesc;
        tmp.deltaInDesc = &deltaInDesc;

        tmp.var_in   = var_in;
        tmp.var_out  = var_out;
        tmp.alpha_in = alpha_in;
        tmp.delta_in = delta_in;

        return tmp;
    }();

    const auto algo    = AlgorithmName{"GradientDescent"};
    const auto solvers = solver::SolverContainer<solver::GradientDescent::GradientDescent>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace GradientDescent

} // namespace miopen
