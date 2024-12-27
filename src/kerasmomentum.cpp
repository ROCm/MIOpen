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
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/kerasmomentum.hpp>
#include <miopen/kerasmomentum/invoke_params.hpp>
#include <miopen/kerasmomentum/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace KerasMomentum {

miopenStatus_t KerasMomentum(Handle& handle,
                             const TensorDescriptor& varInDesc,
                             ConstData_t var_in,
                             const TensorDescriptor& varOutDesc,
                             Data_t var_out,
                             const TensorDescriptor& accumInDesc,
                             ConstData_t accum_in,
                             const TensorDescriptor& accumOutDesc,
                             Data_t accum_out,
                             const TensorDescriptor& lrInDesc,
                             ConstData_t lr_in,
                             const TensorDescriptor& gradInDesc,
                             ConstData_t grad_in,
                             const TensorDescriptor& momentumInDesc,
                             ConstData_t momentum_in,
                             const bool nesterov)
{
    const auto problem = KerasMomentum::ProblemDescription{
        varInDesc, varOutDesc, accumInDesc, accumOutDesc, lrInDesc, gradInDesc, momentumInDesc};

    const auto invoke_params = [&]() {
        auto tmp           = KerasMomentum::InvokeParams{};
        tmp.varInDesc      = &varInDesc;
        tmp.varOutDesc     = &varOutDesc;
        tmp.accumInDesc    = &accumInDesc;
        tmp.accumOutDesc   = &accumOutDesc;
        tmp.lrInDesc       = &lrInDesc;
        tmp.gradInDesc     = &gradInDesc;
        tmp.momentumInDesc = &momentumInDesc;

        tmp.var_in      = var_in;
        tmp.var_out     = var_out;
        tmp.accum_in    = accum_in;
        tmp.accum_out   = accum_out;
        tmp.lr_in       = lr_in;
        tmp.grad_in     = grad_in;
        tmp.momentum_in = momentum_in;
        tmp.nesterov    = nesterov;

        return tmp;
    }();

    const auto algo    = AlgorithmName{"KerasMomentum"};
    const auto solvers = solver::SolverContainer<solver::KerasMomentum::KerasMomentum>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace KerasMomentum

} // namespace miopen
