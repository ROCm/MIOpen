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
#include <miopen/sgd.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/sgd/invoke_params.hpp>
#include <miopen/sgd/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace SGD {

miopenStatus_t SGDForward(Handle& handle,
                          const TensorDescriptor& paramInDesc,
                          ConstData_t paramIn,
                          const TensorDescriptor& paramOutDesc,
                          Data_t paramOut,
                          const TensorDescriptor& gradDesc,
                          ConstData_t grad,
                          const TensorDescriptor& momentumBufferInDesc,
                          ConstData_t momentumBufferIn,
                          const TensorDescriptor& momentumBufferOutDesc,
                          Data_t momentumBufferOut,
                          double lr,
                          double momentum,
                          double dampening,
                          double weightDecay,
                          bool nesterov,
                          bool momentum_initialized)
{
    const auto problem = SGD::ProblemDescription{
        paramInDesc, paramOutDesc, gradDesc, momentumBufferInDesc, momentumBufferOutDesc};

    const auto invoke_params = [&]() {
        auto tmp                  = SGD::InvokeParams{};
        tmp.type                  = InvokeType::Run;
        tmp.paramInDesc           = &paramInDesc;
        tmp.paramIn               = paramIn;
        tmp.paramOutDesc          = &paramOutDesc;
        tmp.paramOut              = paramOut;
        tmp.gradDesc              = &gradDesc;
        tmp.grad                  = grad;
        tmp.momentumBufferInDesc  = &momentumBufferInDesc;
        tmp.momentumBufferIn      = momentumBufferIn;
        tmp.momentumBufferOutDesc = &momentumBufferOutDesc;
        tmp.momentumBufferOut     = momentumBufferOut;
        tmp.lr                    = lr;
        tmp.momentum              = momentum;
        tmp.dampening             = dampening;
        tmp.weightDecay           = weightDecay;
        tmp.nesterov              = nesterov;
        tmp.momentum_initialized  = momentum_initialized;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"SGDForward"};
    const auto solvers = solver::SolverContainer<solver::SGD::SGDForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace SGD

} // namespace miopen
