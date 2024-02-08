/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/kernel_cache.hpp>
#include <miopen/softmax.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/tensor.hpp>

#include <miopen/softmax/invoke_params.hpp>
#include <miopen/softmax/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

extern "C" miopenStatus_t miopenCreateSoftmaxDescriptor(miopenSoftmaxDescriptor_t* activDesc)
{

    MIOPEN_LOG_FUNCTION(activDesc);
    return miopen::try_([&] { miopen::deref(activDesc) = new miopen::ActivationDescriptor(); });
}

extern "C" miopenStatus_t miopenSetSoftmaxDescriptor(miopenSoftmaxDescriptor_t activDesc,
                                                        miopenActivationMode_t mode,
                                                        double activAlpha,
                                                        double activBeta,
                                                        double activGamma)
{

    MIOPEN_LOG_FUNCTION(activDesc, mode, activAlpha, activBeta, activGamma);
    return miopen::try_([&] {
        std::initializer_list<double> parms = {activAlpha, activBeta, activGamma};
        miopen::deref(activDesc)            = miopen::ActivationDescriptor(mode, parms.begin());
    });
}


miopenStatus_t SoftmaxForward(Handle& handle,
                              const void* alpha,
                              const void* beta,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              miopenSoftmaxAlgorithm_t algorithm,
                              miopenSoftmaxMode_t mode,
                              int x_offset,
                              int y_offset)
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    const auto problem = softmax::ProblemDescription{alpha, beta, xDesc, yDesc, algorithm, mode};
    const auto invoke_params =
        softmax::InvokeParams{alpha, beta, xDesc, x, yDesc, y, algorithm, mode, x_offset, y_offset};
    const auto algo    = AlgorithmName{"Softmax"};
    const auto solvers = solver::SolverContainer<solver::softmax::Softmax>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t SoftmaxBackward(Handle& handle,
                               const void* alpha,
                               const TensorDescriptor& yDesc,
                               ConstData_t y,
                               const TensorDescriptor& dyDesc,
                               ConstData_t dy,
                               const void* beta,
                               const TensorDescriptor& dxDesc,
                               Data_t dx,
                               miopenSoftmaxAlgorithm_t algorithm,
                               miopenSoftmaxMode_t mode,
                               int y_offset,
                               int dy_offset,
                               int dx_offset)
{
    if(dx == nullptr || y == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    const auto problem =
        softmax::ProblemDescription{alpha, beta, yDesc, dyDesc, dxDesc, algorithm, mode};
    const auto invoke_params = softmax::InvokeParams{alpha,
                                                     beta,
                                                     yDesc,
                                                     y,
                                                     dyDesc,
                                                     dy,
                                                     dxDesc,
                                                     dx,
                                                     algorithm,
                                                     mode,
                                                     y_offset,
                                                     dy_offset,
                                                     dx_offset};
    const auto algo          = AlgorithmName{"Softmax"};
    const auto solvers       = solver::SolverContainer<solver::softmax::Softmax>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
