/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/fractionalmaxpool.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/fractionalmaxpool/invoke_params.hpp>
#include <miopen/fractionalmaxpool/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace fractionalmaxpool {

miopenStatus_t FractionalMaxPoolForward(Handle& handle,
                                        const TensorDescriptor& inputDesc,
                                        ConstData_t input,
                                        const TensorDescriptor& outputDesc,
                                        Data_t output,
                                        const TensorDescriptor& indicesDesc,
                                        Data_t indices,
                                        const TensorDescriptor& randomSampleDesc,
                                        ConstData_t random_sample,
                                        const bool return_indices,
                                        const int64_t KD,
                                        const int64_t KH,
                                        const int64_t KW)
{
    const auto problem = fractionalmaxpool::FwdProblemDescription{
        inputDesc, outputDesc, indicesDesc, randomSampleDesc, return_indices, KD, KH, KW};
    const auto invoke_params = [&]() {
        auto tmp             = fractionalmaxpool::FwdInvokeParams{};
        tmp.inputDesc        = &inputDesc;
        tmp.input            = input;
        tmp.outputDesc       = &outputDesc;
        tmp.output           = output;
        tmp.indicesDesc      = &indicesDesc;
        tmp.indices          = indices;
        tmp.randomSampleDesc = &randomSampleDesc;
        tmp.random_sample    = random_sample;
        tmp.KD               = KD;
        tmp.KH               = KH;
        tmp.KW               = KW;

        return tmp;
    }();
    const auto algo = AlgorithmName{"FractionalMaxPoolForward"};
    const auto solvers =
        solver::SolverContainer<solver::fractionalmaxpool::FractionalMaxPoolForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t FractionalMaxPoolBackward(Handle& handle,
                                         const TensorDescriptor& indicesDesc,
                                         ConstData_t indices,
                                         const TensorDescriptor& outputGradDesc,
                                         ConstData_t output_grad,
                                         const TensorDescriptor& inputGradDesc,
                                         Data_t input_grad)
{
    const auto problem =
        fractionalmaxpool::BwdProblemDescription{indicesDesc, outputGradDesc, inputGradDesc};
    const auto invoke_params = [&]() {
        auto tmp           = fractionalmaxpool::BwdInvokeParams{};
        tmp.indicesDesc    = &indicesDesc;
        tmp.indices        = indices;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.output_grad    = output_grad;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.input_grad     = input_grad;

        return tmp;
    }();
    const auto algo = AlgorithmName{"FractionalMaxPoolBackward"};
    const auto solvers =
        solver::SolverContainer<solver::fractionalmaxpool::FractionalMaxPoolBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace fractionalmaxpool

} // namespace miopen
