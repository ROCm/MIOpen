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
#include <miopen/lppool.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/lppool/invoke_params.hpp>
#include <miopen/lppool/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace lppool {

miopenStatus_t LPPoolForward(Handle& handle,
                             const TensorDescriptor& inputDesc,
                             ConstData_t input,
                             const TensorDescriptor& outputDesc,
                             Data_t output,
                             const int64_t KD,
                             const int64_t KH,
                             const int64_t SD,
                             const int64_t SH,
                             const float norm_type)
{
    const auto problem = lppool::FwdProblemDescription{inputDesc, outputDesc, norm_type};

    const auto invoke_params = [&]() {
        auto tmp       = lppool::FwdInvokeParams{};
        tmp.inputDesc  = &inputDesc;
        tmp.outputDesc = &outputDesc;

        tmp.input     = input;
        tmp.output    = output;
        tmp.KD        = KD;
        tmp.KH        = KH;
        tmp.SD        = SD;
        tmp.SH        = SH;
        tmp.norm_type = norm_type;

        return tmp;
    }();
    const auto algo = AlgorithmName{"LPPoolForward"};
    const auto solvers =
        solver::SolverContainer<solver::lppool::LPPoolForward1d, solver::lppool::LPPoolForward2d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t LPPoolBackward(Handle& handle,
                              const TensorDescriptor& inputDesc,
                              ConstData_t input,
                              const TensorDescriptor& outputDesc,
                              ConstData_t output,
                              const TensorDescriptor& outputGradDesc,
                              ConstData_t output_grad,
                              const TensorDescriptor& inputGradDesc,
                              Data_t input_grad,
                              const int64_t KD,
                              const int64_t KH,
                              const int64_t SD,
                              const int64_t SH,
                              const float norm_type)
{
    const auto problem = lppool::BwdProblemDescription{
        inputDesc, outputDesc, outputGradDesc, inputGradDesc, norm_type};

    const auto invoke_params = [&]() {
        auto tmp           = lppool::BwdInvokeParams{};
        tmp.inputDesc      = &inputDesc;
        tmp.outputDesc     = &outputDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.inputGradDesc  = &inputGradDesc;

        tmp.input       = input;
        tmp.output      = output;
        tmp.output_grad = output_grad;
        tmp.input_grad  = input_grad;
        tmp.KD          = KD;
        tmp.KH          = KH;
        tmp.SD          = SD;
        tmp.SH          = SH;
        tmp.norm_type   = norm_type;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"LPPoolBackward"};
    const auto solvers = solver::SolverContainer<solver::lppool::LPPoolBackward1d,
                                                 solver::lppool::LPPoolBackward2d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace lppool

} // namespace miopen
