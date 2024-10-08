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
#include <miopen/avgpool.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/avgpool/invoke_params.hpp>
#include <miopen/avgpool/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace avgpool {

miopenStatus_t AvgPoolForward(Handle& handle,
                              const TensorDescriptor& inputDesc,
                              ConstData_t input,
                              const TensorDescriptor& outputDesc,
                              Data_t output,
                              const int64_t KD,
                              const int64_t KH,
                              const int64_t KW,
                              const int64_t SD,
                              const int64_t SH,
                              const int64_t SW,
                              const int64_t PD,
                              const int64_t PH,
                              const int64_t PW,
                              const bool count_include_pad,
                              const int64_t divisor_override)
{
    const auto problem =
        avgpool::FwdProblemDescription{inputDesc, outputDesc, count_include_pad, divisor_override};

    const auto invoke_params = [&]() {
        auto tmp       = avgpool::FwdInvokeParams{};
        tmp.inputDesc  = &inputDesc;
        tmp.outputDesc = &outputDesc;

        tmp.input             = input;
        tmp.output            = output;
        tmp.KD                = KD;
        tmp.KH                = KH;
        tmp.KW                = KW;
        tmp.SD                = SD;
        tmp.SH                = SH;
        tmp.SW                = SW;
        tmp.PD                = PD;
        tmp.PH                = PH;
        tmp.PW                = PW;
        tmp.count_include_pad = count_include_pad;
        tmp.divisor_override  = divisor_override;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"AvgPoolForward"};
    const auto solvers = solver::SolverContainer<solver::avgpool::AvgPoolForward2d,
                                                 solver::avgpool::AvgPoolForward3d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t AvgPoolBackward(Handle& handle,
                               const TensorDescriptor& outputGradDesc,
                               ConstData_t output_grad,
                               const TensorDescriptor& inputGradDesc,
                               Data_t input_grad,
                               const int64_t KD,
                               const int64_t KH,
                               const int64_t KW,
                               const int64_t SD,
                               const int64_t SH,
                               const int64_t SW,
                               const int64_t PD,
                               const int64_t PH,
                               const int64_t PW,
                               const bool count_include_pad,
                               const int64_t divisor_override)
{
    const auto problem = avgpool::BwdProblemDescription{
        outputGradDesc, inputGradDesc, count_include_pad, divisor_override};

    const auto invoke_params = [&]() {
        auto tmp           = avgpool::BwdInvokeParams{};
        tmp.outputGradDesc = &outputGradDesc;
        tmp.inputGradDesc  = &inputGradDesc;

        tmp.output_grad       = output_grad;
        tmp.input_grad        = input_grad;
        tmp.KD                = KD;
        tmp.KH                = KH;
        tmp.KW                = KW;
        tmp.SD                = SD;
        tmp.SH                = SH;
        tmp.SW                = SW;
        tmp.PD                = PD;
        tmp.PH                = PH;
        tmp.PW                = PW;
        tmp.count_include_pad = count_include_pad;
        tmp.divisor_override  = divisor_override;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"AvgPoolBackward"};
    const auto solvers = solver::SolverContainer<solver::avgpool::AvgPoolBackward2d,
                                                 solver::avgpool::AvgPoolBackward3d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace avgpool

} // namespace miopen
