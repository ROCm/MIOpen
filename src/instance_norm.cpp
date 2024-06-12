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

#include "miopen/instance_norm.hpp"
#include "miopen/miopen.h"
#include "miopen/instancenorm/invoke_params.hpp"
#include "miopen/instancenorm/problem_description.hpp"
#include "miopen/instancenorm/solvers.hpp"
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t InstanceNormForward(Handle &handle,
                                    const TensorDescriptor &inputDesc,
                                    ConstData_t input,
                                    const TensorDescriptor &outputDesc,
                                    Data_t output,
                                    const TensorDescriptor &weightDesc,
                                    ConstData_t weight,
                                    const TensorDescriptor &biasDesc,
                                    ConstData_t bias,
                                    const TensorDescriptor &meanInDesc,
                                    ConstData_t meanIn,
                                    const TensorDescriptor &varInDesc,
                                    ConstData_t varIn,
                                    const TensorDescriptor &meanOutDesc,
                                    Data_t meanOut,
                                    const TensorDescriptor &varOutDesc,
                                    Data_t varOut,
                                    const TensorDescriptor &meanVarDesc,
                                    Data_t meanVar,
                                    float epsilon,
                                    float momentum,
                                    bool useInputStats)
{
    const auto problem = instancenorm::InstanceNormFwdProblemDescription{ inputDesc, outputDesc, weightDesc, biasDesc, meanInDesc, varInDesc, meanOutDesc, varOutDesc, meanVarDesc, useInputStats};

    const auto invoke_params = [&]() {
        auto tmp           = instancenorm::InstanceNormInvokeParams{};
        tmp.inputDesc      = &inputDesc;
        tmp.outputDesc      = &outputDesc;
        tmp.weightDesc      = &weightDesc;
        tmp.biasDesc      = &biasDesc;
        tmp.meanInDesc      = &meanInDesc;
        tmp.varInDesc      = &varInDesc;
        tmp.meanOutDesc      = &meanOutDesc;
        tmp.varOutDesc      = &varOutDesc;
        tmp.meanVarDesc      = &meanVarDesc;
        tmp.input      = input;
        tmp.output      = output;
        tmp.weight      = weight;
        tmp.bias      = bias;
        tmp.meanIn      = meanIn;
        tmp.varIn      = varIn;
        tmp.meanOut      = meanOut;
        tmp.varOut      = varOut;
        tmp.meanVar      = meanVar;
        tmp.epsilon      = epsilon;
        tmp.momentum      = momentum;
        tmp.useInputStats      = useInputStats;
        return tmp;
    }();

    const auto algo = AlgorithmName{"InstanceNormFwd"};
    const auto solvers =
        solver::SolverContainer<solver::instancenorm::InstanceNormFwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
