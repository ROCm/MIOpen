/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/adam.hpp>
#include <miopen/adam/invoke_params.hpp>
#include <miopen/adam/solvers.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t Adam(Handle& handle,
                    const TensorDescriptor& paramDesc,
                    Data_t param,
                    const TensorDescriptor& gradDesc,
                    ConstData_t grad,
                    const TensorDescriptor& expAvgDesc,
                    Data_t expAvg,
                    const TensorDescriptor& expAvgSqDesc,
                    Data_t expAvgSq,
                    const TensorDescriptor& stepDesc,
                    Data_t step,
                    const double lr,
                    const double beta1,
                    const double beta2,
                    const double weight_decay,
                    const double eps,
                    const bool amsgrad,
                    const TensorDescriptor* gradScaleDescPtr,
                    ConstData_t gradScale,
                    const TensorDescriptor* foundInfDescPtr,
                    ConstData_t foundInf)
{
    const auto problem = adam::ProblemDescription{paramDesc,
                                                  gradDesc,
                                                  expAvgDesc,
                                                  expAvgSqDesc,
                                                  stepDesc,
                                                  lr,
                                                  beta1,
                                                  beta2,
                                                  weight_decay,
                                                  eps,
                                                  amsgrad,
                                                  gradScaleDescPtr,
                                                  foundInfDescPtr};

    const auto invoke_params = [&]() {
        auto tmp = adam::InvokeParams{};
        tmp.type = InvokeType::Run;

        tmp.paramDesc    = &paramDesc;
        tmp.param        = param;
        tmp.gradDesc     = &gradDesc;
        tmp.grad         = grad;
        tmp.expAvgDesc   = &expAvgDesc;
        tmp.expAvg       = expAvg;
        tmp.expAvgSqDesc = &expAvgSqDesc;
        tmp.expAvgSq     = expAvgSq;
        tmp.stepDesc     = &stepDesc;
        tmp.step         = step;

        tmp.lr           = lr;
        tmp.beta1        = beta1;
        tmp.beta2        = beta2;
        tmp.weight_decay = weight_decay;
        tmp.eps          = eps;
        tmp.amsgrad      = amsgrad;

        tmp.gradScaleDesc = gradScaleDescPtr;
        tmp.gradScale     = gradScale;
        tmp.foundInfDesc  = foundInfDescPtr;
        tmp.foundInf      = foundInf;

        return tmp;
    }();

    const auto algo    = AlgorithmName{"Adam"};
    const auto solvers = solver::SolverContainer<solver::adam::Adam>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
