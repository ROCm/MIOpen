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
                    const TensorDescriptor& paramInDesc,
                    ConstData_t paramIn,
                    const TensorDescriptor& paramOutDesc,
                    Data_t paramOut,
                    const TensorDescriptor& paramOutFloat16Desc,
                    Data_t paramOutFloat16,
                    const TensorDescriptor& gradInDesc,
                    ConstData_t gradIn,
                    const TensorDescriptor& expAvgInDesc,
                    ConstData_t expAvgIn,
                    const TensorDescriptor& expAvgOutDesc,
                    Data_t expAvgOut,
                    const TensorDescriptor& expAvgSqInDesc,
                    ConstData_t expAvgSqIn,
                    const TensorDescriptor& expAvgSqOutDesc,
                    Data_t expAvgSqOut,
                    const TensorDescriptor& maxExpAvgSqInDesc,
                    ConstData_t maxExpAvgSqIn,
                    const TensorDescriptor& maxExpAvgSqOutDesc,
                    Data_t maxExpAvgSqOut,
                    const TensorDescriptor& gradScaleDesc,
                    ConstData_t gradScale,
                    const TensorDescriptor& foundInfDesc,
                    ConstData_t foundInf,
                    const TensorDescriptor& stepInDesc,
                    ConstData_t stepIn,
                    const TensorDescriptor& stepOutDesc,
                    Data_t stepOut,
                    const uint32_t step,
                    const float lr,
                    const float beta1,
                    const float beta2,
                    const float weight_decay,
                    const float eps,
                    const bool amsgrad,
                    const bool maximize,
                    const bool adamw,
                    const bool is_amp)
{
    const auto problem = adam::ProblemDescription{paramInDesc,
                                                  paramOutDesc,
                                                  paramOutFloat16Desc,
                                                  gradInDesc,
                                                  expAvgInDesc,
                                                  expAvgOutDesc,
                                                  expAvgSqInDesc,
                                                  expAvgSqOutDesc,
                                                  maxExpAvgSqInDesc,
                                                  maxExpAvgSqOutDesc,
                                                  gradScaleDesc,
                                                  foundInfDesc,
                                                  stepInDesc,
                                                  stepOutDesc,
                                                  amsgrad,
                                                  true,
                                                  adamw,
                                                  is_amp};

    const auto invoke_params = [&]() {
        auto tmp = adam::AdamInvokeParams{};
        tmp.type = InvokeType::Run;

        tmp.paramDesc       = &paramInDesc;
        tmp.gradDesc        = &gradInDesc;
        tmp.paramIn         = paramIn;
        tmp.paramOut        = paramOut;
        tmp.paramOutFloat16 = paramOutFloat16;
        tmp.gradIn          = gradIn;
        tmp.expAvgIn        = expAvgIn;
        tmp.expAvgOut       = expAvgOut;
        tmp.expAvgSqIn      = expAvgSqIn;
        tmp.expAvgSqOut     = expAvgSqOut;
        tmp.maxExpAvgSqIn   = maxExpAvgSqIn;
        tmp.maxExpAvgSqOut  = maxExpAvgSqOut;
        tmp.gradScale       = gradScale;
        tmp.foundInf        = foundInf;
        tmp.stepIn          = stepIn;
        tmp.stepOut         = stepOut;

        tmp.step         = step;
        tmp.lr           = lr;
        tmp.beta1        = beta1;
        tmp.beta2        = beta2;
        tmp.weight_decay = weight_decay;
        tmp.eps          = eps;
        tmp.amsgrad      = amsgrad;
        tmp.maximize     = maximize;
        tmp.adamw        = adamw;

        return tmp;
    }();

    const auto algo    = AlgorithmName{"Adam"};
    const auto solvers = solver::SolverContainer<solver::adam::Adam>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
