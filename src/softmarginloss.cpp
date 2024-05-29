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

#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/softmarginloss/invoke_params.hpp>
#include <miopen/softmarginloss/solvers.hpp>
#include <miopen/softmarginloss.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t SoftMarginLossUnreducedForward(Handle& handle,
                                              const TensorDescriptor& iDesc,
                                              ConstData_t i,
                                              const TensorDescriptor& tDesc,
                                              ConstData_t t,
                                              const TensorDescriptor& oDesc,
                                              Data_t o)
{
    const auto problem = softmarginloss::ForwardProblemDescription{iDesc, tDesc, oDesc};

    const auto invoke_params = [&]() {
        auto tmp  = softmarginloss::InvokeParams{};
        tmp.type  = InvokeType::Run;
        tmp.iDesc = &iDesc;
        tmp.i     = i;
        tmp.tDesc = &tDesc;
        tmp.t     = t;
        tmp.oDesc = &oDesc;
        tmp.o     = o;
        return tmp;
    }();

    const auto algo = AlgorithmName{"SoftMarginLossUnreducedForward"};
    const auto solvers =
        solver::SolverContainer<solver::softmarginloss::SoftMarginLossUnreducedForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t SoftMarginLossUnreducedBackward(Handle& handle,
                                               const TensorDescriptor& iDesc,
                                               ConstData_t i,
                                               const TensorDescriptor& tDesc,
                                               ConstData_t t,
                                               const TensorDescriptor& dODesc,
                                               ConstData_t dO,
                                               const TensorDescriptor& dIDesc,
                                               Data_t dI)
{
    const auto problem = softmarginloss::BackwardProblemDescription{iDesc, tDesc, dODesc, dIDesc};

    const auto invoke_params = [&]() {
        auto tmp   = softmarginloss::InvokeParams{};
        tmp.type   = InvokeType::Run;
        tmp.iDesc  = &iDesc;
        tmp.i      = i;
        tmp.tDesc  = &tDesc;
        tmp.t      = t;
        tmp.dODesc = &dODesc;
        tmp.dO     = dO;
        tmp.dIDesc = &dIDesc;
        tmp.dI     = dI;
        return tmp;
    }();

    const auto algo = AlgorithmName{"SoftMarginLossUnreducedBackward"};
    const auto solvers =
        solver::SolverContainer<solver::softmarginloss::SoftMarginLossUnreducedBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
