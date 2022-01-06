/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#include <miopen/gemm.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/gemm/invoke_params.hpp>
#include <miopen/gemm/problem_description.hpp>
#include <miopen/gemm/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t GemmNewDescriptor::CallGemm(Handle& handle,
                                            const TensorDescriptor& ADesc,
                                            ConstData_t A,
                                            const TensorDescriptor& BDesc,
                                            ConstData_t B,
                                            const TensorDescriptor& CDesc,
                                            Data_t C)
{
    const auto problem = gemm::ProblemDescription{*this, ADesc, BDesc, CDesc};

    const auto invoke_params = [&]() {
        auto tmp = gemm::InvokeParams{};
        tmp.type = InvokeType::Run;
        tmp.A    = A;
        tmp.B    = B;
        tmp.C    = C;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"miopenGemm"};
    const auto solvers = solver::SolverContainer<solver::gemm::GemmSolver0>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}
} // namespace miopen
