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

#include <miopen/gemm/solvers.hpp>

#include <miopen/gemm/invoke_params.hpp>
#include <miopen/gemm/problem_description.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/gemm_rocblas.hpp>

namespace miopen {

namespace solver {

namespace gemm {

bool GemmSolver1x1::IsApplicable(const ExecutionContext&,
                               const miopen::gemm::ProblemDescription& problem) const
{
    const auto ADesc = problem.GetADesc();
    const auto BDesc = problem.GetBDesc();

    const auto A_elem_sz = ADesc.GetElementSize();
    const auto B_elem_sz = BDesc.GetElementSize();

    if(A_elem_sz != B_elem_sz)
        return false;
    
    return true;
}

ConvSolution GemmSolver1x1::GetSolution(const ExecutionContext&,
                                        const miopen::gemm::ProblemDescription& problem) const
{
    auto solution = ConvSolution{miopenStatusSuccess};

    //decltype(auto) ADesc = problem.GetADesc();
    //decltype(auto) BDesc = problem.GetBDesc();
    //decltype(auto) CDesc = problem.GetCDesc();

    GemmNewDescriptor gemm_desc = problem.GetGemmDescriptor();

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        MIOPEN_LOG_FUNCTION("gemm");

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            decltype(auto) gemm_params = primitive_params.CastTo<miopen::gemm::InvokeParams>();
            const auto& B              = gemm_params.B;
            const auto& A              = gemm_params.A;
            const auto& C              = gemm_params.C;

            MIOPEN_LOG_FUNCTION("gemm");

            miopenStatus_t gemm_status;
            if(gemm_params.type == InvokeType::Run)
            {
                // kernel calls here
                // call rocblas (temp solution)
                gemm_status = CallGemmRocblas(handle,
                                            gemm_desc,
                                            A,
                                            0,
                                            B,
                                            0,
                                            C,
                                            0);               
            }

            if(gemm_status != miopenStatusSuccess)
                MIOPEN_THROW("GEMM execution failure");
        };
    };

    return solution;
}

} // namespace gemm

} // namespace solver

} // namespace miopen
