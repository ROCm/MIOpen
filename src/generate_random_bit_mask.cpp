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

#include <miopen/common.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/generate_random_bit_mask.hpp>
#include <miopen/generate_random_bit_mask/invoke_params.hpp>
#include <miopen/generate_random_bit_mask/solvers.hpp>
#include <miopen/generate_random_bit_mask/problem_description.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace generate_random_bit_mask {

miopenStatus_t
InitPRNGState(Handle& handle, Data_t pstate, const size_t stateSizeInBytes, const uint64_t seed)
{
    const auto problem =
        generate_random_bit_mask::InitPRNGStateProblemDescription{stateSizeInBytes};

    const auto invoke_params = [&]() {
        auto tmp = miopen::generate_random_bit_mask::PStateInvokeParams{};

        tmp.pstate           = pstate;
        tmp.stateSizeInBytes = stateSizeInBytes;
        tmp.seed             = seed;

        return tmp;
    }();

    const auto algo   = AlgorithmName{"InitPRNGState"};
    const auto solver = solver::SolverContainer<solver::generate_random_bit_mask::InitPRNGState>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t GenerateRandomBitMask(Handle& handle,
                                     ConstData_t pstate,
                                     const size_t stateSizeInBytes,
                                     const TensorDescriptor& maskDesc,
                                     Data_t mask,
                                     const float p)
{
    const auto problem =
        generate_random_bit_mask::ProblemDescription{stateSizeInBytes, maskDesc, p};

    const auto invoke_params = [&]() {
        auto tmp     = miopen::generate_random_bit_mask::InvokeParams{};
        tmp.maskDesc = &maskDesc;

        tmp.pstates = pstate;
        tmp.mask    = mask;

        tmp.stateSizeInBytes = stateSizeInBytes;

        tmp.p = p;

        return tmp;
    }();

    const auto algo = AlgorithmName{"GenerateRandomBitMask"};
    const auto solver =
        solver::SolverContainer<solver::generate_random_bit_mask::GenerateRandomBitMask>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace generate_random_bit_mask
} // namespace miopen
