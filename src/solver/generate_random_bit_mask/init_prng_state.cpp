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

#include <miopen/mlo_internal.hpp>
#include <miopen/buffer_info.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/datatype.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/miopen.h>
#include <miopen/kernel_build_params.hpp>
#include <miopen/generate_random_bit_mask.hpp>
#include <miopen/generate_random_bit_mask/solvers.hpp>
#include <miopen/generate_random_bit_mask/invoke_params.hpp>
#include <miopen/generate_random_bit_mask/problem_description.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <rocrand/rocrand_uniform.h>

#include <algorithm>

#define LOCAL_SIZE 256

namespace miopen {
namespace solver {
namespace generate_random_bit_mask {

bool InitPRNGState::IsApplicable(
    const ExecutionContext& context,
    const miopen::generate_random_bit_mask::InitPRNGStateProblemDescription& problem) const
{
    auto& handle = context.GetStream();

    if(problem.GetStateSizeInBytes() > handle.GetMaxMemoryAllocSize())
    {
        MIOPEN_THROW("PRNG state size should not exceed system maximum memory allocation size.");
    }

    return true;
}

ConvSolution InitPRNGState::GetSolution(
    const ExecutionContext& context,
    const miopen::generate_random_bit_mask::InitPRNGStateProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto prng_stateSizeInBytes = problem.GetStateSizeInBytes();

    auto states_num = prng_stateSizeInBytes / sizeof(rocrand_state_xorwow);

    size_t wk_grp_num =
        std::min(static_cast<size_t>(MAX_PRNG_STATE) / 256, (states_num + 255) / 256);

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = wk_grp_num * xlocalsize;
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenInitPState.cpp";
    kernel.kernel_name = "InitKernelStateHIP";

    kernel.comp_options = KernelBuildParameters{}.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    // Start building result.invoker_factory
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::generate_random_bit_mask::PStateInvokeParams>();

            kernel(params.pstate, params.seed, states_num);
        };
    };

    return result;
}

} // namespace generate_random_bit_mask
} // namespace solver
} // namespace miopen
