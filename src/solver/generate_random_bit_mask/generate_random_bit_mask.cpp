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

#include "miopen/mlo_internal.hpp"
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

#define LOCAL_SIZE 256
#define VIEW_DIMS 5

namespace miopen {
namespace solver {
namespace generate_random_bit_mask {

bool GenerateRandomBitMask::IsApplicable(
    const ExecutionContext& context,
    const miopen::generate_random_bit_mask::ProblemDescription& problem) const
{
    return true;
}

ConvSolution GenerateRandomBitMask::GetSolution(
    const ExecutionContext& context,
    const miopen::generate_random_bit_mask::ProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto p             = problem.GetProb();
    auto bitmask_numel = problem.GetMaskDesc().GetElementSize();

    size_t RD_BLCK = 8;

    if(p == 0 || p == 1)
    {
        const size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize        = AlignUp(bitmask_numel, xlocalsize);
        size_t ylocalsize       = 1;
        size_t ygridsize        = 1;
        size_t zlocalsize       = 1;
        size_t zgridsize        = 1;

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenFill.cpp";
        kernel.kernel_name = "FillValContiguous";

        auto build_params = KernelBuildParameters{{"IO_TYPE", "uchar"}, {"VIEW_DIMS", VIEW_DIMS}};

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }
    else
    {
        const std::string READ_DAT_TYPE =
            RD_BLCK == 1 ? "float" : "float" + std::to_string(RD_BLCK);
        size_t wk_grp_num =
            std::min(size_t(MAX_PRNG_STATE) / LOCAL_SIZE, ((bitmask_numel + 255) / 256));

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = wk_grp_num * xlocalsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenGenerateRandomBitMask.cpp";
        kernel.kernel_name = "GenerateRandomBitMask";

        const auto build_params = KernelBuildParameters{

            {"VEC_SIZE", RD_BLCK}};

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    // Start building result.invoker_factory
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::generate_random_bit_mask::InvokeParams>();

            auto p = params.p;

            if(p == 0 || p == 1)
            {
                // Launch fill kernel
                auto val = p == 0 ? static_cast<uchar>(255) : static_cast<uchar>(0);
                kernel(params.mask, val, bitmask_numel);
            }
            else
            {
                kernel(params.pstates, params.mask, bitmask_numel, p);
            }
        };
    };

    return result;
}

} // namespace generate_random_bit_mask
} // namespace solver
} // namespace miopen
