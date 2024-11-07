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
#include "tensor_op_helpers.hpp"

#include <miopen/tensorOp/solvers.hpp>
#include <miopen/tensorOp/invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/datatype.hpp>

namespace miopen {

namespace solver {

namespace tensorOp {

bool Op2dTensorLite::IsApplicable(const ExecutionContext& context,
                                  const miopen::tensorOp::ProblemDescription& problem) const
{
    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    const auto& alens = aTensorDesc.GetLengths();
    const auto& blens = bTensorDesc.GetLengths();
    const auto& clens = cTensorDesc.GetLengths();

    auto asize = alens.size();

    if(aTensorDesc.GetType() == miopenDouble)
    {
        return false;
    }

    if(asize == 3)
    {
        size_t local_threads = 256;
        int max_num_wg       = 4096;

        // for naive tensor ops
        size_t RD_BLCK    = (clens[2] % 4 == 0) ? 4 : (clens[2] % 2 == 0) ? 2 : 1;
        size_t total_work = std::max(clens[2] / RD_BLCK, size_t(1));
        size_t grp_sz     = (total_work + local_threads - 1) / local_threads;

        // opencl kernels are no longer supported, fallback to generic case
        bool lite_applicable = grp_sz <= size_t(max_num_wg);

        bool is_lite = clens[0] == 1 && blens[0] == 1 && alens[0] == 1 &&
                       (blens[1] == clens[1] || blens[1] == 1) && blens[2] == clens[2];

        if(lite_applicable && is_lite)
        {
            return true;
        }
    }

    return false;
}

std::size_t
Op2dTensorLite::GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::tensorOp::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution Op2dTensorLite::GetSolution(const ExecutionContext& context,
                                         const miopen::tensorOp::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    const auto& blens = bTensorDesc.GetLengths();
    const auto& clens = cTensorDesc.GetLengths();

    auto&& [num_wg, work_per_wg, bitmap] = GetBitmapAndWgInfo(blens, clens);

    int max_num_wg = 4096;
    num_wg         = num_wg > max_num_wg ? max_num_wg : num_wg;

    size_t local_threads = 256;

    // for naive tensor ops
    auto&& [RD_BLCK, READ_TYPE] = GetRDBLCKandREADTYPE(clens[2], bTensorDesc.GetType());

    size_t total_work = std::max(clens[2] / RD_BLCK, size_t(1));
    size_t grp_sz     = (total_work + local_threads - 1) / local_threads;

    grp_sz        = std::min(size_t(max_num_wg), grp_sz);
    size_t glb_sz = local_threads * grp_sz;

    size_t local_threads2 = 64;
    size_t total_work2    = clens[1];
    size_t grp_sz2        = (total_work2 + local_threads2 - 1) / local_threads2;
    grp_sz2               = std::min(size_t(max_num_wg / grp_sz), grp_sz2);
    size_t glb_sz2        = local_threads2 * grp_sz2;

    const std::array<size_t, 3> vld{local_threads, 1, 1};
    const std::array<size_t, 3> vgd{glb_sz, glb_sz2, 1};

    KernelBuildParameters build_params = KernelBuildParameters{};

    GetCommonParams(build_params, problem, false);

    build_params.Define("USE_2D_TENSOR_LITE");
    build_params.Define("RD_BLCK", std::to_string(RD_BLCK));
    build_params.Define("READ_TYPE", READ_TYPE);

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});
    kernel.kernel_file  = "MIOpenTensorKernels.cl";
    kernel.kernel_name  = "Op2dTensorLite";

    using std::begin, std::end;

    kernel.l_wk.insert(end(kernel.l_wk), begin(vld), end(vld));
    kernel.g_wk.insert(end(kernel.g_wk), begin(vgd), end(vgd));

    result.invoker_factory = [total_work, total_work2](const std::vector<Kernel> kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::tensorOp::InvokeParams>();

            visit_float(params.bTensorDesc.GetType(), [&](auto as_float) {
                auto miopen_alpha0 = as_float(*(static_cast<const float*>(params.alpha0)));
                auto miopen_alpha1 = as_float(*(static_cast<const float*>(params.alpha1)));
                auto miopen_beta   = as_float(*(static_cast<const float*>(params.beta)));

                const auto& blens = params.bTensorDesc.GetLengths();

                const auto& astrides = params.aTensorDesc.GetStrides();
                const auto& bstrides = params.bTensorDesc.GetStrides();
                const auto& cstrides = params.cTensorDesc.GetStrides();

                kernel(params.ATensor,
                       static_cast<int>(astrides[1]),
                       params.BTensor,
                       static_cast<int>(bstrides[1]),
                       params.CTensor,
                       static_cast<int>(cstrides[1]),
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       static_cast<int64_t>(params.Aoffset),
                       static_cast<int64_t>(params.Boffset),
                       static_cast<int64_t>(params.Coffset),
                       static_cast<int64_t>(total_work),
                       static_cast<int64_t>(total_work2),
                       static_cast<int>(!float_equal(miopen_beta, 0.0)),
                       static_cast<int>(blens[1] == 1));
            });
        };
    };
    result.construction_params.push_back(kernel);

    return result;
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
