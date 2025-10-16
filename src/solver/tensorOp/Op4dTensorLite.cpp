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

bool Op4dTensorLite::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                  const miopen::tensorOp::ProblemDescription& problem) const
{
    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    const auto& alens = aTensorDesc.GetLengths();
    const auto& blens = bTensorDesc.GetLengths();
    const auto& clens = cTensorDesc.GetLengths();

    auto asize = alens.size();

    if(asize == 4)
    {
        auto&& [num_wg, work_per_wg, bitmap] = GetBitmapAndWgInfo(blens, clens);

        // quick fix for btensor = <1, 1, 1, 1>
        if(bTensorDesc.GetElementSize() == 1)
            bitmap = 4;

        bool fwd_conv_bias = (bitmap == (1 << 2));

        bool packed_tensor = true;
        packed_tensor &= aTensorDesc.IsPacked();
        packed_tensor &= bTensorDesc.IsPacked();
        packed_tensor &= cTensorDesc.IsPacked();

        bool packed_equal_tensor =
            packed_tensor && (bTensorDesc.GetElementSize() == cTensorDesc.GetElementSize());

        if(!fwd_conv_bias && packed_equal_tensor)
        {
            return true;
        }
    }

    return false;
}

std::size_t Op4dTensorLite::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::tensorOp::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution Op4dTensorLite::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                         const miopen::tensorOp::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    miopenDataType_t data_type = bTensorDesc.GetType();

    auto&& [num_wg_orig, work_per_wg, incr_wg, bitmap, local_threads, global_threads] =
        Get4dParams(problem, true);

    auto&& [RD_BLCK, READ_TYPE] =
        GetRDBLCKandREADTYPEHIP(cTensorDesc.GetElementSize(), bTensorDesc.GetType());

    size_t total_work = std::max(cTensorDesc.GetElementSize() / RD_BLCK, size_t(1));

    const std::array<size_t, 3> vld{local_threads, 1, 1};
    const std::array<size_t, 3> vgd{global_threads, 1, 1};

    KernelBuildParameters build_params = KernelBuildParameters{};

    GetCommonParams(build_params, problem, false);

    build_params.Define("USE_4D_TENSOR_LITE");
    build_params.Define("RD_BLCK", std::to_string(RD_BLCK));
    build_params.Define("READ_TYPE", READ_TYPE);

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    kernel.kernel_file  = "MIOpenTensorKernelsHip.cpp";
    kernel.kernel_name  = "Op4dTensorLite";

    using std::begin, std::end;

    kernel.l_wk.insert(end(kernel.l_wk), begin(vld), end(vld));
    kernel.g_wk.insert(end(kernel.g_wk), begin(vgd), end(vgd));

    result.invoker_factory = [data_type, total_work](const std::vector<Kernel> kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::tensorOp::InvokeParams>();

            visit_float(data_type, [&](auto as_float) {
                auto miopen_alpha0 = as_float(*(static_cast<const float*>(params.alpha0)));
                auto miopen_alpha1 = as_float(*(static_cast<const float*>(params.alpha1)));
                auto miopen_beta   = as_float(*(static_cast<const float*>(params.beta)));

                kernel(params.ATensor,
                       params.BTensor,
                       params.CTensor,
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       static_cast<int64_t>(params.Aoffset),
                       static_cast<int64_t>(params.Boffset),
                       static_cast<int64_t>(params.Coffset),
                       static_cast<int64_t>(total_work),
                       static_cast<int>(!float_equal(miopen_beta, 0.0)));
            });
        };
    };
    result.construction_params.push_back(kernel);

    return result;
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
