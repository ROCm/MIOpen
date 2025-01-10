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

bool Op5dTensorGeneric::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::tensorOp::ProblemDescription& problem) const
{
    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& alens       = aTensorDesc.GetLengths();
    auto asize              = alens.size();

    if(asize == 5)
    {
        return true;
    }

    return false;
}

std::size_t Op5dTensorGeneric::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::tensorOp::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution
Op5dTensorGeneric::GetSolution([[maybe_unused]] const ExecutionContext& context,
                               const miopen::tensorOp::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    const auto& blens = bTensorDesc.GetLengths();
    const auto& clens = cTensorDesc.GetLengths();

    std::array<size_t, 5> astrides;
    std::array<size_t, 5> bstrides;
    std::array<size_t, 5> cstrides;
    std::tie(astrides[0], astrides[1], astrides[2], astrides[3], astrides[4]) =
        miopen::tien<5>(aTensorDesc.GetStrides());
    std::tie(bstrides[0], bstrides[1], bstrides[2], bstrides[3], bstrides[4]) =
        miopen::tien<5>(bTensorDesc.GetStrides());
    std::tie(cstrides[0], cstrides[1], cstrides[2], cstrides[3], cstrides[4]) =
        miopen::tien<5>(cTensorDesc.GetStrides());

    miopenDataType_t data_type = bTensorDesc.GetType();

    auto&& [num_wg, work_per_wg, bitmap] = GetBitmapAndWgInfo(blens, clens);

    int num_wg_orig = num_wg;
    int max_num_wg  = 4096;
    num_wg          = num_wg > max_num_wg ? max_num_wg : num_wg;

    size_t local_threads  = 256;
    size_t global_threads = num_wg * local_threads;

    const std::array<size_t, 3> vld{local_threads, 1, 1};
    const std::array<size_t, 3> vgd{global_threads, 1, 1};

    KernelBuildParameters build_params = KernelBuildParameters{};

    GetCommonParams(build_params, problem, false);

    build_params.Define("USE_5D_TENSOR_GENERIC");
    build_params.Define("MAX_NUM_WG", std::to_string(max_num_wg));

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});
    kernel.kernel_file  = "MIOpenTensorKernels.cl";
    kernel.kernel_name  = "Op5dTensorGeneric";

    using std::begin, std::end;

    kernel.l_wk.insert(end(kernel.l_wk), begin(vld), end(vld));
    kernel.g_wk.insert(end(kernel.g_wk), begin(vgd), end(vgd));

    result.invoker_factory =
        [data_type, blens, clens, astrides, bstrides, cstrides, bitmap, work_per_wg, num_wg_orig](
            const std::vector<Kernel> kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::tensorOp::InvokeParams>();

                visit_float(data_type, [&](auto as_float) {
                    auto miopen_alpha0 = as_float(*(static_cast<const float*>(params.alpha0)));
                    auto miopen_alpha1 = as_float(*(static_cast<const float*>(params.alpha1)));
                    auto miopen_beta   = as_float(*(static_cast<const float*>(params.beta)));

                    kernel(params.ATensor,
                           static_cast<int>(astrides[0]),
                           static_cast<int>(astrides[1]),
                           static_cast<int>(astrides[2]),
                           static_cast<int>(astrides[3]),
                           params.BTensor,
                           static_cast<int>(blens[1]),    // b_c,
                           static_cast<int>(blens[2]),    // b_d,
                           static_cast<int>(blens[3]),    // b_h,
                           static_cast<int>(blens[4]),    // b_w,
                           static_cast<int>(bstrides[0]), // b_nstride,
                           static_cast<int>(bstrides[1]), // b_cstride,
                           static_cast<int>(bstrides[2]), // b_dstride,
                           static_cast<int>(bstrides[3]), // b_hstride,
                           params.CTensor,
                           static_cast<int>(clens[1]),    // c_c,
                           static_cast<int>(clens[2]),    // c_d,
                           static_cast<int>(clens[3]),    // c_h,
                           static_cast<int>(clens[4]),    // c_w,
                           static_cast<int>(cstrides[0]), // c_nstride,
                           static_cast<int>(cstrides[1]), // c_cstride,
                           static_cast<int>(cstrides[2]), // c_dstride,
                           static_cast<int>(cstrides[3]), // c_hstride,
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           bitmap,
                           work_per_wg,
                           static_cast<int64_t>(params.Aoffset),
                           static_cast<int64_t>(params.Boffset),
                           static_cast<int64_t>(params.Coffset),
                           static_cast<int>(num_wg_orig));
                });
            };
        };
    result.construction_params.push_back(kernel);

    return result;
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
