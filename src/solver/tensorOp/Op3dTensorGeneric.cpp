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

bool Op3dTensorGeneric::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::tensorOp::ProblemDescription& problem) const
{
    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& alens       = aTensorDesc.GetLengths();
    auto asize              = alens.size();

    if(asize == 3)
    {
        return true;
    }

    return false;
}

std::size_t Op3dTensorGeneric::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::tensorOp::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution
Op3dTensorGeneric::GetSolution([[maybe_unused]] const ExecutionContext& context,
                               const miopen::tensorOp::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    const auto& blens = bTensorDesc.GetLengths();
    const auto& clens = cTensorDesc.GetLengths();

    std::array<size_t, 3> astrides;
    std::array<size_t, 3> bstrides;
    std::array<size_t, 3> cstrides;
    std::tie(astrides[0], astrides[1], astrides[2]) = miopen::tien<3>(aTensorDesc.GetStrides());
    std::tie(bstrides[0], bstrides[1], bstrides[2]) = miopen::tien<3>(bTensorDesc.GetStrides());
    std::tie(cstrides[0], cstrides[1], cstrides[2]) = miopen::tien<3>(cTensorDesc.GetStrides());

    miopenDataType_t data_type = bTensorDesc.GetType();

    size_t local_threads = 32;
    size_t max_num_wg    = 4096;

    auto num_wg =
        std::clamp((clens[0] * clens[1] * clens[2]) / local_threads, size_t(1), size_t(max_num_wg));
    size_t global_threads = num_wg * local_threads;

    const std::array<size_t, 3> vld{local_threads, 1, 1};
    const std::array<size_t, 3> vgd{global_threads, 1, 1};

    KernelBuildParameters build_params = KernelBuildParameters{};

    GetCommonParams(build_params, problem, false);

    build_params.Define("USE_3D_TENSOR_GENERIC");

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    kernel.kernel_file  = "MIOpenTensorKernelsHip.cpp";
    kernel.kernel_name  = "Op3dTensorGeneric";

    using std::begin, std::end;

    kernel.l_wk.insert(end(kernel.l_wk), begin(vld), end(vld));
    kernel.g_wk.insert(end(kernel.g_wk), begin(vgd), end(vgd));

    result.invoker_factory =
        [data_type, blens, clens, astrides, bstrides, cstrides](const std::vector<Kernel> kernels) {
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
                           static_cast<uint64_t>(params.Aoffset),
                           static_cast<uint64_t>(params.Boffset),
                           static_cast<uint64_t>(params.Coffset),
                           static_cast<uint32_t>(blens[1] == 1 ? clens[1] : blens[1]), // b_c,
                           static_cast<uint32_t>(blens[2] == 1 ? clens[2] : blens[2]), // b_h,
                           static_cast<uint32_t>(clens[1]),                            // c_c,
                           static_cast<uint32_t>(clens[2]),                            // c_h,
                           static_cast<uint32_t>(astrides[0]),                         // a_nstride,
                           static_cast<uint32_t>(astrides[1]),                         // a_cstride,
                           static_cast<uint32_t>(astrides[2]),                         // a_hstride,
                           static_cast<uint32_t>(blens[0] == 1 ? 0 : bstrides[0]),     // b_nstride,
                           static_cast<uint32_t>(blens[1] == 1 ? 0 : bstrides[1]),     // b_cstride,
                           static_cast<uint32_t>(blens[2] == 1 ? 0 : bstrides[2]),     // b_hstride,
                           static_cast<uint32_t>(cstrides[0]),                         // c_nstride,
                           static_cast<uint32_t>(cstrides[1]),                         // c_cstride,
                           static_cast<uint32_t>(cstrides[2]),                         // c_hstride,
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           static_cast<uint32_t>(clens[0]),
                           !float_equal(miopen_beta, 0.0));
                });
            };
        };
    result.construction_params.push_back(kernel);

    return result;
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
