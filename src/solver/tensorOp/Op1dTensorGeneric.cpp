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

bool Op1dTensorGeneric::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::tensorOp::ProblemDescription& problem) const
{
    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& alens       = aTensorDesc.GetLengths();
    auto asize              = alens.size();

    if(asize == 1)
    {
        return true;
    }

    return false;
}

std::size_t Op1dTensorGeneric::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::tensorOp::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution
Op1dTensorGeneric::GetSolution([[maybe_unused]] const ExecutionContext& context,
                               const miopen::tensorOp::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    const size_t b_n = bTensorDesc.GetLengths()[0];
    const size_t c_n = cTensorDesc.GetLengths()[0];

    const size_t a_nstrides = aTensorDesc.GetStrides()[0];
    const size_t b_nstrides = bTensorDesc.GetStrides()[0];
    const size_t c_nstrides = cTensorDesc.GetStrides()[0];

    miopenDataType_t data_type = bTensorDesc.GetType();
    bool fit_into_int          = aTensorDesc.AllDimsFitIntoInt();

    size_t local_threads = 256;
    size_t max_num_wg    = 4096;

    auto num_wg           = std::clamp(c_n / local_threads, size_t(1), size_t(max_num_wg));
    size_t global_threads = num_wg * local_threads;

    const std::array<size_t, 3> vld{local_threads, 1, 1};
    const std::array<size_t, 3> vgd{global_threads, 1, 1};

    KernelBuildParameters build_params = KernelBuildParameters{};

    GetCommonParams(build_params, problem, true);

    build_params.Define("USE_1D_TENSOR_GENERIC");

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    kernel.kernel_file  = "MIOpenTensorKernelsHip.cpp";
    kernel.kernel_name  = "Op1dTensorGeneric";

    using std::begin, std::end;

    kernel.l_wk.insert(end(kernel.l_wk), begin(vld), end(vld));
    kernel.g_wk.insert(end(kernel.g_wk), begin(vgd), end(vgd));

    result.invoker_factory =
        [data_type, fit_into_int, b_n, c_n, a_nstrides, b_nstrides, c_nstrides](
            const std::vector<Kernel> kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::tensorOp::InvokeParams>();

                visit_float(data_type, [&](auto as_float) {
                    auto miopen_alpha0 = as_float(*(static_cast<const float*>(params.alpha0)));
                    auto miopen_alpha1 = as_float(*(static_cast<const float*>(params.alpha1)));
                    auto miopen_beta   = as_float(*(static_cast<const float*>(params.beta)));

                    if(fit_into_int)
                    {
                        kernel(params.ATensor,
                               params.BTensor,
                               params.CTensor,
                               static_cast<uint64_t>(params.Aoffset),
                               static_cast<uint64_t>(params.Boffset),
                               static_cast<uint64_t>(params.Coffset),
                               static_cast<uint32_t>(a_nstrides),
                               static_cast<uint32_t>(b_n == 1 ? 0 : b_nstrides),
                               static_cast<uint32_t>(c_nstrides),
                               miopen_alpha0,
                               miopen_alpha1,
                               miopen_beta,
                               static_cast<uint32_t>(c_n),
                               !float_equal(miopen_beta, 0.0));
                    }
                    else
                    {
                        kernel(params.ATensor,
                               params.BTensor,
                               params.CTensor,
                               static_cast<uint64_t>(params.Aoffset),
                               static_cast<uint64_t>(params.Boffset),
                               static_cast<uint64_t>(params.Coffset),
                               static_cast<uint64_t>(a_nstrides),
                               static_cast<uint64_t>(b_n == 1 ? 0 : b_nstrides),
                               static_cast<uint64_t>(c_nstrides),
                               miopen_alpha0,
                               miopen_alpha1,
                               miopen_beta,
                               static_cast<uint64_t>(c_n),
                               !float_equal(miopen_beta, 0.0));
                    }
                });
            };
        };
    result.construction_params.push_back(kernel);

    return result;
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
