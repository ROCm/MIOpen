/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/tensor/solvers.hpp>

#include <miopen/tensor/invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/datatype.hpp>

namespace miopen {

namespace solver {

namespace tensor {

bool Op1dTensorGeneric::IsApplicable(const ExecutionContext& context,
                                     const miopen::tensor::ProblemDescription& problem) const
{
    auto aTensorDesc = problem.GetATensorDesc();
    auto bTensorDesc = problem.GetBTensorDesc();
    auto alens       = aTensorDesc.GetLengths();
    auto blens       = bTensorDesc.GetLengths();
    auto asize       = alens.size();

    if(asize == 1)
    {
        return true;
    }
    if(asize == 2 && ((blens[0] == 1 && blens[1] == 1) || (blens[0] > 1 && blens[1] > 1)))
    {
        return true;
    }
    if(asize == 3 && ((blens[0] == 1 && blens[1] == 1 && blens[2] == 1) ||
                      (blens[0] > 1 && blens[1] > 1 && blens[2] > 1)))
    {
        return true;
    }
    return false;
}

std::size_t
Op1dTensorGeneric::GetWorkspaceSize(const ExecutionContext& context,
                                    const miopen::tensor::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution Op1dTensorGeneric::GetSolution(const ExecutionContext& context,
                                            const miopen::tensor::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto aTensorDesc = problem.GetATensorDesc();
    auto bTensorDesc = problem.GetBTensorDesc();
    auto cTensorDesc = problem.GetCTensorDesc();

    auto clens = cTensorDesc.GetLengths();

    size_t local_threads = 256;
    size_t max_num_wg    = 4096;

    auto num_wg           = std::clamp(clens[0] / local_threads, size_t(1), size_t(max_num_wg));
    num_wg                = num_wg > max_num_wg ? max_num_wg : num_wg;
    size_t global_threads = num_wg * local_threads;

    const std::vector<size_t> vld{local_threads, 1, 1};
    const std::vector<size_t> vgd{global_threads, 1, 1};

    KernelBuildParameters build_params =
        KernelBuildParameters{{"MIOPEN_TYPE", GetDataType(bTensorDesc.GetType())}};

    // build_params.Define("MIOPEN_TENSOR_OP", std::to_string(problem.GetTensorOp()));

    switch(problem.GetTensorOp())
    {
    case 0: build_params.Define("MIOPEN_TENSOR_OP", "miopenAdd"); break;
    case 1: build_params.Define("MIOPEN_TENSOR_OP", "miopenMul"); break;
    case 2: build_params.Define("MIOPEN_TENSOR_OP", "miopenMin"); break;
    case 3: build_params.Define("MIOPEN_TENSOR_OP", "miopenMax"); break;
    }

    if(aTensorDesc.AllDimsFitIntoInt())
    {
        build_params.Define("DIM_TYPE", "uint32_t");
    }
    else
    {
        build_params.Define("DIM_TYPE", "uint64_t");
    }

    build_params.Define("USE_1D_TENSOR_GENERIC");

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(
        kbp::HIP{}); // GetDataTypeKBP(aTensorDesc.GetType()).GenerateFor(kbp::HIP{});
    kernel.kernel_file = "MIOpenTensorKernelsHip.cpp";
    kernel.kernel_name = "Op1dTensorGeneric";

    for(uint32_t i = 0; i <= 2; i++)
    {
        kernel.l_wk.push_back(vld[i]);
        kernel.g_wk.push_back(vgd[i]);
    }

    result.invoker_factory = [=](const std::vector<Kernel> kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::tensor::InvokeParams>();

            visit_float(bTensorDesc.GetType(), [&](auto as_float) {
                auto miopen_alpha0 = as_float(*(static_cast<const float*>(params.alpha0)));
                auto miopen_alpha1 = as_float(*(static_cast<const float*>(params.alpha1)));
                auto miopen_beta   = as_float(*(static_cast<const float*>(params.beta)));

                auto blens = params.bTensorDesc.GetLengths();
                auto clens = params.cTensorDesc.GetLengths();

                auto astrides = params.aTensorDesc.GetStrides();
                auto bstrides = params.bTensorDesc.GetStrides();
                auto cstrides = params.cTensorDesc.GetStrides();

                if(aTensorDesc.AllDimsFitIntoInt())
                { // change offsets to 64bit after PR is merged
                    kernel(params.ATensor,
                           params.BTensor,
                           params.CTensor,
                           static_cast<uint32_t>(params.Aoffset),
                           static_cast<uint32_t>(params.Boffset),
                           static_cast<uint32_t>(params.Coffset),
                           static_cast<uint32_t>(astrides[0]),
                           static_cast<uint32_t>(blens[0] == 1 ? 0 : bstrides[0]),
                           static_cast<uint32_t>(cstrides[0]),
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           static_cast<uint32_t>(clens[0]),
                           !float_equal(miopen_beta, 0.0));
                }
                else
                {
                    kernel(params.ATensor,
                           params.BTensor,
                           params.CTensor,
                           static_cast<uint32_t>(params.Aoffset),
                           static_cast<uint32_t>(params.Boffset),
                           static_cast<uint32_t>(params.Coffset),
                           static_cast<uint64_t>(astrides[0]),
                           static_cast<uint64_t>(blens[0] == 1 ? 0 : bstrides[0]),
                           static_cast<uint64_t>(cstrides[0]),
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           static_cast<uint64_t>(clens[0]),
                           !float_equal(miopen_beta, 0.0));
                }
            });
        };
    };
    result.construction_params.push_back(kernel);

    return result;
}

} // namespace tensor

} // namespace solver

} // namespace miopen
