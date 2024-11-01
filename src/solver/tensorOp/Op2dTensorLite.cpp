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
    auto aTensorDesc = problem.GetATensorDesc();
    auto bTensorDesc = problem.GetBTensorDesc();
    auto cTensorDesc = problem.GetCTensorDesc();

    auto alens = aTensorDesc.GetLengths();
    auto blens = bTensorDesc.GetLengths();
    auto clens = cTensorDesc.GetLengths();

    auto asize = alens.size();

    if(asize < 3)
    {
        return false;
    }

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

    if(asize == 3 && lite_applicable && is_lite)
    {
        return true;
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

    auto aTensorDesc = problem.GetATensorDesc();
    auto bTensorDesc = problem.GetBTensorDesc();
    auto cTensorDesc = problem.GetCTensorDesc();

    auto alens = aTensorDesc.GetLengths();
    auto blens = bTensorDesc.GetLengths();
    auto clens = cTensorDesc.GetLengths();

    auto astrides = aTensorDesc.GetStrides();
    auto bstrides = bTensorDesc.GetStrides();
    auto cstrides = cTensorDesc.GetStrides();

    // first_not_one is incorrect if btensor size equal to 1
    auto first_not_one = std::find_if(blens.rbegin(), blens.rend(), [](int i) { return i != 1; });
    auto d             = std::distance(blens.begin(), first_not_one.base());

    // quick fix
    int num_wg = first_not_one != blens.rend()
                     ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                     : 1;

    for(int i = (d - 2); i >= 0; i--)
    {
        if(blens[i] != 1)
        {
            num_wg *= blens[i];
        }
    }
    int max_num_wg = 4096;
    num_wg         = num_wg > max_num_wg ? max_num_wg : num_wg;

    size_t local_threads = 256;

    // for naive tensor ops
    size_t RD_BLCK              = (clens[2] % 4 == 0) ? 4 : (clens[2] % 2 == 0) ? 2 : 1;
    const std::string data_type = GetDataType(bTensorDesc.GetType());
    const std::string READ_TYPE = (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK);

    size_t total_work = std::max(clens[2] / RD_BLCK, size_t(1));
    size_t grp_sz     = (total_work + local_threads - 1) / local_threads;

    grp_sz        = std::min(size_t(max_num_wg), grp_sz);
    size_t glb_sz = local_threads * grp_sz;

    size_t local_threads2 = 64;
    size_t total_work2    = clens[1];
    size_t grp_sz2        = (total_work2 + local_threads2 - 1) / local_threads2;
    grp_sz2               = std::min(size_t(max_num_wg / grp_sz), grp_sz2);
    size_t glb_sz2        = local_threads2 * grp_sz2;

    const std::vector<size_t> vld{local_threads, 1, 1};
    const std::vector<size_t> vgd{glb_sz, glb_sz2, 1};

    KernelBuildParameters build_params =
        KernelBuildParameters{{"MIOPEN_TYPE", GetDataType(bTensorDesc.GetType())}};

    switch(problem.GetTensorOp())
    {
    case 0: build_params.Define("MIOPEN_TENSOR_OP", "miopenAdd"); break;
    case 1: build_params.Define("MIOPEN_TENSOR_OP", "miopenMul"); break;
    case 2: build_params.Define("MIOPEN_TENSOR_OP", "miopenMin"); break;
    case 3: build_params.Define("MIOPEN_TENSOR_OP", "miopenMax"); break;
    }

    build_params.Define("USE_2D_TENSOR_LITE");
    build_params.Define("RD_BLCK", std::to_string(RD_BLCK));
    build_params.Define("READ_TYPE", READ_TYPE);

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    kernel.kernel_file  = "MIOpenTensorKernels.cl";
    kernel.kernel_name  = "Op2dTensorLite";

    for(uint32_t i = 0; i <= 2; i++)
    {
        kernel.l_wk.push_back(vld[i]);
        kernel.g_wk.push_back(vgd[i]);
    }

    result.invoker_factory = [=](const std::vector<Kernel> kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::tensorOp::InvokeParams>();

            visit_float(bTensorDesc.GetType(), [&](auto as_float) {
                auto miopen_alpha0 = as_float(*(static_cast<const float*>(params.alpha0)));
                auto miopen_alpha1 = as_float(*(static_cast<const float*>(params.alpha1)));
                auto miopen_beta   = as_float(*(static_cast<const float*>(params.beta)));

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
