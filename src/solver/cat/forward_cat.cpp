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

#include <miopen/cat/solvers.hpp>

#include <miopen/cat/cat_invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/cat.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

namespace miopen {

namespace solver {

namespace cat {

bool IsUnderXCountLimit(const miopen::cat::ProblemDescription& problem)
{
    constexpr int32_t max_tensor_x_count = 8;
    if(problem.GetXCount() > max_tensor_x_count)
    {
        return false;
    }
    return true;
}

bool IsImprovementOverROCm(const miopen::cat::ProblemDescription& problem)
{
    constexpr size_t min_output_tensor_size = 1000000;
    if(problem.GetYDesc().GetElementSize() < min_output_tensor_size)
    {
        return false;
    }

    return true;
}

bool CatForward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                              const miopen::cat::ProblemDescription& problem) const
{
    if(!IsUnderXCountLimit(problem))
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!IsImprovementOverROCm(problem))
        return false;
    return true;
}

inline size_t AlignUp(size_t num, size_t align) { return (num + align - 1) / align * align; }

ConvSolution CatForward::GetSolution(const ExecutionContext& context,
                                     const miopen::cat::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype  = problem.GetYDesc().GetType();
    auto ydims  = problem.GetYDesc().GetLengths();
    auto dim    = problem.GetDim();
    auto stride = problem.GetYDesc().GetStrides()[dim];
    auto xCount = problem.GetXCount();

    size_t x_dim_size_max = 0;

    for(int i = 0; i < xCount; i++)
    {
        auto& xdims    = problem.GetXDesc(i).GetLengths();
        x_dim_size_max = std::max(x_dim_size_max, xdims[dim]);
    }

    auto& handle = context.GetStream();
    auto numCu   = handle.GetMaxComputeUnits();

    auto outer_size =
        std::accumulate(ydims.begin(), ydims.begin() + dim, 1ULL, std::multiplies<size_t>());

    constexpr size_t local_size = 192;

    auto data_size = get_data_size(dtype);
    size_t max_inner_size =
        (x_dim_size_max * stride * data_size + sizeof(short4) - 1) / sizeof(short4);

    size_t xlocalsize = std::min(max_inner_size, local_size);
    size_t ylocalsize = std::max(static_cast<int>(local_size / xlocalsize), 1);
    size_t zlocalsize = 1;
    size_t ygridsize  = AlignUp(outer_size, ylocalsize);
    size_t xgridsize =
        std::max(static_cast<int>(numCu * 8 / (ygridsize / ylocalsize)), 1) * xlocalsize;
    xgridsize        = std::min(xgridsize, AlignUp(max_inner_size, xlocalsize));
    size_t zgridsize = 1;

    KernelBuildParameters build_params;

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    kernel.kernel_file = "MIOpenCat.cpp";

    int fusion_size = 2;
    while(fusion_size < xCount)
    {
        fusion_size *= 2;
    }

    switch(fusion_size)
    {
    case 2:
        kernel.kernel_name     = "Cat2FwdPacked";
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::cat::CatInvokeParams>();

                auto ydims      = params.yDesc.GetLengths();
                auto dim        = params.dim;
                auto stride     = params.yDesc.GetStrides()[dim];
                auto y_dim_size = ydims[dim];
                auto outer_size = std::accumulate(
                    ydims.begin(), ydims.begin() + dim, 1ULL, std::multiplies<size_t>());
                auto data_size = get_data_size(params.yDesc.GetType());

                kernel(params.GetX(0),
                       params.GetX(1),
                       params.y,
                       params.GetXDimSize(0),
                       params.GetXDimSize(1),
                       outer_size,
                       stride * data_size,
                       y_dim_size);
            };
        };
        break;
    case 4:
        kernel.kernel_name     = "Cat4FwdPacked";
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::cat::CatInvokeParams>();

                auto ydims      = params.yDesc.GetLengths();
                auto dim        = params.dim;
                auto stride     = params.yDesc.GetStrides()[dim];
                auto y_dim_size = ydims[dim];
                auto outer_size = std::accumulate(
                    ydims.begin(), ydims.begin() + dim, 1ULL, std::multiplies<size_t>());
                auto data_size = get_data_size(params.yDesc.GetType());

                kernel(params.GetX(0),
                       params.GetX(1),
                       params.GetX(2),
                       params.GetX(3),
                       params.y,
                       params.GetXDimSize(0),
                       params.GetXDimSize(1),
                       params.GetXDimSize(2),
                       params.GetXDimSize(3),
                       outer_size,
                       stride * data_size,
                       y_dim_size);
            };
        };
        break;
    case 8:
        kernel.kernel_name     = "Cat8FwdPacked";
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::cat::CatInvokeParams>();

                auto ydims      = params.yDesc.GetLengths();
                auto dim        = params.dim;
                auto stride     = params.yDesc.GetStrides()[dim];
                auto y_dim_size = ydims[dim];
                auto outer_size = std::accumulate(
                    ydims.begin(), ydims.begin() + dim, 1ULL, std::multiplies<size_t>());
                auto data_size = get_data_size(params.yDesc.GetType());

                kernel(params.GetX(0),
                       params.GetX(1),
                       params.GetX(2),
                       params.GetX(3),
                       params.GetX(4),
                       params.GetX(5),
                       params.GetX(6),
                       params.GetX(7),
                       params.y,
                       params.GetXDimSize(0),
                       params.GetXDimSize(1),
                       params.GetXDimSize(2),
                       params.GetXDimSize(3),
                       params.GetXDimSize(4),
                       params.GetXDimSize(5),
                       params.GetXDimSize(6),
                       params.GetXDimSize(7),
                       outer_size,
                       stride * data_size,
                       y_dim_size);
            };
        };
        break;
    default: break;
    }
    result.construction_params.push_back(kernel);

    return result;
}

} // namespace cat

} // namespace solver

} // namespace miopen
