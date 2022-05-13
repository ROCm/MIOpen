/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/pooling/solvers.hpp>

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/pooling/problem_description.hpp>
#include <miopen/datatype.hpp>
#include <miopen/pooling.hpp>
#include <miopen/kernel_build_params.hpp>

namespace miopen {

namespace solver {

namespace pooling {

bool PoolingBackward2d::IsApplicable(const ExecutionContext&,
                                     const miopen::pooling::ProblemDescription& problem) const
{
    return problem.GetDirection() == miopen::pooling::Direction::Backward &&
           (problem.GetPooling().GetMode() == miopenPoolingMax ||
            problem.GetPooling().GetMode() == miopenPoolingAverage ||
            problem.GetPooling().GetMode() == miopenPoolingAverageInclusive) &&
           problem.GetXDesc().GetSize() == 4 && problem.GetXDesc().GetLayout("NCHW") == "NCHW" &&
           problem.GetYDesc().GetLayout("NCHW") == "NCHW";
}

ConvSolution
PoolingBackward2d::GetSolution(const ExecutionContext&,
                               const miopen::pooling::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPoolingBwd.cl";

        if(problem.GetPooling().GetMode() == miopenPoolingMax)
        {
            kernel.kernel_name = "mloPoolingMaxBwd";
        }
        else if(problem.GetPooling().GetMode() == miopenPoolingAverage ||
                problem.GetPooling().GetMode() == miopenPoolingAverageInclusive)
        {
            kernel.kernel_name = "mloPoolingAveBwd";
        }

        std::size_t batch_sz, n_inputs, in_height, in_width;
        std::tie(batch_sz, n_inputs, in_height, in_width) =
            miopen::tien<4>(problem.GetXDesc().GetLengths(), 1);

        const int pooling_method = (problem.GetPooling().GetMode() == miopenPoolingMax)
                                       ? MLO_POOLING_OP_MAX
                                       : ((problem.GetPooling().GetMode() == miopenPoolingAverage)
                                              ? MLO_POOLING_OP_AVE
                                              : MLO_POOLING_OP_AVE_INCLUSIVE);

        std::size_t grp_tile0 = 8;
        std::size_t grp_tile1 = 8;

        std::size_t out_pix_tile0 = 1;
        std::size_t out_pix_tile1 = 1;

        if(problem.GetPooling().GetMode() == miopenPoolingMax)
        {
            // clang-format off
            grp_tile0 =
                   in_width <= 8   ? 8
                : (in_width <= 16  ? 4
                : (in_width <= 24  ? 8
                : (in_width <= 32  ? 32
                : (in_width <= 64  ? 8
                : (in_width <= 96  ? 16
                : (in_width <= 128 ? 16
                                   : 32))))));
            grp_tile1 =
                   in_width <= 8   ? 8
                : (in_width <= 16  ? 16
                : (in_width <= 24  ? 8
                : (in_width <= 32  ? 4
                : (in_width <= 64  ? 8
                : (in_width <= 96  ? 4
                : (in_width <= 128 ? 16
                                   : 4))))));
            // clang-format on

            out_pix_tile0 = in_width > 8 && in_width <= 24 ? 4 : 1;
            out_pix_tile1 = in_width <= 24 ? 1 : (in_width > 64 && in_width <= 96 ? 4 : 8);
        }

        int g_wk_width = ((in_width + grp_tile0 * out_pix_tile0 - 1) / (grp_tile0 * out_pix_tile0));
        int g_wk_height =
            ((in_height + grp_tile1 * out_pix_tile1 - 1) / (grp_tile1 * out_pix_tile1));

        const auto kernel_size_w   = problem.GetPooling().lens[0];
        const auto kernel_size_h   = problem.GetPooling().lens[1];
        const auto kernel_stride_w = problem.GetPooling().strides[0];
        const auto kernel_stride_h = problem.GetPooling().strides[1];

        const auto build_params =
            KernelBuildParameters{
                {"MLO_POOLING_OP_ID", static_cast<long long>(pooling_method)},
                {"MLO_POOLING_KERNEL_SZ1", static_cast<long long>(kernel_size_h)},
                {"MLO_POOLING_STRIDE1", static_cast<long long>(kernel_stride_h)},
                {"MLO_POOLING_KERNEL_SZ0", static_cast<long long>(kernel_size_w)},
                {"MLO_POOLING_STRIDE0", static_cast<long long>(kernel_stride_w)},
                {"MLO_POOLBWD_N_HORIZ_OUT_PIX", static_cast<long long>(out_pix_tile0)},
                {"MLO_POOLBWD_N_VERT_OUT_PIX", static_cast<long long>(out_pix_tile1)},
                {"MLO_POOLBWD_GROUP_SZ0", static_cast<long long>(grp_tile0)},
                {"MLO_POOLBWD_GROUP_SZ1", static_cast<long long>(grp_tile1)},
                {"MLO_POOLING_INDEX_TYPE",
                 get_pooling_index_type_name(problem.GetPooling().GetIndexType())},
                {"MLO_POOLING_INDEX_MAX",
                 get_pooling_index_type_max_name(problem.GetPooling().GetIndexType())},
                {"USE_IMG_INDEX",
                 problem.GetPooling().GetWorkspaceIndexMode() == miopenPoolingWorkspaceIndexImage
                     ? 1
                     : 0},
            }
            << GetDataTypeKBP(problem.GetXDesc().GetType());

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk = {grp_tile0, grp_tile1, 1};
        kernel.g_wk = {g_wk_width * grp_tile0, g_wk_height * grp_tile1, n_inputs * batch_sz};

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::BwdInvokeParams>();

            if(params.pooling.GetMode() == miopenPoolingMax)
            {
                kernel(params.dy,
                       params.dx,
                       params.workspace,
                       static_cast<int>(params.pooling.pads[0]),
                       static_cast<int>(params.pooling.pads[1]),
                       static_cast<int>(params.dyDesc.GetLengths()[1]),
                       static_cast<int>(params.dxDesc.GetLengths()[2]),
                       static_cast<int>(params.dxDesc.GetLengths()[3]),
                       static_cast<int>(params.dyDesc.GetLengths()[2]),
                       static_cast<int>(params.dyDesc.GetLengths()[3]),
                       static_cast<int>(params.dxDesc.GetStrides()[0]),
                       static_cast<int>(params.dxDesc.GetStrides()[1]),
                       static_cast<int>(params.dxDesc.GetStrides()[2]),
                       static_cast<int>(params.dyDesc.GetStrides()[0]),
                       static_cast<int>(params.dyDesc.GetStrides()[1]),
                       static_cast<int>(params.dyDesc.GetStrides()[2]));
            }
            else
            {
                kernel(params.dy,
                       params.dx,
                       static_cast<int>(params.pooling.pads[0]),
                       static_cast<int>(params.pooling.pads[1]),
                       static_cast<int>(params.dyDesc.GetLengths()[1]),
                       static_cast<int>(params.dxDesc.GetLengths()[2]),
                       static_cast<int>(params.dxDesc.GetLengths()[3]),
                       static_cast<int>(params.dyDesc.GetLengths()[2]),
                       static_cast<int>(params.dyDesc.GetLengths()[3]),
                       static_cast<int>(params.dxDesc.GetStrides()[0]),
                       static_cast<int>(params.dxDesc.GetStrides()[1]),
                       static_cast<int>(params.dxDesc.GetStrides()[2]),
                       static_cast<int>(params.dyDesc.GetStrides()[0]),
                       static_cast<int>(params.dyDesc.GetStrides()[1]),
                       static_cast<int>(params.dyDesc.GetStrides()[2]));
            }
        };
    };

    return result;
}

std::size_t
PoolingBackward2d::GetWorkspaceSize(const ExecutionContext&,
                                    const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
