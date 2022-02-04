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

bool PoolingForward2d::IsApplicable(const ExecutionContext&,
                                    const miopen::pooling::ProblemDescription& problem) const
{
    return problem.GetDirection() == miopen::pooling::Direction::Forward &&
           problem.GetXDesc().GetSize() == 4 &&
           problem.GetXDesc().GetType() == problem.GetYDesc().GetType() &&
           (problem.GetXDesc().GetType() == miopenFloat ||
            problem.GetXDesc().GetType() == miopenHalf) &&
           problem.GetXDesc().GetLayout("NCHW") == "NCHW" &&
           problem.GetYDesc().GetLayout("NCHW") == "NCHW";
}

ConvSolution PoolingForward2d::GetSolution(const ExecutionContext&,
                                           const miopen::pooling::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPooling.cl";
        kernel.kernel_name = "mloPoolingG";

        int batch_sz, n_outputs, out_height, out_width;
        std::tie(batch_sz, n_outputs, out_height, out_width) =
            miopen::tien<4>(problem.GetYDesc().GetLengths(), 1);

        const auto kernel_size_h   = problem.GetPooling().lens[0];
        const auto kernel_size_w   = problem.GetPooling().lens[1];
        const auto kernel_stride_h = problem.GetPooling().strides[0];
        const auto kernel_stride_w = problem.GetPooling().strides[1];
        const auto _wsp_index      = problem.GetPooling().GetWorkspaceIndexMode();

        const int _out_pix_tile0 = 1;
        int _out_pix_tile1       = out_height <= 8 ? 1 : out_height <= 32 ? 4 : 8;
        if(out_height > 16 && out_height % 32 > 16)
            _out_pix_tile1 = std::min(16, std::max(1, prePow2(_out_pix_tile1 * kernel_stride_h)));

        int _grp_tile0 = out_width <= 8 ? 8 : (out_width % 32 <= 16 ? 16 : 32);
        int _grp_tile1 =
            out_height <= 8
                ? 8
                : out_height < 16 ? 16 : out_height <= 32 ? 32 : out_height <= 64 ? 64 : 128;
        _grp_tile1 /= _out_pix_tile1;
        while(_grp_tile0 * _grp_tile1 > 256 && _grp_tile0 > 1)
            _grp_tile0 >>= 1;

        int pooling_method = (problem.GetPooling().GetMode() == miopenPoolingMax)
                                 ? MLO_POOLING_OP_MAX
                                 : ((problem.GetPooling().GetMode() == miopenPoolingAverage)
                                        ? MLO_POOLING_OP_AVE
                                        : MLO_POOLING_OP_AVE_INCLUSIVE);

        auto build_params = KernelBuildParameters{
            {"MLO_POOLING_OP_ID", static_cast<long long>(pooling_method)},
            {"MLO_POOLING_KERNEL_SZ1", static_cast<long long>(kernel_size_h)},
            {"MLO_POOLING_STRIDE1", static_cast<long long>(kernel_stride_h)},
            {"MLO_POOLING_KERNEL_SZ0", static_cast<long long>(kernel_size_w)},
            {"MLO_POOLING_STRIDE0", static_cast<long long>(kernel_stride_w)},
            {"MLO_POOLING_N_HORIZ_OUT_PIX", static_cast<long long>(_out_pix_tile0)},
            {"MLO_POOLING_N_VERT_OUT_PIX", static_cast<long long>(_out_pix_tile1)},
            {"MLO_POOLING_GROUP_SZ0", static_cast<long long>(_grp_tile0)},
            {"MLO_POOLING_GROUP_SZ1", static_cast<long long>(_grp_tile1)},
            {"MLO_POOLING_INDEX_TYPE",
             get_pooling_index_type_name(problem.GetPooling().GetIndexType())},
            {"MLO_POOLING_INDEX_MAX",
             get_pooling_index_type_max_name(problem.GetPooling().GetIndexType())},
        };

        if(problem.SaveIndex())
        {
            build_params << KernelBuildParameters{
                {"MLO_POOLING_SAVE_INDEX"},
                {"USE_IMG_INDEX", (_wsp_index == miopenPoolingWorkspaceIndexImage ? 1 : 0)},
            };
        }

        build_params << GetDataTypeKBP(problem.GetXDesc().GetType());

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk.push_back(_grp_tile0);
        kernel.l_wk.push_back(_grp_tile1);
        kernel.l_wk.push_back(1);

        int g_wk_width =
            ((out_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
        int g_wk_height =
            ((out_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));

        kernel.g_wk.push_back(g_wk_width * _grp_tile0);
        kernel.g_wk.push_back(g_wk_height * _grp_tile1);
        kernel.g_wk.push_back(n_outputs * batch_sz);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::FwdInvokeParams>();

            kernel(params.x,
                   params.y,
                   params.workspace,
                   static_cast<int>(params.pooling.pads[0]),
                   static_cast<int>(params.pooling.pads[1]),
                   static_cast<int>(params.xDesc.GetLengths()[1]),
                   static_cast<int>(params.xDesc.GetLengths()[2]),
                   static_cast<int>(params.xDesc.GetLengths()[3]),
                   static_cast<int>(params.yDesc.GetLengths()[2]),
                   static_cast<int>(params.yDesc.GetLengths()[3]),
                   static_cast<int>(params.xDesc.GetStrides()[0]),
                   static_cast<int>(params.xDesc.GetStrides()[1]),
                   static_cast<int>(params.xDesc.GetStrides()[2]),
                   static_cast<int>(params.yDesc.GetStrides()[0]),
                   static_cast<int>(params.yDesc.GetStrides()[1]),
                   static_cast<int>(params.yDesc.GetStrides()[2]));
        };
    };

    return result;
}

std::size_t
PoolingForward2d::GetWorkspaceSize(const ExecutionContext&,
                                   const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
