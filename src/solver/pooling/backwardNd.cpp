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

bool PoolingBackwardNd::IsApplicable(const ExecutionContext&,
                                     const miopen::pooling::ProblemDescription& problem) const
{
    return problem.GetDirection() == miopen::pooling::Direction::Backward &&
           (problem.GetPooling().GetMode() == miopenPoolingMax ||
            problem.GetPooling().GetMode() == miopenPoolingAverage ||
            problem.GetPooling().GetMode() == miopenPoolingAverageInclusive) &&
           problem.GetXDesc().GetSize() == 5 && problem.GetXDesc().GetLayout("NCDHW") == "NCDHW" &&
           problem.GetYDesc().GetLayout("NCDHW") == "NCDHW";
}

ConvSolution
PoolingBackwardNd::GetSolution(const ExecutionContext&,
                               const miopen::pooling::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPoolingBwdND.cl";
        kernel.kernel_name = "mloPoolingND";

        if(problem.GetPooling().GetMode() == miopenPoolingMax)
        {
            kernel.kernel_name += "MaxBwd";
        }
        else if(problem.GetPooling().GetMode() == miopenPoolingAverage ||
                problem.GetPooling().GetMode() == miopenPoolingAverageInclusive)
        {
            kernel.kernel_name += "AveBwd";
        }

        std::size_t batch_sz, n_inputs, in_height, in_width;
        std::tie(batch_sz, n_inputs, in_height, in_width) =
            miopen::tien<4>(problem.GetXDesc().GetLengths(), 1);

        const int pooling_method = (problem.GetPooling().GetMode() == miopenPoolingMax)
                                       ? MLO_POOLING_OP_MAX
                                       : ((problem.GetPooling().GetMode() == miopenPoolingAverage)
                                              ? MLO_POOLING_OP_AVE
                                              : MLO_POOLING_OP_AVE_INCLUSIVE);

        int pix_w_per_work = 1;
        int pix_h_per_work = 4;
        int pix_d_per_work = 2;

        int batch = problem.GetDYDesc().GetLengths()[0];
        int chal  = problem.GetDYDesc().GetLengths()[1];

        int bot_d = *(problem.GetDXDesc().GetLengths().rbegin() + 2);
        int bot_h = *(problem.GetDXDesc().GetLengths().rbegin() + 1);
        int bot_w = *(problem.GetDXDesc().GetLengths().rbegin());

        int pix_blk_w = std::max((bot_w + pix_w_per_work - 1) / pix_w_per_work, 1);
        int pix_blk_h = std::max((bot_h + pix_h_per_work - 1) / pix_h_per_work, 1);
        int pix_blk_d = std::max((bot_d + pix_d_per_work - 1) / pix_d_per_work, 1);

        int max_activ_workitem = 65536;
        int total_work         = batch * chal * pix_blk_w * pix_blk_h * pix_blk_d;
        int activ_work         = std::min(total_work, max_activ_workitem);

        size_t lcl_work = 64;
        size_t grp_num  = (activ_work + lcl_work - 1) / lcl_work;

        bool territory_overlap = false;
        for(std::size_t i = 0; i < problem.GetPooling().strides.size(); i++)
            territory_overlap |= (problem.GetPooling().strides[i] < problem.GetPooling().lens[i]);

        const auto build_params =
            KernelBuildParameters{
                {"MLO_POOLING_OP_ID", static_cast<long long>(pooling_method)},
                {"MAX_ACTIV_WORKITEM", static_cast<uint>(max_activ_workitem)},
                {"MLO_POOLING_GROUP_SZ0", static_cast<long long>(lcl_work)},
                {"MLO_POOLING_GROUP_SZ1", 1},
                {"MLO_POOLING_GROUP_SZ2", 1},
                {"PIX_W_PER_WORK", static_cast<uint>(pix_w_per_work)},
                {"PIX_H_PER_WORK", static_cast<uint>(pix_h_per_work)},
                {"PIX_D_PER_WORK", static_cast<uint>(pix_d_per_work)},
                {"KERNEL_SZ_D", static_cast<uint>(problem.GetPooling().lens[0])},
                {"KERNEL_SZ_H", static_cast<uint>(problem.GetPooling().lens[1])},
                {"KERNEL_SZ_W", static_cast<uint>(problem.GetPooling().lens[2])},
                {"STRIDE_D", static_cast<uint>(problem.GetPooling().strides[0])},
                {"STRIDE_H", static_cast<uint>(problem.GetPooling().strides[1])},
                {"STRIDE_W", static_cast<uint>(problem.GetPooling().strides[2])},
                {"TERRITORY_OVERLAP", static_cast<int>(territory_overlap)},
                {"MLO_POOLING_INDEX_TYPE",
                 get_pooling_index_type_name(problem.GetPooling().GetIndexType())},
                {"MLO_POOLING_INDEX_MAX",
                 get_pooling_index_type_max_name(problem.GetPooling().GetIndexType())},
            }
            << GetDataTypeKBP(problem.GetDYDesc().GetType());

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk = {lcl_work, 1, 1};
        kernel.g_wk = {lcl_work * grp_num, 1, 1};

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::BwdInvokeParams>();

            const auto top_d = *(params.dyDesc.GetLengths().rbegin() + 2);
            const auto top_h = *(params.dyDesc.GetLengths().rbegin() + 1);
            const auto top_w = *(params.dyDesc.GetLengths().rbegin());

            int pix_w_per_work = 1;
            int pix_h_per_work = 4;
            int pix_d_per_work = 2;

            int batch = params.dyDesc.GetLengths()[0];
            int chal  = params.dyDesc.GetLengths()[1];

            int bot_d = *(params.dxDesc.GetLengths().rbegin() + 2);
            int bot_h = *(params.dxDesc.GetLengths().rbegin() + 1);
            int bot_w = *(params.dxDesc.GetLengths().rbegin());

            int pix_blk_w = std::max((bot_w + pix_w_per_work - 1) / pix_w_per_work, 1);
            int pix_blk_h = std::max((bot_h + pix_h_per_work - 1) / pix_h_per_work, 1);
            int pix_blk_d = std::max((bot_d + pix_d_per_work - 1) / pix_d_per_work, 1);

            int total_work = batch * chal * pix_blk_w * pix_blk_h * pix_blk_d;

            if(params.pooling.GetMode() == miopenPoolingMax)
            {
                kernel(params.dy,
                       params.dx,
                       params.workspace,
                       static_cast<uint>(params.pooling.pads[0]),
                       static_cast<uint>(params.pooling.pads[1]),
                       static_cast<uint>(params.pooling.pads[2]),
                       static_cast<uint>(batch),
                       static_cast<uint>(chal),
                       static_cast<uint>(params.dxDesc.GetLengths()[2]),
                       static_cast<uint>(params.dxDesc.GetLengths()[3]),
                       static_cast<uint>(params.dxDesc.GetLengths()[4]),
                       static_cast<uint>(top_d),
                       static_cast<uint>(top_h),
                       static_cast<uint>(top_w),
                       static_cast<uint>(params.dxDesc.GetStrides()[0]),
                       static_cast<uint>(params.dxDesc.GetStrides()[1]),
                       static_cast<uint>(params.dxDesc.GetStrides()[2]),
                       static_cast<uint>(params.dxDesc.GetStrides()[3]),
                       static_cast<uint>(params.dyDesc.GetStrides()[0]),
                       static_cast<uint>(params.dyDesc.GetStrides()[1]),
                       static_cast<uint>(params.dyDesc.GetStrides()[2]),
                       static_cast<uint>(params.dyDesc.GetStrides()[3]),
                       static_cast<uint>(total_work));
            }
            else
            {
                kernel(params.dy,
                       params.dx,
                       static_cast<uint>(params.pooling.pads[0]),
                       static_cast<uint>(params.pooling.pads[1]),
                       static_cast<uint>(params.pooling.pads[2]),
                       static_cast<uint>(batch),
                       static_cast<uint>(chal),
                       static_cast<uint>(params.dxDesc.GetLengths()[2]),
                       static_cast<uint>(params.dxDesc.GetLengths()[3]),
                       static_cast<uint>(params.dxDesc.GetLengths()[4]),
                       static_cast<uint>(top_d),
                       static_cast<uint>(top_h),
                       static_cast<uint>(top_w),
                       static_cast<uint>(params.dxDesc.GetStrides()[0]),
                       static_cast<uint>(params.dxDesc.GetStrides()[1]),
                       static_cast<uint>(params.dxDesc.GetStrides()[2]),
                       static_cast<uint>(params.dxDesc.GetStrides()[3]),
                       static_cast<uint>(params.dyDesc.GetStrides()[0]),
                       static_cast<uint>(params.dyDesc.GetStrides()[1]),
                       static_cast<uint>(params.dyDesc.GetStrides()[2]),
                       static_cast<uint>(params.dyDesc.GetStrides()[3]),
                       static_cast<uint>(total_work));
            }
        };
    };

    return result;
}

std::size_t
PoolingBackwardNd::GetWorkspaceSize(const ExecutionContext&,
                                    const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
