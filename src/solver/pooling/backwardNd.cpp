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
#include <miopen/datatype.hpp>
#include <miopen/pooling.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>

#define WORKAROUND_ISSUE_MIFIN_80 1 // https://github.com/ROCm/MIFin/issues/80

namespace miopen {

namespace solver {

namespace pooling {

bool PoolingBackwardNd::IsApplicable(const ExecutionContext&,
                                     const miopen::pooling::ProblemDescription& problem) const
{
    return problem.GetDirection() == miopen::pooling::Direction::Backward          //
           && problem.GetXDesc().GetType() == problem.GetYDesc().GetType()         //
           && (problem.GetXDesc().GetType() == miopenFloat                         //
               || problem.GetXDesc().GetType() == miopenHalf)                      //
           && (problem.GetPooling().GetMode() == miopenPoolingMax                  //
               || problem.GetPooling().GetMode() == miopenPoolingAverage           //
               || problem.GetPooling().GetMode() == miopenPoolingAverageInclusive) //
           && (                                                                    //
                  (problem.GetXDesc().GetNumDims() == 5                            //
                   && problem.GetXDesc().IsPossibleLayout4D5D("NCDHW")             //
                   && problem.GetYDesc().IsPossibleLayout4D5D("NCDHW"))            //
                  ||                                                               //
                  (problem.GetXDesc().GetNumDims() == 4                            //
                   && problem.GetXDesc().IsPossibleLayout4D5D("NCHW")              //
                   && problem.GetYDesc().IsPossibleLayout4D5D("NCHW"))             //
                  )                                                                //
           /// \todo This solver does not support workspace index mask mode yet.
           && !(problem.GetPooling().GetMode() == miopenPoolingMax //
                && problem.GetPooling().GetWorkspaceIndexMode() == miopenPoolingWorkspaceIndexMask);
}

ConvSolution
PoolingBackwardNd::GetSolution(const ExecutionContext&,
                               const miopen::pooling::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto kernel        = KernelInfo{};
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

    const auto& bot = problem.GetXDesc();
    const auto& top = problem.GetYDesc();

    std::size_t batch_sz, n_inputs, in_height, in_width;
    std::tie(batch_sz, n_inputs, in_height, in_width) = miopen::tien<4>(bot.GetLengths(), 1);

    const int pooling_method = (problem.GetPooling().GetMode() == miopenPoolingMax)
                                   ? MLO_POOLING_OP_MAX
                                   : ((problem.GetPooling().GetMode() == miopenPoolingAverage)
                                          ? MLO_POOLING_OP_AVE
                                          : MLO_POOLING_OP_AVE_INCLUSIVE);

    int pix_w_per_work = 1;
    int pix_h_per_work = 4;
    int pix_d_per_work = 2;

    int batch = top.GetLengths()[0];
    int chal  = top.GetLengths()[1];

    const bool is2d = (bot.GetNumDims() == 4);

    int bot_d = is2d ? 1 : *(bot.GetLengths().rbegin() + 2);
    int bot_h = *(bot.GetLengths().rbegin() + 1);
    int bot_w = *(bot.GetLengths().rbegin());

    int pix_blk_w = std::max((bot_w + pix_w_per_work - 1) / pix_w_per_work, 1);
    int pix_blk_h = std::max((bot_h + pix_h_per_work - 1) / pix_h_per_work, 1);
    int pix_blk_d = std::max((bot_d + pix_d_per_work - 1) / pix_d_per_work, 1);

    int max_activ_workitem = 65536;
    int total_work         = batch * chal * pix_blk_w * pix_blk_h * pix_blk_d;
    int activ_work         = std::min(total_work, max_activ_workitem);

#if WORKAROUND_ISSUE_MIFIN_80
    const std::size_t wavesize = 64;
#else
    const std::size_t wavesize = context.GetStream().GetWavefrontWidth();
#endif
    size_t grp_num = (activ_work + wavesize - 1) / wavesize;

    auto strides = problem.GetPooling().strides;
    auto lens    = problem.GetPooling().lens;
    auto pads    = problem.GetPooling().pads;

    if(is2d)
    {
        strides.push_back(strides[1]);
        strides[1] = strides[0];
        lens.push_back(lens[1]);
        lens[1] = lens[0];
        lens[0] = 1;
        pads.push_back(pads[1]);
        pads[1] = pads[0];
        pads[0] = 0;
    }

    bool territory_overlap = false;
    for(std::size_t i = 0; i < strides.size(); i++)
        territory_overlap |= (strides[i] < lens[i]);

    const auto build_params =
        KernelBuildParameters{
            {"MLO_POOLING_OP_ID", pooling_method},
            {"MAX_ACTIV_WORKITEM", max_activ_workitem},
            {"MLO_POOLING_GROUP_SZ0", wavesize},
            {"MLO_POOLING_GROUP_SZ1", 1},
            {"MLO_POOLING_GROUP_SZ2", 1},
            {"PIX_W_PER_WORK", pix_w_per_work},
            {"PIX_H_PER_WORK", pix_h_per_work},
            {"PIX_D_PER_WORK", pix_d_per_work},
            {"KERNEL_SZ_D", lens[0]},
            {"KERNEL_SZ_H", lens[1]},
            {"KERNEL_SZ_W", lens[2]},
            {"STRIDE_D", strides[0]},
            {"STRIDE_H", strides[1]},
            {"STRIDE_W", strides[2]},
            {"TERRITORY_OVERLAP", static_cast<int>(territory_overlap)},
            {"MLO_POOLING_INDEX_TYPE",
             get_pooling_index_type_name(problem.GetPooling().GetIndexType())},
            {"MLO_POOLING_INDEX_MAX",
             get_pooling_index_type_max_name(problem.GetPooling().GetIndexType())},
        }
        << GetDataTypeKBP(problem.GetDYDesc().GetType());

    kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

    kernel.l_wk = {wavesize, 1, 1};
    kernel.g_wk = {wavesize * grp_num, 1, 1};

    result.construction_params.push_back(kernel);

    const auto top_d = is2d ? 1 : *(top.GetLengths().rbegin() + 2);
    const auto top_h = *(top.GetLengths().rbegin() + 1);
    const auto top_w = *(top.GetLengths().rbegin());

    auto unpackStrides = [is2d](const auto& strides) {
        return std::make_tuple(strides[0], // N stride
                               strides[1], // C stride
                               strides[2], // D stride. Same as H_stride in 3D converted from 2D.
                               is2d        //
                                   ? strides[2] // 2D H stride
                                   : strides[3] // 3D H stride
        );
    };

    std::size_t bot_n_stride, bot_c_stride, bot_d_stride, bot_h_stride;
    std::size_t top_n_stride, top_c_stride, top_d_stride, top_h_stride;
    std::tie(bot_n_stride, bot_c_stride, bot_d_stride, bot_h_stride) =
        unpackStrides(bot.GetStrides());
    std::tie(top_n_stride, top_c_stride, top_d_stride, top_h_stride) =
        unpackStrides(top.GetStrides());

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::BwdInvokeParams>();

            if(params.pooling.GetMode() == miopenPoolingMax)
            {
                kernel(params.dy,
                       params.dx,
                       params.workspace,
                       static_cast<unsigned>(pads[0]),
                       static_cast<unsigned>(pads[1]),
                       static_cast<unsigned>(pads[2]),
                       static_cast<unsigned>(batch),
                       static_cast<unsigned>(chal),
                       static_cast<unsigned>(bot_d),
                       static_cast<unsigned>(bot_h),
                       static_cast<unsigned>(bot_w),
                       static_cast<unsigned>(top_d),
                       static_cast<unsigned>(top_h),
                       static_cast<unsigned>(top_w),
                       static_cast<unsigned>(bot_n_stride),
                       static_cast<unsigned>(bot_c_stride),
                       static_cast<unsigned>(bot_d_stride),
                       static_cast<unsigned>(bot_h_stride),
                       static_cast<unsigned>(top_n_stride),
                       static_cast<unsigned>(top_c_stride),
                       static_cast<unsigned>(top_d_stride),
                       static_cast<unsigned>(top_h_stride),
                       static_cast<unsigned>(total_work));
            }
            else
            {
                kernel(params.dy,
                       params.dx,
                       static_cast<unsigned>(pads[0]),
                       static_cast<unsigned>(pads[1]),
                       static_cast<unsigned>(pads[2]),
                       static_cast<unsigned>(batch),
                       static_cast<unsigned>(chal),
                       static_cast<unsigned>(bot_d),
                       static_cast<unsigned>(bot_h),
                       static_cast<unsigned>(bot_w),
                       static_cast<unsigned>(top_d),
                       static_cast<unsigned>(top_h),
                       static_cast<unsigned>(top_w),
                       static_cast<unsigned>(bot_n_stride),
                       static_cast<unsigned>(bot_c_stride),
                       static_cast<unsigned>(bot_d_stride),
                       static_cast<unsigned>(bot_h_stride),
                       static_cast<unsigned>(top_n_stride),
                       static_cast<unsigned>(top_c_stride),
                       static_cast<unsigned>(top_d_stride),
                       static_cast<unsigned>(top_h_stride),
                       static_cast<unsigned>(total_work));
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
