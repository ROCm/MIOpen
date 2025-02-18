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

namespace miopen {

namespace solver {

namespace pooling {

namespace {

constexpr int top_w_per_work = 1;
constexpr int top_h_per_work = 4;
constexpr int top_d_per_work = 2;

struct kernel_params
{
    uint32_t stride_d;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t kernel_sz_d;
    uint32_t kernel_sz_h;
    uint32_t kernel_sz_w;

    kernel_params(const miopen::pooling::ProblemDescription& p)
    {
        const auto& pd = p.GetPooling();
        stride_d       = pd.strides[0];
        stride_h       = pd.strides[1];
        stride_w       = pd.strides[2];
        kernel_sz_d    = pd.lens[0];
        kernel_sz_h    = pd.lens[1];
        kernel_sz_w    = pd.lens[2];
    }
};

std::size_t sizeof_kernel_FLOAT(const miopen::pooling::ProblemDescription& problem)
{
    const auto datatype = problem.GetXDesc().GetType();
    return get_data_size(datatype);
}

inline std::size_t RoundUpToMultiple(std::size_t v, std::size_t m)
{
    assert(m > 0);
    return ((v + m - 1) / m) * m;
}

// Compute amount of private memory required for holding the arrays defined
// in the "mloPoolingNDFwd" kernel:
//
// #define BOT_TILE_W ((TOP_W_PER_WORK - 1) * STRIDE_W + KERNEL_SZ_W)
// #define BOT_TILE_H ((TOP_H_PER_WORK - 1) * STRIDE_H + KERNEL_SZ_H)
// #define BOT_TILE_D ((TOP_D_PER_WORK - 1) * STRIDE_D + KERNEL_SZ_D)
//
// _FLOAT bot_data[BOT_TILE_D][BOT_TILE_H][BOT_TILE_W];
//
std::size_t sizeof_private_memory(const miopen::pooling::ProblemDescription& problem)
{
    const kernel_params kp(problem);

    const std::size_t bot_tile_w = ((top_w_per_work - 1) * kp.stride_w + kp.kernel_sz_w);
    const std::size_t bot_tile_h = ((top_h_per_work - 1) * kp.stride_h + kp.kernel_sz_h);
    const std::size_t bot_tile_d = ((top_d_per_work - 1) * kp.stride_d + kp.kernel_sz_d);

    const auto sizeof_bot_data =
        sizeof_kernel_FLOAT(problem) * bot_tile_d * bot_tile_h * bot_tile_w;
    MIOPEN_LOG_T("sizeof_bot_data " << sizeof_bot_data);

    /// \ref alignment_of_arrays_in_gpu_memory
    return RoundUpToMultiple(sizeof_bot_data, 16);
}

} // namespace

bool PoolingForwardNd::IsApplicable(const ExecutionContext& context,
                                    const miopen::pooling::ProblemDescription& problem) const
{

    return problem.GetDirection() == miopen::pooling::Direction::Forward                      //
           && problem.GetXDesc().GetNumDims() == 5                                            //
           && problem.GetXDesc().IsPossibleLayout4D5D("NCDHW")                                //
           && problem.GetYDesc().IsPossibleLayout4D5D("NCDHW")                                //
           && problem.GetXDesc().GetType() == problem.GetYDesc().GetType()                    //
           && (problem.GetXDesc().GetType() == miopenFloat                                    //
               || problem.GetXDesc().GetType() == miopenHalf)                                 //
           && (problem.GetPooling().GetMode() == miopenPoolingMax                             //
               || problem.GetPooling().GetMode() == miopenPoolingAverage                      //
               || problem.GetPooling().GetMode() == miopenPoolingAverageInclusive)            //
           && sizeof_private_memory(problem) <= TargetProperties::GetMaxWaveScratchSize()     //
                                                    / context.GetStream().GetWavefrontWidth() //
           /// \todo This solver does not support workspace index mask mode yet.
           &&
           !(problem.GetPooling().GetMode() == miopenPoolingMax                                 //
             && problem.GetPooling().GetWorkspaceIndexMode() == miopenPoolingWorkspaceIndexMask //
             && problem.SaveIndex() == true);
}

ConvSolution PoolingForwardNd::GetSolution(const ExecutionContext&,
                                           const miopen::pooling::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const int batch = problem.GetXDesc().GetLengths()[0];
    const int chal  = problem.GetXDesc().GetLengths()[1];

    const kernel_params kp(problem);

    const int top_d = *(problem.GetYDesc().GetLengths().rbegin() + 2);
    const int top_h = *(problem.GetYDesc().GetLengths().rbegin() + 1);
    const int top_w = *(problem.GetYDesc().GetLengths().rbegin());

    const int top_blk_w = std::max((top_w + top_w_per_work - 1) / top_w_per_work, 1);
    const int top_blk_h = std::max((top_h + top_h_per_work - 1) / top_h_per_work, 1);
    const int top_blk_d = std::max((top_d + top_d_per_work - 1) / top_d_per_work, 1);

    const int max_activ_workitem = 65536;
    const int total_work         = batch * chal * top_blk_w * top_blk_h * top_blk_d;
    const int activ_work         = std::min(total_work, max_activ_workitem);

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPoolingND.cl";
        kernel.kernel_name = "mloPoolingNDFwd";

        int pooling_method = (problem.GetPooling().mode == miopenPoolingMax)
                                 ? MLO_POOLING_OP_MAX
                                 : ((problem.GetPooling().mode == miopenPoolingAverage)
                                        ? MLO_POOLING_OP_AVE
                                        : MLO_POOLING_OP_AVE_INCLUSIVE);

        const size_t lcl_work = 64;
        const size_t grp_num  = (activ_work + lcl_work - 1) / lcl_work;

        auto build_params = KernelBuildParameters{
            {"MLO_POOLING_OP_ID", static_cast<long long>(pooling_method)},
            {"MAX_ACTIV_WORKITEM", static_cast<unsigned>(max_activ_workitem)},
            {"MLO_POOLING_GROUP_SZ0", static_cast<long long>(lcl_work)},
            {"MLO_POOLING_GROUP_SZ1", 1},
            {"MLO_POOLING_GROUP_SZ2", 1},
            {"TOP_W_PER_WORK", top_w_per_work},
            {"TOP_H_PER_WORK", top_h_per_work},
            {"TOP_D_PER_WORK", top_d_per_work},
            {"KERNEL_SZ_D", kp.kernel_sz_d},
            {"KERNEL_SZ_H", kp.kernel_sz_h},
            {"KERNEL_SZ_W", kp.kernel_sz_w},
            {"STRIDE_D", kp.stride_d},
            {"STRIDE_H", kp.stride_h},
            {"STRIDE_W", kp.stride_w},
            {"MLO_POOLING_INDEX_TYPE",
             get_pooling_index_type_name(problem.GetPooling().GetIndexType())},
            {"MLO_POOLING_INDEX_MAX",
             get_pooling_index_type_max_name(problem.GetPooling().GetIndexType())},
        };

        if(problem.SaveIndex())
        {
            build_params << KernelBuildParameters{
                {"MLO_POOLING_SAVE_INDEX"},
            };
        }

        build_params << GetDataTypeKBP(problem.GetXDesc().GetType());

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk = {lcl_work, 1, 1};
        kernel.g_wk = {lcl_work * grp_num, 1, 1};

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::FwdInvokeParams>();

            const int batch_ = params.xDesc.GetLengths()[0];
            const int chal_  = params.xDesc.GetLengths()[1];

            const int top_d_ = *(params.yDesc.GetLengths().rbegin() + 2);
            const int top_h_ = *(params.yDesc.GetLengths().rbegin() + 1);
            const int top_w_ = *(params.yDesc.GetLengths().rbegin());

            const int top_blk_w_ = std::max((top_w_ + top_w_per_work - 1) / top_w_per_work, 1);
            const int top_blk_h_ = std::max((top_h_ + top_h_per_work - 1) / top_h_per_work, 1);
            const int top_blk_d_ = std::max((top_d_ + top_d_per_work - 1) / top_d_per_work, 1);

            const int total_work_ = batch_ * chal_ * top_blk_w_ * top_blk_h_ * top_blk_d_;

            kernel(params.x,
                   params.y,
                   params.workspace,
                   static_cast<unsigned>(params.pooling.pads[0]),
                   static_cast<unsigned>(params.pooling.pads[1]),
                   static_cast<unsigned>(params.pooling.pads[2]),
                   static_cast<unsigned>(batch_),
                   static_cast<unsigned>(chal_),
                   static_cast<unsigned>(params.xDesc.GetLengths()[2]),
                   static_cast<unsigned>(params.xDesc.GetLengths()[3]),
                   static_cast<unsigned>(params.xDesc.GetLengths()[4]),
                   static_cast<unsigned>(top_d_),
                   static_cast<unsigned>(top_h_),
                   static_cast<unsigned>(top_w_),
                   static_cast<unsigned>(params.xDesc.GetStrides()[0]),
                   static_cast<unsigned>(params.xDesc.GetStrides()[1]),
                   static_cast<unsigned>(params.xDesc.GetStrides()[2]),
                   static_cast<unsigned>(params.xDesc.GetStrides()[3]),
                   static_cast<unsigned>(params.yDesc.GetStrides()[0]),
                   static_cast<unsigned>(params.yDesc.GetStrides()[1]),
                   static_cast<unsigned>(params.yDesc.GetStrides()[2]),
                   static_cast<unsigned>(params.yDesc.GetStrides()[3]),
                   static_cast<unsigned>(total_work_));
        };
    };

    return result;
}

std::size_t
PoolingForwardNd::GetWorkspaceSize(const ExecutionContext&,
                                   const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
