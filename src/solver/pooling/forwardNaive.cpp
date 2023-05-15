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

#include <miopen/pooling/solvers.hpp>

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/pooling.hpp>
#include <miopen/kernel_build_params.hpp>

namespace miopen {

namespace solver {

namespace pooling {

namespace {

template <typename T, typename Index>
void RunHost(int pooling_method,
             int pad_d,
             int pool_stride_d,
             int filter_size_d,
             int pad_h,
             int pool_stride_h,
             int filter_size_h,
             int pad_w,
             int pool_stride_w,
             int filter_size_w,
             const miopen::TensorDescriptor& bot,
             const miopen::TensorDescriptor& top,
             const T* bot_ptr,
             T* out_ptr,
             bool do_backward,
             Index* ref_mask_ptr,
             int index_position)
{

    const auto spatial_dim = bot.GetLengths().size() - 2;

    int n_batchs, n_outputs, bot_depth, bot_height, bot_width;
    int bot_w_stride, bot_h_stride, bot_d_stride, bot_c_stride, bot_n_stride;

    int top_depth, top_height, top_width;
    int top_w_stride, top_h_stride, top_d_stride, top_c_stride, top_n_stride;

    std::tie(n_batchs, n_outputs, bot_depth, bot_height, bot_width) =
        miopen::GetNCDHW(spatial_dim, bot.GetLengths());
    std::tie(bot_n_stride, bot_c_stride, bot_d_stride, bot_h_stride, bot_w_stride) =
        miopen::GetNCDHW(spatial_dim, bot.GetStrides());

    std::tie(std::ignore, std::ignore, top_depth, top_height, top_width) =
        miopen::GetNCDHW(spatial_dim, top.GetLengths());
    std::tie(top_n_stride, top_c_stride, top_d_stride, top_h_stride, top_w_stride) =
        miopen::GetNCDHW(spatial_dim, top.GetStrides());

    // Mask data is always NCDHW
    constexpr const int mask_w_stride = 1;
    const int mask_h_stride           = mask_w_stride * top_width;
    const int mask_d_stride           = mask_h_stride * top_height;
    const int mask_c_stride           = mask_d_stride * top_depth;
    const int mask_n_stride           = mask_c_stride * n_outputs;

    const T MAX_VAL = std::numeric_limits<T>::max();

    for(int b = 0; b < n_batchs; b++)
    {
        for(int o = 0; o < n_outputs; o++)
        {
            for(int k = 0; k < top_depth; k++)
            {
                for(int j = 0; j < top_height; j++)
                {
                    for(int i = 0; i < top_width; i++)
                    {
                        double res;
                        if(pooling_method == MLO_POOLING_OP_MAX)
                            res = -MAX_VAL;
                        else
                            res = 0;

                        int dstart = k * pool_stride_d - pad_d;
                        int hstart = j * pool_stride_h - pad_h;
                        int wstart = i * pool_stride_w - pad_w;
                        int dend   = std::min(dstart + filter_size_d, bot_depth);
                        int hend   = std::min(hstart + filter_size_h, bot_height);
                        int wend   = std::min(wstart + filter_size_w, bot_width);
                        dstart     = std::max(dstart, 0);
                        hstart     = std::max(hstart, 0);
                        wstart     = std::max(wstart, 0);

                        int pool_size;
                        if(pooling_method == MLO_POOLING_OP_AVE)
                            pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                        else
                            pool_size = filter_size_w * filter_size_h * filter_size_d;
                        pool_size            = (pool_size == 0) ? 1 : pool_size;
                        size_t res_index_gpu = 0;
                        bool found           = false;
                        for(int d = dstart; d < dend; ++d)
                        {
                            for(int h = hstart; h < hend; ++h)
                            {
                                for(int w = wstart; w < wend; ++w)
                                {
                                    size_t bot_index = b * bot_n_stride + o * bot_c_stride +
                                                       d * bot_d_stride + h * bot_h_stride +
                                                       w * bot_w_stride;
                                    if(pooling_method == MLO_POOLING_OP_MAX)
                                    {
                                        if(bot_ptr[bot_index] > res)
                                        {
                                            res = static_cast<T>(bot_ptr[bot_index]);
                                            res_index_gpu =
                                                index_position == 1
                                                    ? (d * bot_height * bot_width + h * bot_width +
                                                       w)
                                                    : ((d - k * pool_stride_d + pad_d) *
                                                       filter_size_w * filter_size_h) +
                                                          ((h - j * pool_stride_h + pad_h) *
                                                           filter_size_w) +
                                                          (w - i * pool_stride_w + pad_w);
                                            if(do_backward)
                                                found = true;
                                        }
                                    }
                                    else // Average
                                    {
                                        res += bot_ptr[bot_index];
                                    }
                                }
                            }
                        }

                        if(pooling_method == MLO_POOLING_OP_MAX && do_backward)
                        {
                            if(!found)
                                res_index_gpu = std::numeric_limits<uint8_t>::max();

                            const size_t mask_gpu_index = b * mask_n_stride + o * mask_c_stride +
                                                          k * mask_d_stride + j * mask_h_stride +
                                                          i * mask_w_stride;
                            ref_mask_ptr[mask_gpu_index] = res_index_gpu;
                        }

                        if(pooling_method == MLO_POOLING_OP_AVE ||
                           pooling_method == MLO_POOLING_OP_AVE_INCLUSIVE)
                        {
                            res /= pool_size;
                        }
                        const size_t top_index = b * top_n_stride + o * top_c_stride +
                                                 k * top_d_stride + j * top_h_stride +
                                                 i * top_w_stride;

                        out_ptr[top_index] = static_cast<T>(res);
                    }
                }
            }
        }
    }
}

struct arguments_t // Syntax sugar.
{
    int pooling_method;
    int pad_d;
    int stride_d;
    int filter_d;
    int pad_h;
    int stride_h;
    int filter_h;
    int pad_w;
    int stride_w;
    int filter_w;
    bool save_index;
    int index_mode;
};

template <typename T, typename Index>
void RunGpuEmulation(miopen::pooling::FwdInvokeParams& params,
                     const arguments_t& args,
                     const miopen::TensorDescriptor& bot,
                     const miopen::TensorDescriptor& top)
{
    const auto bot_nelem  = bot.GetElementSize();
    const auto top_nelem  = top.GetElementSize();
    const auto mask_nelem = args.save_index ? top_nelem : 0;

    std::vector<T> bot_host(bot_nelem);
    std::vector<T> top_host(top_nelem);
    std::vector<Index> mask_host(mask_nelem);

    auto rc = hipDeviceSynchronize();
    MIOPEN_LOG_T("hipDeviceSynchronize 1: " << rc);
    rc = hipMemcpy(
        bot_host.data(), params.x, bot_host.size() * sizeof(bot_host[0]), hipMemcpyDeviceToHost);
    MIOPEN_LOG_T("hipMemcpy bot: " << rc << ' ' << bot_host.data() << ' '
                                   << (bot_host.size() * sizeof(bot_host[0])));

    RunHost<T, Index>(args.pooling_method,
                      args.pad_d,
                      args.stride_d,
                      args.filter_d,
                      args.pad_h,
                      args.stride_h,
                      args.filter_h,
                      args.pad_w,
                      args.stride_w,
                      args.filter_w,
                      bot,
                      top,
                      bot_host.data(),
                      top_host.data(),
                      args.save_index,
                      mask_host.data(),
                      args.index_mode);

    rc = hipMemcpy(
        params.y, top_host.data(), top_host.size() * sizeof(top_host[0]), hipMemcpyHostToDevice);
    MIOPEN_LOG_T("hipMemcpy top: " << rc << ' ' << top_host.data() << ' '
                                   << (top_host.size() * sizeof(top_host[0])));

    if(args.save_index)
    {
        rc = hipMemcpy(params.workspace,
                       mask_host.data(),
                       mask_host.size() * sizeof(mask_host[0]),
                       hipMemcpyHostToDevice);
        MIOPEN_LOG_T("hipMemcpy mask: " << rc << ' ' << mask_host.data() << ' '
                                        << (mask_host.size() * sizeof(mask_host[0])));
    }
    rc = hipDeviceSynchronize();
    MIOPEN_LOG_T("hipDeviceSynchronize 2: " << rc);
}

} // namespace

bool PoolingForwardNaive::IsApplicable(const ExecutionContext&,
                                       const miopen::pooling::ProblemDescription& problem) const
{
    return problem.GetDirection() == miopen::pooling::Direction::Forward   //
           && problem.GetXDesc().GetType() == problem.GetYDesc().GetType() //
           && (problem.GetXDesc().GetType() == miopenFloat                 //
               || problem.GetXDesc().GetType() == miopenHalf)              //
           && (                                                            //
                  (problem.GetXDesc().GetSize() == 5                       //
                   && problem.GetXDesc().GetLayout("NCDHW") == "NCDHW"     //
                   && problem.GetYDesc().GetLayout("NCDHW") == "NCDHW")    //
                  ||                                                       //
                  (problem.GetXDesc().GetSize() == 4                       //
                   && problem.GetXDesc().GetLayout("NCHW") == "NCHW"       //
                   && problem.GetYDesc().GetLayout("NCHW") == "NCHW")      //
              );
}

ConvSolution
PoolingForwardNaive::GetSolution(const ExecutionContext&,
                                 const miopen::pooling::ProblemDescription& problem) const
{
    auto result     = ConvSolution{miopenStatusSuccess};
    const bool is2d = (problem.GetXDesc().GetSize() == 4);

    const auto& pooling = problem.GetPooling();
    const auto& lengths = pooling.GetLengths();
    const auto& strides = pooling.GetStrides();
    const auto& pads    = pooling.GetPads();

    // This also deduces 3D (DHW) parameters from 2D (HW) descriptor.
    const auto filter_w = lengths[is2d ? 1 : 2];
    const auto filter_h = lengths[is2d ? 0 : 1];
    const auto filter_d = is2d ? 1 : lengths[0];
    const auto stride_w = strides[is2d ? 1 : 2];
    const auto stride_h = strides[is2d ? 0 : 1];
    const auto stride_d = is2d ? (stride_h * filter_d) : strides[0];
    const auto pad_w    = pads[is2d ? 1 : 2];
    const auto pad_h    = pads[is2d ? 0 : 1];
    const auto pad_d    = is2d ? 0 : pads[0];

    const int pooling_method = (pooling.GetMode() == miopenPoolingMax) ? MLO_POOLING_OP_MAX
                               : (pooling.GetMode() == miopenPoolingAverage)
                                   ? MLO_POOLING_OP_AVE
                                   : MLO_POOLING_OP_AVE_INCLUSIVE;

    const auto bot = problem.GetXDesc();
    const auto top = problem.GetYDesc();

    const auto save_index = problem.SaveIndex();
    const auto index_mode = pooling.GetWorkspaceIndexMode();
    const auto index_type = pooling.GetIndexType();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
            std::ignore            = kernels;
            std::ignore            = handle;
            auto params            = raw_params.CastTo<miopen::pooling::FwdInvokeParams>();
            const arguments_t args = {pooling_method,
                                      pad_d,
                                      stride_d,
                                      filter_d,
                                      pad_h,
                                      stride_h,
                                      filter_h,
                                      pad_w,
                                      stride_w,
                                      filter_w,
                                      save_index,
                                      index_mode};

            if(bot.GetType() == miopenFloat)
            {
                switch(index_type)
                {
                case miopenIndexUint8:
                    RunGpuEmulation<float, uint8_t>(params, args, bot, top);
                    break;
                case miopenIndexUint16:
                    RunGpuEmulation<float, uint16_t>(params, args, bot, top);
                    break;
                case miopenIndexUint32:
                    RunGpuEmulation<float, uint32_t>(params, args, bot, top);
                    break;
                case miopenIndexUint64:
                    RunGpuEmulation<float, uint64_t>(params, args, bot, top);
                    break;
                }
            }
            else if(bot.GetType() == miopenHalf)
            {
                switch(index_type)
                {
                case miopenIndexUint8:
                    RunGpuEmulation<half_float::half, uint8_t>(params, args, bot, top);
                    break;
                case miopenIndexUint16:
                    RunGpuEmulation<half_float::half, uint16_t>(params, args, bot, top);
                    break;
                case miopenIndexUint32:
                    RunGpuEmulation<half_float::half, uint32_t>(params, args, bot, top);
                    break;
                case miopenIndexUint64:
                    RunGpuEmulation<half_float::half, uint64_t>(params, args, bot, top);
                    break;
                }
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "PoolingForwardNaive: unsupported data type");
            }
        };
    };

    return result;
}

std::size_t
PoolingForwardNaive::GetWorkspaceSize(const ExecutionContext&,
                                      const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax || !problem.SaveIndex())
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
