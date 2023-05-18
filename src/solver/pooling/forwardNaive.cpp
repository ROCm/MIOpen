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

#define HOST_IMPL 1

namespace miopen {

namespace solver {

namespace pooling {

#if HOST_IMPL
namespace {

template <typename T, typename Index>
void RunHost(uint32_t b,
             uint32_t o,
             uint32_t i,
             const T* bot_ptr,
             T* top_ptr,
             Index* mask_ptr,
             int pooling_method,
             bool save_index,
             int index_mode,
             uint32_t pad_d,
             uint32_t pool_d_stride,
             uint32_t filter_d,
             uint32_t pad_h,
             uint32_t pool_h_stride,
             uint32_t filter_h,
             uint32_t pad_w,
             uint32_t pool_w_stride,
             uint32_t filter_w,
             uint32_t bot_d,
             uint32_t bot_h,
             uint32_t bot_w,
             uint32_t bot_w_stride,
             uint32_t bot_h_stride,
             uint32_t bot_d_stride,
             size_t bot_c_stride,
             size_t bot_n_stride,
             uint32_t top_d,
             uint32_t top_h,
             uint32_t top_w_stride,
             uint32_t top_h_stride,
             uint32_t top_d_stride,
             size_t top_c_stride,
             size_t top_n_stride,
             uint32_t mask_w_stride,
             uint32_t mask_h_stride,
             uint32_t mask_d_stride,
             size_t mask_c_stride,
             size_t mask_n_stride)
{
    const T MAX_VAL = std::numeric_limits<T>::max();

    for(uint32_t j = 0; j < top_h; ++j)
    {
        for(uint32_t k = 0; k < top_d; ++k)
        {
            double res;
            if(pooling_method == MLO_POOLING_OP_MAX)
                res = -MAX_VAL;
            else
                res = 0;

            const int int_dstart = k * pool_d_stride - pad_d;
            const int int_hstart = j * pool_h_stride - pad_h;
            const int int_wstart = i * pool_w_stride - pad_w;
            const uint32_t dend =
                std::min(int_dstart + static_cast<int>(filter_d), static_cast<int>(bot_d));
            const uint32_t hend =
                std::min(int_hstart + static_cast<int>(filter_h), static_cast<int>(bot_h));
            const uint32_t wend =
                std::min(int_wstart + static_cast<int>(filter_w), static_cast<int>(bot_w));
            const uint32_t dstart = std::max(int_dstart, 0);
            const uint32_t hstart = std::max(int_hstart, 0);
            const uint32_t wstart = std::max(int_wstart, 0);

            uint32_t pool_size;
            if(pooling_method == MLO_POOLING_OP_AVE)
                pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            else
                pool_size = filter_w * filter_h * filter_d;
            pool_size = (pool_size == 0) ? 1 : pool_size;

            bool found = false; // This may remain false only if the input tensor
                                // contains only NaNs and -INFs.
            uint32_t d_save = 0, h_save = 0, w_save = 0;
            for(uint32_t d = dstart; d < dend; ++d)
            {
                for(uint32_t h = hstart; h < hend; ++h)
                {
                    for(uint32_t w = wstart; w < wend; ++w)
                    {
                        const size_t bot_index = b * bot_n_stride + o * bot_c_stride +
                                                 static_cast<size_t>(d * bot_d_stride) +
                                                 static_cast<size_t>(h * bot_h_stride) +
                                                 static_cast<size_t>(w * bot_w_stride);
                        if(pooling_method == MLO_POOLING_OP_MAX)
                        {
                            if(bot_ptr[bot_index] > res)
                            {
                                res = bot_ptr[bot_index];
                                if(save_index)
                                {
                                    found  = true;
                                    d_save = d;
                                    h_save = h;
                                    w_save = w;
                                }
                            }
                        }
                        else // Average
                        {
                            res += bot_ptr[bot_index];
                        }
                    }
                }
            }

            if(pooling_method == MLO_POOLING_OP_MAX && save_index)
            {
                Index res_index = 0;

                /// The warning happens during computation of res_index when Index is wider than
                /// uint32_t. At the first glance, it seems like in this case we need to cast to
                /// Index before multiplication. But if we cast before AND Index is shorter than
                /// uint32_t, then the range of computation would shrink and that may affect
                /// correctness. However casting to Index before multiplcation is actually not
                /// necessary, see \ref multiply_dims_overflow_assumption. Let's simply shut the
                /// warning.
                if(found)
                {
                    // NOLINTBEGIN(bugprone-misplaced-widening-cast)
                    if(index_mode == 1)
                        res_index =
                            static_cast<Index>(d_save * bot_h * bot_w + h_save * bot_w + w_save);
                    else
                        res_index = static_cast<Index>(                                  //
                            ((d_save - k * pool_d_stride + pad_d) * filter_w * filter_h) //
                            + ((h_save - j * pool_h_stride + pad_h) * filter_w)          //
                            + (w_save - i * pool_w_stride + pad_w)                       //
                        );
                    // NOLINTEND(bugprone-misplaced-widening-cast)
                }

                const size_t mask_index =
                    b * mask_n_stride + o * mask_c_stride + static_cast<size_t>(k * mask_d_stride) +
                    static_cast<size_t>(j * mask_h_stride) + static_cast<size_t>(i * mask_w_stride);
                mask_ptr[mask_index] = res_index;
            }

            if(pooling_method == MLO_POOLING_OP_AVE ||
               pooling_method == MLO_POOLING_OP_AVE_INCLUSIVE)
            {
                res /= pool_size;
            }
            const size_t top_index =
                b * top_n_stride + o * top_c_stride + static_cast<size_t>(k * top_d_stride) +
                static_cast<size_t>(j * top_h_stride) + static_cast<size_t>(i * top_w_stride);

            top_ptr[top_index] = static_cast<T>(res);
        }
    }
}

struct arguments_t // Syntax sugar.
{
    int pooling_method;
    uint32_t pad_d;
    uint32_t stride_d;
    uint32_t filter_d;
    uint32_t pad_h;
    uint32_t stride_h;
    uint32_t filter_h;
    uint32_t pad_w;
    uint32_t stride_w;
    uint32_t filter_w;
    bool save_index;
    int index_mode;
    uint32_t n_batchs, n_outputs, bot_d, bot_h, bot_w;
    uint32_t bot_w_stride, bot_h_stride, bot_d_stride;
    size_t bot_c_stride, bot_n_stride;
    uint32_t top_d, top_h, top_w;
    uint32_t top_w_stride, top_h_stride, top_d_stride;
    size_t top_c_stride, top_n_stride;
    uint32_t mask_w_stride;
    uint32_t mask_h_stride;
    uint32_t mask_d_stride;
    size_t mask_c_stride;
    size_t mask_n_stride;
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

    for(uint32_t b = 0; b < args.n_batchs; ++b)
    {
        for(uint32_t o = 0; o < args.n_outputs; ++o)
        {
            for(uint32_t i = 0; i < args.top_w; ++i)
            {
                RunHost<T, Index>(b,
                                  o,
                                  i,
                                  bot_host.data(),
                                  top_host.data(),
                                  mask_host.data(),
                                  args.pooling_method,
                                  args.save_index,
                                  args.index_mode,
                                  args.pad_d,
                                  args.stride_d,
                                  args.filter_d,
                                  args.pad_h,
                                  args.stride_h,
                                  args.filter_h,
                                  args.pad_w,
                                  args.stride_w,
                                  args.filter_w,
                                  args.bot_d,
                                  args.bot_h,
                                  args.bot_w,
                                  args.bot_w_stride,
                                  args.bot_h_stride,
                                  args.bot_d_stride,
                                  args.bot_c_stride,
                                  args.bot_n_stride,
                                  args.top_d,
                                  args.top_h,
                                  args.top_w_stride,
                                  args.top_h_stride,
                                  args.top_d_stride,
                                  args.top_c_stride,
                                  args.top_n_stride,
                                  args.mask_w_stride,
                                  args.mask_h_stride,
                                  args.mask_d_stride,
                                  args.mask_c_stride,
                                  args.mask_n_stride);
            }
        }
    }

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
#endif // HOST_IMPL

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
    auto result = ConvSolution{miopenStatusSuccess};

    const auto bot  = problem.GetXDesc();
    const auto top  = problem.GetYDesc();
    const bool is2d = (bot.GetSize() == 4);

    // To compact code:
    const auto& pooling = problem.GetPooling();
    const auto& lengths = pooling.GetLengths();
    const auto& strides = pooling.GetStrides();
    const auto& pads    = pooling.GetPads();

    // This also deduces 3D (DHW) parameters from 2D (HW) descriptor.
    const uint32_t filter_w = lengths[is2d ? 1 : 2];
    const uint32_t filter_h = lengths[is2d ? 0 : 1];
    const uint32_t filter_d = is2d ? 1 : lengths[0];
    const uint32_t stride_w = strides[is2d ? 1 : 2];
    const uint32_t stride_h = strides[is2d ? 0 : 1];
    const uint32_t stride_d = is2d ? (stride_h * filter_d) : strides[0];
    const uint32_t pad_w    = pads[is2d ? 1 : 2];
    const uint32_t pad_h    = pads[is2d ? 0 : 1];
    const uint32_t pad_d    = is2d ? 0 : pads[0];

    const int pooling_method = (pooling.GetMode() == miopenPoolingMax) ? MLO_POOLING_OP_MAX
                               : (pooling.GetMode() == miopenPoolingAverage)
                                   ? MLO_POOLING_OP_AVE
                                   : MLO_POOLING_OP_AVE_INCLUSIVE;

    const auto save_index = problem.SaveIndex();
    const auto index_mode = pooling.GetWorkspaceIndexMode();
    const auto index_type = pooling.GetIndexType();

    /// \anchor multiply_dims_overflow_assumption
    ///
    /// Preventing overflow during dimension-related computations:
    /// Let's assume that multiplication of three dims always fits into 32 bits (unsigned).
    /// Then let's use size_t when we need to multiply more than three dims.
    /// For example, in NCDHW layout, the N and C strides are results of multiplication
    /// of >= 3 dims, so we have to use size_t for storing them.
    ///
    /// We need to pay special attention to muls of D stride with some other dims.
    /// The D stride is a result of 2 muls. Therefore (d_stride * dim) does
    /// not require widening to size_t prior mul, but (d_stride * dim * dim)
    /// requires it because the total number of muls is 4.

    const auto spatial_dim = is2d ? 2 : 3;
    uint32_t n_batchs, n_outputs, bot_d, bot_h, bot_w;
    uint32_t bot_w_stride, bot_h_stride, bot_d_stride;
    size_t bot_c_stride, bot_n_stride;
    std::tie(n_batchs, n_outputs, bot_d, bot_h, bot_w) =
        miopen::GetNCDHW(spatial_dim, bot.GetLengths());
    std::tie(bot_n_stride, bot_c_stride, bot_d_stride, bot_h_stride, bot_w_stride) =
        miopen::GetNCDHW(spatial_dim, bot.GetStrides());

    uint32_t top_d, top_h, top_w;
    uint32_t top_w_stride, top_h_stride, top_d_stride;
    size_t top_c_stride, top_n_stride;
    std::tie(std::ignore, std::ignore, top_d, top_h, top_w) =
        miopen::GetNCDHW(spatial_dim, top.GetLengths());
    std::tie(top_n_stride, top_c_stride, top_d_stride, top_h_stride, top_w_stride) =
        miopen::GetNCDHW(spatial_dim, top.GetStrides());

    // Mask data is always NCDHW
    const uint32_t mask_w_stride = 1;
    const uint32_t mask_h_stride = mask_w_stride * top_w;
    const uint32_t mask_d_stride = mask_h_stride * top_h;
    const size_t mask_c_stride   = static_cast<size_t>(mask_d_stride) * top_d;
    const size_t mask_n_stride   = mask_c_stride * n_outputs;

#if !HOST_IMPL
    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPoolingForwardNaive.cl";
        kernel.kernel_name = "mloPoolingForwardNaive";

        auto build_params = KernelBuildParameters{
            {"MLO_POOLING_OP_ID", pooling_method}, // We need this at compile time in order to
                                                   // engage mixed precision only when necessary.
            {"MLO_POOLING_INDEX_TYPE", get_pooling_index_type_name(pooling.GetIndexType())},
        };
        build_params << GetDataTypeKBP(bot.GetType());
        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        // [Informative] The total number of kernels required to cover all the supported pooling
        // configs:
        // - 3: the number of supported operations
        // - 4: the number of supported index types
        // - 2: the number of supported data types
        // The total number of kernels is 3*4*2=24.
        // The solver is dynamic.

        kernel.g_wk.push_back(n_batchs);
        kernel.g_wk.push_back(n_outputs);
        kernel.g_wk.push_back(top_w);
        // local_work_size is not needed as there is no need for synchronization between workitems.
        kernel.l_wk.push_back(0);

        result.construction_params.push_back(kernel);
    }
#endif

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
#if !HOST_IMPL
#else  // HOST_IMPL
            std::ignore            = kernels;
            std::ignore            = handle;
            auto params            = raw_params.CastTo<miopen::pooling::FwdInvokeParams>();
            const arguments_t args = {
                pooling_method, pad_d,         stride_d,      filter_d,      pad_h,
                stride_h,       filter_h,      pad_w,         stride_w,      filter_w,
                save_index,     index_mode,    n_batchs,      n_outputs,     bot_d,
                bot_h,          bot_w,         bot_w_stride,  bot_h_stride,  bot_d_stride,
                bot_c_stride,   bot_n_stride,  top_d,         top_h,         top_w,
                top_w_stride,   top_h_stride,  top_d_stride,  top_c_stride,  top_n_stride,
                mask_w_stride,  mask_h_stride, mask_d_stride, mask_c_stride, mask_n_stride,
            };

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
#endif // HOST_IMPL
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
