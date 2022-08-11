/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#ifndef MLO_POOLINGHOST_H_
#define MLO_POOLINGHOST_H_

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#endif

#include <cmath>
#include <cstring>
#include <iomanip>

#include "calcerr.hpp"

#if 0
template<typename _T>
double CalcErr( _T c_val, _T g_val)
{
	double err = 0;
	if (sizeof(_T) == 4)
	{
		int * c_uval = (int *)&c_val;
		int * g_uval = (int *)&g_val;
		err = (double)std::abs(*c_uval - *g_uval);
	}
	else if (sizeof(_T) == 8)
	{
		int64_t * c_uval = (int64_t *)&c_val;
		int64_t * g_uval = (int64_t *)&g_val;
		err = (double)std::abs(*c_uval - *g_uval);

	}

	//		double delta = abs(c_val - g_val);
	//	double nextafter_delta = nextafterf(min(abs(c_val), abs(g_val)), (_T)INFINITY) - min(abs(c_val), abs(g_val));
	//		err = delta / nextafter_delta;
	return err;
}
#endif

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

#ifndef MLO_POOLING_OP_MAX
#define MLO_POOLING_OP_MAX 0
#define MLO_POOLING_OP_AVE 1
#define MLO_POOLING_OP_STC 2
#define MLO_POOLING_OP_AVE_INCLUSIVE 3
#endif

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */,
          typename Index>
bool mloPoolingForwardRunHostAndVerify(int pooling_method,
                                       int pad_d,
                                       int pool_stride_d,
                                       int filter_size_d,
                                       int pad_h,
                                       int pool_stride_h,
                                       int filter_size_h,
                                       int pad_w,
                                       int pool_stride_w,
                                       int filter_size_w,
                                       const miopenTensorDescriptor_t& bot_,
                                       const miopenTensorDescriptor_t& top_,
                                       const Tgpu_* bot_ptr,
                                       const Tgpu_* top_ptr,
                                       bool do_backward,
                                       size_t* mask_ptr,
                                       Index* mask_gpu,
                                       Tcheck_ allowedEps,
                                       int index_position = 1)
{
    const miopen::TensorDescriptor& bot = miopen::deref(bot_);
    const miopen::TensorDescriptor& top = miopen::deref(top_);

    int n_batchs, n_outputs, bot_depth, bot_height, bot_width;
    int bot_w_stride, bot_h_stride, bot_d_stride, bot_c_stride, bot_n_stride;

    int top_depth, top_height, top_width;
    int top_w_stride, top_h_stride, top_d_stride, top_c_stride, top_n_stride;

    std::tie(n_batchs, n_outputs, bot_depth, bot_height, bot_width) =
        miopen::GetNCDHW(bot.GetSize(), bot.GetLengths());
    std::tie(bot_n_stride, bot_c_stride, bot_d_stride, bot_h_stride, bot_w_stride) =
        miopen::GetNCDHW(bot.GetSize(), bot.GetStrides());

    std::tie(std::ignore, std::ignore, top_depth, top_height, top_width) =
        miopen::GetNCDHW(top.GetSize(), top.GetLengths());
    std::tie(top_n_stride, top_c_stride, top_d_stride, top_h_stride, top_w_stride) =
        miopen::GetNCDHW(top.GetSize(), top.GetStrides());

    // Mask data is always NCDHW
    constexpr const int mask_w_stride = 1;
    const int mask_h_stride           = mask_w_stride * top_width;
    const int mask_d_stride           = mask_h_stride * top_height;
    const int mask_c_stride           = mask_d_stride * top_depth;
    const int mask_n_stride           = mask_c_stride * n_outputs;

    bool match = true;
    Tcheck_ MAX_VAL(3.402823466e+38);
    Tgpu_ G_MAX_VAL = (sizeof(Tgpu_) == 4 || sizeof(Tgpu_) == 8)
                          ? static_cast<Tgpu_>(3.402823466e+38)
                          : static_cast<Tgpu_>(65504);
    // c-emulator
    Tcheck_ res = static_cast<Tcheck_>(0);

    for(int b = 0; b < n_batchs && match; b++)
    {
        for(int o = 0; o < n_outputs && match; o++)
        {
            for(int k = 0; k < top_depth && match; k++)
            {
                for(int j = 0; j < top_height && match; j++)
                {
                    for(int i = 0; i < top_width && match; i++)
                    {
                        // c-emulator
                        if(pooling_method == MLO_POOLING_OP_MAX)
                        {
                            res = -MAX_VAL;
                        }
                        else if(pooling_method == MLO_POOLING_OP_AVE ||
                                pooling_method == MLO_POOLING_OP_AVE_INCLUSIVE)
                        {
                            res = static_cast<Tcheck_>(0);
                        }

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
                        size_t res_index     = 0;
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
                                        if(static_cast<Tcheck_>(bot_ptr[bot_index]) > res)
                                        {
                                            res       = static_cast<Tcheck_>(bot_ptr[bot_index]);
                                            res_index = bot_index;
                                            res_index_gpu =
                                                index_position == 1
                                                    ? (d * bot_height * bot_width + h * bot_width +
                                                       w)
                                                    : ((d - k * pool_stride_d + pad_d) *
                                                       filter_size_w * filter_size_h) +
                                                          ((h - j * pool_stride_h + pad_h) *
                                                           filter_size_w) +
                                                          (w - i * pool_stride_w + pad_w);
                                            found = true;
                                        }
                                    }
                                    else if(pooling_method == MLO_POOLING_OP_AVE ||
                                            pooling_method == MLO_POOLING_OP_AVE_INCLUSIVE)
                                    {

                                        res += static_cast<Tcheck_>(bot_ptr[bot_index]);
                                    }
                                    else
                                    {
                                        std::cout << "ERROR: unknown operator : layer: pooling."
                                                  << std::endl;
                                        match = false;
                                        continue;
                                    }
                                }
                            }
                        }
                        // special index value is used to mark top points which has no associated
                        // bottom
                        // points
                        if(!found)
                        {
                            res_index     = std::numeric_limits<size_t>::max();
                            res_index_gpu = std::numeric_limits<uint8_t>::max();
                        }

                        size_t top_index = b * top_n_stride + o * top_c_stride + k * top_d_stride +
                                           j * top_h_stride + i * top_w_stride;
                        size_t mask_gpu_index = b * mask_n_stride + o * mask_c_stride +
                                                k * mask_d_stride + j * mask_h_stride +
                                                i * mask_w_stride;
                        if(pooling_method == MLO_POOLING_OP_MAX)
                        {
                            // the case with the odd input, the even kernel size and 2*pad == kernel
                            // size
                            mask_ptr[top_index] = res_index;
                            if(do_backward)
                            {
                                size_t mg = mask_gpu[mask_gpu_index];
                                if(mg != res_index_gpu)
                                {
                                    std::cout << "Mask mismatch, gpu " << mg << " cpu "
                                              << res_index_gpu << "(" << res_index << ")"
                                              << std::endl;
                                    match = false;
                                }
                            }
                        }
                        if(pooling_method == MLO_POOLING_OP_AVE ||
                           pooling_method == MLO_POOLING_OP_AVE_INCLUSIVE)
                        {
                            res /= pool_size;
                        }
                        Tcheck_ c_val = res;

                        Tgpu_ gg_val = (top_ptr[top_index]);

                        gg_val = (Tgpu_(gg_val) == Tgpu_(-G_MAX_VAL)) ? Tgpu_(0) : Tgpu_(gg_val);

                        c_val = (c_val == -MAX_VAL) ? 0 : c_val;

                        Tcheck_ g_val(gg_val);

                        double err = std::abs(c_val - g_val);

                        if(err > allowedEps || std::isnan(c_val) || std::isnan(g_val) ||
                           !std::isfinite(c_val) || !std::isfinite(g_val))
                        {
                            std::cout << "Difference " << err << " too large at " << b << ", " << o
                                      << ", " << j << ", " << i << " c_v = " << c_val
                                      << " vs g_val = " << g_val << std::endl;
                            match = false;
                        }
                    }
                }
            }
        }
    }

    return (match);
}

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */>
int mloPoolingBackwardRunHost(
    int pooling_method,
    int filter_size_d,
    int pad_d,
    int pool_stride_d,
    int filter_size_h,
    int pad_h,
    int pool_stride_h,
    int filter_size_w,
    int pad_w,
    int pool_stride_w,

    const miopenTensorDescriptor_t& bot_df_,
    const miopenTensorDescriptor_t& top_df_,
    Tcheck_* bot_df_v_ptr, // the code assumes that bot_df_v_ptr was zeroed
    const Tgpu_* top_df_ptr,
    const size_t* mask_ptr)
{
    const miopen::TensorDescriptor& bot_df = miopen::deref(bot_df_);
    const miopen::TensorDescriptor& top_df = miopen::deref(top_df_);

    int n_outputs, n_batchs;

    int bot_w, bot_h, bot_d;
    int bot_df_n_stride, bot_df_c_stride, bot_df_d_stride, bot_df_h_stride, bot_df_w_stride;

    int top_w, top_h, top_d;
    int top_df_n_stride, top_df_c_stride, top_df_d_stride, top_df_h_stride, top_df_w_stride;

    std::tie(n_batchs, n_outputs, bot_d, bot_h, bot_w) =
        miopen::GetNCDHW(bot_df.GetSize(), bot_df.GetLengths());
    std::tie(bot_df_n_stride, bot_df_c_stride, bot_df_d_stride, bot_df_h_stride, bot_df_w_stride) =
        miopen::GetNCDHW(bot_df.GetSize(), bot_df.GetStrides());

    std::tie(std::ignore, std::ignore, top_d, top_h, top_w) =
        miopen::GetNCDHW(top_df.GetSize(), top_df.GetLengths());
    std::tie(top_df_n_stride, top_df_c_stride, top_df_d_stride, top_df_h_stride, top_df_w_stride) =
        miopen::GetNCDHW(top_df.GetSize(), top_df.GetStrides());

    int ret = 0;

    for(int b = 0; b < n_batchs; b++)
    {
        for(int o = 0; o < n_outputs; o++)
        {
            int bot_df_v_off = b * bot_df_n_stride + o * bot_df_c_stride;
            int top_df_off   = b * top_df_n_stride + o * top_df_c_stride;

            if(pooling_method == MLO_POOLING_OP_MAX)
            {
                for(int k = 0; k < top_d; k++)
                {
                    for(int j = 0; j < top_h; j++)
                    {
                        for(int i = 0; i < top_w; i++)
                        {
                            size_t top_idx = top_df_off + k * top_df_d_stride +
                                             j * top_df_h_stride + i * top_df_w_stride;
                            size_t bot_idx = mask_ptr[top_idx];
                            // skip top points that don't have associated bottom points
                            if(bot_idx == std::numeric_limits<size_t>::max())
                                continue;
                            bot_df_v_ptr[bot_idx] += static_cast<Tcheck_>(top_df_ptr[top_idx]);
                        }
                    }
                }
            }
            else if(pooling_method == MLO_POOLING_OP_AVE ||
                    pooling_method == MLO_POOLING_OP_AVE_INCLUSIVE)
            {

                for(int k = 0; k < bot_d; k++)
                {
                    for(int j = 0; j < bot_h; j++)
                    {
                        for(int i = 0; i < bot_w; i++)
                        {
                            // c-emulator
                            const auto bot_idx = bot_df_v_off + k * bot_df_d_stride +
                                                 j * bot_df_h_stride + i * bot_df_w_stride;
                            bot_df_v_ptr[bot_idx] = static_cast<Tcheck_>(0);

                            int d = k + pad_d;
                            int h = j + pad_h;
                            int w = i + pad_w;
                            int pdstart =
                                (d < filter_size_d) ? 0 : (d - filter_size_d) / pool_stride_d + 1;
                            int pdend = std::min(d / pool_stride_d + 1, top_d);
                            int phstart =
                                (h < filter_size_h) ? 0 : (h - filter_size_h) / pool_stride_h + 1;
                            int phend = std::min(h / pool_stride_h + 1, top_h);
                            int pwstart =
                                (w < filter_size_w) ? 0 : (w - filter_size_w) / pool_stride_w + 1;
                            int pwend        = std::min(w / pool_stride_w + 1, top_w);
                            Tcheck_ gradient = static_cast<Tcheck_>(0);
                            for(int pd = pdstart; pd < pdend; ++pd)
                            {
                                for(int ph = phstart; ph < phend; ++ph)
                                {
                                    for(int pw = pwstart; pw < pwend; ++pw)
                                    {
                                        // figure out the pooling size
                                        int dstart = pd * pool_stride_d - pad_d;
                                        int hstart = ph * pool_stride_h - pad_h;
                                        int wstart = pw * pool_stride_w - pad_w;
                                        int dend   = std::min(dstart + filter_size_d, bot_d);
                                        int hend   = std::min(hstart + filter_size_h, bot_h);
                                        int wend   = std::min(wstart + filter_size_w, bot_w);
                                        dstart     = std::max(dstart, 0);
                                        hstart     = std::max(hstart, 0);
                                        wstart     = std::max(wstart, 0);

                                        int pool_size;
                                        if(pooling_method == MLO_POOLING_OP_AVE)
                                            pool_size = ((dend - dstart) * (hend - hstart) *
                                                             (wend - wstart) ==
                                                         0)
                                                            ? 1
                                                            : (dend - dstart) * (hend - hstart) *
                                                                  (wend - wstart);
                                        else
                                            pool_size =
                                                (filter_size_w * filter_size_h * filter_size_d == 0)
                                                    ? 1
                                                    : filter_size_w * filter_size_h * filter_size_d;

                                        const auto top_idx = top_df_off + pd * top_df_d_stride +
                                                             ph * top_df_h_stride +
                                                             pw * top_df_w_stride;

                                        gradient += static_cast<Tcheck_>(top_df_ptr[top_idx]) /
                                                    static_cast<Tcheck_>(pool_size);
                                    }
                                }
                            }
                            bot_df_v_ptr[bot_idx] = gradient;
                        }
                    }
                }
            }
            else
            {
                std::cout << "ERROR: unknown operator : layer: pooling back-propagation."
                          << std::endl;
                continue;
            }
        }
    }
    return (ret);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
