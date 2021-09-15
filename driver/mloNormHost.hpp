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

#ifndef MLO_NORMHOST_H_
#define MLO_NORMHOST_H_

#include <cmath>
#include <iomanip>

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

#ifndef MLO_LRN_WITHIN_CHANNEL
#define MLO_LRN_WITHIN_CHANNEL 0
#define MLO_LRN_ACROSS_CHANNELS 1
#endif

template <typename _Tgpu /* the data type used in GPU computations (usually half) */,
          typename _Tcheck /* the data type used in CPU checkings (usually double) */>
int mloLRNForwardRunHost(bool do_scale,
                         int norm_region,
                         int pad,
                         int local_area,
                         _Tcheck alphaoverarea,
                         _Tcheck alpha,
                         _Tcheck beta,
                         _Tcheck K,
                         int n_batchs,
                         int n_outputs,
                         int n_inputs,
                         int bot_height,
                         int bot_width,
                         int bot_stride,
                         int bot_channel_stride,
                         int bot_batch_stride,
                         int top_height,
                         int top_width,
                         int top_v_stride,
                         int top_v_channel_stride,
                         int top_v_batch_stride,
                         int scale_v_stride,
                         int scale_v_channel_stride,
                         int scale_v_batch_stride,
                         const _Tgpu* bot_ptr,
                         _Tcheck* scale_v_ptr,
                         _Tcheck* top_v_ptr)
{

    int ret = 0;
    if(local_area < 1 + pad)
    {
        std::cout << "ERROR: Lrn kernel size is insufficient." << std::endl;
        return -1;
    }

    if(norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        for(int b = 0; b < n_batchs; b++)
        {
            for(int j = 0; j < top_height; j++)
            {
                for(int i = 0; i < top_width; i++)
                {
                    // c-emulator
                    _Tcheck accum_scale = _Tcheck{0};
                    int head            = 0;
                    _Tcheck bot_val;
                    while(head < pad)
                    {
                        bot_val = (head < n_inputs)
                                      ? static_cast<_Tcheck>(
                                            bot_ptr[b * bot_batch_stride +
                                                    head * bot_channel_stride + j * bot_stride + i])
                                      : static_cast<_Tcheck>(0);
                        accum_scale += bot_val * bot_val;
                        ++head;
                    }
                    // until we reach size, nothing needs to be subtracted
                    while(head < local_area)
                    {
                        bot_val = (head < n_inputs)
                                      ? static_cast<_Tcheck>(
                                            bot_ptr[b * bot_batch_stride +
                                                    head * bot_channel_stride + j * bot_stride + i])
                                      : static_cast<_Tcheck>(0);
                        accum_scale += bot_val * bot_val;
                        _Tcheck scale = K + accum_scale * alphaoverarea;
                        if((head - pad) >= 0 && (head - pad) < n_outputs && do_scale)
                        {
                            scale_v_ptr[b * scale_v_batch_stride +
                                        (head - pad) * scale_v_channel_stride + j * scale_v_stride +
                                        i] = scale;
                        }
                        bot_val =
                            ((head - pad) >= 0 && (head - pad) < n_inputs)
                                ? static_cast<_Tcheck>(bot_ptr[b * bot_batch_stride +
                                                               (head - pad) * bot_channel_stride +
                                                               j * bot_stride + i])
                                : static_cast<_Tcheck>(0);
                        _Tcheck s     = pow(scale, -beta);
                        _Tcheck c_val = bot_val * s;
                        if((head - pad) >= 0 && (head - pad) < n_outputs)
                        {
                            top_v_ptr[b * top_v_batch_stride + (head - pad) * top_v_channel_stride +
                                      j * top_v_stride + i] = c_val;
                        }
                        ++head;
                    }
                    // both add and subtract
                    while(head < n_inputs)
                    {
                        bot_val = static_cast<_Tcheck>(
                            bot_ptr[b * bot_batch_stride + head * bot_channel_stride +
                                    j * bot_stride + i]);
                        accum_scale += bot_val * bot_val;
                        bot_val = ((head - local_area) >= 0)
                                      ? static_cast<_Tcheck>(
                                            bot_ptr[b * bot_batch_stride +
                                                    (head - local_area) * bot_channel_stride +
                                                    j * bot_stride + i])
                                      : static_cast<_Tcheck>(0);
                        accum_scale -= bot_val * bot_val;
                        _Tcheck scale = K + accum_scale * alphaoverarea;
                        if((head - pad) >= 0 && do_scale)
                        {
                            scale_v_ptr[b * scale_v_batch_stride +
                                        (head - pad) * scale_v_channel_stride + j * scale_v_stride +
                                        i] = scale;
                        }
                        _Tcheck s = pow(scale, -beta);
                        bot_val =
                            ((head - pad) >= 0)
                                ? static_cast<_Tcheck>(bot_ptr[b * bot_batch_stride +
                                                               (head - pad) * bot_channel_stride +
                                                               j * bot_stride + i])
                                : static_cast<_Tcheck>(0);
                        _Tcheck c_val = bot_val * s;
                        if((head - pad) >= 0)
                        {
                            top_v_ptr[b * top_v_batch_stride + (head - pad) * top_v_channel_stride +
                                      j * top_v_stride + i] = c_val;
                        }
                        ++head;
                    }
                    // subtract only
                    while(head < n_inputs + pad)
                    {
                        bot_val = ((head - local_area) >= 0 && (head - local_area) < n_inputs)
                                      ? static_cast<_Tcheck>(
                                            bot_ptr[b * bot_batch_stride +
                                                    (head - local_area) * bot_channel_stride +
                                                    j * bot_stride + i])
                                      : static_cast<_Tcheck>(0);
                        accum_scale -= bot_val * bot_val;
                        _Tcheck scale = K + accum_scale * alphaoverarea;
                        if((head - pad) >= 0 && (head - pad) < n_outputs && do_scale)
                        {
                            scale_v_ptr[b * scale_v_batch_stride +
                                        (head - pad) * scale_v_channel_stride + j * scale_v_stride +
                                        i] = scale;
                        }
                        bot_val =
                            ((head - pad) >= 0 && (head - pad) < n_inputs)
                                ? static_cast<_Tcheck>(bot_ptr[b * bot_batch_stride +
                                                               (head - pad) * bot_channel_stride +
                                                               j * bot_stride + i])
                                : static_cast<_Tcheck>(0);
                        _Tcheck s     = pow(scale, -beta);
                        _Tcheck c_val = bot_val * s;
                        if((head - pad) >= 0 && (head - pad) < n_outputs)
                        {
                            top_v_ptr[b * top_v_batch_stride + (head - pad) * top_v_channel_stride +
                                      j * top_v_stride + i] = c_val;
                        }
                        ++head;
                    }

                } // for (int i = 0; i < top_width; i++)
            }     // for (int j = 0; j < top_height; j++)
        }         // for (int b = 0; b < batch; b++)
    }
    else
    {
        for(int b = 0; b < n_batchs; b++)
        {
            for(int o = 0; o < n_outputs; o++)
            {
                for(int j = 0; j < top_height; j++)
                {
                    for(int i = 0; i < top_width; i++)
                    {
                        // c-emulator
                        _Tcheck scale     = static_cast<_Tcheck>(0);
                        int hstart        = j - (local_area - 1 - pad);
                        int wstart        = i - (local_area - 1 - pad);
                        int hend          = std::min(hstart + local_area, bot_height + pad);
                        int wend          = std::min(wstart + local_area, bot_width + pad);
                        int adj_area_size = (hend - hstart) * (wend - wstart);
                        hstart            = std::max(hstart, 0);
                        wstart            = std::max(wstart, 0);
                        hend              = std::min(hend, bot_height);
                        wend              = std::min(wend, bot_width);
                        _Tcheck accum     = static_cast<_Tcheck>(0);
                        for(int h = hstart; h < hend; ++h)
                        {
                            for(int w = wstart; w < wend; ++w)
                            {

                                _Tcheck bot_val = static_cast<_Tcheck>(
                                    bot_ptr[b * bot_batch_stride + o * bot_channel_stride +
                                            h * bot_stride + w]);
                                accum += bot_val * bot_val;
                            }
                        }

                        alphaoverarea = alpha / adj_area_size;
                        scale         = K + accum * alphaoverarea;
                        if(do_scale)
                        {
                            scale_v_ptr[b * scale_v_batch_stride + o * scale_v_channel_stride +
                                        j * scale_v_stride + i] = scale;
                        }

                        _Tcheck s       = pow(scale, -beta);
                        _Tcheck bot_val = static_cast<_Tcheck>(
                            bot_ptr[b * bot_batch_stride + o * bot_channel_stride + j * bot_stride +
                                    i]);
                        _Tcheck c_val = bot_val * s;

                        top_v_ptr[b * top_v_batch_stride + o * top_v_channel_stride +
                                  j * top_v_stride + i] = c_val;

                    } // for (int i = 0; i < top_width; i++)
                }     // for (int j = 0; j < top_height; j++)
            }         // for (int o = 0; o < outputs; o++)
        }             // for (int b = 0; b < batch; b++)
    }                 // (norm_region == ACROSS_CHANNELS)

    return (ret);
}

template <typename _Tgpu /* the data type used in GPU computations (usually half) */,
          typename _Tcheck /* the data type used in CPU checkings (usually double) */>
int mloLRNBackwardRunHost(int norm_region,
                          int pad,
                          int local_area,
                          _Tcheck /*alphaoverarea*/,
                          _Tcheck alpha,
                          _Tcheck beta,
                          _Tcheck /*K*/,
                          int n_batchs,
                          int /*n_outputs*/,
                          int n_inputs,
                          int bot_height,
                          int bot_width,
                          int bot_stride,
                          int bot_channel_stride,
                          int bot_batch_stride,
                          int bot_df_v_stride,
                          int bot_df_v_channel_stride,
                          int bot_df_v_batch_stride,
                          int top_height,
                          int top_width,
                          int top_stride,
                          int top_channel_stride,
                          int top_batch_stride,
                          int top_df_stride,
                          int top_df_channel_stride,
                          int top_df_batch_stride,
                          int scale_stride,
                          int scale_channel_stride,
                          int scale_batch_stride,
                          const _Tgpu* top_ptr,
                          const _Tgpu* top_df_ptr,
                          const _Tgpu* scale_ptr,
                          const _Tgpu* bot_ptr,
                          _Tcheck* bot_df_v_ptr)
{

    int ret               = 0;
    _Tcheck negative_beta = -beta;
    int pre_pad           = local_area - 1 - pad;
    if(pre_pad < 0)
    {
        std::cout << "ERROR: Lrn kernel size is insufficient." << std::endl;
        return -1;
    }

    if(norm_region == MLO_LRN_ACROSS_CHANNELS)
    {

        _Tcheck ratio_dta_bwd =
            static_cast<_Tcheck>(2.) * alpha * beta / static_cast<_Tcheck>(local_area);

        for(int b = 0; b < n_batchs; b++)
        {
            for(int j = 0; j < bot_height; j++)
            {
                for(int i = 0; i < bot_width; i++)
                {

                    // c-emulator
                    int head            = 0;
                    _Tcheck accum_ratio = static_cast<_Tcheck>(0);

                    // accumulate values
                    while(head < pre_pad)
                    {
                        if(head < n_inputs)
                        {
                            _Tcheck adder =
                                (static_cast<_Tcheck>(top_df_ptr[b * top_df_batch_stride +
                                                                 head * top_df_channel_stride +
                                                                 j * top_df_stride + i]) *
                                 static_cast<_Tcheck>(
                                     top_ptr[b * top_batch_stride + head * top_channel_stride +
                                             j * top_stride + i])) /
                                static_cast<_Tcheck>(
                                    scale_ptr[b * scale_batch_stride + head * scale_channel_stride +
                                              j * scale_stride + i]);

                            accum_ratio += adder;
                        }

                        ++head;
                    }

                    // until we reach size, nothing needs to be subtracted
                    while(head < local_area)
                    {

                        if(head < n_inputs)
                        {
                            _Tcheck adder =
                                (static_cast<_Tcheck>(top_df_ptr[b * top_df_batch_stride +
                                                                 head * top_df_channel_stride +
                                                                 j * top_df_stride + i]) *
                                 static_cast<_Tcheck>(
                                     top_ptr[b * top_batch_stride + head * top_channel_stride +
                                             j * top_stride + i])) /
                                static_cast<_Tcheck>(
                                    scale_ptr[b * scale_batch_stride + head * scale_channel_stride +
                                              j * scale_stride + i]);

                            accum_ratio += adder;
                        }

                        if(head - pre_pad >= 0 && head - pre_pad < n_inputs)
                        {
                            bot_df_v_ptr[b * bot_df_v_batch_stride +
                                         (head - pre_pad) * bot_df_v_channel_stride +
                                         j * bot_df_v_stride + i] =
                                static_cast<_Tcheck>(
                                    top_df_ptr[b * top_df_batch_stride +
                                               (head - pre_pad) * top_df_channel_stride +
                                               j * top_df_stride + i]) *
                                    pow(static_cast<_Tcheck>(
                                            scale_ptr[b * scale_batch_stride +
                                                      (head - pre_pad) * scale_channel_stride +
                                                      j * scale_stride + i]),
                                        negative_beta) -
                                ratio_dta_bwd *
                                    static_cast<_Tcheck>(
                                        bot_ptr[b * bot_batch_stride +
                                                (head - pre_pad) * bot_channel_stride +
                                                j * bot_stride + i]) *
                                    accum_ratio;
                        }
                        ++head;
                    }

                    // both add and subtract
                    while(head < n_inputs)
                    {

                        _Tcheck adder =
                            static_cast<_Tcheck>(
                                top_df_ptr[b * top_df_batch_stride + head * top_df_channel_stride +
                                           j * top_df_stride + i]) *
                            static_cast<_Tcheck>(
                                top_ptr[b * top_batch_stride + head * top_channel_stride +
                                        j * top_stride + i]) /
                            static_cast<_Tcheck>(
                                scale_ptr[b * scale_batch_stride + head * scale_channel_stride +
                                          j * scale_stride + i]);

                        accum_ratio += adder;

                        if(head - local_area >= 0)
                        {
                            _Tcheck subs =
                                (static_cast<_Tcheck>(
                                     top_df_ptr[b * top_df_batch_stride +
                                                (head - local_area) * top_df_channel_stride +
                                                j * top_df_stride + i]) *
                                 static_cast<_Tcheck>(
                                     top_ptr[b * top_batch_stride +
                                             (head - local_area) * top_channel_stride +
                                             j * top_stride + i])) /
                                static_cast<_Tcheck>(
                                    scale_ptr[b * scale_batch_stride +
                                              (head - local_area) * scale_channel_stride +
                                              j * scale_stride + i]);

                            accum_ratio -= subs;
                        }
                        if(head - pre_pad >= 0)
                        {
                            bot_df_v_ptr[b * bot_df_v_batch_stride +
                                         (head - pre_pad) * bot_df_v_channel_stride +
                                         j * bot_df_v_stride + i] =
                                static_cast<_Tcheck>(
                                    top_df_ptr[b * top_df_batch_stride +
                                               (head - pre_pad) * top_df_channel_stride +
                                               j * top_df_stride + i]) *
                                    pow(static_cast<_Tcheck>(
                                            scale_ptr[b * scale_batch_stride +
                                                      (head - pre_pad) * scale_channel_stride +
                                                      j * scale_stride + i]),
                                        negative_beta) -
                                ratio_dta_bwd *
                                    static_cast<_Tcheck>(
                                        bot_ptr[b * bot_batch_stride +
                                                (head - pre_pad) * bot_channel_stride +
                                                j * bot_stride + i]) *
                                    accum_ratio;
                        }

                        ++head;
                    }
                    // subtract only
                    while(head < n_inputs + pre_pad)
                    {
                        if(head - local_area >= 0 && head - local_area < n_inputs)
                        {
                            _Tcheck subs =
                                (static_cast<_Tcheck>(
                                     top_df_ptr[b * top_df_batch_stride +
                                                (head - local_area) * top_df_channel_stride +
                                                j * top_df_stride + i]) *
                                 static_cast<_Tcheck>(
                                     top_ptr[b * top_batch_stride +
                                             (head - local_area) * top_channel_stride +
                                             j * top_stride + i])) /
                                static_cast<_Tcheck>(
                                    scale_ptr[b * scale_batch_stride +
                                              (head - local_area) * scale_channel_stride +
                                              j * scale_stride + i]);

                            accum_ratio -= subs;
                        }
                        if(head - pre_pad >= 0 && head - pre_pad < n_inputs)
                        {
                            bot_df_v_ptr[b * bot_df_v_batch_stride +
                                         (head - pre_pad) * bot_df_v_channel_stride +
                                         j * bot_df_v_stride + i] =
                                static_cast<_Tcheck>(
                                    top_df_ptr[b * top_df_batch_stride +
                                               (head - pre_pad) * top_df_channel_stride +
                                               j * top_df_stride + i]) *
                                    pow(static_cast<_Tcheck>(
                                            scale_ptr[b * scale_batch_stride +
                                                      (head - pre_pad) * scale_channel_stride +
                                                      j * scale_stride + i]),
                                        negative_beta) -
                                ratio_dta_bwd *
                                    static_cast<_Tcheck>(
                                        bot_ptr[b * bot_batch_stride +
                                                (head - pre_pad) * bot_channel_stride +
                                                j * bot_stride + i]) *
                                    accum_ratio;
                        }

                        ++head;
                    }

                } // for (int i = 0; i < bot_width; i++)
            }     // for (int j = 0; j < bot_height; j++)
        }         // for (int b = 0; b < n_batchs; b++)
    }             // if (norm_region == MLO_LRN_ACROSS_CHANNELS)
    else
    {
        for(int b = 0; b < n_batchs; b++)
        {
            for(int o = 0; o < n_inputs; o++)
            {
                for(int j = 0; j < bot_height; j++)
                {

                    for(int i = 0; i < bot_width; i++)
                    {
                        _Tcheck accum_ratio = static_cast<_Tcheck>(0);

                        int hstart        = j - pad;
                        int wstart        = i - pad;
                        int hend          = std::min(hstart + local_area, top_height + pre_pad);
                        int wend          = std::min(wstart + local_area, top_width + pre_pad);
                        int adj_area_size = (hend - hstart) * (wend - wstart);
                        hstart            = std::max(hstart, 0);
                        wstart            = std::max(wstart, 0);
                        hend              = std::min(hend, top_height);
                        wend              = std::min(wend, top_width);
                        for(int h = hstart; h < hend; ++h)
                        {
                            for(int w = wstart; w < wend; ++w)
                            {
                                _Tcheck adder =
                                    static_cast<_Tcheck>(top_df_ptr[b * top_df_batch_stride +
                                                                    o * top_df_channel_stride +
                                                                    h * top_df_stride + w]) *
                                    static_cast<_Tcheck>(
                                        top_ptr[b * top_batch_stride + o * top_channel_stride +
                                                h * top_stride + w]) /
                                    static_cast<_Tcheck>(
                                        scale_ptr[b * scale_batch_stride +
                                                  o * scale_channel_stride + h * scale_stride + w]);

                                accum_ratio += adder;
                            }
                        }

                        _Tcheck ratio_dta_bwd = static_cast<_Tcheck>(2.) * alpha * beta /
                                                static_cast<_Tcheck>(adj_area_size);

                        bot_df_v_ptr[b * bot_df_v_batch_stride + o * bot_df_v_channel_stride +
                                     j * bot_df_v_stride + i] =
                            static_cast<_Tcheck>(
                                top_df_ptr[b * top_df_batch_stride + o * top_df_channel_stride +
                                           j * top_df_stride + i]) *
                                pow(static_cast<_Tcheck>(
                                        scale_ptr[b * scale_batch_stride +
                                                  o * scale_channel_stride + j * scale_stride + i]),
                                    negative_beta) -
                            ratio_dta_bwd *
                                static_cast<_Tcheck>(
                                    bot_ptr[b * bot_batch_stride + o * bot_channel_stride +
                                            j * bot_stride + i]) *
                                accum_ratio;
                    }
                }
            }
        }

    } // if (norm_region == MLO_LRN_ACROSS_CHANNELS)

    return (ret);
}

#endif
