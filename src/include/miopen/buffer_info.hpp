/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_BUFFER_INFO_HPP_
#define GUARD_MIOPEN_BUFFER_INFO_HPP_

#include <string>
#include <cassert>

namespace miopen {

enum class MemLayout_t
{
    NCHW  = 0,
    CNHW  = 1,
    NHWC  = 2,
    CHWN  = 3,
    HWCN  = 4,
    HWNC  = 5,
    NGCHW = 6,
    GNCHW = 7,
    CGNHW = 8,
    GCNHW = 9,
    // Initializers must match values defined in common.inc
};

MemLayout_t GetGroupConvLayout(MemLayout_t layout, bool IsDataBuffer);

MemLayout_t GetMemLayout_t(const std::string& s);
MemLayout_t GetSwappedNCLayout(MemLayout_t layout);

// Parts of MemLayout
enum class LPart_t
{
    LPart_begin = 0,
    W           = 1,
    H           = 2,
    C           = 3,
    N           = 4,
    G           = 5,
    LPart_end   = 6
};

struct BuffInfo
{
    size_t total_byte_size = 0;
    int element_size       = 4;

    struct
    {
        unsigned int nk = 0, g = 0, c = 0, h = 0, w = 0;
    } stride{}, byte_stride{}, size{};
    BuffInfo() {}
    BuffInfo(MemLayout_t layout, int nk, int c, int h, int w, int g, int _element_size);
    BuffInfo(MemLayout_t layout, int nk, int c, int h, int w, int _element_size)
        : BuffInfo(layout, nk, c, h, w, 1, _element_size)
    {
    }
};

namespace LayoutConstructor {
template <LPart_t EnumVal>
inline unsigned int FillStride(BuffInfo*, unsigned int)
{
    assert(0);
    // Unknown LPart_t
    return 0;
}

template <>
inline unsigned int FillStride<LPart_t::H>(BuffInfo* b, unsigned int cum_stride)
{
    b->stride.h      = cum_stride;
    b->byte_stride.h = cum_stride * b->element_size;
    return b->size.h * cum_stride;
}

template <>
inline unsigned int FillStride<LPart_t::W>(BuffInfo* b, unsigned int cum_stride)
{
    b->stride.w      = cum_stride;
    b->byte_stride.w = cum_stride * b->element_size;
    return b->size.w * cum_stride;
}

template <>
inline unsigned int FillStride<LPart_t::C>(BuffInfo* b, unsigned int cum_stride)
{
    b->stride.c      = cum_stride;
    b->byte_stride.c = cum_stride * b->element_size;
    return b->size.c * cum_stride;
}

template <>
inline unsigned int FillStride<LPart_t::N>(BuffInfo* b, unsigned int cum_stride)
{
    b->stride.nk      = cum_stride;
    b->byte_stride.nk = cum_stride * b->element_size;
    return b->size.nk * cum_stride;
}

template <>
inline unsigned int FillStride<LPart_t::G>(BuffInfo* b, unsigned int cum_stride)
{
    b->stride.g      = cum_stride;
    b->byte_stride.g = cum_stride * b->element_size;
    return b->size.g * cum_stride;
}

template <LPart_t first, LPart_t... others>
inline void FillNextLayoutStride(BuffInfo* b, unsigned int cum_stride)
{
    auto sum = FillStride<first>(b, cum_stride);
    FillNextLayoutStride<others...>(b, sum);
}
template <>
inline void FillNextLayoutStride<LPart_t::LPart_begin>(BuffInfo*, unsigned int)
{
}

template <LPart_t... others>
inline void FillLayoutStride(BuffInfo* b)
{
    FillNextLayoutStride<others..., LPart_t::LPart_begin>(b, 1);
}

} // namespace LayoutConstructor

enum class ConvWinoBuffType
{
    Input,
    Weight,
    Output,
};

enum class ConvWinoXformType
{
    // N_G_C_H_W,
    N_1_CThTw_Xh_Xw,
    N_GXhXw_C_Th_Tw
};

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct WinogradBufferInfo
{

    const int WinoDataHW[2] = {WinoDataH, WinoDataW}, WinoFilterHW[2] = {WinoFilterH, WinoFilterW};
    const bool direct[2] = {(WinoDataW == 1) && (WinoFilterW == 1),
                            (WinoDataH == 1) && (WinoFilterH == 1)};

    struct WinoInfo
    {
        size_t wino_tiles_HW[2] = {0, 0}, wino_HW[2] = {0, 0};
    } wino_info;

    BuffInfo buff_info;

    WinogradBufferInfo(int n,
                       int k,
                       int c,
                       int g,
                       int out_h,
                       int out_w,
                       int wei_h,
                       int wei_w,
                       MemLayout_t layout,
                       ConvWinoXformType xform_t,
                       int element_size,
                       ConvWinoBuffType buff_type,
                       int wino_xform_h,
                       int wino_xform_w)
    {
        WinoInfo wino_data, wino_filter;
        const int out_HW[2] = {out_h, out_w};
        const int wei_HW[2] = {wei_h, wei_w};

        const int wino_xtile[2] = {wino_xform_h, wino_xform_w};

        for(int i = 0; i < 2; i++)
        {
            wino_data.wino_tiles_HW[i]   = (out_HW[i] + WinoDataHW[i] - 1) / WinoDataHW[i];
            wino_filter.wino_tiles_HW[i] = (wei_HW[i] + WinoFilterHW[i] - 1) / WinoFilterHW[i];

            wino_filter.wino_HW[i] = wino_xtile[i];
            wino_data.wino_HW[i]   = wino_xtile[i] * wino_data.wino_tiles_HW[i];
        }

        switch(xform_t)
        {
        case ConvWinoXformType::N_GXhXw_C_Th_Tw: {
            const int wino_g = g * wino_xtile[0] * wino_xtile[1];
            switch(buff_type)
            {
            case ConvWinoBuffType::Input:
                buff_info = BuffInfo(layout,
                                     n,
                                     c,
                                     wino_data.wino_tiles_HW[0],
                                     wino_data.wino_tiles_HW[1],
                                     wino_g,
                                     element_size);
                wino_info = wino_data;
                break;
            case ConvWinoBuffType::Weight:
                buff_info = BuffInfo(layout,
                                     k,
                                     c,
                                     wino_filter.wino_tiles_HW[0],
                                     wino_filter.wino_tiles_HW[1],
                                     wino_g,
                                     element_size);
                wino_info = wino_filter;
                break;
            case ConvWinoBuffType::Output:
                buff_info = BuffInfo(layout,
                                     n,
                                     k,
                                     wino_data.wino_tiles_HW[0],
                                     wino_data.wino_tiles_HW[1],
                                     wino_g,
                                     element_size);
                wino_info = wino_data;
                break;
            default: break;
            }
            break;
        }
        case ConvWinoXformType::N_1_CThTw_Xh_Xw: {
            const int wino_c = c * wino_filter.wino_tiles_HW[0] * wino_filter.wino_tiles_HW[1];
            switch(buff_type)
            {
            case ConvWinoBuffType::Input:
                buff_info = BuffInfo(
                    layout, n, wino_c, wino_data.wino_HW[0], wino_data.wino_HW[1], element_size);
                wino_info = wino_data;
                break;
            case ConvWinoBuffType::Weight:
                buff_info = BuffInfo(layout,
                                     k,
                                     wino_c,
                                     wino_filter.wino_HW[0],
                                     wino_filter.wino_HW[1],
                                     element_size);
                wino_info = wino_filter;
                break;
            case ConvWinoBuffType::Output:
                buff_info = BuffInfo(
                    layout, n, k, wino_data.wino_HW[0], wino_data.wino_HW[1], element_size);
                wino_info = wino_data;
                break;
            default: break;
            }
            break;
        }
        default: break;
        }
    }
    WinogradBufferInfo(int n,
                       int k,
                       int c,
                       int out_h,
                       int out_w,
                       int wei_h,
                       int wei_w,
                       MemLayout_t layout,
                       int element_size,
                       ConvWinoBuffType buff_type,
                       int wino_xform_h,
                       int wino_xform_w)
        : WinogradBufferInfo(n,
                             k,
                             c,
                             1,
                             out_h,
                             out_w,
                             wei_h,
                             wei_w,
                             layout,
                             ConvWinoXformType::N_1_CThTw_Xh_Xw,
                             element_size,
                             buff_type,
                             wino_xform_h,
                             wino_xform_w)
    {
    }
};

} // namespace miopen

#endif // GUARD_MIOPEN_BUFFER_INFO_HPP_
