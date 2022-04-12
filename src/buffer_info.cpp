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

#include <miopen/buffer_info.hpp>
#include <miopen/errors.hpp>
#include <cassert>

namespace miopen {

MemLayout_t GetMemLayout_t(const std::string& s)
{
    return s == "NCHW"
               ? MemLayout_t::NCHW
               : (s == "CNHW" ? MemLayout_t::CNHW
                              : (s == "NHWC" ? MemLayout_t::NHWC
                                             : (s == "CHWN" ? MemLayout_t::CHWN
                                                            : (s == "HWCN" ? MemLayout_t::HWCN
                                                                           : MemLayout_t::HWNC))));
}

MemLayout_t GetSwappedNCLayout(MemLayout_t layout)
{
    switch(layout)
    {
    case MemLayout_t::CNHW: return MemLayout_t::NCHW;
    case MemLayout_t::NCHW: return MemLayout_t::CNHW;
    case MemLayout_t::NHWC: return MemLayout_t::CHWN;
    case MemLayout_t::CHWN: return MemLayout_t::NHWC;
    case MemLayout_t::HWCN: return MemLayout_t::HWNC;
    case MemLayout_t::HWNC: return MemLayout_t::HWCN;
    case MemLayout_t::GCNHW: return MemLayout_t::GNCHW;
    case MemLayout_t::GNCHW: return MemLayout_t::GCNHW;
    case MemLayout_t::CGNHW: return MemLayout_t::NGCHW;
    case MemLayout_t::NGCHW: return MemLayout_t::CGNHW;
    default:
        MIOPEN_THROW(std::string("Internal error in GetSwappedNCLayout: Unknown MemLayout_t "));
    }
}

MemLayout_t GetGroupConvLayout(MemLayout_t layout, bool IsDataBuffer)
{
    if(IsDataBuffer)
    {
        switch(layout)
        {
        case MemLayout_t::CNHW: return MemLayout_t::CGNHW;
        case MemLayout_t::NCHW: return MemLayout_t::NGCHW;
        case MemLayout_t::NHWC:
        case MemLayout_t::CHWN:
        case MemLayout_t::HWCN:
        case MemLayout_t::HWNC:
        case MemLayout_t::NGCHW:
        case MemLayout_t::GNCHW:
        case MemLayout_t::CGNHW:
        case MemLayout_t::GCNHW:
        default: break;
        }
    }
    else
    {
        switch(layout)
        {
        case MemLayout_t::CNHW: return MemLayout_t::GCNHW;
        case MemLayout_t::NCHW: return MemLayout_t::GNCHW;
        case MemLayout_t::NHWC:
        case MemLayout_t::CHWN:
        case MemLayout_t::HWCN:
        case MemLayout_t::HWNC:
        case MemLayout_t::NGCHW:
        case MemLayout_t::GNCHW:
        case MemLayout_t::CGNHW:
        case MemLayout_t::GCNHW:
        default: break;
        }
    }
    MIOPEN_THROW(std::string("Internal error in GetGroupConvLayout: Unknown MemLayout_t "));
}

BuffInfo::BuffInfo(MemLayout_t layout, int nk, int c, int h, int w, int g, int _element_size)
{

    element_size       = _element_size;
    const size_t count = nk * c * h * w * g;
    total_byte_size    = count * element_size;
    size.nk            = nk;
    size.g             = g;
    size.c             = c;
    size.h             = h;
    size.w             = w;

    using LayoutConstructor::FillLayoutStride;
    switch(layout)
    {
    case MemLayout_t::NCHW:
        FillLayoutStride<LPart_t::W, LPart_t::H, LPart_t::C, LPart_t::N>(this);
        break;
    case MemLayout_t::CNHW:
        FillLayoutStride<LPart_t::W, LPart_t::H, LPart_t::N, LPart_t::C>(this);
        break;
    case MemLayout_t::CHWN:
        FillLayoutStride<LPart_t::N, LPart_t::W, LPart_t::H, LPart_t::C>(this);
        break;
    case MemLayout_t::NHWC:
        FillLayoutStride<LPart_t::C, LPart_t::W, LPart_t::H, LPart_t::N>(this);
        break;
    case MemLayout_t::HWCN:
        FillLayoutStride<LPart_t::N, LPart_t::C, LPart_t::W, LPart_t::H>(this);
        break;
    case MemLayout_t::HWNC:
        FillLayoutStride<LPart_t::C, LPart_t::N, LPart_t::W, LPart_t::H>(this);
        break;
    case MemLayout_t::NGCHW:
        FillLayoutStride<LPart_t::W, LPart_t::H, LPart_t::C, LPart_t::G, LPart_t::N>(this);
        break;
    case MemLayout_t::GNCHW:
        FillLayoutStride<LPart_t::W, LPart_t::H, LPart_t::C, LPart_t::N, LPart_t::G>(this);
        break;
    case MemLayout_t::CGNHW:
        FillLayoutStride<LPart_t::W, LPart_t::H, LPart_t::N, LPart_t::G, LPart_t::C>(this);
        break;
    case MemLayout_t::GCNHW:
        FillLayoutStride<LPart_t::W, LPart_t::H, LPart_t::N, LPart_t::C, LPart_t::G>(this);
        break;
    default: MIOPEN_THROW(std::string("Internal error in BuffInfo(): Unknown MemLayout_t ")); break;
    }
}

MultiBufferWorkspaceTraits::MultiBufferWorkspaceTraits(std::initializer_list<size_t> v_size_,
                                                       size_t alignment_)
    : v_size(v_size_), alignment(alignment_)
{
    size_t each_offset = 0;
    v_offset.push_back(each_offset);
    for(auto each_size : v_size)
    {
        size_t padding = (alignment - (each_size % alignment)) % alignment;
        each_offset += each_size + padding;
        v_offset.push_back(each_offset);
    }
}

size_t MultiBufferWorkspaceTraits::GetSize() const { return v_offset.back(); }

size_t MultiBufferWorkspaceTraits::GetOffset(size_t index) const
{
    if(index >= v_offset.size())
        MIOPEN_THROW("index given overflows");
    return v_offset[index];
}

} // namespace miopen
