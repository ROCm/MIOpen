
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

BuffInfo::BuffInfo(
    MemLayout_t layout, int nk, int c, int h, int w, int vec_c, int _g, int _data_len_t)
{
    if(!(vec_c != 0))
        MIOPEN_THROW(std::string("Internal error in BuffInfo: (vec_c != 0) "));

    data_len_t = _data_len_t;
    const size_t c_hi  = (c + vec_c - 1) / vec_c;
    const size_t count = nk * c_hi * h * w * _g * vec_c;
    total_byte_size    = count * data_len_t;
    size.nk            = nk;
    size.g             = _g;
    size.c             = c;
    size.h             = h;
    size.w             = w;

    using namespace LayoutConstructor;
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

} // namespace miopen
