
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
    }
    MIOPEN_THROW(std::string("Internal error in GetSwappedNCLayout: Unknown MemLayout_t "));
}

BuffInfo::BuffInfo(MemLayout_t layout, int nk, int c, int h, int w, int vec_c, int data_len_t)
{
    if(!(vec_c != 0))
        MIOPEN_THROW(std::string("Internal error in BuffInfo: (vec_c != 0) "));

    const size_t c_hi  = (c + vec_c - 1) / vec_c;
    const size_t count = nk * c_hi * h * w * vec_c;
    total_byte_size    = count * data_len_t;
    size.nk            = nk;
    size.c             = c;
    size.h             = h;
    size.w             = w;

    switch(layout)
    {
    case MemLayout_t::NCHW:
        stride.w  = 1;
        stride.h  = w;
        stride.c  = w * h;
        stride.nk = w * h * c_hi;
        break;
    case MemLayout_t::CNHW:
        stride.w  = 1;
        stride.h  = w;
        stride.nk = w * h;
        stride.c  = w * h * nk;
        break;
    case MemLayout_t::CHWN:
        stride.nk = 1;
        stride.w  = nk;
        stride.h  = w * nk;
        stride.c  = w * h * nk;
        break;
    case MemLayout_t::NHWC:
        stride.c  = 1;
        stride.w  = c_hi;
        stride.h  = c_hi * w;
        stride.nk = c_hi * w * h;
        break;
    case MemLayout_t::HWCN:
        stride.nk = 1;
        stride.c  = nk;
        stride.w  = nk * c_hi;
        stride.h  = nk * c_hi * w;
        break;
    case MemLayout_t::HWNC:
        stride.c  = 1;
        stride.nk = c_hi;
        stride.w  = c_hi * nk;
        stride.h  = c_hi * nk * w;
        break;
    default: MIOPEN_THROW(std::string("Internal error in BuffInfo(): Unknown MemLayout_t ")); break;
    }
    stride.nk *= vec_c;
    stride.c *= vec_c;
    stride.h *= vec_c;
    stride.w *= vec_c;
    byte_stride.nk = stride.nk * data_len_t;
    byte_stride.c  = stride.c * data_len_t;
    byte_stride.h  = stride.h * data_len_t;
    byte_stride.w  = stride.w * data_len_t;
}

} // namespace miopen
