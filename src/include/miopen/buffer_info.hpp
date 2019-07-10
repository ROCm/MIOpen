#ifndef GUARD_MIOPEN_BUFFER_INFO_HPP_
#define GUARD_MIOPEN_BUFFER_INFO_HPP_

#include <string>

namespace miopen {

enum class MemLayout_t
{
    NCHW = 0,
    CNHW = 1,
    NHWC = 2,
    CHWN = 3,
    HWCN = 4,
    HWNC = 5,
    // Initializers must match values defined in common.inc
};

MemLayout_t GetMemLayout_t(const std::string& s);
MemLayout_t GetSwappedNCLayout(MemLayout_t layout);

struct BuffInfo
{
    size_t total_byte_size = 0;
    struct
    {
        unsigned int nk = 0, c = 0, h = 0, w = 0;
    } stride{}, byte_stride{}, size{};
    BuffInfo() {}
    BuffInfo(MemLayout_t layout, int nk, int c, int h, int w, int vec_c, int data_len_t);
};

enum class ConvWinoBuffType
{
    Input,
    Weight,
    Output,
};
template <int WinoDataW, int WinoFilterW, int WinoDataH = WinoDataW, int WinoFilterH = WinoFilterW>
struct WinogradBufferInfo
{

    const int WinoDataHW[2] = {WinoDataW, WinoDataW}, WinoFilterHW[2] = {WinoFilterW, WinoFilterH};
    const bool direct[2] = {(WinoDataW == 1) && (WinoFilterW == 1),
                            (WinoDataH == 1) && (WinoFilterH == 1)};
    const int wino_xtile[2] = {WinoDataW + WinoFilterW - 1, WinoDataH + WinoFilterH - 1};

    struct WinoInfo
    {
        size_t wino_tiles_HW[2] = {0, 0}, wino_HW[2] = {0, 0};
    } wino_info;
    int wino_c;

    BuffInfo buff_info;

    WinogradBufferInfo(int n,
                       int k,
                       int c,
                       int out_h,
                       int out_w,
                       int wei_h,
                       int wei_w,
                       MemLayout_t layout,
                       int vec_c,
                       int data_len_t,
                       ConvWinoBuffType buff_type)
    {
        WinoInfo wino_in, wino_out, wino_wei;
        int out_HW[2] = {out_w, out_h};
        int wei_HW[2] = {wei_w, wei_h};
        wino_c        = c;
        for(int i = 0; i < 2; i++)
        {
            wino_out.wino_tiles_HW[i] = (out_HW[i] + WinoDataHW[i] - 1) / WinoDataHW[i];
            wino_wei.wino_tiles_HW[i] = (wei_HW[i] + WinoFilterHW[i] - 1) / WinoFilterHW[i];
            wino_in.wino_tiles_HW[i] =
                direct[i] ? (out_HW[i] + wei_HW[i] - 1) : wino_out.wino_tiles_HW[i];

            wino_c *= direct[i] ? 1 : wino_wei.wino_tiles_HW[i];

            wino_in.wino_HW[i]  = wino_xtile[i] * wino_in.wino_tiles_HW[i];
            wino_wei.wino_HW[i] = wino_xtile[i];
            wino_out.wino_HW[i] = wino_xtile[i] * wino_out.wino_tiles_HW[i];
        }
        switch(buff_type)
        {
        case ConvWinoBuffType::Input:
            buff_info = BuffInfo(
                layout, n, wino_c, wino_in.wino_HW[1], wino_in.wino_HW[0], vec_c, data_len_t);
            wino_info = wino_in;
            break;
        case ConvWinoBuffType::Weight:
            buff_info = BuffInfo(
                layout, k, wino_c, wino_wei.wino_HW[1], wino_wei.wino_HW[0], vec_c, data_len_t);
            wino_info = wino_wei;
            break;
        case ConvWinoBuffType::Output:
            buff_info =
                BuffInfo(layout, n, k, wino_out.wino_HW[1], wino_out.wino_HW[0], vec_c, data_len_t);
            wino_info = wino_out;
            break;
        default: break;
        }
    }
};

} // namespace miopen

#endif // GUARD_MIOPEN_BUFFER_INFO_HPP_
