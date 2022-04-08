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

#include <miopen/pooling/problem_description.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/pooling.hpp>

#include <sstream>

namespace miopen {

namespace pooling {

template <typename T>
std::string get_vect_config(const std::vector<T>& v)
{
    std::string str;
    for(auto itr = v.begin(); itr < v.end(); itr++)
    {
        str += (std::to_string(*itr) + (itr == v.end() - 1 ? "" : "x"));
    }
    return str;
}

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    int pooling_method =
        (pooling.GetMode() == miopenPoolingMax)
            ? MLO_POOLING_OP_MAX
            : ((pooling.GetMode() == miopenPoolingAverage) ? MLO_POOLING_OP_AVE
                                                           : MLO_POOLING_OP_AVE_INCLUSIVE);

    ss << "m" + std::to_string(pooling_method);

    if(direction == Direction::Forward)
    {

        ss << "_i" << static_cast<int>(save_index);
        ss << "_dt" << xDesc.GetType();
        ss << "_ker" << get_vect_config(pooling.lens);
        ss << "_str" << get_vect_config(pooling.strides);
        ss << "_it" << pooling.GetIndexType();

        if(xDesc.GetSize() == 4)
        {
            int batch_sz, n_outputs, out_height, out_width;
            std::tie(batch_sz, n_outputs, out_height, out_width) =
                miopen::tien<4>(yDesc.GetLengths(), 1);

            const int _out_pix_tile0 = 1;
            int _out_pix_tile1       = out_height <= 8 ? 1 : out_height <= 32 ? 4 : 8;
            if(out_height > 16 && out_height % 32 > 16)
                _out_pix_tile1 =
                    std::min(16, std::max(1, prePow2(_out_pix_tile1 * pooling.strides[0])));

            int _grp_tile0 = out_width <= 8 ? 8 : (out_width % 32 <= 16 ? 16 : 32);
            int _grp_tile1 =
                out_height <= 8
                    ? 8
                    : out_height < 16 ? 16 : out_height <= 32 ? 32 : out_height <= 64 ? 64 : 128;
            _grp_tile1 /= _out_pix_tile1;
            while(_grp_tile0 * _grp_tile1 > 256 && _grp_tile0 > 1)
                _grp_tile0 >>= 1;

            auto _g_wk = std::vector<std::size_t>{};

            const int g_wk_width =
                ((out_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
            const int g_wk_height =
                ((out_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));

            _g_wk.clear();
            _g_wk.push_back(g_wk_width * _grp_tile0);
            _g_wk.push_back(g_wk_height * _grp_tile1);
            _g_wk.push_back(n_outputs * batch_sz);

            ss << "_nout" << xDesc.GetLengths()[1];
            ss << "_tile" << static_cast<int>(_out_pix_tile1);
            ss << "x" << static_cast<int>(_out_pix_tile0);
            ss << "_grp" << static_cast<uint>(_grp_tile1);
            ss << "x" << static_cast<uint>(_grp_tile0);
            ss << "_glb" << get_vect_config(_g_wk);
            ss << "_wsidx" << pooling.GetWorkspaceIndexMode();
        }
        else
        {
            const int top_w_per_work     = 1;
            const int top_h_per_work     = 4;
            const int top_d_per_work     = 2;
            const int max_activ_workitem = 65536;
            const size_t lcl_work        = 64;

            const int batch = xDesc.GetLengths()[0];
            const int chal  = xDesc.GetLengths()[1];

            const int top_d = *(yDesc.GetLengths().rbegin() + 2);
            const int top_h = *(yDesc.GetLengths().rbegin() + 1);
            const int top_w = *(yDesc.GetLengths().rbegin());

            const int top_blk_w = std::max((top_w + top_w_per_work - 1) / top_w_per_work, 1);
            const int top_blk_h = std::max((top_h + top_h_per_work - 1) / top_h_per_work, 1);
            const int top_blk_d = std::max((top_d + top_d_per_work - 1) / top_d_per_work, 1);

            const int total_work = batch * chal * top_blk_w * top_blk_h * top_blk_d;
            const int activ_work = std::min(total_work, max_activ_workitem);

            const size_t grp_num = (activ_work + lcl_work - 1) / lcl_work;

            ss << "_tile" << static_cast<int>(top_d_per_work);
            ss << "x" << static_cast<int>(top_h_per_work);
            ss << "x" << static_cast<int>(top_w_per_work);
            ss << "_maxwkitm" << static_cast<uint>(max_activ_workitem);
            ss << "_lcl" << static_cast<uint>(lcl_work);
            ss << "_grp" << static_cast<uint>(grp_num);
        }
    }
    else
    {
        ss << "_dt" << dyDesc.GetType();

        if(xDesc.GetSize() == 4)
        {
            ss << "_xd" << get_vect_config(xDesc.GetLengths());
            ss << "_xs" << get_vect_config(xDesc.GetStrides());
            ss << "_yd" << get_vect_config(yDesc.GetLengths());
            ss << "_ys" << get_vect_config(yDesc.GetStrides());
            ss << "_dxd" << get_vect_config(dxDesc.GetLengths());
            ss << "_dxs" << get_vect_config(dxDesc.GetStrides());
            ss << "_dyd" << get_vect_config(dyDesc.GetLengths());
            ss << "_dys" << get_vect_config(dyDesc.GetStrides());
            ss << "_ker" << get_vect_config(pooling.lens);
            ss << "_str" << get_vect_config(pooling.strides);
            ss << "_pad" << get_vect_config(pooling.pads);
            ss << "_it" << pooling.GetIndexType();
            ss << "_wsidx" << pooling.GetWorkspaceIndexMode();
        }
        else
        {
            const int pix_w_per_work     = 1;
            const int pix_h_per_work     = 4;
            const int pix_d_per_work     = 2;
            const int max_activ_workitem = 65536;
            const size_t lcl_work        = 64;

            const int batch = dyDesc.GetLengths()[0];
            const int chal  = dyDesc.GetLengths()[1];

            const int bot_d = *(dxDesc.GetLengths().rbegin() + 2);
            const int bot_h = *(dxDesc.GetLengths().rbegin() + 1);
            const int bot_w = *(dxDesc.GetLengths().rbegin());

            const int pix_blk_w = std::max((bot_w + pix_w_per_work - 1) / pix_w_per_work, 1);
            const int pix_blk_h = std::max((bot_h + pix_h_per_work - 1) / pix_h_per_work, 1);
            const int pix_blk_d = std::max((bot_d + pix_d_per_work - 1) / pix_d_per_work, 1);

            const int total_work = batch * chal * pix_blk_w * pix_blk_h * pix_blk_d;
            const int activ_work = std::min(total_work, max_activ_workitem);
            const size_t grp_num = (activ_work + lcl_work - 1) / lcl_work;

            ss << "_ker" << get_vect_config(pooling.lens);
            ss << "_str" << get_vect_config(pooling.strides);
            ss << "_it" << pooling.GetIndexType();
            ss << "_tile" << static_cast<int>(pix_d_per_work);
            ss << "x" << static_cast<int>(pix_h_per_work);
            ss << "x" << static_cast<int>(pix_w_per_work);
            ss << "_maxwkitm" << static_cast<uint>(max_activ_workitem);
            ss << "_lcl" << static_cast<uint>(lcl_work);
            ss << "_grp" << static_cast<uint>(grp_num);
        }
    }

    return NetworkConfig{ss.str()};
}

} // namespace pooling

} // namespace miopen
