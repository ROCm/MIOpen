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
/**********************************************************************
Copyright (c)2016 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

?	Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
?	Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
// to share code with between CPU and GPU

#define MIOPEN
#include <miopen/mlo_internal.hpp>
#include <miopen/mlo_utils.hpp>

int mlo_construct_pooling2D::mloConstruct()
{
    int ret = 0;

    if(isForwardDirection())
    {

        ret = mloConstructFwd();
    }
    else
    {
        ret = mloConstructBwd();
    }

    return (ret);
}

int mlo_construct_pooling2D::mloConstructFwd()
{
    int ret = 0;

    _grp_tile0 = 8;
    _grp_tile1 = 8;

    _out_pix_tile0 = std::max(1, 8 / _search_params.kernel_stride0);
    _out_pix_tile1 = std::max(1, 8 / _search_params.kernel_stride1);

    while(_out_pix_tile0 * _grp_tile0 > _search_params.out_width * 2 && _out_pix_tile0 > 1)
    {
        _out_pix_tile0 >>= 1;
    }

    while(_out_pix_tile1 * _grp_tile1 > _search_params.out_height * 2 && _out_pix_tile1 > 1)
    {
        _out_pix_tile1 >>= 1;
    }

    _comp_options = std::string(" -DMLO_POOLING_OP_ID=") +
                    std::to_string(static_cast<long long>(_pooling_method)) +
                    std::string(" -DMLO_POOLING_KERNEL_SZ1=") +
                    std::to_string(static_cast<long long>(_search_params.kernel_size1)) +
                    std::string(" -DMLO_POOLING_PAD1=") +
                    std::to_string(static_cast<long long>(_search_params.pad1)) +
                    std::string(" -DMLO_POOLING_STRIDE1=") +
                    std::to_string(static_cast<long long>(_search_params.kernel_stride1)) +
                    std::string(" -DMLO_POOLING_KERNEL_SZ0=") +
                    std::to_string(static_cast<long long>(_search_params.kernel_size0)) +
                    std::string(" -DMLO_POOLING_PAD0=") +
                    std::to_string(static_cast<long long>(_search_params.pad0)) +
                    std::string(" -DMLO_POOLING_STRIDE0=") +
                    std::to_string(static_cast<long long>(_search_params.kernel_stride0)) +
                    std::string(" -DMLO_POOLING_N_OUTPUTS=") +
                    std::to_string(static_cast<long long>(_search_params.n_outputs)) +
                    std::string(" -DMLO_POOLING_N_CHANNELS=") +
                    std::to_string(static_cast<long long>(_search_params.n_inputs)) +
                    std::string(" -DMLO_POOLING_N_HORIZ_OUT_PIX=") +
                    std::to_string(static_cast<long long>(_out_pix_tile0)) +
                    std::string(" -DMLO_POOLING_N_VERT_OUT_PIX=") +
                    std::to_string(static_cast<long long>(_out_pix_tile1)) +
                    std::string(" -DMLO_POOLING_GROUP_SZ0=") +
                    std::to_string(static_cast<long long>(_grp_tile0)) +
                    std::string(" -DMLO_POOLING_GROUP_SZ1=") +
                    std::to_string(static_cast<long long>(_grp_tile1)) +
                    std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") +
                    std::to_string(static_cast<long long>(_search_params.in_batch_stride)) +
                    std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") +
                    std::to_string(static_cast<long long>(_search_params.in_channel_stride)) +
                    std::string(" -DMLO_POOLING_BOT_STRIDE=") +
                    std::to_string(static_cast<long long>(_search_params.in_stride)) +
                    std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") +
                    std::to_string(static_cast<long long>(_search_params.out_batch_stride)) +
                    std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") +
                    std::to_string(static_cast<long long>(_search_params.out_channel_stride)) +
                    std::string(" -DMLO_POOLING_TOP_STRIDE=") +
                    std::to_string(static_cast<long long>(_search_params.out_stride)) +
                    std::string(" -DMLO_POOLING_BOT_WIDTH=") +
                    std::to_string(static_cast<long long>(_search_params.in_width)) +
                    std::string(" -DMLO_POOLING_BOT_HEIGHT=") +
                    std::to_string(static_cast<long long>(_search_params.in_height)) +
                    std::string(" -DMLO_POOLING_TOP_WIDTH=") +
                    std::to_string(static_cast<long long>(_search_params.out_width)) +
                    std::string(" -DMLO_POOLING_TOP_HEIGHT=") +
                    std::to_string(static_cast<long long>(_search_params.out_height)) +
                    std::string(_do_backward ? " -DMLO_POOLING_DO_BACKWARD" : "") +
                    getGeneralCompOptions();

    int g_wk_width = ((_search_params.out_width + _grp_tile0 * _out_pix_tile0 - 1) /
                      (_grp_tile0 * _out_pix_tile0));
    int g_wk_height = ((_search_params.out_height + _grp_tile1 * _out_pix_tile1 - 1) /
                       (_grp_tile1 * _out_pix_tile1));

    _l_wk.clear();
    _l_wk.push_back(_grp_tile0);
    _l_wk.push_back(_grp_tile1);
    _l_wk.push_back(1);

    _g_wk.clear();
    _g_wk.push_back(g_wk_width * _grp_tile0);
    _g_wk.push_back(g_wk_height * _grp_tile1);
    _g_wk.push_back(_search_params.n_outputs * _search_params.batch_sz);

    _kernel_file = "MIOpenPooling.cl";

    _kernel_name = "mloPoolingG";

    return (ret);
}

int mlo_construct_pooling2D::mloConstructBwd()
{
    int ret = 0;

    _grp_tile0 = 8;
    _grp_tile1 = 8;

    //_out_pix_tile0 = _kernel_stride0;
    //_out_pix_tile1 = _kernel_stride1;
    _out_pix_tile0 = (_search_params.out_width < _grp_tile0 * 2) ? 1 : 2;
    _out_pix_tile1 = (_search_params.out_height < _grp_tile1 * 2) ? 1 : 2;

    _comp_options = std::string(" -DMLO_POOLING_KERNEL_SZ1=") +
                    std::to_string(static_cast<long long>(_search_params.kernel_size1)) +
                    std::string(" -DMLO_POOLING_PAD1=") +
                    std::to_string(static_cast<long long>(_search_params.pad1)) +
                    std::string(" -DMLO_POOLING_STRIDE1=") +
                    std::to_string(static_cast<long long>(_search_params.kernel_stride1)) +
                    std::string(" -DMLO_POOLING_KERNEL_SZ0=") +
                    std::to_string(static_cast<long long>(_search_params.kernel_size0)) +
                    std::string(" -DMLO_POOLING_PAD0=") +
                    std::to_string(static_cast<long long>(_search_params.pad0)) +
                    std::string(" -DMLO_POOLING_STRIDE0=") +
                    std::to_string(static_cast<long long>(_search_params.kernel_stride0)) +
                    std::string(" -DMLO_POOLING_N_OUTPUTS=") +
                    std::to_string(static_cast<long long>(_search_params.n_outputs)) +
                    std::string(" -DMLO_POOLBWD_N_HORIZ_OUT_PIX=") +
                    std::to_string(static_cast<long long>(_out_pix_tile0)) +
                    std::string(" -DMLO_POOLBWD_N_VERT_OUT_PIX=") +
                    std::to_string(static_cast<long long>(_out_pix_tile1)) +
                    std::string(" -DMLO_POOLBWD_GROUP_SZ0=") +
                    std::to_string(static_cast<long long>(_grp_tile0)) +
                    std::string(" -DMLO_POOLBWD_GROUP_SZ1=") +
                    std::to_string(static_cast<long long>(_grp_tile1)) +
                    std::string(" -DMLO_POOLBWD_BOT_WIDTH=") +
                    std::to_string(static_cast<long long>(_search_params.in_width)) +
                    std::string(" -DMLO_POOLBWD_BOT_HEIGHT=") +
                    std::to_string(static_cast<long long>(_search_params.in_height)) +
                    std::string(" -DMLO_POOLBWD_TOP_WIDTH=") +
                    std::to_string(static_cast<long long>(_search_params.out_width)) +
                    std::string(" -DMLO_POOLBWD_TOP_HEIGHT=") +
                    std::to_string(static_cast<long long>(_search_params.out_height)) +
                    std::string(" -DMLO_POOLBWD_BOTDF_BATCH_STRIDE=") +
                    std::to_string(static_cast<long long>(_in_df_batch_stride)) +
                    std::string(" -DMLO_POOLBWD_BOTDF_CHANNEL_STRIDE=") +
                    std::to_string(static_cast<long long>(_in_df_channel_stride)) +
                    std::string(" -DMLO_POOLBWD_BOTDF_STRIDE=") +
                    std::to_string(static_cast<long long>(_in_df_stride)) +
                    std::string(" -DMLO_POOLBWD_TOPDF_BATCH_STRIDE=") +
                    std::to_string(static_cast<long long>(_out_df_batch_stride)) +
                    std::string(" -DMLO_POOLBWD_TOPDF_CHANNEL_STRIDE=") +
                    std::to_string(static_cast<long long>(_out_df_channel_stride)) +
                    std::string(" -DMLO_POOLBWD_TOPDF_STRIDE=") +
                    std::to_string(static_cast<long long>(_out_df_stride))

                    + getGeneralCompOptions();

    int g_wk_width = ((_search_params.in_width + _grp_tile0 * _out_pix_tile0 - 1) /
                      (_grp_tile0 * _out_pix_tile0));
    int g_wk_height = ((_search_params.in_height + _grp_tile1 * _out_pix_tile1 - 1) /
                       (_grp_tile1 * _out_pix_tile1));

    _l_wk.clear();
    _l_wk.push_back(_grp_tile0);
    _l_wk.push_back(_grp_tile1);
    _l_wk.push_back(1);

    _g_wk.clear();
    _g_wk.push_back(g_wk_width * _grp_tile0);
    _g_wk.push_back(g_wk_height * _grp_tile1);
    _g_wk.push_back(_search_params.n_inputs * _search_params.batch_sz);

    _kernel_file = "MIOpenPoolingBwd.cl";
    if(_pooling_method == MLO_POOLING_OP_MAX)
    {
        _kernel_name = "mloPoolingMaxBwd";
    }
    else if(_pooling_method == MLO_POOLING_OP_AVE)
    {
        _kernel_name = "mloPoolingAveBwd";
    }
    else
    {
        std::cout << "Layer: %s. Error: unknowm method\n";
        ret = -1;
    }
    return (ret);
}
