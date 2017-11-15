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

// KNOWN ISSUES:
// backward propogagation has a bug in cross map normalization when numper of maps less than
// normalization region

void mlo_construct_norm::mloConstruct()
{
    if(_search_params.direction.IsForward())
    {
        mloConstructFwd();
    }
    else
    {
        mloConstructBwd();
    }
}

int mlo_construct_norm::mloConstructFwd()
{
    int ret = 0;

    size_t localMemSize    = _search_params.GetStream().GetLocalMemorySize();
    size_t maxComputeUnits = _search_params.GetStream().GetMaxComputeUnits();

    _hw_wave_sz       = 64;
    _dev_local_mem_sz = localMemSize; // in bytes

    int pre_pad = (_norm_area - 1) / 2;
    int pad     = _norm_area - pre_pad - 1;

    int top_df_stride         = 1;
    int top_df_channel_stride = 1;
    int top_df_batch_stride   = 1;

    int bot_df_stride         = 1;
    int bot_df_channel_stride = 1;
    int bot_df_batch_stride   = 1;

    _grp_tile0     = (_search_params.out_width <= 16) ? 8 : 16;
    _grp_tile1     = 8;
    _out_pix_tile0 = 1;
    _out_pix_tile1 = 1;

    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        _grp_tile0 = (_search_params.out_width <= 8) ? 8 : 16;
        _grp_tile1 = (_search_params.out_height <= 8) ? 8 : 16;
    }
    else
    {

        _out_pix_tile0 = (_search_params.out_width <= 8) ? 1 : 2;
        _out_pix_tile1 = (_search_params.out_height <= 8) ? 1 : 2;
    }

    auto ocl_group_lg2sz0 =
        static_cast<int>(ceil(log(static_cast<double>(_out_pix_tile0) / log(2.))));
    auto ocl_group_lg2sz1 =
        static_cast<int>(ceil(log(static_cast<double>(_out_pix_tile1) / log(2.))));

    int read_unit = 4;
    int N4S       = 1;
    int MAP_SZ4   = (_search_params.in_width * _search_params.in_height + N4S * read_unit - 1) /
                  (N4S * read_unit);

#if 1
    _kernel_file = "MIOpenLRNFwd.cl";
    _kernel_name = (_norm_region == MLO_LRN_ACROSS_CHANNELS) ? "MIOpenLRNAcrossChannels4"
                                                             : "MIOpenLRNWithinChannel_PS";
    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        _grp_tile0  = 8 * 8;
        _grp_tile1  = 1;
        int n_waves = (_search_params.batch_sz * MAP_SZ4 + _hw_wave_sz - 1) / _hw_wave_sz;
        if(n_waves <= maxComputeUnits * 8)
        {
            read_unit = 2;
            MAP_SZ4   = (_search_params.in_width * _search_params.in_height + N4S * read_unit - 1) /
                      (N4S * read_unit);
        }
    }
#else
    _kernel_file = "MIOpenLRN.cl";
    _kernel_name = (_norm_region == MLO_LRN_ACROSS_CHANNELS) ? "MIOpenLRNAcrossChannels1"
                                                             : "MIOpenLRNWithinChannel";
#endif

    int scale_stride         = _search_params.out_stride;
    int scale_channel_stride = _search_params.out_channel_stride;
    int scale_batch_stride   = _search_params.out_batch_stride;
    int scale                = (doBackward()) ? 1 : 0;

    auto g_wk_width =
        static_cast<int>((_search_params.out_width + _grp_tile0 * _out_pix_tile0 - 1) /
                         (_grp_tile0 * _out_pix_tile0));
    auto g_wk_height =
        static_cast<int>((_search_params.out_height + _grp_tile1 * _out_pix_tile1 - 1) /
                         (_grp_tile1 * _out_pix_tile1));
    int OUT_VERT_ALIGNED =
        (g_wk_height * (_grp_tile1 * _out_pix_tile1) == _search_params.out_height) ? 1 : 0;
    int OUT_HORIZ_ALIGNED =
        (g_wk_width * (_grp_tile0 * _out_pix_tile0) == _search_params.out_width) ? 1 : 0;
    // currently always 1
    int DIVBY4 =
        (MAP_SZ4 * read_unit == _search_params.in_width * _search_params.in_height) ? 1 : 0;
    int C1x1_PIXLEFT = (DIVBY4 == 1) ? 0 : _search_params.in_width * _search_params.in_height -
                                               (MAP_SZ4 - 1) * read_unit;

    std::string READ_TYPE =
        (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(static_cast<long long>(read_unit));

    _comp_options =
        std::string(" -DMLO_LRN_KERNEL_SZ=") + std::to_string(static_cast<long long>(_norm_area)) +
        std::string(" -DMLO_LRN_PAD=") + std::to_string(static_cast<long long>(pad)) +
        std::string(" -DMLO_LRN_KERNEL_SZ1=") + std::to_string(static_cast<long long>(_norm_area)) +
        std::string(" -DMLO_LRN_PAD1=") + std::to_string(static_cast<long long>(pad)) +
        std::string(" -DMLO_LRN_KERNEL_SZ0=") + std::to_string(static_cast<long long>(_norm_area)) +
        std::string(" -DMLO_LRN_PAD0=") + std::to_string(static_cast<long long>(pad)) +
        std::string(" -DMLO_LRN_N_OUTPUTS=") +
        std::to_string(static_cast<long long>(_search_params.n_outputs)) +
        std::string(" -DMLO_LRN_N_INPUTS=") +
        std::to_string(static_cast<long long>(_search_params.n_inputs)) +
        std::string(" -DMLO_LRN_N_HORIZ_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile0)) +
        std::string(" -DMLO_LRN_N_VERT_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile1)) +
        std::string(" -DMLO_LRN_GROUP_SZ0=") + std::to_string(static_cast<long long>(_grp_tile0)) +
        std::string(" -DMLO_LRN_GROUP_SZ1=") + std::to_string(static_cast<long long>(_grp_tile1)) +
        std::string(" -DMLO_LRN_GROUP_LG2SZ0=") +
        std::to_string(static_cast<long long>(ocl_group_lg2sz0)) +
        std::string(" -DMLO_LRN_GROUP_LG2SZ1=") +
        std::to_string(static_cast<long long>(ocl_group_lg2sz1)) +
        std::string(" -DMLO_LRN_BOT_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.in_batch_stride)) +
        std::string(" -DMLO_LRN_BOT_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.in_channel_stride)) +
        std::string(" -DMLO_LRN_BOT_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.in_stride)) +
        std::string(" -DMLO_LRN_TOP_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.out_batch_stride)) +
        std::string(" -DMLO_LRN_TOP_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.out_channel_stride)) +
        std::string(" -DMLO_LRN_TOP_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.out_stride)) +
        std::string(" -DMLO_LRN_BOT_WIDTH=") +
        std::to_string(static_cast<long long>(_search_params.out_width)) +
        std::string(" -DMLO_LRN_BOT_HEIGHT=") +
        std::to_string(static_cast<long long>(_search_params.out_height)) +
        std::string(" -DMLO_LRN_TOP_WIDTH=") +
        std::to_string(static_cast<long long>(_search_params.out_width)) +
        std::string(" -DMLO_LRN_TOP_HEIGHT=") +
        std::to_string(static_cast<long long>(_search_params.out_height)) +
        std::string(" -DMLO_LRN_SCALE_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(scale_batch_stride)) +
        std::string(" -DMLO_LRN_SCALE_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(scale_channel_stride)) +
        std::string(" -DMLO_LRN_SCALE_STRIDE=") +
        std::to_string(static_cast<long long>(scale_stride)) +
        std::string(" -DMLO_LRN_TOPDF_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(top_df_batch_stride)) +
        std::string(" -DMLO_LRN_TOPDF_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(top_df_channel_stride)) +
        std::string(" -DMLO_LRN_TOPDF_STRIDE=") +
        std::to_string(static_cast<long long>(top_df_stride)) +
        std::string(" -DMLO_LRN_BOTDF_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(bot_df_batch_stride)) +
        std::string(" -DMLO_LRN_BOTDF_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(bot_df_channel_stride)) +
        std::string(" -DMLO_LRN_BOTDF_STRIDE=") +
        std::to_string(static_cast<long long>(bot_df_stride)) +
        std::string(" -DMLO_LRN_BATCH_SZ=") +
        std::to_string(static_cast<long long>(_search_params.batch_sz)) +
        std::string(" -DMLO_LRN_N_INPUTS=") +
        std::to_string(static_cast<long long>(_search_params.n_inputs)) +
        std::string(" -DMLO_LRN_N_OUTPUTS=") +
        std::to_string(static_cast<long long>(_search_params.n_outputs)) +
        std::string(" -DMLO_LRN_DO_SCALE=") + std::to_string(static_cast<long long>(scale)) +
        std::string(" -DMLO_OUT_VERT_ALIGNED=") +
        std::to_string(static_cast<long long>(OUT_VERT_ALIGNED)) +
        std::string(" -DMLO_OUT_HORIZ_ALIGNED=") +
        std::to_string(static_cast<long long>(OUT_HORIZ_ALIGNED)) + std::string(" -DMLO_MAP_SZ4=") +
        std::to_string(static_cast<long long>(MAP_SZ4)) + std::string(" -DMLO_C1x1_PIXLEFT=") +
        std::to_string(static_cast<long long>(C1x1_PIXLEFT)) + std::string(" -DMLO_DIVBY4=") +
        std::to_string(static_cast<long long>(DIVBY4)) + std::string(" -DMLO_READ_TYPE=") +
        READ_TYPE + std::string(" -DMLO_READ_UNIT=") +
        std::to_string(static_cast<long long>(read_unit)) + getGeneralCompOptions();

    _l_wk.clear();
    _l_wk.push_back(_grp_tile0);
    _l_wk.push_back(_grp_tile1);
    _l_wk.push_back(1);

    _g_wk.clear();
    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
#if 1
        _g_wk.push_back(MAP_SZ4);
        _g_wk.push_back(1);

#else
        _g_wk.push_back(_out_width);
        _g_wk.push_back(_out_height);
#endif
        _g_wk.push_back(_search_params.batch_sz);
    }
    else
    {

        _g_wk.push_back(g_wk_width * _grp_tile0);
        _g_wk.push_back(g_wk_height * _grp_tile1);
        _g_wk.push_back(_search_params.n_outputs * _search_params.batch_sz);
    }
    int data_len = (_search_params.out_data_type == "FP32" ? 4 : 8);

    // calculate workspace
    size_t scale_sz = _search_params.batch_sz * scale_batch_stride * data_len;
    _workspce_sz    = (doBackward()) ? scale_sz : 0;

    return (ret);
}

int mlo_construct_norm::mloConstructBwd()
{
    int ret = 0;

    _out_pix_tile0 = 1;
    _out_pix_tile1 = 1;
    _grp_tile0     = 8;
    _grp_tile1     = 8;
    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        _grp_tile0 = (_in_df_width <= 8) ? 8 : 16;
        _grp_tile1 = (_in_df_height <= 8) ? 8 : 16;
    }
    else
    {
        _out_pix_tile0 = (_in_df_width <= 8) ? 1 : (_in_df_width <= 16) ? 2 : 4;
        _out_pix_tile1 = (_in_df_height <= 8) ? 1 : (_in_df_height <= 16) ? 2 : 4;
        ;
    }
    auto ocl_group_lg2sz0 = static_cast<int>(ceil(log(static_cast<double>(_grp_tile0)) / log(2.)));
    auto ocl_group_lg2sz1 = static_cast<int>(ceil(log(static_cast<double>(_grp_tile1)) / log(2.)));

    int pre_pad              = (_norm_area - 1) / 2;
    int pad                  = _norm_area - pre_pad - 1;
    int scale_stride         = _search_params.out_stride;
    int scale_channel_stride = _search_params.out_channel_stride;
    int scale_batch_stride   = _search_params.out_batch_stride;

    _comp_options =
        std::string(" -DMLO_LRN_KERNEL_SZ=") + std::to_string(static_cast<long long>(_norm_area)) +
        std::string(" -DMLO_LRN_N_OUTPUTS=") +
        std::to_string(static_cast<long long>(_search_params.n_outputs)) +
        std::string(" -DMLO_LRN_N_CHANNELS=") +
        std::to_string(static_cast<long long>(_search_params.n_inputs)) +
        std::string(" -DMLO_LRN_PAD=") + std::to_string(static_cast<long long>(pad)) +
        std::string(" -DMLO_LRN_N_HORIZ_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile0)) +
        std::string(" -DMLO_LRN_N_VERT_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile1)) +
        std::string(" -DMLO_LRN_GROUP_SZ0=") + std::to_string(static_cast<long long>(_grp_tile0)) +
        std::string(" -DMLO_LRN_GROUP_SZ1=") + std::to_string(static_cast<long long>(_grp_tile1)) +
        std::string(" -DMLO_LRN_GROUP_LG2SZ0=") +
        std::to_string(static_cast<long long>(ocl_group_lg2sz0)) +
        std::string(" -DMLO_LRN_GROUP_LG2SZ1=") +
        std::to_string(static_cast<long long>(ocl_group_lg2sz1)) +
        std::string(" -DMLO_LRN_BOT_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.in_batch_stride)) +
        std::string(" -DMLO_LRN_BOT_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.in_channel_stride)) +
        std::string(" -DMLO_LRN_BOT_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.in_stride)) +
        std::string(" -DMLO_LRN_TOP_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.out_batch_stride)) +
        std::string(" -DMLO_LRN_TOP_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.out_channel_stride)) +
        std::string(" -DMLO_LRN_TOP_STRIDE=") +
        std::to_string(static_cast<long long>(_search_params.out_stride)) +
        std::string(" -DMLO_LRN_BOT_WIDTH=") +
        std::to_string(static_cast<long long>(_search_params.in_width)) +
        std::string(" -DMLO_LRN_BOT_HEIGHT=") +
        std::to_string(static_cast<long long>(_search_params.in_height)) +
        std::string(" -DMLO_LRN_TOP_WIDTH=") +
        std::to_string(static_cast<long long>(_search_params.out_width)) +
        std::string(" -DMLO_LRN_TOP_HEIGHT=") +
        std::to_string(static_cast<long long>(_search_params.out_height)) +
        std::string(" -DMLO_LRN_SCALE_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(scale_batch_stride)) +
        std::string(" -DMLO_LRN_SCALE_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(scale_channel_stride)) +
        std::string(" -DMLO_LRN_SCALE_STRIDE=") +
        std::to_string(static_cast<long long>(scale_stride)) +
        std::string(" -DMLO_LRN_TOPDF_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_out_df_batch_stride)) +
        std::string(" -DMLO_LRN_TOPDF_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_out_df_channel_stride)) +
        std::string(" -DMLO_LRN_TOPDF_STRIDE=") +
        std::to_string(static_cast<long long>(_out_df_stride)) +
        std::string(" -DMLO_LRN_BOTDF_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(_in_df_batch_stride)) +
        std::string(" -DMLO_LRN_BOTDF_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(_in_df_channel_stride)) +
        std::string(" -DMLO_LRN_BOTDF_STRIDE=") +
        std::to_string(static_cast<long long>(_in_df_stride)) +
        std::string(" -DMLO_LRN_BATCH_SZ=") +
        std::to_string(static_cast<long long>(_search_params.batch_sz)) +
        std::string(" -DMLO_LRN_N_INPUTS=") +
        std::to_string(static_cast<long long>(_search_params.n_inputs)) +
        std::string(" -DMLO_LRN_N_OUTPUTS=") +
        std::to_string(static_cast<long long>(_search_params.n_outputs)) + getGeneralCompOptions();

    _kernel_file = "MIOpenLRNBwd.cl";

    _l_wk.clear();
    _g_wk.clear();
    _l_wk.push_back(_grp_tile0);
    _l_wk.push_back(_grp_tile1);
    _l_wk.push_back(1);

    if(_norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        _g_wk.push_back(_in_df_width);
        _g_wk.push_back(_in_df_height);
        _g_wk.push_back(_search_params.batch_sz);
        _kernel_name = "MIOpenLRNAcrossChannelsBwd1";
    }
    else
    {
        int g_wk_width =
            ((_in_df_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
        int g_wk_height =
            ((_in_df_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));

        _g_wk.push_back(g_wk_width * _grp_tile0);
        _g_wk.push_back(g_wk_height * _grp_tile1);
        _g_wk.push_back(_search_params.n_inputs * _search_params.batch_sz);
        _kernel_name = "MIOpenLRNWithinChannelBwd";
    }

    return (ret);
}
