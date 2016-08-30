/**********************************************************************
Copyright (c)2016 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

?	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
?	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
// to share code with between CPU and GPU

#define MLOPEN
#include <mlopen/mlo_internal.hpp>
#include <mlopen/mlo_utils.hpp>

// KNOWN ISSUES:
// backward propogagation has a bug in cross map normalization when numper of maps less than normalization region

int mlo_construct_norm::mloConstruct(void)
{
	int ret = 0;
	if (_direction == 1)
	{
		ret = mloConstructFwd();
	}
	else
	{
		ret = mloConstructBwd();
	}
	return(ret);
}

int mlo_construct_norm::mloConstructFwd(void)
{
	int ret = 0;

	int pre_pad = (_norm_area - 1) / 2;
	int pad = _norm_area - pre_pad - 1;

	int top_df_stride = 1;
	int top_df_channel_stride = 1;
	int	top_df_batch_stride = 1;

	int bot_df_stride = 1;
	int bot_df_channel_stride = 1;
	int	bot_df_batch_stride = 1;


	_grp_tile0 = 8;
	_grp_tile1 = 8;
	_out_pix_tile0 = 1;
	_out_pix_tile1 = 1;

	if (_norm_region == MLO_LRN_ACROSS_CHANNELS)
	{
		_grp_tile0 = (_out_width <= 8) ? 8 : 16;
		_grp_tile1 = (_out_height <= 8) ? 8 : 16;

	}
	else
	{

		_out_pix_tile0 = (_out_width <= 8) ? 1 : (_out_width <= 16) ? 2 : 4;
		_out_pix_tile1 = (_out_height <= 8) ? 1 : (_out_height <= 16) ? 2 : 4;;
	}
	int ocl_group_lg2sz0 = (int)ceil(log((double)_out_pix_tile0) / log(2.));
	int ocl_group_lg2sz1 = (int)ceil(log((double)_out_pix_tile1) / log(2.));

	// workspace size !!!!
	int scale_stride = _out_stride;
	int scale_channel_stride = _out_channel_stride;
	int	scale_batch_stride = _out_batch_stride;
	int scale = (doBackward()) ? 1 : 0;

	_comp_options =
		std::string(" -D MLO_LRN_KERNEL_SZ=") + std::to_string((long long)_norm_area)
		+ std::string(" -D MLO_LRN_N_OUTPUTS=") + std::to_string((long long)_n_outputs)
		+ std::string(" -D MLO_LRN_N_CHANNELS=") + std::to_string((long long)_n_inputs)
		+ std::string(" -D MLO_LRN_PAD=") + std::to_string((long long)pad)
		+ std::string(" -D MLO_LRN_N_HORIZ_OUT_PIX=") + std::to_string((long long)_out_pix_tile0)
		+ std::string(" -D MLO_LRN_N_VERT_OUT_PIX=") + std::to_string((long long)_out_pix_tile1)
		+ std::string(" -D MLO_LRN_GROUP_SZ0=") + std::to_string((long long)_grp_tile0)
		+ std::string(" -D MLO_LRN_GROUP_SZ1=") + std::to_string((long long)_grp_tile1)
		+ std::string(" -D MLO_LRN_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
		+ std::string(" -D MLO_LRN_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
		+ std::string(" -D MLO_LRN_BOT_BATCH_STRIDE=") + std::to_string((long long)_in_batch_stride)
		+ std::string(" -D MLO_LRN_BOT_CHANNEL_STRIDE=") + std::to_string((long long)_in_channel_stride)
		+ std::string(" -D MLO_LRN_BOT_STRIDE=") + std::to_string((long long)_in_stride)
		+ std::string(" -D MLO_LRN_TOP_BATCH_STRIDE=") + std::to_string((long long)_out_batch_stride)
		+ std::string(" -D MLO_LRN_TOP_CHANNEL_STRIDE=") + std::to_string((long long)_out_channel_stride)
		+ std::string(" -D MLO_LRN_TOP_STRIDE=") + std::to_string((long long)_out_stride)
		+ std::string(" -D MLO_LRN_BOT_WIDTH=") + std::to_string((long long)_out_width)
		+ std::string(" -D MLO_LRN_BOT_HEIGHT=") + std::to_string((long long)_out_height)
		+ std::string(" -D MLO_LRN_TOP_WIDTH=") + std::to_string((long long)_out_width)
		+ std::string(" -D MLO_LRN_TOP_HEIGHT=") + std::to_string((long long)_out_height)
		+ std::string(" -D MLO_LRN_SCALE_BATCH_STRIDE=") + std::to_string((long long)scale_batch_stride)
		+ std::string(" -D MLO_LRN_SCALE_CHANNEL_STRIDE=") + std::to_string((long long)scale_channel_stride)
		+ std::string(" -D MLO_LRN_SCALE_STRIDE=") + std::to_string((long long)scale_stride)
		+ std::string(" -D MLO_LRN_TOPDF_BATCH_STRIDE=") + std::to_string((long long)top_df_batch_stride)
		+ std::string(" -D MLO_LRN_TOPDF_CHANNEL_STRIDE=") + std::to_string((long long)top_df_channel_stride)
		+ std::string(" -D MLO_LRN_TOPDF_STRIDE=") + std::to_string((long long)top_df_stride)
		+ std::string(" -D MLO_LRN_BOTDF_BATCH_STRIDE=") + std::to_string((long long)bot_df_batch_stride)
		+ std::string(" -D MLO_LRN_BOTDF_CHANNEL_STRIDE=") + std::to_string((long long)bot_df_channel_stride)
		+ std::string(" -D MLO_LRN_BOTDF_STRIDE=") + std::to_string((long long)bot_df_stride)
		+ std::string(" -D MLO_LRN_BATCH_SZ=") + std::to_string((long long)_batch_sz)
		+ std::string(" -D MLO_LRN_N_INPUTS=") + std::to_string((long long)_n_inputs)
		+ std::string(" -D MLO_LRN_N_OUTPUTS=") + std::to_string((long long)_n_outputs)
		+ std::string(" -D MLO_LRN_DO_SCALE=") + std::to_string((long long)scale)
		+ getGeneralCompOptions()
		;



	_kernel_file = "MLOpenLRN.cl";
	_kernel_name = (_norm_region == MLO_LRN_ACROSS_CHANNELS) ? "MLOpenLRNAcrossChannels1" : "MLOpenLRNWithinChannel";

	_l_wk.clear();
	_l_wk.push_back(_grp_tile0);
	_l_wk.push_back(_grp_tile1);
	_l_wk.push_back(1);

	_g_wk.clear();
	if (_norm_region == MLO_LRN_ACROSS_CHANNELS)
	{
		_g_wk.push_back(_out_width);
		_g_wk.push_back(_out_height);
		_g_wk.push_back(_batch_sz);
	}
	else
	{
		int g_wk_width = (int)((_out_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
		int g_wk_height = (int)((_out_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));

		_g_wk.push_back(g_wk_width * _grp_tile0);
		_g_wk.push_back(g_wk_height * _grp_tile1);
		_g_wk.push_back(_n_outputs * _batch_sz);

	}
	int data_len = (!_out_data_type.compare("FP32") ? 4 : 8);

	// calculate workspace
	size_t scale_sz = _batch_sz * scale_batch_stride * data_len;
	_workspce_sz = (doBackward()) ? scale_sz : 0;

	return(ret);
}

int mlo_construct_norm::mloConstructBwd(void)
{
	int ret = 0;

	_out_pix_tile0 = 1;
	_out_pix_tile1 = 1;
	_grp_tile0 = 8;
	_grp_tile1 = 8;
	if (_norm_region == MLO_LRN_ACROSS_CHANNELS)
	{
		_grp_tile0 = (_in_df_width <= 8) ? 8 : 16;
		_grp_tile1 = (_in_df_height <= 8) ? 8 : 16;


	}
	else
	{
		_out_pix_tile0 = (_in_df_width <= 8) ? 1 : (_in_df_width <= 16) ? 2 : 4;
		_out_pix_tile1 = (_in_df_height <= 8) ? 1 : (_in_df_height <= 16) ? 2 : 4;;
	}
	int ocl_group_lg2sz0 = (int)ceil(log((double)_grp_tile0) / log(2.));
	int ocl_group_lg2sz1 = (int)ceil(log((double)_grp_tile1) / log(2.));

	int pre_pad = (_norm_area - 1) / 2;
	int pad = _norm_area - pre_pad - 1;
	int scale_stride = _out_stride;
	int scale_channel_stride = _out_channel_stride;
	int	scale_batch_stride = _out_batch_stride;

	_comp_options =
		std::string(" -D MLO_LRN_KERNEL_SZ=") + std::to_string((long long)_norm_area)
		+ std::string(" -D MLO_LRN_N_OUTPUTS=") + std::to_string((long long)_n_outputs)
		+ std::string(" -D MLO_LRN_N_CHANNELS=") + std::to_string((long long)_n_inputs)
		+ std::string(" -D MLO_LRN_PAD=") + std::to_string((long long)pad)
		+ std::string(" -D MLO_LRN_N_HORIZ_OUT_PIX=") + std::to_string((long long)_out_pix_tile0)
		+ std::string(" -D MLO_LRN_N_VERT_OUT_PIX=") + std::to_string((long long)_out_pix_tile1)
		+ std::string(" -D MLO_LRN_GROUP_SZ0=") + std::to_string((long long)_grp_tile0)
		+ std::string(" -D MLO_LRN_GROUP_SZ1=") + std::to_string((long long)_grp_tile1)
		+ std::string(" -D MLO_LRN_GROUP_LG2SZ0=") + std::to_string((long long)ocl_group_lg2sz0)
		+ std::string(" -D MLO_LRN_GROUP_LG2SZ1=") + std::to_string((long long)ocl_group_lg2sz1)
		+ std::string(" -D MLO_LRN_BOT_BATCH_STRIDE=") + std::to_string((long long)_in_batch_stride)
		+ std::string(" -D MLO_LRN_BOT_CHANNEL_STRIDE=") + std::to_string((long long)_in_channel_stride)
		+ std::string(" -D MLO_LRN_BOT_STRIDE=") + std::to_string((long long)_in_stride)
		+ std::string(" -D MLO_LRN_TOP_BATCH_STRIDE=") + std::to_string((long long)_out_batch_stride)
		+ std::string(" -D MLO_LRN_TOP_CHANNEL_STRIDE=") + std::to_string((long long)_out_channel_stride)
		+ std::string(" -D MLO_LRN_TOP_STRIDE=") + std::to_string((long long)_out_stride)
		+ std::string(" -D MLO_LRN_BOT_WIDTH=") + std::to_string((long long)_in_width)
		+ std::string(" -D MLO_LRN_BOT_HEIGHT=") + std::to_string((long long)_in_height)
		+ std::string(" -D MLO_LRN_TOP_WIDTH=") + std::to_string((long long)_out_width)
		+ std::string(" -D MLO_LRN_TOP_HEIGHT=") + std::to_string((long long)_out_height)
		+ std::string(" -D MLO_LRN_SCALE_BATCH_STRIDE=") + std::to_string((long long)scale_batch_stride)
		+ std::string(" -D MLO_LRN_SCALE_CHANNEL_STRIDE=") + std::to_string((long long)scale_channel_stride)
		+ std::string(" -D MLO_LRN_SCALE_STRIDE=") + std::to_string((long long)scale_stride)
		+ std::string(" -D MLO_LRN_TOPDF_BATCH_STRIDE=") + std::to_string((long long)_out_df_batch_stride)
		+ std::string(" -D MLO_LRN_TOPDF_CHANNEL_STRIDE=") + std::to_string((long long)_out_df_channel_stride)
		+ std::string(" -D MLO_LRN_TOPDF_STRIDE=") + std::to_string((long long)_out_df_stride)
		+ std::string(" -D MLO_LRN_BOTDF_BATCH_STRIDE=") + std::to_string((long long)_in_df_batch_stride)
		+ std::string(" -D MLO_LRN_BOTDF_CHANNEL_STRIDE=") + std::to_string((long long)_in_df_channel_stride)
		+ std::string(" -D MLO_LRN_BOTDF_STRIDE=") + std::to_string((long long)_in_df_stride)
		+ std::string(" -D MLO_LRN_BATCH_SZ=") + std::to_string((long long)_batch_sz)
		+ std::string(" -D MLO_LRN_N_INPUTS=") + std::to_string((long long)_n_inputs)
		+ std::string(" -D MLO_LRN_N_OUTPUTS=") + std::to_string((long long)_n_outputs)
		+ getGeneralCompOptions()
		;

	_kernel_file = "MLOpenLRN.cl";

	_l_wk.clear();
	_g_wk.clear();
	_l_wk.push_back(_grp_tile0);
	_l_wk.push_back(_grp_tile1);
	_l_wk.push_back(1);

	if (_norm_region == MLO_LRN_ACROSS_CHANNELS)
	{
		_g_wk.push_back(_in_df_width);
		_g_wk.push_back(_in_df_height);
		_g_wk.push_back(_batch_sz);
		_kernel_name = "MLOpenLRNAcrossChannelsBwd1";
	}
	else
	{
		int g_wk_width = (int)((_in_df_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
		int g_wk_height = (int)((_in_df_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));

		_g_wk.push_back(g_wk_width * _grp_tile0);
		_g_wk.push_back(g_wk_height * _grp_tile1);
		_g_wk.push_back(_n_inputs * _batch_sz);
		_kernel_name = "MLOpenLRNWithinChannelBwd";

	}

	return(ret);
}
