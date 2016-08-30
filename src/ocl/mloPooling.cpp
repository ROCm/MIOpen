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

int mlo_construct_pooling2D::mloConstruct()
{
	int ret = 0;


	if (getDirectcion())
	{

		ret = mloConstructFwd();
	}
	else
	{
		ret = mloConstructBwd();
	}
	return(ret);
} 

int mlo_construct_pooling2D::mloConstructFwd()
{
	int ret = 0;

	_grp_tile0 = 8;
	_grp_tile1 = 8;
	int ocl_group_lg2sz0 = static_cast<int>(ceil(log(static_cast<double>(_grp_tile0)) / log(2.)));
	int ocl_group_lg2sz1 = static_cast<int>(ceil(log(static_cast<double>(_grp_tile1)) / log(2.)));;


	_out_pix_tile0 = (_out_width < _grp_tile0 * 2) ? 1 : 2;
	_out_pix_tile1 = (_out_height < _grp_tile1 * 2) ? 1 : 2;

	_comp_options =
		std::string(" -D MLO_POOLING_OP_ID=") + std::to_string(static_cast<long long>(_pooling_method))
		+ std::string(" -D MLO_POOLING_KERNEL_SZ1=") + std::to_string(static_cast<long long>(_kernel_size1))
		+ std::string(" -D MLO_POOLING_PAD1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -D MLO_POOLING_STRIDE1=") + std::to_string(static_cast<long long>(_kernel_stride1))
		+ std::string(" -D MLO_POOLING_KERNEL_SZ0=") + std::to_string(static_cast<long long>(_kernel_size0))
		+ std::string(" -D MLO_POOLING_PAD0=") + std::to_string(static_cast<long long>(_pad0))
		+ std::string(" -D MLO_POOLING_STRIDE0=") + std::to_string(static_cast<long long>(_kernel_stride0))
		+ std::string(" -D MLO_POOLING_N_OUTPUTS=") + std::to_string(static_cast<long long>(_n_outputs))
		+ std::string(" -D MLO_POOLING_N_CHANNELS=") + std::to_string(static_cast<long long>(_n_inputs))
		+ std::string(" -D MLO_POOLING_N_HORIZ_OUT_PIX=") + std::to_string(static_cast<long long>(_out_pix_tile0))
		+ std::string(" -D MLO_POOLING_N_VERT_OUT_PIX=") + std::to_string(static_cast<long long>(_out_pix_tile1))
		+ std::string(" -D MLO_POOLING_GROUP_SZ0=") + std::to_string(static_cast<long long>(_grp_tile0))
		+ std::string(" -D MLO_POOLING_GROUP_SZ1=") + std::to_string(static_cast<long long>(_grp_tile1))
		+ std::string(" -D MLO_POOLING_GROUP_LG2SZ0=") + std::to_string(static_cast<long long>(ocl_group_lg2sz0))
		+ std::string(" -D MLO_POOLING_GROUP_LG2SZ1=") + std::to_string(static_cast<long long>(ocl_group_lg2sz1))
		+ std::string(" -D MLO_POOLING_BOT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -D MLO_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -D MLO_POOLING_BOT_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		+ std::string(" -D MLO_POOLING_TOP_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -D MLO_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -D MLO_POOLING_TOP_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -D MLO_POOLING_BOT_WIDTH=") + std::to_string(static_cast<long long>(_in_width))
		+ std::string(" -D MLO_POOLING_BOT_HEIGHT=") + std::to_string(static_cast<long long>(_in_height))
		+ std::string(" -D MLO_POOLING_TOP_WIDTH=") + std::to_string(static_cast<long long>(_out_width))
		+ std::string(" -D MLO_POOLING_TOP_HEIGHT=") + std::to_string(static_cast<long long>(_out_height))
		+ getGeneralCompOptions()
		;


	int g_wk_width = ((_out_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
	int g_wk_height = ((_out_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));

	_l_wk.clear();
	_l_wk.push_back(_grp_tile0);
	_l_wk.push_back(_grp_tile1);
	_l_wk.push_back(1);

	_g_wk.clear();
	_g_wk.push_back(g_wk_width * _grp_tile0);
	_g_wk.push_back(g_wk_height * _grp_tile1);
	_g_wk.push_back(_n_outputs * _batch_sz);


	_kernel_file = "MLOpenPooling.cl";

	_kernel_name = "mloPooling";


	return(ret);
}


int mlo_construct_pooling2D::mloConstructBwd()
{
	int ret = 0;

	_out_pix_tile0 = _kernel_stride0;
	_out_pix_tile1 = _kernel_stride1;
	_grp_tile0 = 8;
	_grp_tile1 = 8;
	int ocl_group_lg2sz0 = static_cast<int>(ceil(log(static_cast<double>(_grp_tile0)) / log(2.)));
	int ocl_group_lg2sz1 = static_cast<int>(ceil(log(static_cast<double>(_grp_tile1)) / log(2.)));


	_comp_options =
		std::string(" -D MLO_POOLING_KERNEL_SZ1=") + std::to_string(static_cast<long long>(_kernel_size1))
		+ std::string(" -D MLO_POOLING_PAD1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -D MLO_POOLING_STRIDE1=") + std::to_string(static_cast<long long>(_kernel_stride1))
		+ std::string(" -D MLO_POOLING_KERNEL_SZ0=") + std::to_string(static_cast<long long>(_kernel_size0))
		+ std::string(" -D MLO_POOLING_PAD0=") + std::to_string(static_cast<long long>(_pad0))
		+ std::string(" -D MLO_POOLING_STRIDE0=") + std::to_string(static_cast<long long>(_kernel_stride0))
		+ std::string(" -D MLO_POOLING_N_OUTPUTS=") + std::to_string(static_cast<long long>(_n_outputs))
		+ std::string(" -D MLO_POOLBWD_N_HORIZ_OUT_PIX=") + std::to_string(static_cast<long long>(_out_pix_tile0))
		+ std::string(" -D MLO_POOLBWD_N_VERT_OUT_PIX=") + std::to_string(static_cast<long long>(_out_pix_tile1))
		+ std::string(" -D MLO_POOLBWD_GROUP_SZ0=") + std::to_string(static_cast<long long>(_grp_tile0))
		+ std::string(" -D MLO_POOLBWD_GROUP_SZ1=") + std::to_string(static_cast<long long>(_grp_tile1))
		+ std::string(" -D MLO_POOLBWD_GROUP_LG2SZ0=") + std::to_string(static_cast<long long>(ocl_group_lg2sz0))
		+ std::string(" -D MLO_POOLBWD_GROUP_LG2SZ1=") + std::to_string(static_cast<long long>(ocl_group_lg2sz1))
		+ std::string(" -D MLO_POOLBWD_BOT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -D MLO_POOLBWD_BOT_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -D MLO_POOLBWD_BOT_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		+ std::string(" -D MLO_POOLBWD_TOP_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -D MLO_POOLBWD_TOP_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -D MLO_POOLBWD_TOP_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -D MLO_POOLBWD_BOT_WIDTH=") + std::to_string(static_cast<long long>(_in_width))
		+ std::string(" -D MLO_POOLBWD_BOT_HEIGHT=") + std::to_string(static_cast<long long>(_in_height))
		+ std::string(" -D MLO_POOLBWD_TOP_WIDTH=") + std::to_string(static_cast<long long>(_out_width))
		+ std::string(" -D MLO_POOLBWD_TOP_HEIGHT=") + std::to_string(static_cast<long long>(_out_height))
		+ std::string(" -D MLO_POOLBWD_BOTDF_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_df_batch_stride))
		+ std::string(" -D MLO_POOLBWD_BOTDF_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_in_df_channel_stride))
		+ std::string(" -D MLO_POOLBWD_BOTDF_STRIDE=") + std::to_string(static_cast<long long>(_in_df_stride))
		+ std::string(" -D MLO_POOLBWD_TOPDF_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_df_batch_stride))
		+ std::string(" -D MLO_POOLBWD_TOPDF_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_out_df_channel_stride))
		+ std::string(" -D MLO_POOLBWD_TOPDF_STRIDE=") + std::to_string(static_cast<long long>(_out_df_stride))

		+ getGeneralCompOptions()
		;


	int g_wk_width = ((_in_width + _grp_tile0 * _out_pix_tile0 - 1) / (_grp_tile0 * _out_pix_tile0));
	int g_wk_height = ((_in_height + _grp_tile1 * _out_pix_tile1 - 1) / (_grp_tile1 * _out_pix_tile1));

	_l_wk.clear();
	_l_wk.push_back(_grp_tile0);
	_l_wk.push_back(_grp_tile1);
	_l_wk.push_back(1);

	_g_wk.clear();
	_g_wk.push_back(g_wk_width * _grp_tile0);
	_g_wk.push_back(g_wk_height * _grp_tile1);
	_g_wk.push_back(_n_inputs * _batch_sz);


	_kernel_file = "MLOpenPoolingBwd.cl";
	if (_pooling_method == MLO_POOLING_OP_MAX)
	{
		_kernel_name = "mloPoolingMaxBwd";
	}
	else if (_pooling_method == MLO_POOLING_OP_AVE)
	{
		_kernel_name = "mloPoolingAveBwd";
	}
	else
	{
		std::cout << "Layer: %s. Error: unknowm method\n";
		ret = -1;
	}
	return(ret);
}
