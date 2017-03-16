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

//#define MLOPEN
#include <mlopen/mlo_internal.hpp>
#include <mlopen/mlo_utils.hpp>

int mlo_construct_neuron::mloConstruct()
{
	int ret = 0;


	if (isForwardDirection())
	{

		ret = mloConstructFwd();
	}
	else
	{
		ret = mloConstructBwd();
	}
	return(ret);
} 

int mlo_construct_neuron::mloConstructFwd()
{
	int ret = 0;
	_hw_wave_sz = 64;

	int data_len = (!_in_data_type.compare("FP32") ? 4 : 8);

	int read_unit = 4;
	std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((read_unit));

	size_t map_size = _in_width*_in_height;
	size_t map_size_aligned = (map_size + read_unit - 1) / read_unit;
	int N_PIXS_OFF = map_size - (map_size / read_unit) * read_unit;


	size_t glbl_wk = map_size_aligned * _batch_sz;



	_grp_tile0 = std::min((int)((glbl_wk + _hw_wave_sz-1)/ _hw_wave_sz)*_hw_wave_sz, (int)256);
	_grp_tile1 = 1;

	_comp_options =
		std::string(" -DMLO_NRN_GROUP_SZ0=") + std::to_string(static_cast<long long>(_grp_tile0))
		+ std::string(" -DMLO_NRN_GROUP_SZ1=") + std::to_string(static_cast<long long>(_grp_tile1))
		+ std::string(" -DMLO_NRN_OP_ID=") + std::to_string(static_cast<long long>(_neuron_type))
		+ std::string(" -DMLO_N_PIXS_OFF=") + std::to_string(static_cast<long long>(N_PIXS_OFF))
		+ std::string(" -DMLO_MAP_SIZE_ALIGNED=") + std::to_string(static_cast<long long>(map_size_aligned))
		+ std::string(" -DMLO_READ_UNIT=") + std::to_string(static_cast<long long>(read_unit))
		+ getGeneralCompOptions()
		;

	_l_wk.clear();
	_l_wk.push_back(_grp_tile0);
	_l_wk.push_back(_grp_tile1);
	_l_wk.push_back(1);

	_g_wk.clear();
	_g_wk.push_back(glbl_wk);
	_g_wk.push_back(1);
	_g_wk.push_back(1);


	_kernel_file = "MLOpenNeuron.cl";
	_kernel_name = (isForwardDirection())? "MLOpenNeuron4" : "MLOpenNeuron4_Bwd";

	return(ret);
}


int mlo_construct_neuron::mloConstructBwd()
{
	int ret = 0;

#if 1
	ret = mloConstructFwd();
#else
	int data_len = (!_in_data_type.compare("FP32") ? 4 : 8);

	size_t size = (_bot_sz / data_len);
	if (((size / 4) * 4) != size)
	{
		printf("Error: buffer size is not multipel of 4.\n");
	//	ret = -1;
	//	return(ret);
	}

	size_t glbl_wk = size / 4;

	_grp_tile0 = 256;
	_grp_tile1 = 1;

	_comp_options =
		std::string(" -DMLO_NRN_GROUP_SZ0=") + std::to_string(static_cast<long long>(_grp_tile0))
		+ std::string(" -DMLO_NRN_GROUP_SZ1=") + std::to_string(static_cast<long long>(_grp_tile1))
		+ std::string(" -DMLO_NRN_OP_ID=") + std::to_string(static_cast<long long>(_neuron_type))
		+ getGeneralCompOptions()
		;

	_l_wk.clear();
	_l_wk.push_back(_grp_tile0);
	_l_wk.push_back(_grp_tile1);
	_l_wk.push_back(1);

	_g_wk.clear();
	_g_wk.push_back(glbl_wk);
	_g_wk.push_back(1);
	_g_wk.push_back(1);
	_kernel_file = "MLOpenNeuron.cl";
	_kernel_name = "MLOpenNeuron4_Bwd";
#endif
	return(ret);
}
