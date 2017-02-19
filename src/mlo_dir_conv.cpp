/**********************************************************************
  Copyright (c)2017 Advanced Micro Devices, Inc. All rights reserved.

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

#include <mlopen/db.hpp>

static int mloLg2(int v)
{
	int ret = static_cast<int>(std::ceil(std::log(v) / std::log(2)));
	return(ret);
}


/*
   the search db is a text file with the name defined by the device characteristics.
   each line is a key/value pair, separated by a space:
   32x16x16x3x3x64x16x16x100xNCHWxFP32x1 16.16.16.16.1.4.8.4.1
   or
   64x8x8x5x5x32x8x8x100xNCHWxFP32x0 16.16.8.8.2.4.1.1.4

   key format (all values are separted by x):
   n input maps
   input height
   input width
   filter height
   filter width
   n output maps
   output height
   output width
   batch size
   tensors' layout
   tensprs' data type
   direction (1 - forward, 0 - backward)

Note:
for backward direction - input and output are reversed.

value format (all values are separated by .):
vertical group size
horizontal group size
input block vertical size
input block horizontal size
output tile vertical size
output tile horizaontal size
n of output tiles
n of input blocks
n batchs (stacks) processed by the group
*/



	static
int mloBuildConf_Val(
		std::string & conf_val,
		int grp_tile1,
		int grp_tile0,
		int in_tile1,
		int in_tile0,
		int out_pix_tile1,
		int out_pix_tile0,
		int n_out_pix_tiles,
		int n_in_data_tiles,
		int n_stacks
		)
{
	conf_val = std::to_string(static_cast<long long>(grp_tile1)) + std::string(".")
		+ std::to_string(static_cast<long long>(grp_tile0)) + std::string(".")
		+ std::to_string(static_cast<long long>(in_tile1)) + std::string(".")
		+ std::to_string(static_cast<long long>(in_tile0)) + std::string(".")
		+ std::to_string(static_cast<long long>(out_pix_tile1)) + std::string(".")
		+ std::to_string(static_cast<long long>(out_pix_tile0)) + std::string(".")
		+ std::to_string(static_cast<long long>(n_out_pix_tiles)) + std::string(".")
		+ std::to_string(static_cast<long long>(n_in_data_tiles)) + std::string(".")
		+ std::to_string(static_cast<long long>(n_stacks))
		;
	return(0);

}

	static
int mloParseConf(const std::string & conf_val,
		int & grp_tile1,
		int & grp_tile0,
		int & in_tile1,
		int & in_tile0,
		int & out_pix_tile1,
		int & out_pix_tile0,
		int & n_out_pix_tiles,
		int & n_in_data_tiles,
		int & n_stacks
		)
{
	std::vector<std::string> conf_val_vec;
	tokenize(conf_val,
			conf_val_vec,
			std::string("."));
	grp_tile1 = std::stoi(conf_val_vec[0]);
	grp_tile0 = std::stoi(conf_val_vec[1]);
	in_tile1 = std::stoi(conf_val_vec[2]);
	in_tile0 = std::stoi(conf_val_vec[3]);
	out_pix_tile1 = std::stoi(conf_val_vec[4]);
	out_pix_tile0 = std::stoi(conf_val_vec[5]);
	n_out_pix_tiles = std::stoi(conf_val_vec[6]);
	n_in_data_tiles = std::stoi(conf_val_vec[7]);
	n_stacks = std::stoi(conf_val_vec[8]);
	return(0);

}

	static
int mloReadDb(
		const std::string confreq_db_name,
		std::vector<std::string> &db
		)
{
	int ret = 0;


	mloFile f;

	ret = f.readBinaryFromFile(confreq_db_name.c_str());

	tokenize(f.source(),
			db,
			std::string("\n\r"));

	return(ret);
}

	static
int mloUpdateDb(const std::string  &file_nm, const std::vector<std::string> & db)
{
	mloFile f;
	// serialize
	std::string serial;
	std::vector<std::string>::const_iterator it;
	for (it = db.begin(); it != db.end(); ++it)
	{
		serial += (*it) + "\n";
	}

	int ret = f.writeBinaryToFile(file_nm.c_str(), serial.c_str(), serial.length());


	return(ret);
}



	static
bool mloFindConfigReq(
		const std::string confreq_db_name,
		const std::string & conf_key,
		std::vector<std::string> &req_conf_db,
		std::vector<std::string>::iterator &it
		)
{
	bool ret = true;

	mloReadDb(confreq_db_name,
			req_conf_db
			);

	// find req string
	ret = false;
	for (it = req_conf_db.begin(); it != req_conf_db.end(); ++it)
	{
		if (!(*it).compare(conf_key))
		{
			ret = true;
			break;
		}
	}
	return(ret);
}

	static
bool mloSearchConfigDB(
		std::map<std::string, std::string> & conf_db,
		std::string & conf_key,
		std::string & conf_val,
		std::map<std::string, std::string>::iterator & it
		)
{

	bool found = false;

	it = conf_db.find(conf_key);
	if (it != conf_db.end())
	{
		found = true;
		conf_val = (*it).second;

		//			std::cout << "Found : " << conf_val << std::endl;
	}
	return(found);
}


/************************************************************************************************************************
 **
 **			CONSTRUCT CONVOLUTIONAL LAYER
 **
 ************************************************************************************************************************/

/*
   construction has been split into 2
   generic convlution forward 
   non-generic stride = 1, forward and backward
   */
int mlo_construct_direct2D::mloConstruct()
{
	int ret = 0;
	_gen = (_kernel_size0 > 11 || _kernel_size1 > 11 || _kernel_stride0 > 1 || _kernel_stride1 > 1);

#if MLOPEN_BACKEND_OPENCL
	const auto use_precompiled_binaries_env_p = std::getenv("MLOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES");
	const auto use_precompiled_binaries = ((use_precompiled_binaries_env_p == nullptr) || (std::strcmp(use_precompiled_binaries_env_p, "disable") != 0));

	/*
	Our testing shows that for some corner cases (i.e. specific problem descriptions),
	assembly-written kernels may have worse performance than kernels written in high-level language, e.g. OpenCL.
	For example, forward 3x3 convolution kernel employing Winograd algorithm (conv_3x3_wheel) is very slow when InputWidth x InputHeight is 7x7.
	miOpen avoids asm kernels in such corner cases. This setting overrides that.
	*/
	const auto use_asm_kernels_perf_filtering_env_p = std::getenv("MLOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING");
	const auto use_asm_kernels_perf_filtering = ((use_asm_kernels_perf_filtering_env_p == nullptr) || (std::strcmp(use_asm_kernels_perf_filtering_env_p, "disable") != 0));

	if (use_precompiled_binaries && mloCheckWinograd3x3FwdConvCondition() && (!use_asm_kernels_perf_filtering || mloCheckWinograd3x3FwdConvPerfFilter()))
	{
		return (mloConstructWinograd3x3FwdConv());
	}
#endif

	if (_gen && getDirection())
	{
		ret = mloConstructDirect2DFwdGen();
	}
#if 1
	else if ((_kernel_size0 == 3 && _kernel_size1 == 3 && _pad1 == 1 && _pad0 == 1 && getDirection()) && (_out_width == 512 || _out_width == 64 || _out_width == 128 || _out_width == 256))
	{
		return(mloConstructDirect2D3x3());
	}
#endif
	else
	{
		// search known configurations
		bool known_config = mloGetConfig();
		// if not known and the saerch is alloed - search

		if (!known_config)
		{
			if (doSearch())
			{
				mloSearchDirect2D();
			}

		}
#ifdef MLOPEN_LOG_CONVOLUTION
		std::cout << "Selected run : "
			<< _grp_tile1 << ", "
			<< _grp_tile0 << ", "
			<< _in_tile1 << ", "
			<< _in_tile0 << ", "
			<< _out_pix_tile1 << ", "
			<< _out_pix_tile0 << ", "
			<< _n_out_pix_tiles << ", "
			<< _n_in_data_tiles << ", "
			<< _n_stacks
			<< std::endl;
#endif

		// construct found configuration

		ret = mloConstructDirect2DFwd();

	}

	return(ret);
}


/*
 * constructs found configuration
 */
int mlo_construct_direct2D::mloConstructDirect2DFwd()
{
	int ret = 0;

	bool unaligned = (_out_height < 8 || _out_width < 8 || (_out_height > 8 && _out_height < 16) || (_out_width > 8 && _out_width < 16)
		|| (_out_height > 16 && _out_height < 32) || (_out_width > 16 && _out_width < 32));

	// no 1x1 backward yet
	if (_kernel_size0 == 1 && _kernel_size1 == 1 )
	{

		return(mloConstructDirect2D1x1());
	}

	else if (unaligned)
	{
		return(mloConstructDirect2DFwdC());
	}

	std::size_t localMemSize = _stream->GetLocalMemorySize();

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes

	if (_direction == 0)
	{
		// backward
		_pad0 = _kernel_size0 - 1 - _pad0;
		_pad1 = _kernel_size1 - 1 - _pad1;
	}

	_n_in_data_tiles = std::min(_n_inputs, _n_in_data_tiles);
	_n_out_pix_tiles = std::min(_n_outputs, _n_out_pix_tiles);


	int alu_tile0 = (_in_tile0 + _out_pix_tile0 - 1) / _out_pix_tile0;
	int alu_tile1 = (_in_tile1 + _out_pix_tile1 - 1)/ _out_pix_tile1;
	int alu_tiles_sz = (alu_tile0*alu_tile1);
	if (alu_tiles_sz > 256)
	{
		//			std::cout << "ERROR: need out pix size ajustments\n";
		return(-1);
	}

	int n_alus_total = (_grp_tile0 * _grp_tile1);

	_n_stacks = std::min(_n_stacks, (n_alus_total + alu_tiles_sz - 1) / alu_tiles_sz);
	_n_stacks = std::min(_batch_sz, _n_stacks);


	int n_alus_perstack = (n_alus_total + _n_stacks - 1) / _n_stacks;

	int n_read_procs;
	if ((_grp_tile1 * _grp_tile0) <= static_cast<float>(_in_tile1 * _in_tile0))
	{
		n_read_procs = _grp_tile1 * _grp_tile0;
	}
	else
	{
		float proc_data_ratio = static_cast<float>(_in_tile1 * _in_tile0) / static_cast<float>(_grp_tile1 * _grp_tile0);
		n_read_procs = (proc_data_ratio <= 0.25) ? (_grp_tile1 * _grp_tile0) / 4 : (proc_data_ratio <= 0.5) ? (_grp_tile1 * _grp_tile0) / 2 : (_grp_tile1 * _grp_tile0);
	}

	int n_out_tile_blocks0 = (_out_width + _in_tile0 - 1) / (_in_tile0);
	int n_out_tile_blocks1 = (_out_height + _in_tile1 - 1) / (_in_tile1);

	int n_alu_tiles_perstack = (n_alus_perstack + alu_tiles_sz - 1)/ alu_tiles_sz;
	int n_out_tiles_perstack = n_alu_tiles_perstack * _n_out_pix_tiles;

	n_out_tiles_perstack = std::min(n_out_tiles_perstack, _n_outputs);


	_comp_options =
		std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(static_cast<long long>(_hw_wave_sz))
		+ std::string(" -DMLO_DIR_FORWARD=") + std::to_string(static_cast<long long>(_direction))
		+ std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(static_cast<long long>(_kernel_size0))
		+ std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(static_cast<long long>(_kernel_size1))
		+ std::string(" -DMLO_FILTER_PAD0=") + std::to_string(static_cast<long long>(_pad0))
		+ std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(static_cast<long long>(_kernel_stride0))
		+ std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(static_cast<long long>(_kernel_stride1))
		+ std::string(" -DMLO_N_OUTPUTS=") + std::to_string(static_cast<long long>(_n_outputs))
		+ std::string(" -DMLO_N_INPUTS=") + std::to_string(static_cast<long long>(_n_inputs))
		+ std::string(" -DMLO_BATCH_SZ=") + std::to_string(static_cast<long long>(_batch_sz))
		+ std::string(" -DMLO_OUT_WIDTH=") + std::to_string(static_cast<long long>(_out_width))
		+ std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(static_cast<long long>(_out_height))
		+ std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -DMLO_OUT_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -DMLO_IN_WIDTH=") + std::to_string(static_cast<long long>(_in_width))
		+ std::string(" -DMLO_IN_HEIGHT=") + std::to_string(static_cast<long long>(_in_height))
		+ std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -DMLO_IN_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		// algorithm parameters
		+std::string(" -DMLO_IN_TILE0=") + std::to_string(static_cast<long long>(_in_tile0))  // size of input data per ALU plane
		+ std::string(" -DMLO_IN_TILE1=") + std::to_string(static_cast<long long>(_in_tile1))  // size of input data per ALU plane
		+ std::string(" -DMLO_GRP_TILE0=") + std::to_string(static_cast<long long>(_grp_tile0)) // # of ALUs (group size)
		+ std::string(" -DMLO_GRP_TILE1=") + std::to_string(static_cast<long long>(_grp_tile1)) //
		+ std::string(" -DMLO_OUT_TILE0=") + std::to_string(static_cast<long long>(_out_pix_tile0))  // size of ouptput tile per wk-item (ALU))
		+ std::string(" -DMLO_OUT_TILE1=") + std::to_string(static_cast<long long>(_out_pix_tile1))  //
		+ std::string(" -DMLO_N_STACKS=") + std::to_string(static_cast<long long>(_n_stacks)) // # of diff stacks (part of batch).
		+ std::string(" -DMLO_N_OUT_TILES=") + std::to_string(static_cast<long long>(_n_out_pix_tiles))  // # output pixel tiles per wk-item (ALU)
		+ std::string(" -DMLO_N_OUT_TILES_PERSTACK=") + std::to_string(static_cast<long long>(n_out_tiles_perstack))
		+ std::string(" -DMLO_N_IN_TILES_PERSTACK=") + std::to_string(static_cast<long long>(_n_in_data_tiles)) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_N_READ_PROCS=") + std::to_string(static_cast<long long>(n_read_procs))
		+ std::string(" -DMLO_CONV_BIAS=") + std::to_string(static_cast<long long>(_bias))
		+ std::string(" -DMLO_ALU_VTILE0=") + std::to_string(static_cast<long long>(alu_tile0))
		+ std::string(" -DMLO_ALU_VTILE1=") + std::to_string(static_cast<long long>(alu_tile1))
		+ getGeneralCompOptions()
		;

	_l_wk.clear();
	_l_wk.push_back(_grp_tile1 * _grp_tile0);
	_l_wk.push_back(1);
	_l_wk.push_back(1);

	size_t gbl_wk0 = n_out_tile_blocks0 * n_out_tile_blocks1;

	size_t gbl_wk1 = (_n_outputs + n_out_tiles_perstack - 1) / n_out_tiles_perstack;
	size_t gbl_wk2 = (_batch_sz + _n_stacks - 1) / _n_stacks;

	_g_wk.clear();
	_g_wk.push_back(gbl_wk0 * _l_wk[0]);
	_g_wk.push_back(gbl_wk1);
	_g_wk.push_back(gbl_wk2);

	_kernel_file = "MLOpenConvDirUni.cl";
	_kernel_name = "MLOpenConvUni";



	return(ret);
}

#if MLOPEN_BACKEND_OPENCL
bool mlo_construct_direct2D::mloCheckWinograd3x3FwdConvCondition() const
{
	const auto dev = mlopen::GetDevice(_stream->GetStream());
	const auto platform = mlopen::GetDeviceInfo<CL_DEVICE_PLATFORM>(dev);
	const auto vendor_id = mlopen::GetDeviceInfo<CL_DEVICE_VENDOR_ID>(dev);
	const auto name = mlopen::GetDeviceInfo<CL_DEVICE_NAME>(dev);
	const auto driver = mlopen::GetDeviceInfo<CL_DRIVER_VERSION>(dev);
	const auto platform_vendor = mlopen::GetPlatformInfo<CL_PLATFORM_VENDOR>(platform);
	const auto grid_workgroup_count_x = _stream->GetMaxComputeUnits();

	const auto device_is_opencl_on_rocm =
		   (driver.find("(LC)") != std::string::npos) // Indicates ROCm - our binaries are in OpenCL-on-ROCm Code Object format
		|| (driver.find("(LC,") != std::string::npos)
		|| (driver.find(",LC)") != std::string::npos)
		|| (driver.find("(LC ") != std::string::npos)
		|| (driver.find(" LC)") != std::string::npos)
		|| (driver.find(" LC,") != std::string::npos)
		|| (driver.find(",LC ") != std::string::npos)
		|| (driver.find(" LC ") != std::string::npos)
		|| (driver.find(",LC,") != std::string::npos);

	const auto driver_is_v1_or_v2 =
		   (driver[0] == '1' || driver[0] == '2')
		&& driver[1] == '.'; // Both shall support Metadata for Runtime v1.0 we are using for now

	const auto device_is_opencl_on_rocm_supports_metadata_1_0 =
		   device_is_opencl_on_rocm
		&& driver_is_v1_or_v2;

	const auto device_is_gfx8_no_xnack =
		   name == "gfx800"
		|| name == "gfx802"
		|| name == "gfx803"
		|| name == "gfx804";

	const auto platform_is_amd = platform_vendor == "Advanced Micro Devices, Inc.";
	const auto device_is_amd = vendor_id == 0x1002;

	const auto device_is_gfx8_no_xnack_with_amd_opencl_on_rocm_supports_metadata_1_0 =
		   device_is_opencl_on_rocm_supports_metadata_1_0
		&& device_is_gfx8_no_xnack
		&& device_is_amd
		&& platform_is_amd;

	assert(_weights_layout.length() == 0); // FIXME: Uncomment validation below when _weights_layout content will be updated anywahere.

	const auto kernel_is_valid_for_problem_description =
		   _in_layout == "NCHW"
		// && _weights_layout						== "NKCHW" // FIXME see above
		&& _kernel_size0 == 3
		&& _kernel_size1 == 3
		&& _kernel_stride0 == 1
		&& _kernel_stride1 == 1
		&& _pad0 == 1
		&& _pad1 == 1
		&& _batch_sz							<	std::pow(2, 16)
		&& _n_inputs							<	std::pow(2, 16)
		&& _n_outputs							<	std::pow(2, 16)
		&& _in_height							<	std::pow(2, 16)
		&& _in_width							<	std::pow(2, 16)
		&& grid_workgroup_count_x				<	std::pow(2, 16)
		&& _n_inputs * _in_height * _in_width	<=	std::pow(2, 28)
		&& _n_outputs * _in_height * _in_width	<=	std::pow(2, 28)
		&& _n_inputs % 2 == 0
		&& _n_inputs >= 16;

	return device_is_gfx8_no_xnack_with_amd_opencl_on_rocm_supports_metadata_1_0
		&& kernel_is_valid_for_problem_description;
}

bool mlo_construct_direct2D::mloCheckWinograd3x3FwdConvPerfFilter() const
{
	return
		   _in_width	!= 7
		|| _in_height	!= 7;
}

int mlo_construct_direct2D::mloConstructWinograd3x3FwdConv()
{
	int ret = 0;

	const auto n_groups = _stream->GetMaxComputeUnits();

	_g_wk.clear();
	_g_wk.push_back(512 * n_groups);
	_g_wk.push_back(1);
	_g_wk.push_back(1);

	_l_wk.clear();
	_l_wk.push_back(512);
	_l_wk.push_back(1);
	_l_wk.push_back(1);

	_kernel_file = "conv_3x3_wheel_alpha_v2_0b_gfx803.so";
	_kernel_name = "sp3AsmConv3x3F";

	return (ret);
}
#endif

int mlo_construct_direct2D::mloConstructDirect2DFwdC()
{
	int ret = 0;

	size_t localMemSize = _stream->GetLocalMemorySize();

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes

	if (_direction == 0)
	{
		// backward
		_pad0 = _kernel_size0 - 1 - _pad0;
		_pad1 = _kernel_size1 - 1 - _pad1;
	}



	int in_tile0 = std::min(_out_width, _in_tile0);
	int in_tile1 = std::min(_out_height, _in_tile1);


	int alu_tile0 = (in_tile0 + _out_pix_tile0 - 1) / _out_pix_tile0;
	int alu_tile1 = (in_tile1 + _out_pix_tile1 - 1) / _out_pix_tile1;

	int alu_tiles_sz = (alu_tile0*alu_tile1);
	if (alu_tiles_sz > _grp_tile0 *_grp_tile1)
	{
		//			std::cout << "ERROR: need out pix size ajustments\n";
		return(-1);
	}

	int n_real_alus = std::max(1, (_grp_tile0 * _grp_tile1) / alu_tiles_sz) * alu_tiles_sz;

	_n_in_data_tiles = std::min(_n_inputs, _n_in_data_tiles);
	_n_out_pix_tiles = std::min(_n_outputs, _n_out_pix_tiles);


	int n_read_procs;
	if ((_grp_tile1 * _grp_tile0) <= static_cast<float>(in_tile1 * in_tile0))
	{
		n_read_procs = _grp_tile1 * _grp_tile0;
	}
	else
	{
		float proc_data_ratio = static_cast<float>(in_tile1 * in_tile0) / static_cast<float>(_grp_tile1 * _grp_tile0);
		n_read_procs = (proc_data_ratio <= 0.25) ? (_grp_tile1 * _grp_tile0) / 4 : (proc_data_ratio <= 0.5) ? (_grp_tile1 * _grp_tile0) / 2 : (_grp_tile1 * _grp_tile0);
	}

	int n_out_tile_blocks0 = (_out_width + in_tile0 - 1) / (in_tile0);
	int n_out_tile_blocks1 = (_out_height + in_tile1 - 1) / (in_tile1);


	int n_alu_tiles = (n_real_alus / alu_tiles_sz);


	_n_stacks = std::min(_batch_sz, _n_stacks);
	int n_alu_tiles_perstack = std::max(1, n_alu_tiles / _n_stacks);
	_n_stacks = std::min(std::max(1, n_alu_tiles / n_alu_tiles_perstack), _n_stacks);
	n_real_alus = n_alu_tiles_perstack * _n_stacks * alu_tiles_sz;
	int n_out_tiles_perstack = n_alu_tiles_perstack * _n_out_pix_tiles;

	n_out_tiles_perstack = std::min(n_out_tiles_perstack, _n_outputs);

	_in_tile0 = in_tile0;
	_in_tile1 = in_tile1;

	_comp_options =
		std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(static_cast<long long>(_hw_wave_sz))
		+ std::string(" -DMLO_DIR_FORWARD=") + std::to_string(static_cast<long long>(_direction))
		+ std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(static_cast<long long>(_kernel_size0))
		+ std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(static_cast<long long>(_kernel_size1))
		+ std::string(" -DMLO_FILTER_PAD0=") + std::to_string(static_cast<long long>(_pad0))
		+ std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -DMLO_N_OUTPUTS=") + std::to_string(static_cast<long long>(_n_outputs))
		+ std::string(" -DMLO_N_INPUTS=") + std::to_string(static_cast<long long>(_n_inputs))
		+ std::string(" -DMLO_BATCH_SZ=") + std::to_string(static_cast<long long>(_batch_sz))
		+ std::string(" -DMLO_OUT_WIDTH=") + std::to_string(static_cast<long long>(_out_width))
		+ std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(static_cast<long long>(_out_height))
		+ std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -DMLO_OUT_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -DMLO_IN_WIDTH=") + std::to_string(static_cast<long long>(_in_width))
		+ std::string(" -DMLO_IN_HEIGHT=") + std::to_string(static_cast<long long>(_in_height))
		+ std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -DMLO_IN_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		// algorithm parameters
		+ std::string(" -DMLO_IN_TILE0=") + std::to_string(static_cast<long long>(_in_tile0))  // size of input data per ALU plane
		+ std::string(" -DMLO_IN_TILE1=") + std::to_string(static_cast<long long>(_in_tile1))  // size of input data per ALU plane
		+ std::string(" -DMLO_OUT_TILE0=") + std::to_string(static_cast<long long>(_in_tile0))  // size of input data per ALU plane
		+ std::string(" -DMLO_OUT_TILE1=") + std::to_string(static_cast<long long>(_in_tile1))  // size of input data per ALU plane
		+ std::string(" -DMLO_GRP_TILE0=") + std::to_string(static_cast<long long>(_grp_tile0)) // # of ALUs (group size)
		+ std::string(" -DMLO_GRP_TILE1=") + std::to_string(static_cast<long long>(_grp_tile1)) //
		+ std::string(" -DMLO_ACTIVE_ALUS=") + std::to_string(static_cast<long long>(n_real_alus)) // total number of active alus
		+ std::string(" -DMLO_N_ALUTILES_PERSTACK=") + std::to_string(static_cast<long long>(n_alu_tiles_perstack)) // alu tiles per stack
		+ std::string(" -DMLO_OUT_PIX_TILE0=") + std::to_string(static_cast<long long>(_out_pix_tile0))  // size of ouptput tile per wk-item (ALU))
		+ std::string(" -DMLO_OUT_PIX_TILE1=") + std::to_string(static_cast<long long>(_out_pix_tile1))  //
		+ std::string(" -DMLO_N_STACKS=") + std::to_string(static_cast<long long>(_n_stacks)) // # of diff stacks (part of batch).
		+ std::string(" -DMLO_N_OUT_TILES=") + std::to_string(static_cast<long long>(_n_out_pix_tiles))  // # output pixel tiles per wk-item (ALU)
		+ std::string(" -DMLO_N_OUT_TILES_PERSTACK=") + std::to_string(static_cast<long long>(n_out_tiles_perstack))
		+ std::string(" -DMLO_N_IN_TILES_PERSTACK=") + std::to_string(static_cast<long long>(_n_in_data_tiles)) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_N_READ_PROCS=") + std::to_string(static_cast<long long>(n_read_procs))
		+ std::string(" -DMLO_CONV_BIAS=") + std::to_string(static_cast<long long>(_bias))
		+ std::string(" -DMLO_ALU_VTILE0=") + std::to_string(static_cast<long long>(alu_tile0))
		+ std::string(" -DMLO_ALU_VTILE1=") + std::to_string(static_cast<long long>(alu_tile1))
		+ getGeneralCompOptions()
		;

	_l_wk.clear();
	_l_wk.push_back(_grp_tile1 * _grp_tile0);
	_l_wk.push_back(1);
	_l_wk.push_back(1);

	size_t gbl_wk0 = n_out_tile_blocks0 * n_out_tile_blocks1 * _l_wk[0];

	//	gbl_wk0 = ((gbl_wk0 + n_real_alus - 1) / n_real_alus) * n_real_alus;


	size_t gbl_wk1 = (_n_outputs + n_out_tiles_perstack - 1) / n_out_tiles_perstack;
	size_t gbl_wk2 = (_batch_sz + _n_stacks - 1) / _n_stacks;

	_g_wk.clear();
	_g_wk.push_back(gbl_wk0);
	_g_wk.push_back(gbl_wk1);
	_g_wk.push_back(gbl_wk2);

	_kernel_file = "MLOpenConvDirUniC.cl";
	_kernel_name = "MLOpenConvUniC";

	return(ret);
}

int mlo_construct_direct2D::mloConstructDirect2D1x1()
{
	int ret = 0;

	// to restore to the previous version just comment this line
	// currently runs previous version
	//	return(mloConstructDirect2DFwd2());

	size_t localMemSize = _stream->GetLocalMemorySize();

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes

	_in_tile0 = 4;
	_in_tile1 = 1;
	_out_pix_tile0 = 4;
	_out_pix_tile1 = 1;

	int wei_cstride = _kernel_size0*_kernel_size1;
	// backward: inputs are forward outputs
	int wei_bstride = ((getDirection() == 1) ? _n_inputs : _n_outputs)*wei_cstride;
    int read_unit = 4;
    std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(static_cast<long long>(read_unit));
    

	// currently always 1
	int N4S = 1;

    int MAP_SZ4 = (_in_width * _in_height + N4S * read_unit - 1) / (N4S * read_unit);

    int DIVBY4 = (MAP_SZ4 * read_unit == _in_width * _in_height) ? 1 : 0;

    int C1x1_PIXLEFT = (DIVBY4 == 1) ? 0 : _in_width * _in_height - (MAP_SZ4 - 1) * read_unit;
    
	bool small_map = false;
	int GRP_SZ = _grp_tile0;
	int N_MAPS_PERGROUP = 1;
	// exchange step is a number of partial sums that can be exchanged in the kernel in one pass
	// it's used for small maps at the end of the kerenl to reduce partial sums
	// the number is kept in and passed through _n_in_data_tiles (with abused semantics).
	int exchange_step = 6;
	if (MAP_SZ4 <= GRP_SZ / 2)
	{
		N_MAPS_PERGROUP = GRP_SZ / MAP_SZ4;
		exchange_step = _n_in_data_tiles;
		_n_in_data_tiles = 1;
		small_map = true;
	}

	// number of inputs inside wk-items
	_n_in_data_tiles = std::min(_n_inputs, _n_in_data_tiles);
	// scale input by n of map per wk_item
	int n_input_scaled = (_n_inputs + _n_in_data_tiles - 1) / _n_in_data_tiles;

	// number of outputs inside wk_item
	_n_out_pix_tiles = std::min(_n_outputs, _n_out_pix_tiles);


	if (small_map)
	{
		exchange_step = std::min(std::min(exchange_step, _n_out_pix_tiles), N_MAPS_PERGROUP);
		_n_out_pix_tiles = (_n_out_pix_tiles / exchange_step) * exchange_step;
	}
	// n of input map per group
	N_MAPS_PERGROUP = std::min(N_MAPS_PERGROUP, n_input_scaled);
	// number of input loops
	int n_in_loop = (n_input_scaled + N_MAPS_PERGROUP - 1) / N_MAPS_PERGROUP;

	// number of batches inside wk_item
	_n_stacks = std::min(_batch_sz, _n_stacks);

	int n_out_tiles_pergroup = _n_out_pix_tiles * _n_stacks;

	int batch_aligned = 0;
	int output_aligned = 0;
	if ((_batch_sz / _n_stacks) *_n_stacks == _batch_sz)
	{
		batch_aligned = 1;
	}
	if ((_n_outputs / _n_out_pix_tiles) * _n_out_pix_tiles == _n_outputs)
	{
		output_aligned = 1;
	}

	_comp_options =
		std::string(" -DMLO_DIR_FORWARD=") + std::to_string(static_cast<long long>(_direction))
		+ std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -DMLO_N_OUTPUTS=") + std::to_string(static_cast<long long>(_n_outputs))
		+ std::string(" -DMLO_N_INPUTS=") + std::to_string(static_cast<long long>(_n_inputs))
		+ std::string(" -DMLO_BATCH_SZ=") + std::to_string(static_cast<long long>(_batch_sz))
		+ std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -DMLO_OUT_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -DMLO_IN_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		+ std::string(" -DMLO_WEI_BSTRIDE=") + std::to_string(static_cast<long long>(wei_bstride))
		+ std::string(" -DMLO_WEI_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(wei_cstride))
		// algorithm parameters
		+ std::string(" -DMLO_GRP_SZ0=") + std::to_string(static_cast<long long>(GRP_SZ))
		+ std::string(" -DMLO_MAP_SZ4=") + std::to_string(static_cast<long long>(MAP_SZ4))
		+ std::string(" -DMLO_C1x1_PIXLEFT=") + std::to_string(static_cast<long long>(C1x1_PIXLEFT))
		+ std::string(" -DMLO_DIVBY4=") + std::to_string(static_cast<long long>(DIVBY4))
		+ std::string(" -DMLO_IN_LOOP=") + std::to_string(static_cast<long long>(n_in_loop))
		+ std::string(" -DMLO_N_LCL_BATCHS=") + std::to_string(static_cast<long long>(_n_stacks)) // # of diff stacks (part of batch).
		+ std::string(" -DMLO_N_LCL_OUT_MAPS=") + std::to_string(static_cast<long long>(_n_out_pix_tiles))  // # output pixel tiles per wk-item (ALU)
		+ std::string(" -DMLO_N_OUT_TILES_PERGROUP=") + std::to_string(static_cast<long long>(n_out_tiles_pergroup))
		+ std::string(" -DMLO_N_LCL_IN_MAPS=") + std::to_string(static_cast<long long>(_n_in_data_tiles)) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_N_MAPS_PERGROUP=") + std::to_string(static_cast<long long>(N_MAPS_PERGROUP)) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_CONV_BIAS=") + std::to_string(static_cast<long long>(_bias))
		+ std::string(" -DMLO_BATCH_ALIGNED=") + std::to_string(static_cast<long long>(batch_aligned))
		+ std::string(" -DMLO_OUTPUTS_ALIGNED=") + std::to_string(static_cast<long long>(output_aligned))
		+ std::string(" -DMLO_EXCHANGE_STEP=") + std::to_string(static_cast<long long>(exchange_step))
        + std::string(" -DMLO_READ_TYPE=") + READ_TYPE
        + std::string(" -DMLO_READ_UNIT=") + std::to_string(static_cast<long long>(read_unit))
		+ getGeneralCompOptions()
		;

	_l_wk.clear();
	_l_wk.push_back(_grp_tile0);
	_l_wk.push_back(_grp_tile1);
	_l_wk.push_back(1);

	size_t gbl_wk0 = (GRP_SZ < MAP_SZ4) ? ((MAP_SZ4 + GRP_SZ - 1) / GRP_SZ) *GRP_SZ : GRP_SZ;


	size_t gbl_wk1 = (_n_outputs + _n_out_pix_tiles - 1) / _n_out_pix_tiles;
	size_t gbl_wk2 = (_batch_sz + _n_stacks - 1) / _n_stacks;

	_g_wk.clear();
	_g_wk.push_back(gbl_wk0);
	_g_wk.push_back(gbl_wk1);
	_g_wk.push_back(gbl_wk2);

	//	_kernel_file = "MLOpenConv1x1.cl";
	//	_kernel_name = "MLOpenConv1x1";
	// too much overhead for small maps and few inputs

	if ((getDirection() == 0) || (small_map && (_in_width <= 8 || _in_height <= 8)) || (small_map && _n_inputs <= 256))
	{
		_kernel_file = "MLOpenConv1x1PS.cl";
		_kernel_name = "MLOpenConv1x1PS";
	}
	else
	{
		_kernel_file = "MLOpenConv1x1PS_LW.cl";
		_kernel_name = "MLOpenConv1x1PS_LW";
	}
	// see above comment
	if (small_map)
	{
		_n_in_data_tiles = exchange_step;
	}

	return(ret);
}


int mlo_construct_direct2D::mloConstructDirect2D3x3()
{
	int ret = 0;

	size_t localMemSize = _stream->GetLocalMemorySize();

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes
	int n_waves = 4;

	int wei_cstride = _kernel_size0*_kernel_size1;
	int wei_bstride = _n_inputs*wei_cstride;

	_out_pix_tile0 = 4;
	_out_pix_tile1 = 2;
	_n_stacks = 1;
	_n_out_pix_tiles = 4;
	int read_unit = _out_pix_tile0;
//	std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(static_cast<long long>(read_unit));
	// MD: read_unit is never == 1
	std::string READ_TYPE = "_FLOAT" + std::to_string(static_cast<long long>(read_unit));

	int GRP_SZ = _hw_wave_sz * n_waves;

	int ALU_EXTENT_X = (_out_width + read_unit - 1) / read_unit;
	int LG2ALU_EXTENT_X = static_cast<int>(std::ceil(std::log(ALU_EXTENT_X) / std::log(2)));
	int ALU_EXTENT_Y = (GRP_SZ >> LG2ALU_EXTENT_X);
	int LG2ALU_EXTENT_Y = static_cast<int>(std::ceil(std::log(ALU_EXTENT_Y) / std::log(2)));

	// the wave is logical is a unit of shareing weights in SGPRs
	// it cannot be less than HW_WAVE_SIZE = 64
	// it cannot be larger than the group size.

	int LG2_WAVE_SZ0 = std::ceil(std::log(ALU_EXTENT_X) / std::log(2));
	int logical_wave_sz = std::max(1, ALU_EXTENT_X / _hw_wave_sz) * _hw_wave_sz;
	if (logical_wave_sz > GRP_SZ)
	{
		printf("Conv3x3 conf error\n");
		return(-1);
	}
	int logical_n_waves = std::max(1, GRP_SZ / logical_wave_sz);
	int LG2_WAVE_SZ = std::ceil(std::log(logical_wave_sz) / std::log(2));
	int WAVE_SZ1 = (logical_wave_sz >> LG2_WAVE_SZ0);
	int lg2_n_waves = std::ceil(std::log(logical_n_waves) / std::log(2));
	int N_WAVES_MASK = (1 << lg2_n_waves) - 1;

	int OUT_EXTENT1 = _out_pix_tile1 * WAVE_SZ1;
	int OUT_EXTENT0 = (_out_pix_tile0 << LG2_WAVE_SZ0);

	int total_out_maps = _n_out_pix_tiles * logical_n_waves;

	// number of batches inside wk_item
	_n_stacks = std::min(_batch_sz, _n_stacks);

	int N_HEIGHT_EXTENTS = (_out_height + OUT_EXTENT1 - 1) / OUT_EXTENT1;
	int N_WIDTH_EXTENTS = (_out_width + OUT_EXTENT0 - 1) / OUT_EXTENT0;
	int N_GROUPS_PER_MAP = N_HEIGHT_EXTENTS*N_WIDTH_EXTENTS;


	_grp_tile0 = GRP_SZ;
	_grp_tile1 = 1;
	int grp_tile2 = 1;
	_in_tile0 = OUT_EXTENT0;
	_in_tile1 = OUT_EXTENT1;
	_n_in_data_tiles = 1;

//	_gen_comp_options += std::string(" -limit-vector-registers=64 ");

	_comp_options =
		std::string(" -DMLO_DIR_FORWARD=") + std::to_string(static_cast<long long>(_direction))
		+ std::string(" -DMLO_GRP_SZ=") + std::to_string(static_cast<long long>(GRP_SZ))
		+ std::string(" -DMLO_GRP_SZ0=") + std::to_string(static_cast<long long>(_grp_tile0))
		+ std::string(" -DMLO_GRP_SZ1=") + std::to_string(static_cast<long long>(_grp_tile1))
		+ std::string(" -DMLO_GRP_SZ2=") + std::to_string(static_cast<long long>(grp_tile2))
		+ std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(static_cast<long long>(_kernel_size0))
		+ std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(static_cast<long long>(_kernel_size1))
		+ std::string(" -DMLO_FILTER_PAD0=") + std::to_string(static_cast<long long>(_pad0))
		+ std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -DMLO_N_OUTPUTS=") + std::to_string(static_cast<long long>(_n_outputs))
		+ std::string(" -DMLO_N_INPUTS=") + std::to_string(static_cast<long long>(_n_inputs))
		+ std::string(" -DMLO_BATCH_SZ=") + std::to_string(static_cast<long long>(_batch_sz))
		+ std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -DMLO_OUT_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -DMLO_IN_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		+ std::string(" -DMLO_WEI_BATCH_STRIDE=") + std::to_string(static_cast<long long>(wei_bstride))
		+ std::string(" -DMLO_WEI_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(wei_cstride))
		+ std::string(" -DMLO_IN_WIDTH=") + std::to_string(static_cast<long long>(_in_width))
		+ std::string(" -DMLO_IN_HEIGHT=") + std::to_string(static_cast<long long>(_in_height))
		+ std::string(" -DMLO_N_LCL_BATCHS=") + std::to_string(static_cast<long long>(_n_stacks)) // # of diff stacks (part of batch).
		+ std::string(" -DMLO_N_LCL_OUT_MAPS=") + std::to_string(static_cast<long long>(_n_out_pix_tiles))  // # output pixel tiles per wk-item (ALU)
		+ std::string(" -DMLO_N_LCL_IN_MAPS=") + std::to_string(static_cast<long long>(_n_in_data_tiles)) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_OUT_TILE0=") + std::to_string(static_cast<long long>(_out_pix_tile0))  // size of ouptput tile per wk-item (ALU))
		+ std::string(" -DMLO_OUT_TILE1=") + std::to_string(static_cast<long long>(_out_pix_tile1))  //
		+ std::string(" -DMLO_ALU_EXTENT_X=") + std::to_string(static_cast<long long>(ALU_EXTENT_X))
		+ std::string(" -DMLO_LG2ALU_EXTENT_X=") + std::to_string(static_cast<long long>(LG2ALU_EXTENT_X))
		+ std::string(" -DMLO_ALU_EXTENT_Y=") + std::to_string(static_cast<long long>(ALU_EXTENT_Y))
		+ std::string(" -DMLO_LG2ALU_EXTENT_Y=") + std::to_string(static_cast<long long>(LG2ALU_EXTENT_Y))
		+ std::string(" -DMLO_OUT_EXTENT1=") + std::to_string(static_cast<long long>(OUT_EXTENT1))
		+ std::string(" -DMLO_OUT_EXTENT0=") + std::to_string(static_cast<long long>(OUT_EXTENT0))
		+ std::string(" -DMLO_N_WAVES=") + std::to_string(static_cast<long long>(logical_n_waves))
		+ std::string(" -DMLO_N_WAVES_MASK=") + std::to_string(static_cast<long long>(N_WAVES_MASK))
		+ std::string(" -DMLO_LG2_WAVE_SZ=") + std::to_string(static_cast<long long>(LG2_WAVE_SZ))
		+ std::string(" -DMLO_LG2_WAVE_SZ0=") + std::to_string(static_cast<long long>(LG2_WAVE_SZ0))
		+ std::string(" -DMLO_READ_TYPE=") + READ_TYPE
		+ std::string(" -DMLO_READ_UNIT=") + std::to_string(static_cast<long long>(read_unit))
		+ std::string(" -DMLO_CONV_BIAS=") + std::to_string(static_cast<long long>(_bias))
		+getGeneralCompOptions()
		;

	_l_wk.clear();
	_l_wk.push_back(_grp_tile0);
	_l_wk.push_back(_grp_tile1);
	_l_wk.push_back(grp_tile2);

	size_t gbl_wk0 = N_GROUPS_PER_MAP;

	size_t gbl_wk1 = (_n_outputs + total_out_maps - 1) / total_out_maps;
	size_t gbl_wk2 = (_batch_sz + _n_stacks - 1) / _n_stacks;

	_g_wk.clear();
	_g_wk.push_back(gbl_wk0 * _grp_tile0);
	_g_wk.push_back(gbl_wk1);
	_g_wk.push_back(gbl_wk2);

	_kernel_file = "MLOpenConvD3x3.cl";
	_kernel_name = "MLOpenCvD3x3_WSR0";
	return(ret);
}


int mlo_construct_direct2D::mloConstructDirect2D_11x11()
{
	int ret = 0;

	// to restore to the previous version just comment this line
	// currently runs previous version
	//	return(mloConstructDirect2DFwd2());

	size_t localMemSize = _stream->GetLocalMemorySize();

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes

	_in_tile0 = 4;
	_in_tile1 = 1;
	_out_pix_tile0 = 4;
	_out_pix_tile1 = 1;

	int wei_cstride = _kernel_size0*_kernel_size1;
	// backward: inputs are forward outputs
	int wei_bstride = ((getDirection() == 1) ? _n_inputs : _n_outputs)*wei_cstride;
	int read_unit = 4;
	std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(static_cast<long long>(read_unit));


	// currently always 1
	int N4S = 1;

	int MAP_SZ4 = (_in_width * _in_height + N4S * read_unit - 1) / (N4S * read_unit);

	int DIVBY4 = (MAP_SZ4 * read_unit == _in_width * _in_height) ? 1 : 0;

	int C1x1_PIXLEFT = (DIVBY4 == 1) ? 0 : _in_width * _in_height - (MAP_SZ4 - 1) * read_unit;

	bool small_map = false;
	int GRP_SZ = _grp_tile0;
	int N_MAPS_PERGROUP = 1;
	// exchange step is a number of partial sums that can be exchanged in the kernel in one pass
	// it's used for small maps at the end of the kerenl to reduce partial sums
	// the number is kept in and passed through _n_in_data_tiles (with abused semantics).
	int exchange_step = 6;
	if (MAP_SZ4 <= GRP_SZ / 2)
	{
		N_MAPS_PERGROUP = GRP_SZ / MAP_SZ4;
		exchange_step = _n_in_data_tiles;
		_n_in_data_tiles = 1;
		small_map = true;
	}

	// number of inputs inside wk-items
	_n_in_data_tiles = std::min(_n_inputs, _n_in_data_tiles);
	// scale input by n of map per wk_item
	int n_input_scaled = (_n_inputs + _n_in_data_tiles - 1) / _n_in_data_tiles;

	// number of outputs inside wk_item
	_n_out_pix_tiles = std::min(_n_outputs, _n_out_pix_tiles);


	if (small_map)
	{
		exchange_step = std::min(std::min(exchange_step, _n_out_pix_tiles), N_MAPS_PERGROUP);
		_n_out_pix_tiles = (_n_out_pix_tiles / exchange_step) * exchange_step;
	}
	// n of input map per group
	N_MAPS_PERGROUP = std::min(N_MAPS_PERGROUP, n_input_scaled);
	// number of input loops
	int n_in_loop = (n_input_scaled + N_MAPS_PERGROUP - 1) / N_MAPS_PERGROUP;

	// number of batches inside wk_item
	_n_stacks = std::min(_batch_sz, _n_stacks);

	int n_out_tiles_pergroup = _n_out_pix_tiles * _n_stacks;

	int batch_aligned = 0;
	int output_aligned = 0;
	if ((_batch_sz / _n_stacks) *_n_stacks == _batch_sz)
	{
		batch_aligned = 1;
	}
	if ((_n_outputs / _n_out_pix_tiles) * _n_out_pix_tiles == _n_outputs)
	{
		output_aligned = 1;
	}

	_comp_options =
		std::string(" -DMLO_DIR_FORWARD=") + std::to_string(static_cast<long long>(_direction))
		+ std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -DMLO_N_OUTPUTS=") + std::to_string(static_cast<long long>(_n_outputs))
		+ std::string(" -DMLO_N_INPUTS=") + std::to_string(static_cast<long long>(_n_inputs))
		+ std::string(" -DMLO_BATCH_SZ=") + std::to_string(static_cast<long long>(_batch_sz))
		+ std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -DMLO_OUT_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -DMLO_IN_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		+ std::string(" -DMLO_WEI_BSTRIDE=") + std::to_string(static_cast<long long>(wei_bstride))
		+ std::string(" -DMLO_WEI_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(wei_cstride))
		// algorithm parameters
		+ std::string(" -DMLO_GRP_SZ0=") + std::to_string(static_cast<long long>(GRP_SZ))
		+ std::string(" -DMLO_MAP_SZ4=") + std::to_string(static_cast<long long>(MAP_SZ4))
		+ std::string(" -DMLO_C1x1_PIXLEFT=") + std::to_string(static_cast<long long>(C1x1_PIXLEFT))
		+ std::string(" -DMLO_DIVBY4=") + std::to_string(static_cast<long long>(DIVBY4))
		+ std::string(" -DMLO_IN_LOOP=") + std::to_string(static_cast<long long>(n_in_loop))
		+ std::string(" -DMLO_N_LCL_BATCHS=") + std::to_string(static_cast<long long>(_n_stacks)) // # of diff stacks (part of batch).
		+ std::string(" -DMLO_N_LCL_OUT_MAPS=") + std::to_string(static_cast<long long>(_n_out_pix_tiles))  // # output pixel tiles per wk-item (ALU)
		+ std::string(" -DMLO_N_OUT_TILES_PERGROUP=") + std::to_string(static_cast<long long>(n_out_tiles_pergroup))
		+ std::string(" -DMLO_N_LCL_IN_MAPS=") + std::to_string(static_cast<long long>(_n_in_data_tiles)) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_N_MAPS_PERGROUP=") + std::to_string(static_cast<long long>(N_MAPS_PERGROUP)) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_CONV_BIAS=") + std::to_string(static_cast<long long>(_bias))
		+ std::string(" -DMLO_BATCH_ALIGNED=") + std::to_string(static_cast<long long>(batch_aligned))
		+ std::string(" -DMLO_OUTPUTS_ALIGNED=") + std::to_string(static_cast<long long>(output_aligned))
		+ std::string(" -DMLO_EXCHANGE_STEP=") + std::to_string(static_cast<long long>(exchange_step))
		+ std::string(" -DMLO_READ_TYPE=") + READ_TYPE
		+ std::string(" -DMLO_READ_UNIT=") + std::to_string(static_cast<long long>(read_unit))
		+ getGeneralCompOptions()
		;

	_l_wk.clear();
	_l_wk.push_back(_grp_tile0);
	_l_wk.push_back(_grp_tile1);
	_l_wk.push_back(1);

	size_t gbl_wk0 = (GRP_SZ < MAP_SZ4) ? ((MAP_SZ4 + GRP_SZ - 1) / GRP_SZ) *GRP_SZ : GRP_SZ;


	size_t gbl_wk1 = (_n_outputs + _n_out_pix_tiles - 1) / _n_out_pix_tiles;
	size_t gbl_wk2 = (_batch_sz + _n_stacks - 1) / _n_stacks;

	_g_wk.clear();
	_g_wk.push_back(gbl_wk0);
	_g_wk.push_back(gbl_wk1);
	_g_wk.push_back(gbl_wk2);

	//	_kernel_file = "MLOpenConv1x1.cl";
	//	_kernel_name = "MLOpenConv1x1";
	// too much overhead for small maps and few inputs

	if ((getDirection() == 0) || (small_map && (_in_width <= 8 || _in_height <= 8)) || (small_map && _n_inputs <= 256))
	{
		_kernel_file = "MLOpenConv1x1PS.cl";
		_kernel_name = "MLOpenConv1x1PS";
	}
	else
	{
		_kernel_file = "MLOpenConv1x1PS_LW.cl";
		_kernel_name = "MLOpenConv1x1PS_LW";
	}
	// see above comment
	if (small_map)
	{
		_n_in_data_tiles = exchange_step;
	}

	return(ret);
}


/*
* construct generic forward configuration
* it's mostly used for strides > 1
* right now
* it ustilizes several super-tiles from different batches.
* loads them in parallel
* loads weights (16 or 32)
* apply to a different batchs
* might use group size more than 256
* possible improvement - use different set of weights with super-tiles from different batches
*
*/
int mlo_construct_direct2D::mloConstructDirect2DFwdGen()
{
	_hw_wave_sz = 64;

	int n_in_stacks = 0;
	if ((_kernel_size0 == 11 && _kernel_size0 == 11) || (_kernel_size1 == 3 && _kernel_size0 == 3))
	{
		n_in_stacks = ((_batch_sz / 4) * 4 == _batch_sz) ? 4 : ((_batch_sz / 2) * 2 == _batch_sz) ? 2 : 1;  // n of input batches
	}
	else
	{
		n_in_stacks = ((_batch_sz / 2) * 2 == _batch_sz) ? 2 : 1;  // n of input batches
	}

	int n_proc_supertiles = n_in_stacks; // n of prosessing groups
	int lg2n_proc_supertiles = static_cast<int>(std::ceil(std::log(n_proc_supertiles) / std::log(2)));
	int n_out_stacks = 1; // n of output sets
	int n_proc_supertile0 = ((n_in_stacks > 1) ? 32 : 16) / _kernel_stride0; // n  processor in process supertile
	int n_proc_supertile1 = ((n_in_stacks > 1 && (_kernel_size1 >= 11 || _kernel_size0 >= 11)) ? 32 : 16) / n_in_stacks;
	int lg2n_proc_supertile1 = static_cast<int>(std::ceil(std::log(n_proc_supertile1) / std::log(2)));
	int ocl_group_sz0 = n_proc_supertile0;
	int ocl_group_sz1 = n_proc_supertile1 * n_proc_supertiles;
	int ocl_group_sz2 = 1;
	int gbl0 = 0;
	int gbl1 = 0;
	int gbl2 = 0;

	int n_ins0 = 1; // number of inputs each a from different stack along dim 0
	int n_ins1 = 1; // number of inputs each a from different stack along dim 1

	int n_outs = (_in_width >= 384 || (_kernel_size0 >= 11 && _kernel_stride0 >= 4)) ? 16 : 32;  // n outputs per a single input: major parameter
	int n_out_pix_horiz = (_in_width < 320 || (_kernel_size0 >= 11 && _kernel_stride0 >= 4)) ? 1 : 2; // n of output px horix per wk-item: major parameter
	int n_out_pix_vert = 1; // n of output px horix per wk-item: major parameter

	int n_in_pix_horiz = n_out_pix_horiz; // n of input pix per wk_item
	int n_in_pix_vert = n_out_pix_vert; // n of input pix per wk_item
	int n_v_proc0 = (_out_width + n_out_pix_horiz - 1) / n_out_pix_horiz;
	int n_v_proc1 = (_out_height + n_out_pix_vert - 1) / n_out_pix_vert;
	
	int big = 1;

	int n_procs0 = n_proc_supertile0 / n_ins0;
	int n_procs1 = n_proc_supertile1 / n_ins1;

	int in_sz0 = (n_procs0 * n_out_pix_horiz - 1) * _kernel_stride0 + 1/* + kernel_size0 - 2 * pad0*/;
	int in_sz1 = (n_procs1 * n_out_pix_vert - 1) * _kernel_stride1 + 1/* + kernel_size1 - 2 * pad1*/;

	int n_ins = n_ins0 * n_ins1; // number of inputs each a from different stack

	n_outs = std::min(n_outs, _n_outputs);
	n_ins = std::min(n_ins, _batch_sz);

	n_out_stacks = (n_outs * n_out_stacks < _n_outputs) ? n_out_stacks : 1;
	n_in_stacks = (n_ins * n_in_stacks < _batch_sz) ? n_in_stacks : 1;
	int total_ins = n_ins * n_in_stacks;
	int total_outs = n_outs * n_out_stacks;


	int n_out_blocks = ((_n_outputs + total_outs - 1) / total_outs);
	int n_stack_blocks = ((_batch_sz + total_ins - 1) / total_ins);


	int batch_aligned = 0;
#if 1
	if ((_batch_sz / n_stack_blocks) * n_stack_blocks == _batch_sz)
	{
		batch_aligned = 1;
	}
#endif
	int out_aligned = 0;
#if 1
	if ((_n_outputs / total_outs) * total_outs == _n_outputs)
	{
		out_aligned = 1;
	}
#endif


	// global work size
	gbl0 = n_ins0 * ((n_v_proc0 + n_procs0 - 1) / (n_procs0)) *n_procs0;
	gbl1 = n_ins1 * ((n_v_proc1 + n_procs1 - 1) / (n_procs1)) *n_procs1 * n_proc_supertiles;
	gbl2 = n_out_blocks * n_stack_blocks;


	int aligned_out = 1;

	if (gbl0 != n_ins0 * n_v_proc0 || gbl1 != n_ins1 * n_v_proc1)
	{
		aligned_out = 0;
	}

	int bias = _bias;

	_comp_options =
		std::string("-DMLO_GRP_SZ=") + std::to_string(static_cast<long long>(ocl_group_sz0 * ocl_group_sz1 * ocl_group_sz2))
		+ std::string(" -DMLO_GRP_SZ0=") + std::to_string(static_cast<long long>(ocl_group_sz0))
		+ std::string(" -DMLO_GRP_SZ1=") + std::to_string(static_cast<long long>(ocl_group_sz1))
		+ std::string(" -DMLO_GRP_SZ2=") + std::to_string(static_cast<long long>(ocl_group_sz2))
		+ std::string(" -DMLO_LCL_N_IN_CHNLS=") + std::to_string(static_cast<long long>(n_ins))
		+ std::string(" -DMLO_LCL_N_OUT_CHNLS=") + std::to_string(static_cast<long long>(n_outs))
		+ std::string(" -DMLO_OUT_STACKS=") + std::to_string(static_cast<long long>(n_out_stacks))
		+ std::string(" -DMLO_IN_STACKS=") + std::to_string(static_cast<long long>(n_in_stacks))
		+ std::string(" -DMLO_BATCH_SZ=") + std::to_string(static_cast<long long>(_batch_sz))
		+ std::string(" -DMLO_FLTR_SZ0=") + std::to_string(static_cast<long long>(_kernel_size0))
		+ std::string(" -DMLO_FLTR_PAD_SZ0=") + std::to_string(static_cast<long long>(_pad0))
		+ std::string(" -DMLO_FLTR_STRIDE0=") + std::to_string(static_cast<long long>(_kernel_stride0))
		+ std::string(" -DMLO_FLTR_SZ1=") + std::to_string(static_cast<long long>(_kernel_size1))
		+ std::string(" -DMLO_FLTR_PAD_SZ1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -DMLO_FLTR_STRIDE1=") + std::to_string(static_cast<long long>(_kernel_stride1))
		+ std::string(" -DMLO_N_OUT_CHNLS=") + std::to_string(static_cast<long long>(_n_outputs))			//total number of output channels
		+ std::string(" -DMLO_OUT_WIDTH=") + std::to_string(static_cast<long long>(_out_width))
		+ std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(static_cast<long long>(_out_height))
		+ std::string(" -DMLO_OUT_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -DMLO_OUT_CHNL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -DMLO_N_OUT_PIX_SZ0=") + std::to_string(static_cast<long long>(n_out_pix_horiz))
		+ std::string(" -DMLO_N_OUT_PIX_SZ1=") + std::to_string(static_cast<long long>(n_out_pix_vert))
		+ std::string(" -DMLO_N_IN_CHNLS=") + std::to_string(static_cast<long long>(_n_inputs))
		+ std::string(" -DMLO_IN_WIDTH=") + std::to_string(static_cast<long long>(_in_width))
		+ std::string(" -DMLO_IN_HEIGHT=") + std::to_string(static_cast<long long>(_in_height))
		+ std::string(" -DMLO_IN_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		+ std::string(" -DMLO_IN_CHNL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -DMLO_N_IN_PIX_SZ0=") + std::to_string(static_cast<long long>(n_in_pix_horiz))         // size of output processing group in 0 dim
		+ std::string(" -DMLO_N_IN_PIX_SZ1=") + std::to_string(static_cast<long long>(n_in_pix_vert))         // size of output processing group in 1 dim
		+ std::string(" -DMLO_WEI_SZ=") + std::to_string(static_cast<long long>(_n_outputs * _n_inputs * _kernel_size0 * _kernel_size1))
		+ std::string(" -DMLO_WEIGHTS_STRIDE=") + std::to_string(static_cast<long long>(_n_inputs * _kernel_size0 * _kernel_size1))		//	weights stride
		+ std::string(" -DMLO_N_STACKS=") + std::to_string(static_cast<long long>(n_stack_blocks))          // n of separate data stacks
		+ std::string(" -DMLO_N_PROCS0=") + std::to_string(static_cast<long long>(n_procs0))         // n of processors per stack
		+ std::string(" -DMLO_N_PROCS1=") + std::to_string(static_cast<long long>(n_procs1))         // n of processors per stack
		+ std::string(" -DMLO_ALIGNED=") + std::to_string(static_cast<long long>(aligned_out))		//	dimesions aligned
		+ std::string(" -DMLO_BATCH_ALIGNED=") + std::to_string(static_cast<long long>(batch_aligned))      // batch is multiple of n_ins
		+ std::string(" -DMLO_OUT_ALINED=") + std::to_string(static_cast<long long>(out_aligned))        // outputs is multiple of n_outs
		+ std::string(" -DMLO_IN_SZ0=") + std::to_string(static_cast<long long>(in_sz0))			// horizontal read dim 0
		+ std::string(" -DMLO_IN_SZ1=") + std::to_string(static_cast<long long>(in_sz1))			// vertical read dim 1
		+ std::string(" -DMLO_LG2N_PROC_TILES=") + std::to_string(static_cast<long long>(lg2n_proc_supertiles))
		+ std::string(" -DMLO_LG2N_PROC_TILE1=") + std::to_string(static_cast<long long>(lg2n_proc_supertile1))
		+ std::string(" -DMLO_BIG=") + std::to_string(static_cast<long long>(big))		//	resolution > 32 x 32
		+ std::string(" -DMLO_CONV_BIAS=") + std::to_string(static_cast<long long>(bias))

		//		+ std::string(" -limit-vector-registers=64 ")

		+getGeneralCompOptions()
		;


	_kernel_file = "MlOpenConvDirGenFwd.cl";
	_kernel_name = (n_proc_supertiles == 1) ? "MLOpenCDFGen" : "MLOpenCDFGen4";

	_l_wk.clear();

	_l_wk.push_back(ocl_group_sz0);
	_l_wk.push_back(ocl_group_sz1);
	_l_wk.push_back(ocl_group_sz2);
	
	_g_wk.push_back(gbl0);
	_g_wk.push_back(gbl1);
	_g_wk.push_back(gbl2);

	return(0);

}


/*
* backward with regard to weights
* inputs == output, outputs == input
*/
// TODO: search params

int mlo_construct_BwdWrW2D::mloConstruct1x1()
{

	int ret = 0;
	size_t localMemSize = 64 * 1024;

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes
									  // major parameters

									  // inpout are outputs
	int wei_cstride = _kernel_size0*_kernel_size1;
	int wei_bstride = _n_outputs*wei_cstride;


	// number  of batch iterations
	_n_stacks = 1;
	_n_stacks = std::min(_batch_sz, _n_stacks);
	// defines how to proceed : 1 grouop per batch or with a loop over all batches
	// loop over al batches make sense in 2 cases: a lot of small inputs/outputs or few batches
	// param
	int N_BATCH_LOOPS = (_n_inputs*_n_outputs <= 8 * 1024) ? 1 : _batch_sz / _n_stacks;
	int n_batch_blks = (_batch_sz + N_BATCH_LOOPS * _n_stacks - 1) / (N_BATCH_LOOPS * _n_stacks);

	_out_pix_tile0 = 1;
	_out_pix_tile1 = 1;
	_in_tile1 = 1;
	_in_tile0 = 1;

	// n of wvaefront in a group
	// param
	int n_waves = (_in_width <=  8) ? 1: 4;
	int GRP_SZ = _hw_wave_sz * n_waves;
	// number of input maps per group

	int map_sz = _in_width*_in_height;
	// define a special size for a specific width as a devisor to avoid dealing with out of range
	// param
	int read_unit = (_in_width == 7 || _in_width == 14) ? 7 : (_in_width == 28) ? 14 : (((map_sz / 8) * 8) == map_sz) ? 8 : (((map_sz / 4) * 4) == map_sz) ? 4 : (((map_sz / 2) * 2) == map_sz) ? 2 : 1;

	int MAP_WK_SZ = ((map_sz + read_unit - 1) / read_unit);

// to avoid exeeding the group size but trying to keep multiple of the same unit
	while (MAP_WK_SZ > GRP_SZ)
	{
		read_unit *= 2;
		MAP_WK_SZ = ((map_sz + read_unit - 1) / read_unit);
	}

// this one is valid only till _FLOAT8
// but it's not an error, the kernel does not use these types at all 
	std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((read_unit));

    int POW2_MAP_WK_SZ = (1 << mloLg2(MAP_WK_SZ));
	// number of output maps fetched into LDS and to be shred with the input mapped kept in registers
	// param
	int n_out_stacks = (_in_width == 28)? 10 : ((_in_width == 7) || (_in_width == 14)) ? 8 : (GRP_SZ / MAP_WK_SZ);
	int lcl_size_limit = (_in_width <= 8) ? n_out_stacks*MAP_WK_SZ*read_unit : _dev_local_mem_sz / (2 * sizeof(float));

// not to exeed local memory size
	while ((_in_width > 8) && n_out_stacks*MAP_WK_SZ*read_unit > lcl_size_limit)
	{
		n_out_stacks--;
	}

	// number input maps stacks.
	// n_in_stacks input map wil be written in teh local memory sequentially
	int n_in_stacks = (GRP_SZ / MAP_WK_SZ);

	n_out_stacks = std::min(_n_inputs, n_out_stacks);
	n_in_stacks = std::min(_n_outputs, n_in_stacks);

	// param
	// this is 1 currently
	_n_out_pix_tiles = std::min(1, (_n_inputs + n_out_stacks - 1) / n_out_stacks);

	// number of maps in a stack or number of input read blocks written into 1 wk-item (lane)
	// param
	_n_in_data_tiles = std::min(((_in_width == 28) ? 2 : 4), (_n_outputs + n_in_stacks - 1) / n_in_stacks);
// to be able to do an easy final transform and summation
	while ((_in_width > 8) && n_in_stacks*_n_in_data_tiles* n_out_stacks > GRP_SZ)
	{
		n_in_stacks--;
	}

	 // total maps per group
	int total_out_maps = _n_out_pix_tiles * n_out_stacks;
	int total_in_maps = _n_in_data_tiles * n_in_stacks;

	_grp_tile0 = GRP_SZ;
	_grp_tile1 = 1;
	int grp_tile2 = 1;


	// utility parameters
	int n_ut_waves = 4;
	int UT_GRP_SZ0 = _hw_wave_sz * n_ut_waves;
	int ut_read_unit = ((wei_cstride / 4) * 4 == wei_cstride) ? 4 : ((wei_cstride / 2) * 2 == wei_cstride) ? 2 : 1;
	std::string UT_READ_TYPE = (ut_read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((ut_read_unit));


	// it's backward - inputs are outputs and vs versa
	_comp_options =
		std::string(" -DMLO_DIR_FORWARD=") + std::to_string(_direction)
		+ std::string(" -DMLO_GRP_SZ=") + std::to_string(GRP_SZ)
		+ std::string(" -DMLO_GRP_SZ0=") + std::to_string(_grp_tile0)
		+ std::string(" -DMLO_GRP_SZ1=") + std::to_string(_grp_tile1)
		+ std::string(" -DMLO_GRP_SZ2=") + std::to_string(grp_tile2)
		+ std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(_kernel_size0)
		+ std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(_kernel_size1)
		+ std::string(" -DMLO_FILTER_PAD0=") + std::to_string(_pad0)
		+ std::string(" -DMLO_FILTER_PAD1=") + std::to_string(_pad1)
		+ std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(_kernel_stride0)
		+ std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(_kernel_stride1)
		+ std::string(" -DSTRIDE_W=") + std::to_string(_kernel_stride0)
		+ std::string(" -DSTRIDE_H=") + std::to_string(_kernel_stride1)
		+ std::string(" -DMLO_N_OUTPUTS=") + std::to_string(_n_inputs)
		+ std::string(" -DMLO_N_INPUTS=") + std::to_string(_n_outputs)
		+ std::string(" -DMLO_BATCH_SZ=") + std::to_string(_batch_sz)
		+ std::string(" -DMLO_N_BATCH_LOOPS=") + std::to_string(N_BATCH_LOOPS)
		+ std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(_in_batch_stride)
		+ std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(_in_channel_stride)
		+ std::string(" -DMLO_OUT_STRIDE=") + std::to_string(_in_stride)
		+ std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(_out_batch_stride)
		+ std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(_out_channel_stride)
		+ std::string(" -DMLO_IN_STRIDE=") + std::to_string(_out_stride)
		+ std::string(" -DMLO_WEI_BATCH_STRIDE=") + std::to_string(wei_bstride)
		+ std::string(" -DMLO_WEI_CHANNEL_STRIDE=") + std::to_string(wei_cstride)
		+ std::string(" -DMLO_IN_WIDTH=") + std::to_string(_out_width)
		+ std::string(" -DMLO_IN_HEIGHT=") + std::to_string(_out_height)
		+ std::string(" -DMLO_OUT_WIDTH=") + std::to_string(_in_width)
		+ std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(_in_height)
		+ std::string(" -DMLO_IN_TILE1=") + std::to_string(_in_tile1)
		+ std::string(" -DMLO_IN_TILE0=") + std::to_string(_in_tile0)
		+ std::string(" -DMLO_N_LCL_BATCHS=") + std::to_string(_n_stacks) // # of diff stacks (part of batch).
		+ std::string(" -DMLO_N_LCL_OUT_MAPS=") + std::to_string(_n_out_pix_tiles)  // # output pixel tiles per wk-item (ALU)
		+ std::string(" -DMLO_N_LCL_IN_MAPS=") + std::to_string(_n_in_data_tiles) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_OUT_TILE0=") + std::to_string(_out_pix_tile0)  // size of ouptput tile per wk-item (ALU)
		+ std::string(" -DMLO_OUT_TILE1=") + std::to_string(_out_pix_tile1)  //
		+ std::string(" -DMLO_OUT_STACKS=") + std::to_string(n_out_stacks)
		+ std::string(" -DMLO_IN_STACKS=") + std::to_string(n_in_stacks)
		+ std::string(" -DMLO_N_WAVES=") + std::to_string(n_waves)
		+ std::string(" -DMLO_MAP_WK_SZ=") + std::to_string(MAP_WK_SZ)
		+ std::string(" -DMLO_POW2_MAP_WK_SZ=") + std::to_string(POW2_MAP_WK_SZ)
		+ std::string(" -DMLO_LCL_MEM_SZ=") + std::to_string(lcl_size_limit)

		+ std::string(" -DMLO_READ_TYPE=") + READ_TYPE
		+ std::string(" -DMLO_READ_UNIT=") + std::to_string(read_unit)
		+ std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(_hw_wave_sz)
		+ std::string(" -DMLO_LG2_PHYS_WAVE_SZ=") + std::to_string(mloLg2(_hw_wave_sz))

		+ std::string(" -DMLO_CONV_BIAS=") + std::to_string(_bias)

		+ std::string(" -DMLO_UT_READ_TYPE=") + UT_READ_TYPE
		+ std::string(" -DMLO_UT_READ_UNIT=") + std::to_string(ut_read_unit)

		+ std::string(" -DMLO_UT_GRP_SZ0=") + std::to_string(UT_GRP_SZ0)

		//		+ std::string(" -limit-vector-registers=64 ")
		+ getGeneralCompOptions()
		;


	_mlo_kernels_info.clear();
	// wrt to W
	{
		_l_wk.clear();
		_l_wk.push_back(_grp_tile0);
		_l_wk.push_back(_grp_tile1);
		_l_wk.push_back(grp_tile2);
		// input is output

		size_t gbl_wk0 = GRP_SZ * ((_n_inputs + total_out_maps - 1) / total_out_maps);
		size_t gbl_wk1 = ((_n_outputs + total_in_maps - 1) / total_in_maps);
		size_t gbl_wk2 =  n_batch_blks;


		_g_wk.clear();
		_g_wk.push_back(gbl_wk0);
		_g_wk.push_back(gbl_wk1);
		_g_wk.push_back(gbl_wk2);

		_kernel_file = "MLOpenConvBwdWrW_GxL_1x1.cl";
		_kernel_name = "MLOpenCvBwdWrW";

		auto kern_info = std::make_tuple(_kernel_name, _kernel_file, _comp_options, _g_wk, _l_wk);
		_mlo_kernels_info.push_back(kern_info);

		_workspce_sz = 0;

	}

	// sum over batch
	if (n_batch_blks > 1)
	{


		std::string kernel_file = "MLOpenConvBwdWrW_GxL_1x1.cl";
		std::string kernel_name = "MLOpenCvBwdWrW_rdc";

		std::vector<size_t> l_wk;
		l_wk.clear();
		l_wk.push_back(UT_GRP_SZ0);
		l_wk.push_back(1);
		l_wk.push_back(1);

		int gbl_ut_wk0 = wei_bstride * _n_inputs / ut_read_unit;

		std::vector<size_t> g_wk;
		g_wk.push_back(gbl_ut_wk0);
		g_wk.push_back(1);
		g_wk.push_back(1);
		auto kern_info = std::make_tuple(kernel_name, kernel_file, _comp_options, g_wk, l_wk);
		_mlo_kernels_info.push_back(kern_info);

		int data_len = (!_out_data_type.compare("FP32") ? 4 : 8);
		_workspce_sz = wei_bstride * _n_inputs * n_batch_blks * data_len;
	}

	return(ret);
}


int mlo_construct_BwdWrW2D::mloConstruct53()
{

	int ret = 0;
	size_t localMemSize = 64 * 1024;

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes
									  // major parameters

									  // inpout are outputs
	int wei_cstride = _kernel_size0*_kernel_size1;
	int wei_bstride = _n_outputs*wei_cstride;

	int read_unit = 4;
	std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((read_unit));

	// number  of batch iterations
	_n_stacks = 1;
	_n_stacks = std::min(_batch_sz, _n_stacks);
	// defines how to proceed : 1 grouop per batch or with a loop over all batches
	// loop over al batches make sense in 2 cases: a lot of small inputs/outputs or few batches
	// param
	int N_BATCH_LOOPS = (_n_inputs*_n_outputs <= 8 * 1024) ? 1 : _batch_sz / _n_stacks;
	int n_batch_blks = (_batch_sz + N_BATCH_LOOPS * _n_stacks - 1) / (N_BATCH_LOOPS * _n_stacks);

	_out_pix_tile0 = _kernel_size0;
	_out_pix_tile1 = _kernel_size1;
	_in_tile1 = 1;

	// span size
	// param
	_in_tile0 = ((_out_pix_tile0 *_out_pix_tile1) <= 16 && (_in_width > 8)) ? 4 : 2;
	int n_spans = (_in_width + _in_tile0 - 1) / _in_tile0;

	// n of wvaefront in a group
	// param
	int n_waves = ((_out_pix_tile0 *_out_pix_tile1) <= 16 && (_in_width > 8)) ? 4 : (_in_width <= 16) ? 1 : 2;
	int GRP_SZ = _hw_wave_sz * n_waves;


	_n_out_pix_tiles = 1;
	int n_out_stacks = std::min(_n_inputs, std::max(1, GRP_SZ / n_spans));
	// number of input maps per group
	// param
	_n_in_data_tiles = ((_in_width <= 8 || (_in_width >= 28 && _in_width <= 32)) && (_out_pix_tile0 *_out_pix_tile1) <= 16) ? 2 : 1;

// calculate number of input scans in the input block
// max LDS size is 8K
	int in_lcl_width = ((_in_width + read_unit - 1) / read_unit) * read_unit + 2 * _pad0;
	// number of input map blocks being process at once
	// param
	int in_n_vert_reads = (_in_height > 32 && _in_width <= 64 && (_out_pix_tile0 *_out_pix_tile1) <= 16) ? _in_height/2 : _in_height;
	while (in_lcl_width * in_n_vert_reads * _n_in_data_tiles > (_dev_local_mem_sz/(2*sizeof(float))))
	{
		in_n_vert_reads = (in_n_vert_reads + 1)/2;
		if (in_n_vert_reads < 2 && _n_in_data_tiles >= 2)
		{
			in_n_vert_reads = _in_height;
			_n_in_data_tiles /= 2;
		}
		else if (in_n_vert_reads < 2)
		{
			printf("CONFIG ERROR: not enough local memory for the configuration\n");
			return(-1);
		}
	}
	int in_n_vert_read_loops = (_in_height + in_n_vert_reads - 1) / in_n_vert_reads;

	int ALIGNED_OUT_SCAN_LN = ((_in_width + read_unit - 1) / read_unit); // image aligned scan

// select output mapping
	int total_out_maps = _n_out_pix_tiles * n_out_stacks;

	total_out_maps = (total_out_maps > _n_inputs) ? _n_inputs : total_out_maps;

	_grp_tile0 = GRP_SZ;
	_grp_tile1 = 1;
	int grp_tile2 = 1;


	// utility parameters
	int n_ut_waves = 4;
	int UT_GRP_SZ0 = _hw_wave_sz * n_ut_waves;
	int ut_read_unit = ((wei_cstride / 4) * 4 == wei_cstride) ? 4 : ((wei_cstride / 2) * 2 == wei_cstride) ? 2 : 1;
	std::string UT_READ_TYPE = (ut_read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((ut_read_unit));


	// it's backward - inputs are outputs and vs versa
	_comp_options =
		std::string(" -DMLO_DIR_FORWARD=") + std::to_string(_direction)
		+ std::string(" -DMLO_GRP_SZ=") + std::to_string(GRP_SZ)
		+ std::string(" -DMLO_GRP_SZ0=") + std::to_string(_grp_tile0)
		+ std::string(" -DMLO_GRP_SZ1=") + std::to_string(_grp_tile1)
		+ std::string(" -DMLO_GRP_SZ2=") + std::to_string(grp_tile2)
		+ std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(_kernel_size0)
		+ std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(_kernel_size1)
		+ std::string(" -DMLO_FILTER_PAD0=") + std::to_string(_pad0)
		+ std::string(" -DMLO_FILTER_PAD1=") + std::to_string(_pad1)
		+ std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(_kernel_stride0)
		+ std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(_kernel_stride1)
		+ std::string(" -DSTRIDE_W=") + std::to_string(_kernel_stride0)
		+ std::string(" -DSTRIDE_H=") + std::to_string(_kernel_stride1)
		+ std::string(" -DMLO_N_OUTPUTS=") + std::to_string(_n_inputs)
		+ std::string(" -DMLO_N_INPUTS=") + std::to_string(_n_outputs)
		+ std::string(" -DMLO_BATCH_SZ=") + std::to_string(_batch_sz)
		+ std::string(" -DMLO_N_BATCH_LOOPS=") + std::to_string(N_BATCH_LOOPS)
		+ std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(_in_batch_stride)
		+ std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(_in_channel_stride)
		+ std::string(" -DMLO_OUT_STRIDE=") + std::to_string(_in_stride)
		+ std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(_out_batch_stride)
		+ std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(_out_channel_stride)
		+ std::string(" -DMLO_IN_STRIDE=") + std::to_string(_out_stride)
		+ std::string(" -DMLO_WEI_BATCH_STRIDE=") + std::to_string(wei_bstride)
		+ std::string(" -DMLO_WEI_CHANNEL_STRIDE=") + std::to_string(wei_cstride)
		+ std::string(" -DMLO_IN_WIDTH=") + std::to_string(_out_width)
		+ std::string(" -DMLO_IN_HEIGHT=") + std::to_string(_out_height)
		+ std::string(" -DMLO_OUT_WIDTH=") + std::to_string(_in_width)
		+ std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(_in_height)
		+ std::string(" -DMLO_IN_TILE1=") + std::to_string(_in_tile1)
		+ std::string(" -DMLO_IN_TILE0=") + std::to_string(_in_tile0)
		+ std::string(" -DMLO_N_LCL_BATCHS=") + std::to_string(_n_stacks) // # of diff stacks (part of batch).
		+ std::string(" -DMLO_N_LCL_OUT_MAPS=") + std::to_string(_n_out_pix_tiles)  // # output pixel tiles per wk-item (ALU)
		+ std::string(" -DMLO_N_LCL_IN_MAPS=") + std::to_string(_n_in_data_tiles) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_OUT_TILE0=") + std::to_string(_out_pix_tile0)  // size of ouptput tile per wk-item (ALU)
		+ std::string(" -DMLO_OUT_TILE1=") + std::to_string(_out_pix_tile1)  //
		+ std::string(" -DMLO_OUT_STACKS=") + std::to_string(n_out_stacks)
		+ std::string(" -DMLO_N_WAVES=") + std::to_string(n_waves)
		+ std::string(" -DMLO_READ_TYPE=") + READ_TYPE
		+ std::string(" -DMLO_READ_UNIT=") + std::to_string(read_unit)
		+ std::string(" -DMLO_ALIGNED_OUT_SCAN_LN=") + std::to_string(ALIGNED_OUT_SCAN_LN) // image aligned scan
		+ std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(_hw_wave_sz)
		+ std::string(" -DMLO_LG2_PHYS_WAVE_SZ=") + std::to_string(mloLg2(_hw_wave_sz))
		+ std::string(" -DMLO_IN_EXTENT1=") + std::to_string(in_n_vert_reads)
		+ std::string(" -DMLO_IN_N_VERT_LOOPS=") + std::to_string(in_n_vert_read_loops)

		+ std::string(" -DMLO_CONV_BIAS=") + std::to_string(_bias)

		+ std::string(" -DMLO_UT_READ_TYPE=") + UT_READ_TYPE
		+ std::string(" -DMLO_UT_READ_UNIT=") + std::to_string(ut_read_unit)

		+ std::string(" -DMLO_UT_GRP_SZ0=") + std::to_string(UT_GRP_SZ0)

		//		+ std::string(" -limit-vector-registers=64 ")
		+ getGeneralCompOptions()
		;


	_mlo_kernels_info.clear();
	// wrt to W
	{
		_l_wk.clear();
		_l_wk.push_back(_grp_tile0);
		_l_wk.push_back(_grp_tile1);
		_l_wk.push_back(grp_tile2);
		// input is output

		size_t gbl_wk0 = GRP_SZ;
		size_t gbl_wk1 = (_n_outputs + _n_in_data_tiles - 1) / _n_in_data_tiles;
		size_t gbl_wk2 = ((_n_inputs + total_out_maps - 1) / total_out_maps) * n_batch_blks;


		_g_wk.clear();
		_g_wk.push_back(gbl_wk0);
		_g_wk.push_back(gbl_wk1);
		_g_wk.push_back(gbl_wk2);

		_kernel_file = (_kernel_size0 == 5 && _kernel_size1 == 5 && in_n_vert_read_loops == 1) ? "MLOpenConvBwdWrW_LxG_5x5.cl" : "MLOpenConvBwdWrW_LxG_P53.cl";
		_kernel_name = "MLOpenCvBwdWrW";

		auto kern_info = std::make_tuple(_kernel_name, _kernel_file, _comp_options, _g_wk, _l_wk);
		_mlo_kernels_info.push_back(kern_info);

		_workspce_sz = 0;

	}

	// sum over batch
	if (n_batch_blks > 1)
	{


		std::string kernel_file = (_kernel_size0 == 5 && _kernel_size1 == 5 && in_n_vert_read_loops == 1) ? "MLOpenConvBwdWrW_LxG_5x5.cl" : "MLOpenConvBwdWrW_LxG_P53.cl";
		std::string kernel_name = "MLOpenCvBwdWrW_rdc";

		std::vector<size_t> l_wk;
		l_wk.clear();
		l_wk.push_back(UT_GRP_SZ0);
		l_wk.push_back(1);
		l_wk.push_back(1);

		int gbl_ut_wk0 = wei_bstride * _n_inputs / ut_read_unit;

		std::vector<size_t> g_wk;
		g_wk.push_back(gbl_ut_wk0);
		g_wk.push_back(1);
		g_wk.push_back(1);
		auto kern_info = std::make_tuple(kernel_name, kernel_file, _comp_options, g_wk, l_wk);
		_mlo_kernels_info.push_back(kern_info);

		int data_len = (!_out_data_type.compare("FP32") ? 4 : 8);
		_workspce_sz = wei_bstride * _n_inputs * n_batch_blks * data_len;
	}

	return(ret);
}


int mlo_construct_BwdWrW2D::mloConstruct2()
{
	int ret = 0;
	size_t localMemSize = 64 * 1024;

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes

									  // number  of batch iterations
	_n_stacks = 1;
	_n_stacks = std::min(_batch_sz, _n_stacks);
	int N_BATCH_LOOPS = 1; // _batch_sz / _n_stacks;
	int n_batch_blks = (_batch_sz + N_BATCH_LOOPS * _n_stacks - 1) / (N_BATCH_LOOPS * _n_stacks);
	// number of filter taps in the processing wk_item
	int WEI_WKITEM = (_kernel_size0 <= 7 || (((_kernel_size0 / 2)*2) != _kernel_size0) )? _kernel_size0 : _kernel_size0 / 2;

	_in_tile0 = 1;
	_in_tile1 = 1;
	_out_pix_tile0 = 1;
	_out_pix_tile1 = (_kernel_size0 == 20) ? 1 : 2; //700 = 1, 350 = 2

						// major parameters
	int n_waves = (_out_width > 512) ? 4 : 2; // 700 = 4, 350 == 2

	_n_in_data_tiles = 1;
	// n of out blocks in lcl memory
	_n_out_pix_tiles = (_kernel_size0 == 20) ? 2 : 4; // 700 = 2, 350 = 4

						  // select output mapping
	int total_out_maps = _n_out_pix_tiles * _out_pix_tile1;
	_out_pix_tile1 = (total_out_maps > _n_inputs) ? 1 : _out_pix_tile1;
	total_out_maps = _n_out_pix_tiles * _out_pix_tile1;
	_n_out_pix_tiles = (total_out_maps > _n_inputs) ? _n_inputs : _n_out_pix_tiles;
	int N_OUT_BLK_GRP = _out_pix_tile1;
	total_out_maps = _n_out_pix_tiles * _out_pix_tile1;


	// each wave is a filter row
	int GRP_SZ = _hw_wave_sz * n_waves;


	// inpout are outputs
	int wei_cstride = _kernel_size0*_kernel_size1;
	int wei_bstride = _n_outputs*wei_cstride;

	int read_unit = 4;
	std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((read_unit));


	// input is output
	int ALIGNED_OUT_SCAN_LN = ((_in_width + read_unit - 1) / read_unit); // image aligned scan
	int N_ALIGNED_OUT_SCAN_BLK = 2;
	int N_OUT_BLK = (_in_height + N_ALIGNED_OUT_SCAN_BLK - 1) / N_ALIGNED_OUT_SCAN_BLK;


	int OUT_N_PIXS_OFF = _in_width - (_in_width  / read_unit) * read_unit;


	_grp_tile0 = GRP_SZ;
	_grp_tile1 = 1;
	int grp_tile2 = 1;


	// utility parameters
	int n_ut_waves = 4;
	int UT_GRP_SZ0 = _hw_wave_sz * n_ut_waves;
	int ut_read_unit = ((wei_cstride / 4) * 4 == wei_cstride) ? 4 : ((wei_cstride / 2) * 2 == wei_cstride) ? 2 : 1;
	std::string UT_READ_TYPE = (ut_read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((ut_read_unit));


	// it's backward - inputs are outputs and vs versa
	_comp_options =
		std::string(" -DMLO_DIR_FORWARD=") + std::to_string((_direction))
		+ std::string(" -DMLO_GRP_SZ=") + std::to_string((GRP_SZ))
		+ std::string(" -DMLO_GRP_SZ0=") + std::to_string((_grp_tile0))
		+ std::string(" -DMLO_GRP_SZ1=") + std::to_string((_grp_tile1))
		+ std::string(" -DMLO_GRP_SZ2=") + std::to_string((grp_tile2))
		+ std::string(" -DMLO_FILTER_SIZE0=") + std::to_string((_kernel_size0))
		+ std::string(" -DMLO_FILTER_SIZE1=") + std::to_string((_kernel_size1))
		+ std::string(" -DMLO_FILTER_PAD0=") + std::to_string((_pad0))
		+ std::string(" -DMLO_FILTER_PAD1=") + std::to_string((_pad1))
		+ std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string((_kernel_stride0))
		+ std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string((_kernel_stride1))
		+ std::string(" -DSTRIDE_W=") + std::to_string((_kernel_stride0))
		+ std::string(" -DSTRIDE_H=") + std::to_string((_kernel_stride1))
		+ std::string(" -DMLO_N_OUTPUTS=") + std::to_string((_n_inputs))
		+ std::string(" -DMLO_N_INPUTS=") + std::to_string((_n_outputs))
		+ std::string(" -DMLO_BATCH_SZ=") + std::to_string(_batch_sz)
		+ std::string(" -DMLO_N_BATCH_LOOPS=") + std::to_string(N_BATCH_LOOPS)
		+ std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string((_in_batch_stride))
		+ std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string((_in_channel_stride))
		+ std::string(" -DMLO_OUT_STRIDE=") + std::to_string((_in_stride))
		+ std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string((_out_batch_stride))
		+ std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string((_out_channel_stride))
		+ std::string(" -DMLO_IN_STRIDE=") + std::to_string((_out_stride))
		+ std::string(" -DMLO_WEI_BATCH_STRIDE=") + std::to_string((wei_bstride))
		+ std::string(" -DMLO_WEI_CHANNEL_STRIDE=") + std::to_string((wei_cstride))
		+ std::string(" -DMLO_IN_WIDTH=") + std::to_string((_out_width))
		+ std::string(" -DMLO_IN_HEIGHT=") + std::to_string(_out_height)
		+ std::string(" -DMLO_OUT_WIDTH=") + std::to_string(_in_width)
		+ std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(_in_height)
		+ std::string(" -DMLO_IN_TILE1=") + std::to_string(_in_tile1)
		+ std::string(" -DMLO_IN_TILE0=") + std::to_string(_in_tile0)
		+ std::string(" -DMLO_N_LCL_BATCHS=") + std::to_string(_n_stacks) // # of diff stacks (part of batch).
		+ std::string(" -DMLO_N_LCL_OUT_MAPS=") + std::to_string(_n_out_pix_tiles)  // # output pixel tiles per wk-item (ALU)
		+ std::string(" -DMLO_N_LCL_IN_MAPS=") + std::to_string(_n_in_data_tiles) // total # of blocks of different inputs in LDS
		+ std::string(" -DMLO_OUT_TILE0=") + std::to_string((_out_pix_tile0))  // size of ouptput tile per wk-item (ALU))
		+ std::string(" -DMLO_OUT_TILE1=") + std::to_string(_out_pix_tile1)  //
		+ std::string(" -DMLO_N_WAVES=") + std::to_string(n_waves)
		+ std::string(" -DMLO_READ_TYPE=") + READ_TYPE
		+ std::string(" -DMLO_READ_UNIT=") + std::to_string(read_unit)
		+ std::string(" -DMLO_ALIGNED_OUT_SCAN_LN=") + std::to_string(ALIGNED_OUT_SCAN_LN) // image aligned scan
		+ std::string(" -DMLO_N_ALIGNED_OUT_SCAN_BLK=") + std::to_string(N_ALIGNED_OUT_SCAN_BLK)
		+ std::string(" -DMLO_WEI_WKITEM=") + std::to_string(WEI_WKITEM)
		+ std::string(" -DMLO_N_OUT_BLK_GRP=") + std::to_string(N_OUT_BLK_GRP)
		+ std::string(" -DMLO_N_OUT_BLK=") + std::to_string(N_OUT_BLK)
		+ std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(_hw_wave_sz)
		+ std::string(" -DMLO_LG2_PHYS_WAVE_SZ=") + std::to_string(mloLg2(_hw_wave_sz))
		+ std::string(" -DMLO_OUT_N_PIXS_OFF=") + std::to_string(OUT_N_PIXS_OFF)

		+ std::string(" -DMLO_CONV_BIAS=") + std::to_string(_bias)

		+ std::string(" -DMLO_UT_READ_TYPE=") + UT_READ_TYPE
		+ std::string(" -DMLO_UT_READ_UNIT=") + std::to_string(ut_read_unit)

		+ std::string(" -DMLO_UT_GRP_SZ0=") + std::to_string((UT_GRP_SZ0))

		//		+ std::string(" -limit-vector-registers=64 ")
		+ getGeneralCompOptions()
		;


	_mlo_kernels_info.clear();
	// wrt to W
	{
		_l_wk.clear();
		_l_wk.push_back(_grp_tile0);
		_l_wk.push_back(_grp_tile1);
		_l_wk.push_back(grp_tile2);
		// input is output

		size_t gbl_wk0 = GRP_SZ;
		size_t gbl_wk1 = _n_outputs;
		size_t gbl_wk2 = ((_n_inputs + total_out_maps - 1) / total_out_maps) * n_batch_blks;


		_g_wk.clear();
		_g_wk.push_back(gbl_wk0);
		_g_wk.push_back(gbl_wk1);
		_g_wk.push_back(gbl_wk2);

		_kernel_file = (_pad0 > 0 || _pad1 > 0) ? "MLOpenConvBwdWrW_LxL_P.cl" : "MLOpenConvBwdWrW_LxL.cl";
		_kernel_name = "MLOpenCvBwdWrW";

		auto kern_info = std::make_tuple(_kernel_name, _kernel_file, _comp_options, _g_wk, _l_wk);
		_mlo_kernels_info.push_back(kern_info);

		_workspce_sz = 0;

	}

	// sum over batch
	if (n_batch_blks > 1)
	{


		std::string kernel_file = (_pad0 > 0 || _pad1 > 0) ? "MLOpenConvBwdWrW_LxL_P.cl" : "MLOpenConvBwdWrW_LxL.cl";
		std::string kernel_name = "MLOpenCvBwdWrW_rdc";

		std::vector<size_t> l_wk;
		l_wk.clear();
		l_wk.push_back(UT_GRP_SZ0);
		l_wk.push_back(1);
		l_wk.push_back(1);

		int gbl_ut_wk0 = wei_bstride * _n_inputs / ut_read_unit;

		std::vector<size_t> g_wk;
		g_wk.push_back(gbl_ut_wk0);
		g_wk.push_back(1);
		g_wk.push_back(1);
		auto kern_info = std::make_tuple(kernel_name, kernel_file, _comp_options, g_wk, l_wk);
		_mlo_kernels_info.push_back(kern_info);

		int data_len = (!_out_data_type.compare("FP32") ? 4 : 8);
		_workspce_sz = wei_bstride * _n_inputs * n_batch_blks * data_len;
	}

	return(ret);
}


int mlo_construct_BwdWrW2D::mloConstruct()
{
	int ret = 0;
	_workspce_sz = 0;

    if (((_kernel_size0>=_kernel_size1) && (_kernel_stride0 > 1 || _kernel_stride1 > 1)) || ((_pad0 == 0 || _pad1 == 0) && (_kernel_size0 != 1 || _kernel_size1 != 1)))
	{
		ret = mloConstruct2();
	}
	else if (_kernel_size0 >= _kernel_size1)
	{
		if ((_kernel_size0 >= 2) || (_kernel_size1 >= 2))
		{
			ret = mloConstruct53();
		}
		else if ( _in_width * _in_height <= (8*1024))
		{
			ret = mloConstruct1x1();
		}
	}
	return(ret);
}



/*
 * makes a unique key that represent the current kernel c0onfiguration
 */

int mlo_construct_direct2D::mloMakeKernelHash(std::string & hash) const
{

	std::string conf_key, conf_val;
	mloBuildConf_Key(conf_key);
	int grp_tile1;
	int grp_tile0;
	int in_tile1;
	int in_tile0;
	int out_pix_tile1;
	int out_pix_tile0;
	int n_out_pix_tiles;
	int n_in_data_tiles;
	int n_stacks;

	getConfigParameters(
			grp_tile1,
			grp_tile0,
			in_tile1,
			in_tile0,
			out_pix_tile1,
			out_pix_tile0,
			n_out_pix_tiles,
			n_in_data_tiles,
			n_stacks
			);
	mloBuildConf_Val(
			conf_val,
			grp_tile1,
			grp_tile0,
			in_tile1,
			in_tile0,
			out_pix_tile1,
			out_pix_tile0,
			n_out_pix_tiles,
			n_in_data_tiles,
			n_stacks
			);
	hash = conf_key + std::string(" ") + conf_val;
	return(0);
}

/***********************************************************************************************************

 * Internal implementation of the direct conv configuration search

 ************************************************************************************************************/



/*
   the search db is a text file with the name defined by the device characteristics.
   each line is a key/value pair, separated by a space:
   32x16x16x3x3x64x16x16x100xNCHWxFP32x1 16.16.16.16.1.4.8.4.1
   or
   64x8x8x5x5x32x8x8x100xNCHWxFP32x0 16.16.8.8.2.4.1.1.4

   key format (all values are separted by x):
   n input maps
   input height
   input width
   filter height
   filter width
   n output maps
   output height
   output width
   batch size
   tensors' layout
   tensprs' data type
   direction (1 - forward, 0 - backward)

Note: 
for backward direction - input and output are reversed.

value format (all values are separated by .):
vertical group size
horizontal group size
input block vertical size
input block horizontal size
output tile vertical size
output tile horizaontal size
n of output tiles
n of input blocks
n batchs (stacks) processed by the group
*/

int mlo_construct_direct2D :: mloSetConf(const std::string & conf_val)
{
	mloParseConf(conf_val,
			_grp_tile1,
			_grp_tile0,
			_in_tile1,
			_in_tile0,
			_out_pix_tile1,
			_out_pix_tile0,
			_n_out_pix_tiles,
			_n_in_data_tiles,
			_n_stacks
			);

	return(0);

}

int mlo_construct_direct2D::mloBuildConf_Key(std::string & conf_key) const
{

	conf_key = std::to_string(static_cast<long long>(_n_inputs))
		+ std::string("x") + std::to_string(static_cast<long long>(_in_height))
		+ std::string("x") + std::to_string(static_cast<long long>(_in_width))
		+ std::string("x") + std::to_string(static_cast<long long>(_kernel_size1))
		+ std::string("x") + std::to_string(static_cast<long long>(_kernel_size0))
		+ std::string("x") + std::to_string(static_cast<long long>(_n_outputs))
		+ std::string("x") + std::to_string(static_cast<long long>(_out_height))
		+ std::string("x") + std::to_string(static_cast<long long>(_out_width))
		+ std::string("x") + std::to_string(static_cast<long long>(_batch_sz))
		+ std::string("x") + _in_layout
		+ std::string("x") + _in_data_type
		+ std::string("x") + std::to_string(static_cast<long long>(_direction))
		;
	return(0);
}


/*
 * select defult configuration if a known configuration has not been found.
 */
int mlo_construct_direct2D::mloSelectDefaultConfig(std::string & conf_val)
{

	//
	_in_tile0 = (_in_width <= 8) ? 8 : (_in_width <= 16) ? 16 : 32; // size of input data per ALU plane
	_in_tile1 = (_in_height <= 8) ? 8 : (_in_height <= 16) ? 16 : 8; // size of input data per ALU plane

	_grp_tile0 = (_in_tile0 == 8) ? 8 : 16;
	_grp_tile1 = (_in_tile1 == 8) ? 8 : 16;

	_out_pix_tile0 = 2;  // size of ouptput tile per wk-item (ALU))
	_out_pix_tile1 = 2; // 


	_n_out_pix_tiles = 8; // # output pixel tiles per wk-item (ALU)
	_n_in_data_tiles = 2; // # of blocks of different inputs in LDS

	_n_stacks = 1; // # of diff stacks (part of batch).

	if (_kernel_size0 == 1 && _kernel_size1 == 1)
	{

		_in_tile0 = 4; // size of input data per ALU plane
		_in_tile1 = 1; // size of input data per ALU plane

		int out_len4 = (_out_height * _out_width + 3) / 4;

		_grp_tile0 = (out_len4 > 192) ? 256 : (out_len4 > 128) ? 192 : (out_len4 > 64) ? 128 : 64;
		_grp_tile1 = 1;

		_out_pix_tile0 = 4;  // size of ouptput tile per wk-item (ALU))
		_out_pix_tile1 = 1; // 4; //


		_n_out_pix_tiles = 16; // 2;  // # output pixel tiles per wk-item (ALU)
		_n_in_data_tiles = 2; // 4; // # of blocks of different inputs in LDS

		_n_stacks = (_batch_sz > 1) ? 2 : 1; // # of diff stacks (part of batch).

	}

	mloBuildConf_Val(
		conf_val,
		_grp_tile1,
		_grp_tile0,
		_in_tile1,
		_in_tile0,
		_out_pix_tile1,
		_out_pix_tile0,
		_n_out_pix_tiles,
		_n_in_data_tiles,
		_n_stacks
	);

	mloSetConf(conf_val);

	return(0);
}

/*
 * mesure the current onfiguration pefformance
 */
int mlo_construct_direct2D :: mloMeasuredLoop(mlopen::Handle* profile_h,
		Data_t bot_ocl_buf,
		Data_t top_ocl_buf,
		Data_t wei_ocl_buf,
		Data_t bias_ocl_buf,
		double &processing_time
		)
{
	int ret = 0;

	ret = mloConstructDirect2DFwd();
	if (ret != 0)
	{
		return(ret);
	}

	std::string compiler_options = _gen_comp_options + _comp_options;

	// Creating OCLKernel obj
	try {

		float padding_value = 0;
		
		double s= 0, e = 0;
		int iter = 1;

		if (profile_h)
		{
			processing_time = std::numeric_limits<float>::max();

			auto k = profile_h->GetKernel("", "", _kernel_file, _kernel_name, _l_wk, _g_wk, compiler_options);

			if(_bias) {
				k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
			} else {
				k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
			}
			processing_time = profile_h->GetKernelTime();
		}
		else
		{
			iter = (_n_timer_iter <= 0) ? 1 : _n_timer_iter;

			auto k = _stream->GetKernel("", "", _kernel_file, _kernel_name, _l_wk, _g_wk, compiler_options);

			if(_bias) {
				k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
			} else {
				k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
			}

			_stream->Finish();

			s = mlopen_mach_absolute_time();

			for (int i = 0; i < iter && ret == 0; i++)
			{
				if(_bias) {
					k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
				} else {
					k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
				}
			}

			_stream->Finish();
			e = mlopen_mach_absolute_time();

			processing_time = subtractTimes(e, s) / iter;
		}
    }
    catch(mlopen::Exception&) {
        return -1;
    }

	return (ret);
}



/*
 * request cofiguraion db management
 * request configuration db is a text file
 * each line is a key (in cofig db format) that has not been found in teh configuratio db
 */


int mlo_construct_direct2D :: mloAddConfigReq(const std::string & conf_key) const
{
	int ret = 0;
	std::vector<std::string> req_conf_db;
	std::string conf_file = (_kernel_path == "") ? mlopen::GetDbPath() : _kernel_path;

	conf_file += std::string("/") + _stream->GetDeviceName() + "." + std::string("cd.rdb.txt");
#ifdef MLOPEN_LOG_CONVOLUTION
	printf("file %s\n", conf_file.c_str());
#endif
	std::vector<std::string>::iterator it;
	bool found = mloFindConfigReq(conf_file, conf_key, req_conf_db, it);


	if (!found)
	{
		req_conf_db.push_back(conf_key);
		ret = mloUpdateDb(conf_file, req_conf_db);
	}
	return(ret);
}

int mlo_construct_direct2D :: mloRemoveConfigReq(
		const std::string & conf_key
		) const
{
	int ret = 0;
	std::vector<std::string> req_conf_db;

	std::vector<std::string>::iterator it;

	std::string conf_file = (_kernel_path == "") ? mlopen::GetDbPath() : _kernel_path;
	conf_file += std::string("/") + _stream->GetDeviceName() + "." + std::string("cd.rdb.txt");

	bool found = mloFindConfigReq(conf_file, conf_key, req_conf_db, it);


	if (found)
	{
		req_conf_db.erase(it);
		ret = mloUpdateDb(conf_file, req_conf_db);
	}
	return(ret);
}

int mlo_construct_direct2D :: mloReadConfigDB(
		std::map<std::string, std::string> & conf_db
		) const
{

	int ret = 0;
	std::string conf_file = (_kernel_path == "") ? mlopen::GetDbPath() : _kernel_path;

	conf_file += std::string("/") + _stream->GetDeviceName() + "." + std::string("cd.pdb.txt");

	std::vector<std::string> db;
	mloReadDb(conf_file, db);

	// build searchable db

	std::vector<std::string>::iterator it;
	for (it = db.begin(); it != db.end(); ++it)
	{
		std::vector<std::string> v_key_val;
		tokenize((*it),
				v_key_val,
				std::string(" "));

		conf_db[v_key_val[0]] = v_key_val[1];
	}
	return(ret);
}

int mlo_construct_direct2D :: mloWriteConfigDB(
		const std::map<std::string, std::string> & conf_db
		) const
{

	int ret = 0;
	//serialize
	std::string conf_file = (_kernel_path == "") ? mlopen::GetDbPath() : _kernel_path;

	conf_file += std::string("/") + _stream->GetDeviceName() + "." + std::string("cd.pdb.txt");

	std::vector<std::string> db;

	std::map<std::string, std::string>::const_iterator it;

	for (it = conf_db.begin(); it != conf_db.end(); ++it)
	{
		db.push_back((*it).first + std::string(" ") + (*it).second + std::string("\n"));
	}

	ret = mloUpdateDb(conf_file, db);

	return(ret);
}

int mlo_construct_direct2D :: mloAddConfig(
		std::string & conf_key,
		std::string & conf_val
		) const
{
	int ret = 0;

	// build searchable db
	std::map<std::string, std::string> conf_db;

	mloReadConfigDB(
			conf_db
			);
	// add config

	conf_db[conf_key] = conf_val;
	//serialize
	ret = mloWriteConfigDB(
			conf_db
			);

	// remove request
	mloRemoveConfigReq(
			conf_key
			);

	return(ret);
}





bool mlo_construct_direct2D :: mloSearchConfigInDB(
		std::string & conf_key,
		std::string & conf_val
		) const
{
	bool known_config = false;
	// build searchable db
	std::map<std::string, std::string> conf_db;

	mloReadConfigDB(
			conf_db
			);

	mloBuildConf_Key(conf_key);

	std::map<std::string, std::string>::iterator m_it;
	known_config = mloSearchConfigDB(
			conf_db,
			conf_key,
			conf_val,
			m_it
			);

	return(known_config);
}

/*
 * return a known or default configuration
 */
bool mlo_construct_direct2D :: mloGetConfig()
{
	bool known_config = false;
	std::string conf_key;
	std::string conf_val;

	// find a db and configuration in it
	known_config = mloSearchConfigInDB(
			conf_key,
			conf_val
			);

	// if found save it

	if (known_config)
	{
		mloSetConf(conf_val);
	}
	else
		// otherwise
	{
		// select default
		mloSelectDefaultConfig(conf_val);
		// save the unknown configuration
		// if allowed
		if (_save_srch_req)
		{
			mloAddConfigReq(conf_key);
		}
	}

	return(known_config);

}

/*
 * search utility
 * defines a configurati spce 
 * search by maesure performabce per each configuration and saves the a current minimum


*/
int mlo_construct_direct2D :: mloSearchDirect2D()
{
	int ret = 0;

	mlopen::Handle profile_h;
	double processing_time;
	std::string conf_key;
	std::string conf_val;

	int min_grp_tile0 = 16;
	int min_grp_tile1 = 16;
	// tile 0
	int min_in_tile0 = 16;
	// tile 1
	int min_in_tile1 = 16;
	// out pix 0
	int min_out_pix_tile0 = 1;
	// out pix 1
	int min_out_pix_tile1 = 1;
	int min_n_out_pix_tiles = 2;
	int min_n_in_data_tiles = 3;
	int min_n_stacks = 1;

	size_t localMemSize = profile_h.GetLocalMemorySize();

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes

	// if it is not known
	bool known_config = mloSearchConfigInDB(
			conf_key,
			conf_val
			);

	// proceed
	if (!known_config)
	{

		// allocate tem input/output buffers
		size_t bot_sz = _bot_sz / sizeof(float);
		std::vector<float> bot_sys_buf(bot_sz);

		for (int i = 0; i < bot_sz; i++) {
			bot_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
		}

		auto bot_ocl_buf = profile_h.Write(bot_sys_buf);

		size_t top_sz = _top_sz / sizeof(float);
		std::vector<float> top_sys_buf(top_sz);

		auto top_ocl_buf = profile_h.Write(top_sys_buf);

		size_t weights_sz = _weights_sz / sizeof(float);
		std::vector<float> wei_sys_buf(weights_sz);
		for (int i = 0; i < weights_sz; i++) {
			wei_sys_buf[i] = static_cast<float>((rand() * (1.0 / RAND_MAX) - 0.5) * 0.001);
		}

		auto wei_ocl_buf = profile_h.Write(wei_sys_buf);

		std::vector<float> bias_sys_buf;
		ManageDataPtr bias_ocl_buf = nullptr;

		if (_bias)
		{
			size_t bias_sz = _bias_sz / sizeof(float);
			bias_sys_buf = std::vector<float>(bias_sz);
			for (int i = 0; i < bias_sz; i++) {
				bias_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
			}

			bias_ocl_buf = profile_h.Write(bias_sys_buf);
		}


		// search loop here
		int grp_tl_ln[4] = { 8, 16 };
		int tile_sz[3] = { 8, 16, 32 };
		int tile_sz1[3] = { 8, 16, 32 };
		int tile_sz0[3] = { 8, 16, 32 };
		int out_pix_tile_sz[3] = { 1, 2, 4 };
		int n_out_tiles_rg[2] = { 1, 8 };
		int n_in_tiles_rg[2] = { 1, 4 };
		int n_in_stacks_sz[3] = { 1, 2, 4 };
		/*
		std::vector<int> v_tile_sz;
		std::vector<int> v_out_pix_tile_sz;
		std::vector<int> v_n_out_tiles_rg;
		std::vector<int> v_n_in_tiles_rg;
		std::vector<int> v_n_in_stacks_sz;
		*/
		//

		double min_proc_time = std::numeric_limits<float>::max();

#if 1
		std::cout << "Searching the best solution in the 9 dim space. Please, be patient it may take few minutes." << std::endl;

		size_t run_counter = 0;
		int n_grp_tiles1 = 2;
		int n_grp_tiles0 = 2;

		int out_pix_tl_cnt = 3; // out_pix_tile_sz[1];
		int n_out_tls = n_out_tiles_rg[1];
		int stack_cnt = 2;
		int n_tile0_sz = 3;
		int n_tile1_sz = 3;

		n_out_tls = std::min(_n_outputs, n_out_tls);

		if (_in_width <= 8)
		{
			n_tile0_sz = 1;
			n_in_tiles_rg[1] = 16;
		}
		else
			if (_in_width <= 16)
			{
				n_tile0_sz = 1;
				tile_sz0[0] = 16;
				n_in_tiles_rg[1] = 8;

			}
			else
				if (_in_width <= 32)
				{
					n_tile0_sz = 2;
					tile_sz0[0] = 16;
					tile_sz0[1] = 32;

				}


		if (_in_height <= 8)
		{
			n_tile1_sz = 1;
			n_in_tiles_rg[1] = 16;
		}
		else
			if (_in_height <= 16)
			{
				n_tile1_sz = 1;
				tile_sz1[0] = 16;
				n_in_tiles_rg[1] = 8;

			}
			else
				if (_in_width <= 32)
				{
					n_tile1_sz = 2;
					tile_sz1[0] = 16;
					tile_sz1[1] = 32;

				}

		bool unaligned = (_out_height < 8 || _out_width < 8 || (_out_height > 8 && _out_height < 16) || (_out_width > 8 && _out_width < 16)
			|| (_out_height > 16 && _out_height < 32) || (_out_width > 16 && _out_width < 32));

		if (unaligned)
		{
			out_pix_tile_sz[1] = 6;
			out_pix_tl_cnt = out_pix_tile_sz[1];
		}

		int n_grp_tiles = n_grp_tiles1 *  n_grp_tiles0;

		int n_tiles_cnt = n_tile0_sz * n_tile1_sz;
		n_grp_tiles = (_out_height > 16 && _out_width > 16) ? n_grp_tiles - 1 : n_grp_tiles;
		n_tiles_cnt = (_out_height > 16 && _out_width > 16) ? n_tiles_cnt - 1 : n_tiles_cnt;
		size_t report_inteval = 100;
		//			_n_timer_iter = 250;

		if (_kernel_size0 == 1 && _kernel_size1 == 1)
		{
			grp_tl_ln[0] = 64;
			grp_tl_ln[1] = 128;
			grp_tl_ln[2] = 192;
			grp_tl_ln[3] = 256;
			n_grp_tiles1 = 1;
			n_grp_tiles0 = 4;

			tile_sz1[0] = 1;
			tile_sz0[0] = 4;
			n_tile0_sz = n_tile1_sz = 1;
			n_tiles_cnt = n_tile0_sz * n_tile1_sz;
			out_pix_tile_sz[0] = (unaligned) ? 0 : out_pix_tile_sz[0];
			out_pix_tile_sz[1] = 1;
			n_out_tiles_rg[1] = 16;
			n_in_tiles_rg[1] = 8;
			stack_cnt = 3;
			out_pix_tl_cnt = out_pix_tile_sz[1];
			n_out_tls = n_out_tiles_rg[1];
			n_grp_tiles = n_grp_tiles1 *  n_grp_tiles0;

			report_inteval = 20;

		}


		long long runs_left = n_grp_tiles * n_tiles_cnt * out_pix_tl_cnt * out_pix_tl_cnt * n_out_tls * n_in_tiles_rg[1] * stack_cnt;


		for (int g1 = 0; g1 < n_grp_tiles1; g1++)
		{
			_grp_tile1 = (_kernel_size0 == 1 && _kernel_size1 == 1) ? 1 : grp_tl_ln[g1];
			for (int g0 = 0; g0 < n_grp_tiles0; ++g0)
			{
				_grp_tile0 = grp_tl_ln[g0];

				// tile1
				for (int j = 0; j < n_tile1_sz; ++j)
				{
					_in_tile1 = tile_sz1[j];
					if (_out_height * 2 <= _in_tile1 && _in_tile1 > tile_sz[0])
					{
						runs_left--;
						runs_left = (runs_left < 0) ? 0 : runs_left;
						continue;
					}

					// tile 0
					for (int i = 0; i < n_tile0_sz; ++i)
					{
						_in_tile0 = tile_sz0[i];
						if ((_out_width * 2 <= _in_tile0 &&  _in_tile0 > tile_sz[0])
							)
						{
							runs_left--;
							runs_left = (runs_left < 0) ? 0 : runs_left;
							continue;
						}
						if (_out_height > 16 && _out_width > 16 && ((_in_tile1 == 8 && _in_tile0 == 8) || (_grp_tile0 == 8 && _grp_tile1 == 8)))
						{
							runs_left--;
							runs_left = (runs_left < 0) ? 0 : runs_left;
							continue;
						}
						if (_out_width > 32 && _in_tile1 > _in_tile0)
						{
							runs_left--;
							runs_left = (runs_left < 0) ? 0 : runs_left;
							continue;
						}
						// out pix 1

						for (int k = (unaligned) ? out_pix_tile_sz[0] : 0; k < out_pix_tl_cnt; ++k)
						{
							_out_pix_tile1 = (unaligned) ? k : out_pix_tile_sz[k];
							if (_out_pix_tile1 > _in_tile1)
							{
								runs_left--;
								runs_left = (runs_left < 0) ? 0 : runs_left;
								continue;
							}
							// out pix 0

							for (int l = (unaligned) ? out_pix_tile_sz[0] : 0; l < out_pix_tl_cnt; ++l)
							{
								_out_pix_tile0 = (_kernel_size0 == 1 && _kernel_size1 == 1) ? 4 : (unaligned) ? l : out_pix_tile_sz[l];

								if (_out_pix_tile0 > _in_tile0)
								{
									runs_left--;
									runs_left = (runs_left < 0) ? 0 : runs_left;
									continue;
								}

								int o_l = n_out_tiles_rg[1];
								for (int o_t = n_out_tiles_rg[0]; o_t <= o_l; ++o_t)
								{
									_n_out_pix_tiles = o_t;
									if (_n_outputs < _n_out_pix_tiles)
									{
										runs_left--;
										runs_left = (runs_left < 0) ? 0 : runs_left;
										continue;
									}
#if 1
									if (_kernel_size0 == 1 && _kernel_size1 == 1)
									{
										int N4S = 1;

										int MAP_SZ4 = (_in_width * _in_height + N4S * 4 - 1) / (N4S * 4);

										int GRP_SZ = _grp_tile0;
										int N_MAPS_PERGROUP = 1;
										int exchange_step;

										if (MAP_SZ4 <= GRP_SZ / 2)
										{
											N_MAPS_PERGROUP = GRP_SZ / MAP_SZ4;
											int lcl_mem_avial = (_grp_tile0 <= 192) ? (_dev_local_mem_sz / 4) / 2 : (_dev_local_mem_sz / 4);

											exchange_step = lcl_mem_avial / (N_MAPS_PERGROUP* MAP_SZ4 * 4);
											exchange_step = std::min(std::min(exchange_step, _n_out_pix_tiles), N_MAPS_PERGROUP);
											if (exchange_step < _n_out_pix_tiles)
											{
												int tmp_stp = static_cast<int>(ceil(sqrt(static_cast<float>(exchange_step))));
												n_in_tiles_rg[0] = tmp_stp;
												n_in_tiles_rg[1] = exchange_step;
											}
											else
											{
												n_in_tiles_rg[0] = 1;
												n_in_tiles_rg[1] = 1;
											}
										}

									}
#endif
									for (int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1]; ++i_t)
									{
										_n_in_data_tiles = i_t;
										if (_n_inputs < _n_in_data_tiles)
										{
											runs_left--;
											runs_left = (runs_left < 0) ? 0 : runs_left;
											continue;
										}

										for (int s = 0; s < stack_cnt; ++s)
										{

											_n_stacks = n_in_stacks_sz[s];
											if (_kernel_size0 == 1 && _kernel_size1 == 1)
											{

											}
											else
											{
												int alu_tile0 = std::max(1, _in_tile0 / _out_pix_tile0);
												int alu_tile1 = std::max(1, _in_tile1 / _out_pix_tile1);
												int alu_tiles_sz = (alu_tile0*alu_tile1);
												int n_alus_total = (_grp_tile0 * _grp_tile1);

												if (alu_tiles_sz > n_alus_total/* || _n_in_data_tiles*_n_out_pix_tiles*_out_pix_tile1*_out_pix_tile0 > 240*/)
												{
													runs_left--;
													runs_left = (runs_left < 0) ? 0 : runs_left;
													continue;
												}
											}

											if (_n_stacks > _batch_sz)
											{
												runs_left--;
												runs_left = (runs_left < 0) ? 0 : runs_left;
												continue;

											}
											ret = mloMeasuredLoop(&profile_h,
													bot_ocl_buf.get(),
													top_ocl_buf.get(),
													wei_ocl_buf.get(),
													bias_ocl_buf.get(),
													processing_time
													);

											if (ret != 0)
											{
												std::cout << "Failed run." << std::endl;
												runs_left--;
												runs_left = (runs_left < 0) ? 0 : runs_left;
												continue;
											}


											if (run_counter != 0 && run_counter % report_inteval == 0)
											{
												std::cout << "Runs left : " << runs_left << ", "
													<< "min time so far : " << min_proc_time << ", "
													<< "curr time : " << processing_time
#if 1
													<< ", " << _grp_tile1 << ", "
													<< _grp_tile0 << ", "
													<< _in_tile1 << ", "
													<< _in_tile0 << ", "
													<< _out_pix_tile1 << ", "
													<< _out_pix_tile0 << ", "
													<< _n_out_pix_tiles << ", "
													<< _n_in_data_tiles << ", "
													<< _n_stacks
#endif
													<< std::endl;
											}

											run_counter++;
											runs_left--;
											runs_left = (runs_left < 0) ? 0 : runs_left;
											if (min_proc_time > processing_time)
											{
												min_proc_time = processing_time;
												min_grp_tile0 = _grp_tile0;
												min_grp_tile1 = _grp_tile1;
												min_in_tile0 = _in_tile0;
												min_in_tile1 = _in_tile1;
												min_out_pix_tile0 = _out_pix_tile0;
												min_out_pix_tile1 = _out_pix_tile1;
												min_n_out_pix_tiles = _n_out_pix_tiles;
												min_n_in_data_tiles = _n_in_data_tiles;
												min_n_stacks = _n_stacks;
											}

										}  // for (int s = 0; s < 3; ++s)
									} // for (int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1]; ++i_t)
								} // if (_out_pix_tile0 > _in_tile0)
							} // for (int l = 0; l < l_l; ++l)
						} // for (int k = 0; k < k_l; ++k)
						}  // for (int i = 0; i < 3; ++i)
					} // for (int j = 0; j < 3; ++j)
				} // for (int g0 = 0; g0 < 2; ++g0)
			} // for (int g1 = 0; g1 < 2; g1++) 

		std::cout << std::endl << "Score: " << min_proc_time << std::endl;
#endif

		mloBuildConf_Val(conf_val,
				min_grp_tile1,
				min_grp_tile0,
				min_in_tile1,
				min_in_tile0,
				min_out_pix_tile1,
				min_out_pix_tile0,
				min_n_out_pix_tiles,
				min_n_in_data_tiles,
				min_n_stacks
				);


		mloAddConfig(
				conf_key,
				conf_val
				);
		// set the learnt data fo the current run.
		mloSetConf(conf_val);

	}

	return(ret);
}

// Tensor Helper APIs

size_t mlo_construct_direct2D::setWeightDescFromMLDesc(const mlopen::TensorDescriptor &weight_tensor) {

	int nWei;
	int cWei;
	int hWei;
	int wWei;
	int nWeiStride;
	int cWeiStride;
	int hWeiStride;
	int wWeiStride;

	std::tie(nWei, cWei, hWei, wWei) = mlopen::tie4(weight_tensor.GetLengths());
	std::tie(nWeiStride, cWeiStride, hWeiStride, wWeiStride) = mlopen::tie4(weight_tensor.GetStrides());

	setWeightsDescr(
			"NCHW",
			"FP32",
			nWei,
			cWei,
			hWei,
			wWei,
			nWeiStride,
			cWeiStride,
			hWeiStride,
			wWeiStride
			);

	size_t weights_sz = nWei * cWei * hWei * wWei * sizeof(float);
	return weights_sz;

}

size_t mlo_construct_direct2D::setOutputDescFromMLDesc(const mlopen::TensorDescriptor &output_tensor) {

	int nOut;
	int cOut;
	int hOut;
	int wOut;
	int nOutStride;
	int cOutStride;
	int hOutStride;
	int wOutStride;

	std::tie(nOut, cOut, hOut, wOut) = mlopen::tie4(output_tensor.GetLengths());
	std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = mlopen::tie4(output_tensor.GetStrides());

	setOutputDescr(
			"NCHW",
			"FP32",
			nOut,
			cOut,
			hOut,
			wOut,
			nOutStride,
			cOutStride,
			hOutStride,
			wOutStride);

	size_t output_sz = nOut * cOut * hOut * wOut * sizeof(float);
	return output_sz;
}

size_t mlo_construct_direct2D::setInputDescFromMLDesc(const mlopen::TensorDescriptor &input_tensor) {

	int nIn;
	int cIn;
	int hIn;
	int wIn;
	int nInStride;
	int cInStride;
	int hInStride;
	int wInStride;

	std::tie(nIn, cIn, hIn, wIn) = mlopen::tie4(input_tensor.GetLengths());
	std::tie(nInStride, cInStride, hInStride, wInStride) = mlopen::tie4(input_tensor.GetStrides());

	setInputDescr(
			"NCHW",
			"FP32",
			nIn,
			cIn,
			hIn,
			wIn,
			nInStride,
			cInStride,
			hInStride,
			wInStride);

	size_t input_sz = nIn * cIn * hIn * wIn * sizeof(float);

	return input_sz;
}
