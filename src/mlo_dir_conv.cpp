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

#include <mlopen/db.hpp>

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

/*
 * build the confiuration db file name base:
 * system device name_number of compute units_engine frequency
 */
	static
std::string mloConfFileBaseNm(cl_device_id dev
		)
{
	int maxComputeUnits;
	int maxClockFrequency;
	std::string deviceName;

	maxComputeUnits = mlopen::GetDeviceInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(dev);
	maxClockFrequency = mlopen::GetDeviceInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>(dev);
	deviceName = mlopen::GetDeviceInfo<CL_DEVICE_NAME>(dev);

	std::string conf_file_base_nm = deviceName + "_"
		+ std::to_string(static_cast<long long>(maxComputeUnits)) + "_"
		+ std::to_string(static_cast<long long>(maxClockFrequency));
	;
	return(conf_file_base_nm);
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
	_gen = (_kernel_size0 >= 11 || _kernel_size1 >= 11 || _kernel_stride0 > 1 || _kernel_stride1 > 1);
	if (_gen && getDirectcion())
	{
		ret = mloConstructDirect2DFwdGen();
	}
	else if (_gen)
	{
		std::cout << "Error: backward generic direct conv is not implemented. " << std::endl;
		return (-1);
	}
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

	cl_device_id dev = mlopen::GetDevice(reinterpret_cast<cl_command_queue>(_stream));

	size_t localMemSize = mlopen::GetDeviceInfo<CL_DEVICE_LOCAL_MEM_SIZE>(dev);

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes

	if (_direction == 0)
	{
		// backward
		_pad0 = (_pad0 == 0) ? _kernel_size0 - 1 : _pad0;
		_pad1 = (_pad1 == 0) ? _kernel_size1 - 1 : _pad1;
	}

	_n_in_data_tiles = std::min(_n_inputs, _n_in_data_tiles);
	_n_out_pix_tiles = std::min(_n_outputs, _n_out_pix_tiles);


	int alu_tile0 = (_in_tile0 / _out_pix_tile0);
	int alu_tile1 = (_in_tile1 / _out_pix_tile1);
	int alu_tiles_sz = (alu_tile0*alu_tile1);
	if (alu_tiles_sz > 256 || _grp_tile0 < alu_tile0 || _grp_tile1 < alu_tile1)
	{
		//			std::cout << "ERROR: need out pix size ajustments\n";
		return(-1);
	}

	_n_stacks = std::min(_n_stacks, (_grp_tile1*_grp_tile0) / alu_tiles_sz);
	_n_stacks = std::min(_batch_sz, _n_stacks);


	int n_alus_total = (_grp_tile0 * _grp_tile1);
	int n_alus_perstack = (n_alus_total / _n_stacks);

	int n_read_procs = _grp_tile1 * _grp_tile0;
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

	int n_alu_tiles_perstack = (n_alus_perstack / alu_tiles_sz);
	int n_out_tiles_perstack = n_alu_tiles_perstack * _n_out_pix_tiles;

	n_out_tiles_perstack = std::min(n_out_tiles_perstack, _n_outputs);


	_comp_options =
		std::string(" -D MLO_HW_WAVE_SZ=") + std::to_string(static_cast<long long>(_hw_wave_sz))
		+ std::string(" -D MLO_DIR_FORWARD=") + std::to_string(static_cast<long long>(_direction))
		+ std::string(" -D MLO_FILTER_SIZE0=") + std::to_string(static_cast<long long>(_kernel_size0))
		+ std::string(" -D MLO_FILTER_SIZE1=") + std::to_string(static_cast<long long>(_kernel_size1))
		+ std::string(" -D MLO_FILTER_PAD0=") + std::to_string(static_cast<long long>(_pad0))
		+ std::string(" -D MLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -D MLO_N_OUTPUTS=") + std::to_string(static_cast<long long>(_n_outputs))
		+ std::string(" -D MLO_N_INPUTS=") + std::to_string(static_cast<long long>(_n_inputs))
		+ std::string(" -D MLO_BATCH_SZ=") + std::to_string(static_cast<long long>(_batch_sz))
		+ std::string(" -D MLO_OUT_WIDTH=") + std::to_string(static_cast<long long>(_out_width))
		+ std::string(" -D MLO_OUT_HEIGHT=") + std::to_string(static_cast<long long>(_out_height))
		+ std::string(" -D MLO_OUT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -D MLO_OUT_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -D MLO_OUT_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -D MLO_IN_WIDTH=") + std::to_string(static_cast<long long>(_in_width))
		+ std::string(" -D MLO_IN_HEIGHT=") + std::to_string(static_cast<long long>(_in_height))
		+ std::string(" -D MLO_IN_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -D MLO_IN_CHANNEL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -D MLO_IN_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		// algorithm parameters
		+std::string(" -D MLO_IN_TILE0=") + std::to_string(static_cast<long long>(_in_tile0))  // size of input data per ALU plane
		+ std::string(" -D MLO_IN_TILE1=") + std::to_string(static_cast<long long>(_in_tile1))  // size of input data per ALU plane
		+ std::string(" -D MLO_GRP_TILE0=") + std::to_string(static_cast<long long>(_grp_tile0)) // # of ALUs (group size)
		+ std::string(" -D MLO_GRP_TILE1=") + std::to_string(static_cast<long long>(_grp_tile1)) //
		+ std::string(" -D MLO_OUT_TILE0=") + std::to_string(static_cast<long long>(_out_pix_tile0))  // size of ouptput tile per wk-item (ALU))
		+ std::string(" -D MLO_OUT_TILE1=") + std::to_string(static_cast<long long>(_out_pix_tile1))  //
		+ std::string(" -D MLO_N_STACKS=") + std::to_string(static_cast<long long>(_n_stacks)) // # of diff stacks (part of batch).
		+ std::string(" -D MLO_N_OUT_TILES=") + std::to_string(static_cast<long long>(_n_out_pix_tiles))  // # output pixel tiles per wk-item (ALU)
		+ std::string(" -D MLO_N_OUT_TILES_PERSTACK=") + std::to_string(static_cast<long long>(n_out_tiles_perstack))
		+ std::string(" -D MLO_N_IN_TILES_PERSTACK=") + std::to_string(static_cast<long long>(_n_in_data_tiles)) // total # of blocks of different inputs in LDS
		+ std::string(" -D MLO_N_READ_PROCS=") + std::to_string(static_cast<long long>(n_read_procs))
		+ std::string(" -D MLO_CONV_BIAS=") + std::to_string(static_cast<long long>(_bias))
		+ std::string(" -D MLO_ALU_VTILE0=") + std::to_string(static_cast<long long>(alu_tile0))
		+ std::string(" -D MLO_ALU_VTILE1=") + std::to_string(static_cast<long long>(alu_tile1))
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
	_kernel_name = "aDNNConvUni";



	return(ret);
}

/*
 * construct generic forward configuration
 */
int mlo_construct_direct2D::mloConstructDirect2DFwdGen()
{


	int ocl_group_sz0 = 16;
	int ocl_group_sz1 = 16;
	int ocl_group_sz2 = 1;
	int gbl0 = 0;
	int gbl1 = 0;
	int gbl2 = 0;

	int n_ins0 = 1; // number of inputs each a from different stack along dim 0
	int n_ins1 = 1; // number of inputs each a from different stack along dim 1
	int n_ins = n_ins0 * n_ins1; // number of inputs each a from different stack

	// should be a combination of # of CUs, batch size.
	// these is an aprox for Fiji
	int n_outs = (_batch_sz <= 8) ? ((_kernel_size0 < 5) ? 2 : 4) : (_batch_sz <= 16) ? ((_kernel_size0 < 5) ? 4 : 6) : ((_n_outputs <= 32) ? 4 : 8); // (kernel_size0 == 3 && width_out < 64 && height_out < 64) ? 14 : 12; // n outputs per a single input: major parameter
	int n_out_pix_horiz = 2; // n of output px horix per wk-item: major parameter
	int n_out_pix_vert = 2; // n of output px horix per wk-item: major parameter

	if (_gen)
	{
		n_outs = (_kernel_size1 < 7) ? 12 : (_kernel_size1 < 9) ? 8 : ((_kernel_size1 < 11) ? 4 : 2); // n outputs per a single input: major parameter
		n_out_pix_horiz = (_kernel_stride0 <= 4) ? 2 : 1; // n of output px horix per wk-item: major parameter
		n_out_pix_vert = (_kernel_stride1 < 4 && _kernel_size1 < 7) ? 2 : 1; // n of output px horix per wk-item: major parameter
		ocl_group_sz0 = 8; // (stride0 < 4) ? 16 : 8;
		ocl_group_sz1 = 8; //  (stride1 < 4) ? 16 : 8;

	}

	n_outs = std::min(n_outs, _n_outputs);

	int n_in_pix_horiz = n_out_pix_horiz; // n of input pix per wk_item
	int n_in_pix_vert = n_out_pix_vert; // n of input pix per wk_item
	int n_v_proc0 = (_out_width + n_out_pix_horiz - 1) / n_out_pix_horiz;
	int n_v_proc1 = (_out_height + n_out_pix_vert - 1) / n_out_pix_vert;

	for (int proc0 = ocl_group_sz0 / 2; n_v_proc0 <= proc0 && proc0 > 1; proc0 /= 2)
	{
		n_ins0 *= 2;
	}
	for (int proc1 = ocl_group_sz1 / 2; n_v_proc1 <= proc1 && proc1 > 1; proc1 /= 2)
	{
		n_ins1 *= 2;
	}

	n_ins = n_ins0 * n_ins1;
	if (n_ins > _batch_sz)
	{
		ocl_group_sz1 /= 2;
		n_ins1 = 1;
		for (int proc1 = ocl_group_sz1 / 2; n_v_proc1 <= proc1 && proc1 > 1; proc1 /= 2)
		{
			n_ins1 *= 2;
		}
		n_ins = n_ins0 * n_ins1;
	}

	if (n_ins > _batch_sz)
	{
		ocl_group_sz0 /= 2;
		n_ins0 = 1;
		for (int proc0 = ocl_group_sz0 / 2; n_v_proc0 <= proc0 && proc0 > 1; proc0 /= 2)
		{
			n_ins0 *= 2;
		}
		n_ins = n_ins0 * n_ins1;
	}


	int batch_aligned = 0;
#if 1
	if ((_batch_sz / n_ins) * n_ins == _batch_sz)
	{
		batch_aligned = 1;
	}
#endif
	int out_aligned = 0;
#if 1
	if ((_n_outputs / n_outs) * n_outs == _n_outputs)
	{
		out_aligned = 1;
	}
#endif
	int big = 0;
	if (ocl_group_sz0 * n_in_pix_horiz < _in_width || ocl_group_sz1 * n_in_pix_vert < _in_height)
	{
		big = 1;
	}
	int n_procs0 = ocl_group_sz0 / n_ins0;
	int n_procs1 = ocl_group_sz1 / n_ins1;

	int in_sz0 = (n_procs0 * n_out_pix_horiz) * _kernel_stride0/* + kernel_size0 - 2 * pad0*/;
	int in_sz1 = (n_procs1 * n_out_pix_vert) * _kernel_stride1/* + kernel_size1 - 2 * pad1*/;


	int n_out_blocks = ((_n_outputs + n_outs - 1) / n_outs);
	int n_stack_blocks = ((_batch_sz + n_ins - 1) / n_ins);

	// global work size
	gbl0 = n_ins0 * ((n_v_proc0 + n_procs0 - 1) / (n_procs0)) *n_procs0;
	gbl1 = n_ins1 * ((n_v_proc1 + n_procs1 - 1) / (n_procs1)) *n_procs1;
	gbl2 = n_out_blocks * n_stack_blocks;


	int aligned_out = 1;

	if (gbl0 != n_ins0 * n_v_proc0 || gbl1 != n_ins1 * n_v_proc1)
	{
		aligned_out = 0;
	}

	int bias = 1;

	_comp_options =
		std::string("-D ADNN_GRP_SZ=") + std::to_string(static_cast<long long>(ocl_group_sz0) * ocl_group_sz1 * ocl_group_sz2)
		+ std::string(" -D ADNN_GRP_SZ0=") + std::to_string(static_cast<long long>(ocl_group_sz0))
		+ std::string(" -D ADNN_GRP_SZ1=") + std::to_string(static_cast<long long>(ocl_group_sz1))
		+ std::string(" -D ADNN_GRP_SZ2=") + std::to_string(static_cast<long long>(ocl_group_sz2))
		+ std::string(" -D ADNN_LCL_N_IN_CHNLS=") + std::to_string(static_cast<long long>(n_ins))
		+ std::string(" -D ADNN_LCL_N_OUT_CHNLS=") + std::to_string(static_cast<long long>(n_outs))
		+ std::string(" -D ADNN_BATCH_SZ=") + std::to_string(static_cast<long long>(_batch_sz))
		+ std::string(" -D ADNN_FLTR_SZ0=") + std::to_string(static_cast<long long>(_kernel_size0))
		+ std::string(" -D ADNN_FLTR_PAD_SZ0=") + std::to_string(static_cast<long long>(_pad0))
		+ std::string(" -D ADNN_FLTR_STRIDE0=") + std::to_string(static_cast<long long>(_kernel_stride0))
		+ std::string(" -D ADNN_FLTR_SZ1=") + std::to_string(static_cast<long long>(_kernel_size1))
		+ std::string(" -D ADNN_FLTR_PAD_SZ1=") + std::to_string(static_cast<long long>(_pad1))
		+ std::string(" -D ADNN_FLTR_STRIDE1=") + std::to_string(static_cast<long long>(_kernel_stride1))
		+ std::string(" -D ADNN_N_OUT_CHNLS=") + std::to_string(static_cast<long long>(_n_outputs))			//total number of output channels
		+ std::string(" -D ADNN_OUT_WIDTH=") + std::to_string(static_cast<long long>(_out_width))
		+ std::string(" -D ADNN_OUT_HEIGHT=") + std::to_string(static_cast<long long>(_out_height))
		+ std::string(" -D ADNN_OUT_STRIDE=") + std::to_string(static_cast<long long>(_out_stride))
		+ std::string(" -D ADNN_OUT_CHNL_STRIDE=") + std::to_string(static_cast<long long>(_out_channel_stride))
		+ std::string(" -D ADNN_OUT_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_out_batch_stride))
		+ std::string(" -D ADNN_N_OUT_PIX_SZ0=") + std::to_string(static_cast<long long>(n_out_pix_horiz))
		+ std::string(" -D ADNN_N_OUT_PIX_SZ1=") + std::to_string(static_cast<long long>(n_out_pix_vert))
		+ std::string(" -D ADNN_N_IN_CHNLS=") + std::to_string(static_cast<long long>(_n_inputs))
		+ std::string(" -D ADNN_IN_WIDTH=") + std::to_string(static_cast<long long>(_in_width))
		+ std::string(" -D ADNN_IN_HEIGHT=") + std::to_string(static_cast<long long>(_in_height))
		+ std::string(" -D ADNN_IN_STRIDE=") + std::to_string(static_cast<long long>(_in_stride))
		+ std::string(" -D ADNN_IN_CHNL_STRIDE=") + std::to_string(static_cast<long long>(_in_channel_stride))
		+ std::string(" -D ADNN_IN_BATCH_STRIDE=") + std::to_string(static_cast<long long>(_in_batch_stride))
		+ std::string(" -D ADNN_N_IN_PIX_SZ0=") + std::to_string(static_cast<long long>(n_in_pix_horiz))         // size of output processing group in 0 dim
		+ std::string(" -D ADNN_N_IN_PIX_SZ1=") + std::to_string(static_cast<long long>(n_in_pix_vert))         // size of output processing group in 1 dim
		+ std::string(" -D ADNN_WEI_SZ=") + std::to_string(static_cast<long long>(_n_outputs * _n_inputs * _kernel_size0 * _kernel_size1))
		+ std::string(" -D ADNN_WEIGHTS_STRIDE=") + std::to_string(static_cast<long long>(_n_inputs * _kernel_size0 * _kernel_size1))		//	weights stride
		+ std::string(" -D ADNN_N_STACKS=") + std::to_string(static_cast<long long>(n_stack_blocks))          // n of separate data stacks
		+ std::string(" -D ADNN_N_PROCS0=") + std::to_string(static_cast<long long>(n_procs0))         // n of processors per stack
		+ std::string(" -D ADNN_N_PROCS1=") + std::to_string(static_cast<long long>(n_procs1))         // n of processors per stack
		+ std::string(" -D ADNN_ALIGNED=") + std::to_string(static_cast<long long>(aligned_out))		//	dimesions aligned
		+ std::string(" -D ADNN_BATCH_ALIGNED=") + std::to_string(static_cast<long long>(batch_aligned))      // batch is multiple of n_ins
		+ std::string(" -D ADNN_OUT_ALINED=") + std::to_string(static_cast<long long>(out_aligned))        // outputs is multiple of n_outs
		+ std::string(" -D ADNN_IN_SZ0=") + std::to_string(static_cast<long long>(in_sz0))			// horizontal read dim 0
		+ std::string(" -D ADNN_IN_SZ1=") + std::to_string(static_cast<long long>(in_sz1))			// vertical read dim 1

		+ std::string(" -D ADNN_BIG=") + std::to_string(static_cast<long long>(big))		//	resolution > 32 x 32
		+ std::string(" -D ADNN_CONV_BIAS=") + std::to_string(static_cast<long long>(bias))

		+ getGeneralCompOptions()
		;


	_kernel_file = "MlOpenConvDirGenFwd.cl";
	_kernel_name = "MLOpenCDFGen";

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
int mlo_construct_direct2D :: mloSelectDefaultConfig(std::string & conf_val)
{

	//
	int in_tile0 = (_in_width < 12) ? 8 : 16; //(_in_width < 12) ? 8 : (_in_width < 24 || (_in_width > 32 && _in_width < 48)) ? 16 : 32; // size of input data per ALU plane
	int in_tile1 = (_in_height < 12) ? 8 : 16; // (_in_height < 12) ? 8 : (_in_height < 24 || (_in_height > 32 && _in_height < 48)) ? 16 : 32; // size of input data per ALU plane

	int grp_tile0 = in_tile0;
	int grp_tile1 = in_tile1;

	int out_pix_tile0 = 2;  // size of ouptput tile per wk-item (ALU))
	int out_pix_tile1 = 4; //


	int n_out_pix_tiles = 2;  // # output pixel tiles per wk-item (ALU)
	int n_in_data_tiles = 4; // # of blocks of different inputs in LDS

	int n_stacks = 1; // # of diff stacks (part of batch).

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

	mloSetConf(conf_val);

	return(0);
}

/*
 * mesure the current onfiguration pefformance
 */
int mlo_construct_direct2D :: mloMeasuredLoop(cl_command_queue profile_q,
		cl_mem bot_ocl_buf,
		cl_mem top_ocl_buf,
		cl_mem wei_ocl_buf,
		cl_mem bias_ocl_buf,
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
	auto program = mlopen::LoadProgram(mlopen::GetContext(profile_q), mlopen::GetDevice(profile_q), _kernel_file, compiler_options);
	mlopen::OCLKernel kernel{mlopen::CreateKernel(program, _kernel_name), _l_wk, _g_wk};
	// pass all arguments

	float padding_value = 0;
	
	double s= 0, e = 0;
	int iter = 1;

	if (profile_q)
	{
		processing_time = CL_MAXFLOAT;

		auto k = kernel.Invoke(profile_q, [&] (cl_event profile_e) {
			    size_t st, end;
			    clGetEventProfilingInfo(profile_e, CL_PROFILING_COMMAND_START, sizeof(size_t), &st, nullptr);
		        clGetEventProfilingInfo(profile_e, CL_PROFILING_COMMAND_END, sizeof(size_t), &end, nullptr);
		        processing_time = (end-st)*1e-6;
		});

		if(_bias) {
			k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
		} else {
			k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
}
	}
	else
	{
		iter = (_n_timer_iter <= 0) ? 1 : _n_timer_iter;

		cl_command_queue q = reinterpret_cast<cl_command_queue>(_stream);
		auto k = kernel.Invoke(q);

		if(_bias) {
			k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
		} else {
			k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
}

		clFinish(q);

		s = mlopen_mach_absolute_time();

		for (int i = 0; i < iter && ret == 0; i++)
		{
			if(_bias) {
				k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
			} else {
				k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
}
		}

		clFinish(q);
		e = mlopen_mach_absolute_time();

		processing_time = subtractTimes(e, s) / iter;
	}

	return (ret);
}


/*
 * request cofiguraion db management
 * request configuration db is a text file
 * each line is a key (in cofig db format) that has not been found in teh configuratio db
 */


int mlo_construct_direct2D :: mloAddConfigReq(cl_device_id dev, const std::string & conf_key) const
{
	int ret = 0;
	std::vector<std::string> req_conf_db;
	std::string conf_file = (_kernel_path == "") ? mlopen::GetDbPath() : _kernel_path;

	conf_file += std::string("/") + mloConfFileBaseNm(dev) + "." + std::string("cd.rdb.txt");

	printf("file %s\n", conf_file.c_str());
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
		cl_device_id dev,
		const std::string & conf_key
		) const
{
	int ret = 0;
	std::vector<std::string> req_conf_db;

	std::vector<std::string>::iterator it;

	std::string conf_file = (_kernel_path == "") ? mlopen::GetDbPath() : _kernel_path;
	conf_file += std::string("/") + mloConfFileBaseNm(dev) + "." + std::string("cd.rdb.txt");

	bool found = mloFindConfigReq(conf_file, conf_key, req_conf_db, it);


	if (found)
	{
		req_conf_db.erase(it);
		ret = mloUpdateDb(conf_file, req_conf_db);
	}
	return(ret);
}

int mlo_construct_direct2D :: mloReadConfigDB(
		cl_device_id dev,
		std::map<std::string, std::string> & conf_db
		) const
{

	int ret = 0;
	std::string conf_file = (_kernel_path == "") ? mlopen::GetDbPath() : _kernel_path;

	conf_file += std::string("/") + mloConfFileBaseNm(dev) + "." + std::string("cd.pdb.txt");

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
		cl_device_id dev,
		const std::map<std::string, std::string> & conf_db
		) const
{

	int ret = 0;
	//serialize
	std::string conf_file = (_kernel_path == "") ? mlopen::GetDbPath() : _kernel_path;

	conf_file += std::string("/") + mloConfFileBaseNm(dev) + "." + std::string("cd.pdb.txt");

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
		cl_device_id dev,
		std::string & conf_key,
		std::string & conf_val
		) const
{
	int ret = 0;

	// build searchable db
	std::map<std::string, std::string> conf_db;

	ret = mloReadConfigDB(
			dev,
			conf_db
			);
	// add config

	conf_db[conf_key] = conf_val;
	//serialize
	ret = mloWriteConfigDB(
			dev,
			conf_db
			);

	// remove request
	mloRemoveConfigReq(
			dev,
			conf_key
			);

	return(ret);
}





bool mlo_construct_direct2D :: mloSearchConfigInDB(
		cl_device_id dev,
		std::string & conf_key,
		std::string & conf_val
		) const
{
	bool known_config = false;
	// build searchable db
	std::map<std::string, std::string> conf_db;

	mloReadConfigDB(
			dev,
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

	// get device id
	cl_device_id dev = mlopen::GetDevice(reinterpret_cast<cl_command_queue>(_stream));

	// find a db and configuration in it
	known_config = mloSearchConfigInDB(
			dev,
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
			mloAddConfigReq(dev, conf_key);
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

	mlopen::ClAqPtr profile_q;
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

	cl_context ctxt = mlopen::GetContext(reinterpret_cast<cl_command_queue>(_stream));
	cl_device_id dev = mlopen::GetDevice(reinterpret_cast<cl_command_queue>(_stream));
	profile_q = mlopen::CreateQueueWithProfiling(ctxt, dev);

	size_t localMemSize = mlopen::GetDeviceInfo<CL_DEVICE_LOCAL_MEM_SIZE>(dev);

	_hw_wave_sz = 64;
	_dev_local_mem_sz = localMemSize; // in bytes

	// if it is not known
	bool known_config = mloSearchConfigInDB(
			dev,
			conf_key,
			conf_val
			);

	// proceed
	if (!known_config)
	{

		// allocate tem input/output buffers
		size_t bot_sz = _bot_sz / sizeof(float);
		auto * bot_sys_buf = new float[bot_sz];
		assert(bot_sys_buf);

		for (int i = 0; i < bot_sz; i++) {
			bot_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
		}

		cl_mem bot_ocl_buf = clCreateBuffer(ctxt, CL_MEM_COPY_HOST_PTR, _bot_sz, bot_sys_buf, &ret);

		assert(bot_ocl_buf);

		size_t top_sz = _top_sz / sizeof(float);
		auto * top_sys_buf = new float[top_sz];
		assert(top_sys_buf);

		cl_mem top_ocl_buf = clCreateBuffer(ctxt, CL_MEM_COPY_HOST_PTR, _top_sz, top_sys_buf, &ret);
		assert(top_ocl_buf);

		size_t weights_sz = _weights_sz / sizeof(float);
		auto * wei_sys_buf = new float[weights_sz];
		assert(wei_sys_buf);
		for (int i = 0; i < weights_sz; i++) {
			wei_sys_buf[i] = static_cast<float>((rand() * (1.0 / RAND_MAX) - 0.5) * 0.001);
		}

		cl_mem wei_ocl_buf = clCreateBuffer(ctxt, CL_MEM_COPY_HOST_PTR, _weights_sz, wei_sys_buf, &ret);
		assert(wei_ocl_buf);

		float * bias_sys_buf = nullptr;
		cl_mem bias_ocl_buf = nullptr;

		if (_bias)
		{
			size_t bias_sz = _bias_sz / sizeof(float);
			bias_sys_buf = new float[_bias_sz / sizeof(float)];
			assert(bias_sys_buf);
			for (int i = 0; i < bias_sz; i++) {
				bias_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
			}

			bias_ocl_buf = clCreateBuffer(ctxt, CL_MEM_COPY_HOST_PTR, _bias_sz, bias_sys_buf, &ret);
			assert(bias_ocl_buf);
		}


		// search loop here
		//		int grp_sz[3] = { 64, 128, 256 };
		int grp_tl_ln[2] = { 8, 16 };
		int tile_sz[3] = { 8, 16, 32 };
		int out_pix_tile_sz[4] = { 1, 2, 4, 8 };
		int n_out_tiles_rg[2] = { 1, 12 };
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

		double min_proc_time = CL_MAXFLOAT;

#if 1
		std::cout << "Searching the best solution in the 9 dim space. Please, be patient it may take few minutes." << std::endl;

		size_t run_counter = 0;
		int n_grp_tiles0 = (_out_height >= 32) ? 1 : 2;
		int n_grp_tiles1 = (_out_width >= 32) ? 1 : 2;
		int out_pix_tl_cnt = (_kernel_size0 != 3 || _kernel_size1 != 3) ? 3 : 4;
		int n_out_tls = (_kernel_size0 != 3 || _kernel_size1 != 3) ? 6 : n_out_tiles_rg[1];
		n_out_tls = std::min(_n_outputs, n_out_tls);
		int n_tile0_sz = (_out_width * 2 <= 16) ? 1 : (_out_width * 2 <= 32) ? 2 : 3;
		int n_tile1_sz = (_out_height * 2 <= 16) ? 1 : (_out_height * 2 <= 32) ? 2 : 3;
        long long runs_left = n_grp_tiles0 * n_grp_tiles1 * n_tile0_sz * n_tile1_sz * out_pix_tl_cnt * out_pix_tl_cnt * n_out_tls * n_in_tiles_rg[1] * 3;

		size_t report_inteval = 25;
		//			_n_timer_iter = 250;

		for (int g1 = 0; g1 < 2; g1++)
		{
			_grp_tile1 = grp_tl_ln[g1];
			if (_out_height >= 32 && _grp_tile1 == 8)
			{
				continue;
			}

			for (int g0 = 0; g0 < 2; ++g0)
			{
				_grp_tile0 = grp_tl_ln[g0];

				if (_out_width >= 32 && _grp_tile0 == 8)
				{
					continue;
				}
				// tile1
				for (int j = 0; j < 3; ++j)
				{
					_in_tile1 = tile_sz[j];
					if (_out_height * 2 <= _in_tile1 && _in_tile1 > tile_sz[0])
					{
						runs_left--;
						runs_left = (runs_left < 0) ? 0 : runs_left;
						continue;
					}
					// tile 0
					for (int i = 0; i < 3; i++) {
						_in_tile0 = tile_sz[i];
						if (_out_width * 2 <= _in_tile0 &&  _in_tile0 > tile_sz[0])
						{
							runs_left--;
							runs_left = (runs_left < 0) ? 0 : runs_left;
							continue;
						}
						// out pix 1

						int k_l = (_kernel_size1 == 3) ? 4 : 3;
						for (int k = 0; k < k_l; ++k)
						{
							_out_pix_tile1 = out_pix_tile_sz[k];
							if (_out_pix_tile1 > _in_tile1)
							{
								runs_left--;
								runs_left = (runs_left < 0) ? 0 : runs_left;
								continue;
							}
							// out pix 0
							int l_l = (_kernel_size0 == 3) ? 4 : 3;
							for (int l = 0; l < l_l; ++l)
							{
								_out_pix_tile0 = out_pix_tile_sz[l];

								if (_out_pix_tile0 > _in_tile0)
								{
									runs_left--;
									runs_left = (runs_left < 0) ? 0 : runs_left;
									continue;
								}

								int o_l = (_kernel_size0 != 3 || _kernel_size1 != 3) ? 6 : n_out_tiles_rg[1];
								for (int o_t = n_out_tiles_rg[0]; o_t <= o_l; ++o_t)
								{

									if ((_out_pix_tile1 == 8 || _out_pix_tile0 == 8) && o_t > 4)
									{
										runs_left--;
										runs_left = (runs_left < 0) ? 0 : runs_left;
										continue;
									}

									_n_out_pix_tiles = o_t;
									if (_n_outputs < _n_out_pix_tiles)
									{
										runs_left--;
										runs_left = (runs_left < 0) ? 0 : runs_left;
										continue;
									}

									for (int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1]; ++i_t)
									{
										_n_in_data_tiles = i_t;
										if (_n_inputs < _n_in_data_tiles)
										{
											runs_left--;
											runs_left = (runs_left < 0) ? 0 : runs_left;
											continue;
										}

										for (int s = 0; s < 3; ++s)
										{

											_n_stacks = n_in_stacks_sz[s];
#if 1
											if ((_in_tile1 > 16 || _in_tile0 > 16)
													&& i_t > 4
													&& _n_stacks > 2)

											{
												runs_left--;
												runs_left = (runs_left < 0) ? 0 : runs_left;

												continue;
											}

#endif
											ret = mloMeasuredLoop(profile_q.get(),
													bot_ocl_buf,
													top_ocl_buf,
													wei_ocl_buf,
													bias_ocl_buf,
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

		ret = clReleaseMemObject(bot_ocl_buf);
		ret = clReleaseMemObject(top_ocl_buf);
		ret = clReleaseMemObject(wei_ocl_buf);
		if (_bias)
		{
			ret = clReleaseMemObject(bias_ocl_buf);
			delete[] bias_sys_buf;
		}

		delete[] bot_sys_buf;
		delete[] top_sys_buf;
		delete[] wei_sys_buf;

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
				dev,
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
