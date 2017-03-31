#include <mlopen/convolution.hpp>
#include <mlopen/convolution_fft.hpp>
#include <mlopen/util.hpp>

namespace mlopen {

static std::string make_config_prefix(int in_n, int out_c)
{
	std::string config_prefix = "FFT_x";
	config_prefix += "_in_n_";
	config_prefix += std::to_string(in_n);
	config_prefix += "_out_c_";
	config_prefix += std::to_string(out_c);
	config_prefix += "_kernel_";

	return config_prefix;
}

int ConvolutionDescriptor::FindFwdFFTKernel(Handle& handle,
		const TensorDescriptor&			xDesc,
		const TensorDescriptor&			wDesc,
		const TensorDescriptor&			yDesc,
        std::vector<KernelInvoke>&      kernels) const {

	size_t wSize = ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc);
	if(wSize == 0)
		return -1;

	int in_n, in_c;
	std::tie(in_n, in_c, std::ignore, std::ignore) = mlopen::tie4(xDesc.GetLengths());

	int out_n, out_c;
	std::tie(out_n, out_c, std::ignore, std::ignore) = mlopen::tie4(yDesc.GetLengths());

	const int N = FFTConvParams::N;
	const int NumKernels = 	FFTConvParams::NumKernels;

	size_t global_work_size[NumKernels][3];
	size_t  local_work_size[NumKernels][3];

	for(int ik=0; ik<NumKernels; ik++)
	{
		global_work_size[ik][0] = local_work_size[ik][0] = 1;
		global_work_size[ik][1] = local_work_size[ik][1] = 1;
		global_work_size[ik][2] = local_work_size[ik][2] = 1;
	}

	 local_work_size[0][0] = 64;
	global_work_size[0][0] = in_c * out_n * local_work_size[0][0];

	 local_work_size[1][0] = 64;
	global_work_size[1][0] = in_c * out_c * local_work_size[1][0];

	 local_work_size[2][0] = 256;
	global_work_size[2][0] = (1 + N / 64) * (in_c*out_n / 64) * local_work_size[2][0];

	 local_work_size[3][0] = 256;
	global_work_size[3][0] = (1 + N / 64) * (in_c*out_c / 64) * local_work_size[3][0];

	//cgemm
	{
		/* grid sizes */
		const unsigned int threadTile[2] = { 4, 4 };

		local_work_size[4][0] = 16;
		local_work_size[4][1] = 16;

		global_work_size[4][2] = 1;
		global_work_size[4][2] *= N;


		unsigned int sizeOfC0 = out_c;
		unsigned int sizeOfC1 = out_n;
		unsigned int macroTile0 = static_cast<unsigned int>(local_work_size[4][0] * threadTile[0]);
		unsigned int macroTile1 = static_cast<unsigned int>(local_work_size[4][1] * threadTile[1]);
		unsigned int totalWorkGroups0 = sizeOfC0 / macroTile0;
		unsigned int totalWorkGroups1 = sizeOfC1 / macroTile1;
		// b/c single kernel, add extra work-group here if edge needed
		if (totalWorkGroups0*macroTile0 < sizeOfC0) { totalWorkGroups0++; }
		if (totalWorkGroups1*macroTile1 < sizeOfC1) { totalWorkGroups1++; }
		global_work_size[4][0] = totalWorkGroups0*local_work_size[4][0];
		global_work_size[4][1] = totalWorkGroups1*local_work_size[4][1];
	}
	
	 local_work_size[5][0] = 256;
	global_work_size[5][0] = (1 + N / 64) * (out_n*out_c / 64) * local_work_size[5][0];

	 local_work_size[6][0] = 64;
	global_work_size[6][0] = out_n * out_c * local_work_size[6][0];


    const std::string algorithm = "mlopenConvolutionFwdAlgoFFT";
    const std::string program_name = "MLOpenConvFFT.cl";

	std::string parms = "";
	parms += " -D CFF_BATCH=";
	parms += std::to_string(in_n);
	parms += " -D CFF_NFILTER=";
	parms += std::to_string(out_c);
	parms += " -D CFF_CHANNELS=";
	parms += std::to_string(in_c);
	parms += " -D CFF_HALFW=";
	parms += std::to_string(wSize/(2*2*sizeof(float)));

	const std::string config_prefix = make_config_prefix(in_n, out_c);

	for(int ik=0; ik<NumKernels; ik++)
	{
		std::string kernel_name = ""; 

		switch(ik)
		{
		case 0: kernel_name += "MLOpenConvFFT_fwd_in";			break;
		case 1: kernel_name += "MLOpenConvFFT_fwd_we";			break;
		case 2: kernel_name += "MLOpenConvFFT_transpose_in";	break;
		case 3: kernel_name += "MLOpenConvFFT_transpose_we";	break;
		case 4: kernel_name += "MLOpenConvFFT_cgemm";			break;
		case 5: kernel_name += "MLOpenConvFFT_transpose_out";	break;
		case 6: kernel_name += "MLOpenConvFFT_inv_out";			break;
		}
		
		std::string network_config = config_prefix + std::to_string(ik);

		std::vector<size_t> vld(3);
		std::vector<size_t> vgd(3);

		vld[0] =  local_work_size[ik][0];
		vld[1] =  local_work_size[ik][1];
		vld[2] =  local_work_size[ik][2];

		vgd[0] = global_work_size[ik][0];
		vgd[1] = global_work_size[ik][1];
		vgd[2] = global_work_size[ik][2];

		auto k = handle.GetKernel(algorithm,
							network_config,
							program_name,
							kernel_name,
							vld,
							vgd,
							parms);

		kernels.push_back(k);
	}

	return 0;
}


float ConvolutionDescriptor::ExecuteFwdFFTKernel(Handle& handle,
		const TensorDescriptor&			xDesc,
		ConstData_t						x,
		const TensorDescriptor&			wDesc,
		ConstData_t						w,
		const TensorDescriptor&			yDesc,
		Data_t							y,
		Data_t							workSpace,
		size_t							workSpaceSize,
		bool							timed) const {


	(void)wDesc; // suppress warning
	(void)workSpaceSize; // suppress warning

	int in_n, in_c;
	std::tie(in_n, in_c, std::ignore, std::ignore) = mlopen::tie4(xDesc.GetLengths());

	int out_n, out_c;
	std::tie(out_n, out_c, std::ignore, std::ignore) = mlopen::tie4(yDesc.GetLengths());

	const int N = FFTConvParams::N;
	const int Padding = FFTConvParams::TransposePadding;
	const int NumKernels = 	FFTConvParams::NumKernels;

	float time_fft = 0;
	const std::string config_prefix = make_config_prefix(in_n, out_c);
	for(int ik=0; ik<NumKernels; ik++)
	{
		std::string network_config = config_prefix + std::to_string(ik);

		auto k = handle.GetKernel("mlopenConvolutionFwdAlgoFFT", network_config);

		switch(ik)
		{
		case 0:	k(x, workSpace); break;
		case 1: k(w, workSpace); break;
		case 2: k(workSpace); break;
		case 3: k(workSpace); break;
		case 4:
			{
				k(
					workSpace,
					0,
					N*(out_n*out_c + Padding) + N*(in_n*in_c + Padding),
					N*(out_n*out_c + Padding) + 0,
					out_c,
					out_n*out_c + Padding,
					in_c,
					in_c*out_c + Padding,
					in_c,
					in_n*in_c + Padding,
					out_c,
					in_n,
					N,
					in_c
				);
			}
			break;
		case 5: k(workSpace); break;
		case 6: k(workSpace, y); break;
		}

		if(timed)
		{
			time_fft += handle.GetKernelTime();
		}
    }
	
	return time_fft;
}


}  // namespace mlopen
