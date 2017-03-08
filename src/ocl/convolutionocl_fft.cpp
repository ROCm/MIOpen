#include <mlopen/convolution.hpp>
#include <mlopen/convolution_fft.hpp>
#include <mlopen/util.hpp>

namespace mlopen {

int ConvolutionDescriptor::FindFwdFFTKernel(Handle& handle,
		const TensorDescriptor&			xDesc,
		const TensorDescriptor&			wDesc,
		const TensorDescriptor&			yDesc,
        std::vector<KernelInvoke>&      kernels) const {


	if(ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc) == 0)
		return -1;

	int in_c;
	std::tie(std::ignore, in_c, std::ignore, std::ignore) = mlopen::tie4(xDesc.GetLengths());

	int out_n, out_c;
	std::tie(out_n, out_c, std::ignore, std::ignore) = mlopen::tie4(yDesc.GetLengths());

	const int N = FFTConvParams::N;


	const int FFT_NUM_KERNELS = 7;
	size_t global_work_size[FFT_NUM_KERNELS][3];
	size_t  local_work_size[FFT_NUM_KERNELS][3];

	for(int ik=0; ik<FFT_NUM_KERNELS; ik++)
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
		const unsigned int workDim = 3;
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


    std::string algorithm = "mlopenConvolutionFwdAlgoFFT";
    std::string program_name = "MLOpenConvFFT.cl";
    std::string parms = ""; 
	for(int ik=0; ik<FFT_NUM_KERNELS; ik++)
	{
		std::string kernel_name = ""; 

		switch(ik)
		{
		case 0: kernel_name += "fft_fwd_in";		break;
		case 1: kernel_name += "fft_fwd_we";		break;
		case 2: kernel_name += "fft_transpose1";	break;
		case 3: kernel_name += "fft_transpose2";	break;
		case 4: kernel_name += "cgemm";				break;
		case 5: kernel_name += "fft_transpose3";	break;
		case 6: kernel_name += "fft_back";			break;
		}
		
		std::string network_config = "FFT_"; network_config += std::to_string(ik);

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


void ConvolutionDescriptor::ExecuteFwdFFTKernel(Handle& handle,
		const TensorDescriptor&			xDesc,
		ConstData_t						x,
		const TensorDescriptor&			wDesc,
		ConstData_t						w,
		const TensorDescriptor&			yDesc,
		Data_t							y,
		Data_t							workSpace,
		size_t							workSpaceSize,
		std::vector<KernelInvoke>&      kernels,
		float&							timev) const {


	int in_n, in_c;
	std::tie(in_n, in_c, std::ignore, std::ignore) = mlopen::tie4(xDesc.GetLengths());

	int out_n, out_c;
	std::tie(out_n, out_c, std::ignore, std::ignore) = mlopen::tie4(yDesc.GetLengths());

	const int N = FFTConvParams::N;
	const int Padding = FFTConvParams::TransposePadding;

	int ik = 0;
    for(auto &k : kernels) {

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

		timev += handle.GetKernelTime();
		ik++;
    }
	
}


}  // namespace mlopen
