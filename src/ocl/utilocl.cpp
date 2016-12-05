#include <mlopen/util.hpp>
#include <mlopen/kernel_cache.hpp>
#include <cmath>

namespace mlopen {

mlopenStatus_t Im2ColGPU(
		Handle	&handle,
		const cl_mem im, size_t im_offset,
		const int c, const int h, const int w,
		const int wei_h, const int wei_w,
		const int out_h, const int out_w,
		const int pad_h, const int	pad_w,
		const int stride_h, const int stride_w,
		cl_mem col) 
{
	std::string program_name = "MLOpenUtilKernels.cl";
	std::string kernel_name = "Im2Col";
	std::string network = "placeholder";

//	int col_m = c * wei_h * wei_w;
//	int grid_size = col_m * out_h * out_w;

	std::string params;
	int num_ch_per_wg;
	if(out_h <= 8 && out_w <= 8)
		num_ch_per_wg = 4;
	else 
		num_ch_per_wg = 1;
	
	int num_blks_x = std::ceil(static_cast<float>(out_w)/16);
	int num_blks = std::ceil(static_cast<float>(out_w)/16) * std::ceil(static_cast<float>(out_h)/16);

	params += " -DNUM_CH_PER_WG=" + std::to_string(num_ch_per_wg);
	params += " -DNUM_IM_BLKS_X=" + std::to_string(num_blks_x);
	params += " -DNUM_IM_BLKS=" + std::to_string(num_blks);
	params += " -DLOCAL_MEM_SIZE=" + std::to_string((16+2*pad_w)*(16+2*pad_h));

	const std::vector<size_t> vld(1, 256);
//	const std::vector<size_t> vgd(1, grid_size);
	const std::vector<size_t> vgd(1, 256*(c/num_ch_per_wg)*num_blks);

	handle.GetKernel("mlopenIm2Col",
			network,
			program_name,
			kernel_name,
			vld,
			vgd,
			params)(im, im_offset, h, w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, stride_h, stride_w, col);

	return mlopenStatusSuccess;
}

} // namespace mlopen
