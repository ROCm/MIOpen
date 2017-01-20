#include <mlopen/util.hpp>
#include <mlopen/kernel_cache.hpp>
#include <cmath>

namespace mlopen {

mlopenStatus_t Im2ColGPU(
		Handle	&handle,
		ConstData_t im, size_t im_offset,
		const int c, const int h, const int w,
		const int wei_h, const int wei_w,
		const int out_h, const int out_w,
		const int pad_h, const int	pad_w,
		const int stride_h, const int stride_w,
		Data_t col) 
{
	std::string program_name = "MLOpenUtilKernels.cl";
	std::string kernel_name = "Im2Col";
	std::string network = "placeholder";

//	int col_m = c * wei_h * wei_w;
//	int grid_size = col_m * out_h * out_w;

	std::string params;
	int num_ch_per_wg;
	if((out_h <= 8 && out_w <= 8) && (stride_h == 1 && stride_w==1))
		num_ch_per_wg = 4;
	else 
		num_ch_per_wg = 1;

	//int num_blks_x = std::ceil(static_cast<float>(out_w)/16);
	//int num_blks = num_blks_x * std::ceil(static_cast<float>(out_h)/16);
	//int local_mem_sz = (16*stride_w+wei_w)*(16*stride_h+wei_h);

	int tile_sz_x = 32;
	int tile_sz_y = 8;
	int num_blks_x = std::ceil(static_cast<float>(out_w)/tile_sz_x);
	int num_blks = num_blks_x * std::ceil(static_cast<float>(out_h)/tile_sz_y);
	int local_mem_sz = (tile_sz_x*stride_w+wei_w)*(tile_sz_y*stride_h+wei_h);

	params += " -DNUM_CH_PER_WG=" + std::to_string(num_ch_per_wg);
	params += " -DNUM_IM_BLKS_X=" + std::to_string(num_blks_x);
	params += " -DNUM_IM_BLKS=" + std::to_string(num_blks);
	params += " -DLOCAL_MEM_SIZE=" + std::to_string(local_mem_sz);
	params += " -DSTRIDE_GT_1=" + std::to_string(stride_h*stride_w > 1);
	params += " -DTILE_SZ_X=" + std::to_string(tile_sz_x);
	params += " -DTILE_SZ_Y=" + std::to_string(tile_sz_y);

	const std::vector<size_t> vld(1, 256);
//	const std::vector<size_t> vgd(1, grid_size);
	const std::vector<size_t> vgd(1, 256*(c/num_ch_per_wg)*num_blks);
//	const std::vector<size_t> vgd(1, 256*(c/num_ch_per_wg));


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
