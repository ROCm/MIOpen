#include <mlopen/util.hpp>
#include <mlopen/kernel_cache.hpp>

namespace mlopen {

mlopenStatus_t Im2ColGPU(
		Handle					&handle,
		const TensorDescriptor&	imDesc,
		const cl_mem			im,
		const TensorDescriptor&	wDesc,
		const int				pad_h,
		const int				pad_w,
		const int				stride_h,
		const int				stride_w,
		cl_mem					col) 
{
	int n, c, h, w;
	std::tie(n, c, h, w) = tie4(imDesc.GetLengths());

	int wei_h, wei_w;
	std::tie(std::ignore, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

	int out_h = (h - wei_h + 2*pad_h)/stride_h + 1;
	int out_w = (w - wei_w + 2*pad_w)/stride_w + 1;

	std::string program_name = "MLOpenUtilKernels.cl";
	std::string kernel_name = "Im2Col";
	std::string network = "placeholder";

	int col_m = c * wei_h * wei_w;
	int grid_size = col_m * out_h * out_w;

	const std::vector<size_t> vld(1, 256);
	const std::vector<size_t> vgd(1, grid_size);

	handle.GetKernel("mlopenIm2Col",
			network,
			program_name,
			kernel_name,
			vld,
			vgd,
			"")(im, h, w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, stride_h, stride_w, col);

	return mlopenStatusSuccess;
}

} // namespace mlopen
