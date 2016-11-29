#include <mlopen/softmax.hpp>
#include <mlopen/kernel_cache.hpp>

namespace mlopen {

int nextPow2(int v) {
	
	if(v == 1) {
		return (v << 1);
	}
	else {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
	}
}

mlopenStatus_t SoftmaxForward(
		Handle						&handle,
		const void					* /*alpha*/,
		const void					* /*beta*/,
		const TensorDescriptor		&yDesc,
		Data_t						y) 
{
	int n, c, h, w;
	std::tie(n, c, h, w) = tie4(yDesc.GetLengths());
	
	std::string program_name = "MLOpenSoftmax.cl";
	std::string kernel_name = "SoftmaxForward";

	// using workgroup size of 256 by default
	int grid_size = n*h*w;
	int spatial_dim = h*w;
	int num_batch = c < 256 ? nextPow2(256/c) : 1;

	const std::vector<size_t> vld(1, 256);

	// compile parameters
	std::string parms = "-DNUM_BATCH=" + std::to_string(num_batch);

	// See Kernels/MLOpenSoftmax.cl for description
	if(num_batch == 1) { // CSR-Vector like approach

		// Control the max. number of workgroups launched so that we do not
		// start getting workgroup scheduling overheads
		size_t workgroups = std::min(grid_size, 64*40*8);
		const std::vector<size_t> vgd(1, workgroups*vld[0]);

		handle.GetKernel("mlopenSoftmaxForward",
				"",
				program_name,
				kernel_name,
				vld,
				vgd,
				parms)(y, c, grid_size, spatial_dim);
	}
	else { // CSR-Stream like approach

		int batch_size = 256/num_batch;
		int u_batch_size = c > batch_size ? nextPow2(c/batch_size) : 1;

		size_t workgroups = grid_size % num_batch == 0 ? grid_size/num_batch : grid_size/num_batch + 1;
		const std::vector<size_t> vgd(1, workgroups*vld[0]);

		parms += " -DBATCH_SIZE=" + std::to_string(batch_size) + 
			" -DU_BATCH_SIZE=" + std::to_string(u_batch_size);

		handle.GetKernel("mlopenSoftmaxForward",
				"",
				program_name,
				kernel_name,
				vld,
				vgd,
				parms)(y, c, grid_size, spatial_dim);

	}
	return mlopenStatusSuccess;
}

mlopenStatus_t SoftmaxBackward(
		Handle						&handle,
		const void					* /*alpha*/,
		const TensorDescriptor		&yDesc,
		ConstData_t				y,
		const void					* /*beta*/,
		const TensorDescriptor		&dxDesc,
		Data_t						dx) 
{
	if(yDesc != dxDesc) {
		MLOPEN_THROW(mlopenStatusBadParm);
	}
	int n, c, h, w;
	std::tie(n, c, h, w) = tie4(dxDesc.GetLengths());
	
	std::string program_name = "MLOpenSoftmax.cl";
	std::string kernel_name = "SoftmaxBackward";

	// using workgroup size of 256 by default
	int grid_size = n*h*w;
	int spatial_dim = h*w;
	int num_batch = c < 256 ? nextPow2(256/c) : 1;

	const std::vector<size_t> vld(1, 256);

	// compile parameters
	std::string parms = "-DNUM_BATCH=" + std::to_string(num_batch);

	// See Kernels/MLOpenSoftmax.cl for description
	if(num_batch == 1) { // CSR-Vector like approach

		// Control the max. number of workgroups launched so that we do not
		// start getting workgroup scheduling overheads
		size_t workgroups = std::min(grid_size, 64*40*8);
		const std::vector<size_t> vgd(1, workgroups*vld[0]);

		handle.GetKernel("mlopenSoftmaxBackward",
				"",
				program_name,
				kernel_name,
				vld,
				vgd,
				parms)(y, dx, c, grid_size, spatial_dim);
	}
	else { // CSR-Stream like approach

		int batch_size = 256/num_batch;
		int u_batch_size = c > batch_size ? nextPow2(c/batch_size) : 1;

		size_t workgroups = grid_size % num_batch == 0 ? grid_size/num_batch : grid_size/num_batch + 1;
		const std::vector<size_t> vgd(1, workgroups*vld[0]);

		parms += " -DBATCH_SIZE=" + std::to_string(batch_size) + 
			" -DU_BATCH_SIZE=" + std::to_string(u_batch_size);

		handle.GetKernel("mlopenSoftmaxBackward",
				"",
				program_name,
				kernel_name,
				vld,
				vgd,
				parms)(y, dx, c, grid_size, spatial_dim);

	}

	return mlopenStatusSuccess;
}

} // namespace mlopen
