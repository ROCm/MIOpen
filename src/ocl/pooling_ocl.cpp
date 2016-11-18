#include <mlopen/pooling.hpp>
#include <mlopen/mlo_internal.hpp>
#include <mlopen/kernel_cache.hpp>

namespace mlopen {

std::size_t PoolingDescriptor::GetWorkSpaceSize(const TensorDescriptor& tensorDesc) const
{
	return tensorDesc.GetElementSize() * sizeof(int16_t);
}

mlopenStatus_t PoolingDescriptor::Forward(
		Handle								&handle,
		const void							* /*alpha*/,
		const TensorDescriptor				&xDesc,
		const cl_mem						x,
		const void							* /*beta*/,
		const TensorDescriptor				&yDesc,
		cl_mem								y,
		bool								do_backward,
		cl_mem								workSpace,
		size_t								 /*workSpaceSize*/) const {

	mlo_construct_pooling2D construct_params(1); // forward

	construct_params.setStream(handle.GetStream());

	int nOut;
	int cOut;
	int hOut;
	int wOut;
	int nOutStride;
	int cOutStride;
	int hOutStride;
	int wOutStride;

	std::tie(nOut, cOut, hOut, wOut) = tie4(yDesc.GetLengths());
	std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tie4(yDesc.GetStrides());


	construct_params.setTopDescr(
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
	int nIn;
	int cIn;
	int hIn;
	int wIn;
	int nInStride;
	int cInStride;
	int hInStride;
	int wInStride;

	std::tie(nIn, cIn, hIn, wIn) = tie4(xDesc.GetLengths());
	std::tie(nInStride, cInStride, hInStride, wInStride) = tie4(xDesc.GetStrides());

	if (((hIn * wIn) > std::numeric_limits<uint16_t>::max()) && do_backward) {
		MLOPEN_THROW("Height and width to large to do backwards");
	}

	construct_params.setBotDescr(
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

	if (mode == mlopenPoolingMax && do_backward && workSpace == nullptr)
	{
		throw std::invalid_argument("workSpace cannot be NULL in Forward Pooling MAX mode when backward pass is requested");
	}
	int pooling_method = (mode == mlopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;
	construct_params.setPoolingDescr(pooling_method, lens[0], lens[1], pads[0], pads[1], strides[0], strides[1]);

	construct_params.doBackward(do_backward);

	construct_params.mloConstruct();

	std::string program_name = construct_params.getKernelFile();  // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	handle.GetKernel("mlopenPooling2dDForward",
		"",
		program_name,
		kernel_name,
		vld,
		vgd,
		parms)(x, y, workSpace);

	return mlopenStatusSuccess;
}

mlopenStatus_t PoolingDescriptor::Backward(
		Handle								&handle,
		const void							* /*alpha*/,
		const TensorDescriptor				&yDesc,
		const cl_mem						/*y*/,
		const TensorDescriptor				&dyDesc,
		const cl_mem						dy,
		const TensorDescriptor				&xDesc,
		const cl_mem						/*x*/,
		const void							* /*beta*/,
		const TensorDescriptor				&dxDesc,
		cl_mem								dx,
		const cl_mem						workSpace) const {


	mlopenStatus_t status = mlopenStatusSuccess;
	mlo_construct_pooling2D construct_params(0); // backward

	construct_params.setStream(handle.GetStream());

	int ndOut;
	int cdOut;
	int hdOut;
	int wdOut;
	int ndOutStride;
	int cdOutStride;
	int hdOutStride;
	int wdOutStride;

	std::tie(ndOut, cdOut, hdOut, wdOut) = tie4(dyDesc.GetLengths());
	std::tie(ndOutStride, cdOutStride, hdOutStride, wdOutStride) = tie4(dyDesc.GetStrides());

	construct_params.setTopDfDescr(
			"NCHW",
			"FP32",
			ndOut,
			cdOut,
			hdOut,
			wdOut,
			ndOutStride,
			cdOutStride,
			hdOutStride,
			wdOutStride);

	int nOut;
	int cOut;
	int hOut;
	int wOut;
	int nOutStride;
	int cOutStride;
	int hOutStride;
	int wOutStride;

	std::tie(nOut, cOut, hOut, wOut) = tie4(yDesc.GetLengths());
	std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tie4(yDesc.GetStrides());

	construct_params.setTopDescr(
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

	int ndIn;
	int cdIn;
	int hdIn;
	int wdIn;
	int ndInStride;
	int cdInStride;
	int hdInStride;
	int wdInStride;

	std::tie(ndIn, cdIn, hdIn, wdIn) = tie4(dxDesc.GetLengths());
	std::tie(ndInStride, cdInStride, hdInStride, wdInStride) = tie4(dxDesc.GetStrides());

	construct_params.setBotDfDescr(
			"NCHW",
			"FP32",
			ndIn,
			cdIn,
			hdIn,
			wdIn,
			ndInStride,
			cdInStride,
			hdInStride,
			wdInStride);

	int nIn;
	int cIn;
	int hIn;
	int wIn;
	int nInStride;
	int cInStride;
	int hInStride;
	int wInStride;

	std::tie(nIn, cIn, hIn, wIn) = tie4(xDesc.GetLengths());
	std::tie(nInStride, cInStride, hInStride, wInStride) = tie4(xDesc.GetStrides());

	if (((hIn * wIn) > std::numeric_limits<uint16_t>::max())) {
		MLOPEN_THROW("Height and width to large to do backwards");
	}

	construct_params.setBotDescr(
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

	if (mode == mlopenPoolingMax && workSpace == nullptr)
	{
		throw std::invalid_argument("workSpace cannot be NULL in Backward Pooling MAX mode");
	}
	int pooling_method = (mode == mlopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;
	construct_params.setPoolingDescr(pooling_method, lens[0], lens[1], pads[0], pads[1], strides[0], strides[1]);

	status = static_cast<mlopenStatus_t>(construct_params.mloConstruct());

	std::string program_name = construct_params.getKernelFile();  // CL kernel filename
	std::string kernel_name = construct_params.getKernelName(); // kernel name
	std::string parms = construct_params.getCompilerOptions(); // kernel parameters

	std::string network_config;
	construct_params.mloBuildConf_Key(network_config);

	const std::vector<size_t> & vld = construct_params.getLocalWkSize();
	const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

	// Compile the kernel if not aleady compiled
	auto k = handle.GetKernel("mlopenPooling2dBackward", "", program_name, kernel_name, vld, vgd, parms);

	// Set kernel arguments
	// Use proper arguments
	if(mode == mlopenPoolingMax)
	{
		k(dy, dx, workSpace);
	}
	else
	{
		k(dy, dx);
	}

	return(status);
}
} // namespace mlopen
