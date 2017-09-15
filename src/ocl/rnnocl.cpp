/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/rnn.hpp>
#include <miopen/env.hpp>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>

#if MIOPEN_USE_MIOPENGEMM
#include <miopen/gemm.hpp>
#endif

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT)

struct AutoEnableProfiling
{
    AutoEnableProfiling(Handle& x) : h(x)
    {
        prev_state = h.IsProfilingEnabled();
        h.EnableProfiling();
    }

    ~AutoEnableProfiling()
    {
        h.EnableProfiling(prev_state);
        h.ResetKernelTime();
    }

    private:
    Handle& h;
    bool prev_state;
};


void RNNDescriptor::RNNForwardTraining(Handle& handle,
	const int seqLength,
	const TensorDescriptor& xDesc,
	ConstData_t x,
	const TensorDescriptor& hxDesc,
	ConstData_t hx,
	const TensorDescriptor& cxDesc,
	ConstData_t cx,
	const TensorDescriptor& wDesc,
	ConstData_t w,
	const TensorDescriptor& yDesc,
	Data_t y,
	const TensorDescriptor& hyDesc,
	Data_t hy,
	const TensorDescriptor& cyDesc,
	Data_t cy,
	Data_t workSpace,
	size_t workSpaceSize,
	Data_t reserveSpace,
	size_t reserveSpaceSize) const
{/*
	if (x == nullptr || w == nullptr || y == nullptr)
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}
	if (xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != wDesc.GetSize())
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}
	if (xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != wDesc.GetType())
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}
*/
	int in_n, in_c, in_h, in_w;
	//		std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

	int wei_n, wei_h, wei_w;
	//		std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

	int out_h, out_w;
	//		std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());
/*
	if (workSpace == nullptr ||
		workSpaceSize < ForwardGetWorkSpaceSize(handle, wDesc, xDesc, yDesc) || reserveSpaceSize < ForwardGetReserveSpaceSize(handle, wDesc, xDesc, yDesc))
	{
		MIOPEN_THROW("Workspace is required");
	}

	
	if (mode == miopenRNNRELU || mode == miopenRNNTANH)
	{				
		std::string network_config;
#if MIOPEN_USE_MIOPENGEMM
		CreateGemmGeometryConvFwd(xDesc, wDesc, yDesc, false, network_config);
		GemmGeometry gg = GetGemmGeometry("miopenConvolutionFwdAlgoGEMM", network_config);

		float time_0 = 0;
		float t1 = 0;
		for (int i = 0; i < in_n; i++)
		{
			int out_offset = i * wei_n * out_h * out_w;
			if (wei_h != 1 || wei_w != 1 || v != 1 || u != 1)
			{
				size_t in_offset = i * in_c * in_h * in_w;
				Im2ColGPU(handle,
					xDesc.GetElementSize(),
					x,
					in_offset,
					in_c,
					in_h,
					in_w,
					wei_h,
					wei_w,
					out_h,
					out_w,
					pad_h,
					pad_w,
					v,
					u,
					dilation_h,
					dilation_w,
					workSpace);
				if (handle.IsProfilingEnabled())
					t1 = handle.GetKernelTime();

				gg.RunGemm(handle, workSpace, w, y, 0, 0, out_offset);

				// Update times for both the kernels
				if (handle.IsProfilingEnabled())
				{
					if (i == in_n - 1)
						handle.AccumKernelTime(t1 + time_0);
					else
						handle.AccumKernelTime(t1);
					time_0 += handle.GetKernelTime();
				}
			}
			else if (wei_h == 1 && wei_w == 1 && v == 1 && u == 1)
			{
				int in_offset = i * in_c * in_h * in_w;
				gg.RunGemm(handle, x, w, y, in_offset, 0, out_offset);
				if (handle.IsProfilingEnabled())
				{
					if (i == in_n - 1)
						handle.AccumKernelTime(time_0);
					time_0 += handle.GetKernelTime();
				}
			}
		}
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
	}
	else if (mode == miopenLSTM)
	{

	}
	else if (mode == miopenGRU)
	{

	}
	*/
};

void RNNDescriptor::RNNBackwardData(Handle& handle,
	const int seqLength,
	const TensorDescriptor& yDesc,
	ConstData_t y,
	const TensorDescriptor& dyDesc,
	ConstData_t dy,
	const TensorDescriptor& dhyDesc,
	ConstData_t dhy,
	const TensorDescriptor& dcyDesc,
	ConstData_t dcy,
	const TensorDescriptor& wDesc,
	ConstData_t w,
	const TensorDescriptor& hxDesc,
	ConstData_t hx,
	const TensorDescriptor& cxDesc,
	ConstData_t cx,
	const TensorDescriptor& dxDesc,
	Data_t dx,
	const TensorDescriptor& dhxDesc,
	Data_t dhx,
	const TensorDescriptor& dcxDesc,
	Data_t dcx,
	Data_t workSpace,
	size_t workSpaceSize,
	ConstData_t reserveSpace,
	size_t reserveSpaceSize) const
{
	/*
	if (dx == nullptr || w == nullptr || dy == nullptr)
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}
	if (dyDesc.GetSize() != dxDesc.GetSize() || dyDesc.GetSize() != wDesc.GetSize())
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}
	if (dyDesc.GetType() != dxDesc.GetType() || dyDesc.GetType() != wDesc.GetType())
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}


	*/

        int in_n, in_c, in_h, in_w;
	//		std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

	int wei_n, wei_h, wei_w;
	//		std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

	int out_h, out_w;
	//		std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());
};
/*
void RNNBackwardWeights(Handle& handle,
	const int seqLength,
	const TensorDescriptor& xDesc,
	ConstData_t x,
	const TensorDescriptor& hxDesc,
	ConstData_t hx,
	const TensorDescriptor& dyDesc,
	ConstData_t dy,
	ConstData_t workSpace,
	size_t workSpaceSize,
	const TensorDescriptor& dwDesc,
	Data_t dw,
	ConstData_t reserveSpace,
	size_t reserveSpaceSize) const
{

        int in_n, in_c, in_h, in_w;
	//		std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

	int wei_n, wei_h, wei_w;
	//		std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

	int out_h, out_w;
	//		std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());
};
*/
} // namespace miopen
