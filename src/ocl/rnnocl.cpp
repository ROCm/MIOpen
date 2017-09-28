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

#include <../driver/activ_driver.hpp>
#include <miopen/activ.hpp>
#include <miopen/rnn.hpp>
#include <miopen/env.hpp>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <vector>
#include <numeric>

#include <miopengemm/gemm.hpp>

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
	const int seqLen,
//	const TensorDescriptor& xDesc,
	ConstData_t x,
//	const TensorDescriptor& hxDesc,
	ConstData_t hx,
//	const TensorDescriptor& cxDesc,
	ConstData_t cx,
//	const TensorDescriptor& wDesc,
	ConstData_t w,
//	const TensorDescriptor& yDesc,
	Data_t y,
//	const TensorDescriptor& hyDesc,
	Data_t hy,
//	const TensorDescriptor& cyDesc,
	Data_t cy,
	Data_t workSpace,
	size_t workSpaceSize,
	Data_t reserveSpace,
	size_t reserveSpaceSize,
	const std::vector<int> &in_n,
	const int in_h,
	const int hy_d,
	const int hy_n,
	const int hy_h,
	const int out_h) const
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
//	int in_n, in_c, in_h, in_w;
	//		std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

//	int out_h, out_w;
	//		std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());
/*
	if (workSpace == nullptr ||
		workSpaceSize < ForwardGetWorkSpaceSize(handle, wDesc, xDesc, yDesc) || reserveSpaceSize < ForwardGetReserveSpaceSize(handle, wDesc, xDesc, yDesc))
	{
		MIOPEN_THROW("Workspace is required");
	}
	*/

//	miopenRNNMode_t mode;
//	int seqLength;
//	int layer;
//	int bidir;
//	int bias;

	int batch_n = std::accumulate(in_n.begin(), in_n.end(), 0);

	bool bidirection = (bidir != 0);
	bool biased = (bias != 0);
	int numlayer = layer;
	int bacc, baccbi;
	int bi = bidirection ? 2 : 1;

	int wei_len = (bi * (in_h + hy_h + out_h) + (numlayer - 1) * bi * (bi + 1) * hy_h) * hy_h;
	if (biased)
	{
		wei_len += (bi * 2 + (numlayer - 1) * bi * (bi + 1)) * hy_h + bi * out_h;
	}

	int wei_shift_bias =
		((in_h + hy_h + out_h) * bi + (bi * hy_h + hy_h) * bi * (numlayer - 1)) * hy_h;
	int in_stride = in_h;
	int hy_stride = hy_h * bi;
	int h_stride = hy_h * bi;
	int out_stride = out_h;
	int wei_stride = hy_h * bi;

	if (mode == miopenRNNRELU || mode == miopenRNNTANH)
	{				
#if MIOPEN_USE_MIOPENGEMM

		printf("rnn gpu \n");
		cl_command_queue Q = (cl_command_queue)handle.GetStream();

		for (int li = 0; li < numlayer; li++)
		{
			int hid_shift = li * batch_n * hy_h * bi;
			int hx_shift = li * bi * in_n[0] * hy_h;

			// from input
			if (li == 0)
			{
				MIOpenGEMM::gemm0<float>(false,
					false,
					false,
					batch_n,
					hy_h * bi,
					in_h,
					1,
					x,
					0,
					in_stride,
					w,
					0,
					wei_stride,
					1,
					reserveSpace,
					hid_shift,
					hy_stride,
					&Q,
					0,
					nullptr,
					nullptr);
			}
			else
			{
				int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
				int prelayer_shift = (li - 1) * batch_n * hy_h * bi;

				MIOpenGEMM::gemm0<float>(false,
					false,
					false,
					batch_n,
					hy_h * bi,
					hy_h * bi,
					1,
					workSpace,
					prelayer_shift,
					hy_stride,
					w,
					wei_shift,
					wei_stride,
					1,
					reserveSpace,
					hid_shift,
					hy_stride,
					&Q,
					0,
					nullptr,
					nullptr);
			}
			

			// from hidden state
			bacc = 0;
			baccbi = batch_n;
			for (int ti = 0; ti < seqLen; ti++)
			{
				baccbi -= in_n[seqLen - 1 - ti];

				int wei_shift =
					li == 0 ? (in_h * hy_h * bi)
					: (bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
						bi * hy_h * hy_stride);

				if (ti == 0)
				{
					MIOpenGEMM::gemm0<float>(false,
						false,
						false,
						in_n[ti],
						hy_h,
						hy_h,
						1,
						hx,
						hx_shift,
						h_stride,
						w,
						wei_shift,
						wei_stride,
						1,
						reserveSpace,
						hid_shift + bacc * hy_stride,
						hy_stride,
						&Q,
						0,
						nullptr,
						nullptr);

					if (bidirection)
					{
						MIOpenGEMM::gemm0<float>(false,
							false,
							false,
							in_n[seqLen - 1 - ti],
							hy_h,
							hy_h,
							1,
							hx,
							hx_shift + hy_h,
							h_stride,
							w,
							wei_shift + hy_h,
							wei_stride,
							1,
							reserveSpace,
							hid_shift + baccbi * hy_stride + hy_h,
							hy_stride,
							&Q,
							0,
							nullptr,
							nullptr);
					}
				}
				else
				{
					MIOpenGEMM::gemm0<float>(false,
						false,
						false,
						in_n[ti],
						hy_h,
						hy_h,
						1,
						hy,
						hx_shift,
						h_stride,
						w,
						wei_shift,
						wei_stride,
						1,
						reserveSpace,
						hid_shift + bacc * hy_stride,
						hy_stride,
						&Q,
						0,
						nullptr,
						nullptr);

					if (bidirection)
					{
						MIOpenGEMM::gemm0<float>(false,
							false,
							false,
							in_n[seqLen - 1 - ti],
							hy_h,
							hy_h,
							1,
							hy,
							hx_shift + hy_h,
							h_stride,
							w,
							wei_shift + hy_h,
							wei_stride,
							1,
							reserveSpace,
							hid_shift + baccbi * hy_stride + hy_h,
							hy_stride,
							&Q,
							0,
							nullptr,
							nullptr);
					}
				}
				
				int rsv_sz = batch_n * hy_d * hy_h;
				std::vector<int> rsv_size(3, 1);
				rsv_size.push_back(rsv_sz);

				miopenTensorDescriptor_t rsvTensor;
				miopenCreateTensorDescriptor(&rsvTensor);
				SetTensor4d(rsvTensor, rsv_size);
				
				miopenActivationMode_t amode;
				amode = (mode == miopenRNNRELU) ? miopenActivationRELU : miopenActivationLOGISTIC;
				
				ActivationDescriptor activDesc = {amode, 1, 1, 1};

				float alpha = 1, beta = 1;
				
				activDesc.Forward(handle,
					&alpha,
					miopen::deref(rsvTensor),
					reserveSpace,
					&beta,
					miopen::deref(rsvTensor),
					workSpace);
				
				bacc += in_n[ti];
			}

			

		}

		int prelayer_shift = (numlayer - 1) * batch_n * hy_h * bi;
		int wei_shift = bi * (in_h + hy_h) * hy_h + (numlayer - 1) * bi * (bi * hy_h + hy_h) * hy_h;

		MIOpenGEMM::gemm0<float>(false,
			false,
			true,
			batch_n,
			out_h,
			hy_h * bi,
			1,
			workSpace,
			prelayer_shift,
			hy_stride,
			w,
			wei_shift,
			wei_stride,
			1,
			y,
			0,
			out_stride,
			&Q,
			0,
			nullptr,
			nullptr);

		clFinish(Q);
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
	}
	else if (mode == miopenLSTM)
	{
		printf("lstm gpu \n");
	}
	else if (mode == miopenGRU)
	{
		printf("gru gpu \n");
	}
	
};

void RNNDescriptor::RNNBackwardData(Handle& handle,
	const int seqLen,
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

void RNNDescriptor::RNNBackwardWeights(Handle& handle,
	const int seqLen,
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

} // namespace miopen
