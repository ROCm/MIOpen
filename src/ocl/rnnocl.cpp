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
	cl_command_queue Q = (cl_command_queue)handle.GetStream();

	if (mode == miopenRNNRELU || mode == miopenRNNTANH)
	{
		std::string network_config;
#if MIOPEN_USE_MIOPENGEMM
		printf("rnn gpu fwd \n");

		GemmGeometry gg;
		for (int li = 0; li < numlayer; li++)
		{
			int hid_shift = li * batch_n * hy_h * bi;
			int hx_shift = li * bi * in_n[0] * hy_h;

			// from input
			if (li == 0)
			{
				gg = CreateGemmGeometryRNNfwdfull(batch_n,
					hy_h * bi,
					in_h,
					false,
					network_config);

				gg.FindSolution(.003, handle, x, w, reserveSpace, false);
//				gg = GetGemmGeometry("miopenRNNFwdAlgoGEMMfull", network_config);
				
				gg.RunGemm(handle, x, w, reserveSpace, 0, 0, hid_shift);

				if (biased)
				{

				}
			}
			else
			{
				int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
				int prelayer_shift = (li - 1) * batch_n * hy_h * bi;

				gg = CreateGemmGeometryRNNfwdfull(batch_n,
					hy_h * bi,
					hy_h * bi,
					false,
					network_config);

				gg.FindSolution(.003, handle, workSpace, w, reserveSpace, false);
				//gg = GetGemmGeometry("miopenRNNFwdAlgoGEMMfull", network_config);

				gg.RunGemm(handle, workSpace, w, reserveSpace, prelayer_shift, wei_shift, hid_shift);

				if (biased)
				{

				}
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
					gg = CreateGemmGeometryRNNfwdpartial(in_n[ti],
						hy_h,
						hy_h,
						false,
						network_config);

					gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
					//gg = GetGemmGeometry("miopenRNNFwdAlgoGEMMpartial", network_config);

					gg.RunGemm(handle, 
						hx, 
						w, 
						reserveSpace, 
						hx_shift, 
						wei_shift, hid_shift + bacc * hy_stride);

					if (bidirection)
					{
						gg = CreateGemmGeometryRNNfwdpartial(in_n[seqLen - 1 - ti],
							hy_h,
							hy_h,
							false,
							network_config);

						gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
						//gg = GetGemmGeometry("miopenRNNFwdAlgoGEMMpartial", network_config);

						gg.RunGemm(handle, 
							hx, 
							w, 
							reserveSpace, 
							hx_shift + hy_h, 
							wei_shift + hy_h, 
							hid_shift + baccbi * hy_stride + hy_h);
					}
				}
				else
				{
					gg = CreateGemmGeometryRNNfwdpartial(in_n[ti],
						hy_h,
						hy_h,
						false,
						network_config);

					gg.FindSolution(.003, handle, workSpace, w, reserveSpace, false);
					//gg = GetGemmGeometry("miopenRNNFwdAlgoGEMMpartial", network_config);

					gg.RunGemm(handle, 
						workSpace, 
						w, 
						reserveSpace, 
						hid_shift + (bacc - in_n[ti - 1]) * hy_stride, 
						wei_shift, 
						hid_shift + bacc * hy_stride);

					if (bidirection)
					{
						gg = CreateGemmGeometryRNNfwdpartial(in_n[seqLen - ti],
							hy_h,
							hy_h,
							false,
							network_config);

						gg.FindSolution(.003, handle, workSpace, w, reserveSpace, false);
						//gg = GetGemmGeometry("miopenRNNFwdAlgoGEMMpartial", network_config);

						gg.RunGemm(handle, 
							workSpace, 
							w, 
							reserveSpace, 
							hid_shift + (baccbi + in_n[seqLen - 1 - ti]) * hy_stride + hy_h, 
							wei_shift + hy_h, 
							hid_shift + baccbi * hy_stride + hy_h);
					}
				}
				
				int rsv_sz = batch_n * hy_d * hy_h;
				std::vector<int> rsv_size(3, 1);
				rsv_size.push_back(rsv_sz);

				miopenTensorDescriptor_t rsvTensor;
				miopenCreateTensorDescriptor(&rsvTensor);
				SetTensor4d(rsvTensor, rsv_size);
				
				float alpha = 1, beta = 0;
				ActivationDescriptor activDesc;

				if (mode == miopenRNNRELU)
				{
					activDesc = { miopenActivationRELU, 1, 0, 1 };
				}
				else if (mode == miopenRNNTANH)
				{
					activDesc = { miopenActivationTANH, 1, 1, 1 };
				}

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

		// output
		int prelayer_shift = (numlayer - 1) * batch_n * hy_h * bi;
		int wei_shift = bi * (in_h + hy_h) * hy_h + (numlayer - 1) * bi * (bi * hy_h + hy_h) * hy_h;

		gg = CreateGemmGeometryRNNbwddatafull(batch_n,
			out_h,
			hy_h * bi,
			false,
			network_config);

		gg.FindSolution(.003, handle, workSpace, w, y, false);
		//gg = GetGemmGeometry("miopenRNNBwdDataAlgoGEMMfull", network_config);

		gg.RunGemm(handle,
			workSpace,
			w,
			y,
			prelayer_shift,
			wei_shift,
			0);

		if (biased)
		{

		}
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
	}
	else if (mode == miopenLSTM)
	{
		printf("lstm gpu fwd \n");
	}
	else if (mode == miopenGRU)
	{
		printf("gru gpu fwd \n");
	}
	
	clFinish(Q);
};

void RNNDescriptor::RNNBackwardData(Handle& handle,
	const int seqLen,
//	const TensorDescriptor& yDesc,
	ConstData_t y,
//	const TensorDescriptor& dyDesc,
	ConstData_t dy,
//	const TensorDescriptor& dhyDesc,
	ConstData_t dhy,
//	const TensorDescriptor& dcyDesc,
	ConstData_t dcy,
//	const TensorDescriptor& wDesc,
	ConstData_t w,
//	const TensorDescriptor& hxDesc,
	ConstData_t hx,
//	const TensorDescriptor& cxDesc,
	ConstData_t cx,
//	const TensorDescriptor& dxDesc,
	Data_t dx,
//	const TensorDescriptor& dhxDesc,
	Data_t dhx,
//	const TensorDescriptor& dcxDesc,
	Data_t dcx,
	Data_t workSpace,
	size_t workSpaceSize,
	ConstData_t reserveSpace,
	size_t reserveSpaceSize,
	const std::vector<int> &in_n,
	const int in_h,
	const int hy_d,
	const int hy_n,
	const int hy_h,
	const int out_h) const
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

//        int in_n, in_c, in_h, in_w;
	//		std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

//	int wei_n, wei_h, wei_w;
	//		std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

//	int out_h, out_w;
	//		std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());

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

	int in_stride = in_h;
	int hy_stride = hy_h * bi;
	int h_stride = hy_h * bi;
	int out_stride = out_h;
	int wei_stride = hy_h * bi;
	cl_command_queue Q = (cl_command_queue)handle.GetStream();

	if (mode == miopenRNNRELU || mode == miopenRNNTANH)
	{
#if MIOPEN_USE_MIOPENGEMM
		printf("rnn gpu bwd data \n");
/*
		for (int li = numlayer - 1; li >= 0; li--)
		{
			int wei_shift = bi * (in_h + hy_h) * hy_h + li * bi * (bi * hy_h + hy_h) * hy_h;
			int hid_shift = li * batch_n * hy_h * bi;
			int hx_shift = li * bi * in_n[0] * hy_h;

			// feedback from output
			if (li == numlayer - 1)
			{
				MIOpenGEMM::gemm0<float>(false,
					false,
					false,
					batch_n,
					hy_h * bi,
					out_h,
					1,
					dy,
					0,
					out_stride,
					w,
					wei_shift,
					wei_stride,
					1,
					workSpace,
					hid_shift,
					hy_stride,
					&Q,
					0,
					nullptr,
					nullptr);
			}
			else
			{
				int prelayer_shift = (li + 1) * batch_n * hy_h * bi;

				MIOpenGEMM::gemm0<float>(false,
					false,
					true,
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
					workSpace,
					hid_shift,
					hy_stride,
					&Q,
					0,
					nullptr,
					nullptr);
			}


			// from hidden state
			bacc = batch_n;
			baccbi = 0;
			for (int ti = seqLen - 1; ti >= 0; ti--)
			{
				bacc -= in_n[ti];
				wei_shift = li == 0 ? (in_h * hy_stride) : (bi * (in_h + hy_h) * hy_h +
					(li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
					bi * hy_h * hy_stride);


				MIOpenGEMM::gemm0<float>(false,
					false,
					true,
					in_n[ti],
					hy_h,
					hy_h,
					1,
					workSpace,
					hid_shift + bacc * hy_stride,
					hy_stride,
					w,
					wei_shift,
					wei_stride,
					0,
					dhx,
					hx_shift,
					hy_stride,
					&Q,
					0,
					nullptr,
					nullptr);

				if (bidirection)
				{


					MIOpenGEMM::gemm0<float>(false,
						false,
						true,
						in_n[seqLen - 1 - ti],
						hy_h,
						hy_h,
						1,
						workSpace,
						hid_shift + baccbi * hy_stride + hy_h,
						hy_stride,
						w,
						wei_shift + hy_h,
						wei_stride,
						0,
						dhx,
						hx_shift + hy_h,
						hy_stride,
						&Q,
						0,
						nullptr,
						nullptr);
				}

				baccbi += in_n[seqLen - 1 - ti];
			}
		}

		// dinput
		MIOpenGEMM::gemm0<float>(false,
			false,
			true,
			batch_n,
			in_h,
			hy_h * bi,
			1,
			workSpace,
			0,
			hy_stride,
			w,
			0,
			wei_stride,
			1,
			dx,
			0,
			in_stride,
			&Q,
			0,
			nullptr,
			nullptr);*/
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
	}
	else if (mode == miopenLSTM)
	{
		printf("lstm gpu bwd data \n");
	}
	else if (mode == miopenGRU)
	{
		printf("gru gpu bwd data \n");
	}

	clFinish(Q);
};

void RNNDescriptor::RNNBackwardWeights(Handle& handle,
	const int seqLen,
//	const TensorDescriptor& xDesc,
	ConstData_t x,
//	const TensorDescriptor& hxDesc,
	ConstData_t hx,
//	const TensorDescriptor& dyDesc,
	ConstData_t dy,
	ConstData_t workSpace,
	size_t workSpaceSize,
//	const TensorDescriptor& dwDesc,
	Data_t dw,
	ConstData_t reserveSpace,
	size_t reserveSpaceSize,
	const std::vector<int> &in_n,
	const int in_h,
	const int hy_d,
	const int hy_n,
	const int hy_h,
	const int out_h) const
{

//        int in_n, in_c, in_h, in_w;
	//		std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

//	int wei_n, wei_h, wei_w;
	//		std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

//	int out_h, out_w;
	//		std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());


	int batch_n = std::accumulate(in_n.begin(), in_n.end(), 0);

	bool bidirection = (bidir != 0);
	bool biased = (bias != 0);
	int numlayer = layer;
	int bacc;
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
	cl_command_queue Q = (cl_command_queue)handle.GetStream();

	if (mode == miopenRNNRELU || mode == miopenRNNTANH)
	{
#if MIOPEN_USE_MIOPENGEMM
		printf("rnn gpu bwd weights \n");
/*
		int rsv_sz = batch_n * hy_d * hy_h;
		std::vector<int> rsv_size(3, 1);
		rsv_size.push_back(rsv_sz);

		miopenTensorDescriptor_t rsvTensor;
		miopenCreateTensorDescriptor(&rsvTensor);
		SetTensor4d(rsvTensor, rsv_size);

		float alpha = 1, beta = 0;
		ActivationDescriptor activDesc;

		if (mode == miopenRNNRELU)
		{
			activDesc = { miopenActivationRELU, 1, 0, 1 };
		}
		else if (mode == miopenRNNTANH)
		{
			activDesc = { miopenActivationTANH, 1, 1, 1 };
		}

		activDesc.Forward(handle,
			&alpha,
			miopen::deref(rsvTensor),
			reserveSpace,
			&beta,
			miopen::deref(rsvTensor),
			reserveSpace);

		for (int li = 0; li <= numlayer; li++)
		{
			// between layers
			if (li == 0)
			{
				MIOpenGEMM::gemm0<float>(false,
					true,
					false,
					in_h,
					hy_h * bi,
					batch_n,
					1,
					x,
					0,
					in_stride,
					workSpace,
					0,
					hy_stride,
					1,
					dw,
					0,
					wei_stride,
					&Q,
					0,
					nullptr,
					nullptr);

				if (biased)
				{

				}
			}
			else if (li == numlayer)
			{
				int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
				int prelayer_shift = (li - 1) * bi * batch_n * hy_h;

				MIOpenGEMM::gemm0<float>(false,
					true,
					false,
					out_h,
					hy_h * bi,
					batch_n,
					1,
					dy,
					0,
					out_stride,
					reserveSpace,
					prelayer_shift,
					hy_stride,
					1,
					dw,
					wei_shift,
					wei_stride,
					&Q,
					0,
					nullptr,
					nullptr);

				if (biased)
				{

				}
			}
			else
			{
				int prelayer_shift = (li - 1) * bi * batch_n * hy_h;
				int hid_shift = li * bi * batch_n * hy_h;
				int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;

				MIOpenGEMM::gemm0<float>(false,
					true,
					false,
					hy_h * bi,
					hy_h * bi,
					batch_n,
					1,
					reserveSpace,
					prelayer_shift,
					hy_stride,
					workSpace,
					hid_shift,
					hy_stride,
					1,
					dw,
					wei_shift,
					wei_stride,
					&Q,
					0,
					nullptr,
					nullptr);

				if (biased)
				{

				}
			}

			// between time
			if (li < numlayer)
			{
				bacc = 0;
				for (int ti = 0; ti < seqLen; ti++)
				{
					int hid_shift = li * bi * batch_n * hy_h + bacc * hy_stride;
					int hx_shift = li * bi * in_n[0] * hy_h;
					int wei_shift;
					int pretime_shift;

					wei_shift =
						li == 0 ? (in_h * hy_stride)
						: (bi * (in_h + hy_h) * hy_h +
						(li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_stride);

					if (ti == 0)
					{
						MIOpenGEMM::gemm0<float>(false,
							true,
							false,
							hy_h,
							hy_h,
							in_n[ti],
							1,
							hx,
							hx_shift,
							h_stride,
							workSpace,
							hid_shift,
							hy_stride,
							1,
							dw,
							wei_shift,
							wei_stride,
							&Q,
							0,
							nullptr,
							nullptr);
					}
					else
					{
						pretime_shift = li * bi * batch_n * hy_h + (bacc - in_n[ti - 1]) * hy_stride;

						MIOpenGEMM::gemm0<float>(false,
							true,
							false,
							hy_h,
							hy_h,
							in_n[ti],
							1,
							reserveSpace,
							pretime_shift,
							hy_stride,
							workSpace,
							hid_shift,
							hy_stride,
							1,
							dw,
							wei_shift,
							wei_stride,
							&Q,
							0,
							nullptr,
							nullptr);
					}

					if (bidirection)
					{
						if (ti == seqLen - 1)
						{
							MIOpenGEMM::gemm0<float>(false,
								true,
								false,
								hy_h,
								hy_h,
								in_n[ti],
								1,
								hx,
								hx_shift + hy_h,
								h_stride,
								workSpace,
								hid_shift + hy_h,
								hy_stride,
								1,
								dw,
								wei_shift + hy_h,
								wei_stride,
								&Q,
								0,
								nullptr,
								nullptr);
						}
						else
						{
							pretime_shift = li * bi * batch_n * hy_h + (bacc + in_n[ti]) * hy_stride;

							MIOpenGEMM::gemm0<float>(false,
								true,
								false,
								hy_h,
								hy_h,
								in_n[ti + 1],
								1,
								reserveSpace,
								pretime_shift + hy_h,
								hy_stride,
								workSpace,
								hid_shift + hy_h,
								hy_stride,
								1,
								dw,
								wei_shift + hy_h,
								wei_stride,
								&Q,
								0,
								nullptr,
								nullptr);
						}
					}

					bacc += in_n[ti];
				}
			}
		}*/
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
	}
	else if (mode == miopenLSTM)
	{
		printf("lstm gpu bwd weights \n");
	}
	else if (mode == miopenGRU)
	{
		printf("gru gpu bwd weights \n");
	}

	clFinish(Q);
};

} // namespace miopen
