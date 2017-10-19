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

// Assuming sequence length is set to > 0 otherwise throw exception.
void RNNDescriptor::RNNForwardInference(Handle& handle,
                                        const int seqLen,
                                        c_array_view<miopenTensorDescriptor_t> xDesc,
                                        ConstData_t x,
                                        const TensorDescriptor& hxDesc,
                                        ConstData_t hx,
                                        const TensorDescriptor& cxDesc,
                                        ConstData_t cx,
                                        const TensorDescriptor& wDesc,
                                        ConstData_t w,
                                        c_array_view<miopenTensorDescriptor_t> yDesc,
                                        Data_t y,
                                        const TensorDescriptor& hyDesc,
                                        Data_t hy,
                                        const TensorDescriptor& cyDesc,
                                        Data_t cy,
                                        Data_t workSpace,
                                        size_t workSpaceSize) const
{
    std::cout << "RNNForwardInference. Nothing to do here!\n" << std::endl;
}

void RNNDescriptor::RNNForwardTraining(Handle& handle,
                                       const int seqLen,
                                       c_array_view<miopenTensorDescriptor_t> xDesc,
                                       ConstData_t x,
                                       const TensorDescriptor& hxDesc,
                                       ConstData_t hx,
                                       const TensorDescriptor& cxDesc,
                                       ConstData_t cx,
                                       const TensorDescriptor& wDesc,
                                       ConstData_t w,
                                       c_array_view<miopenTensorDescriptor_t> yDesc,
                                       Data_t y,
                                       const TensorDescriptor& hyDesc,
                                       Data_t hy,
                                       const TensorDescriptor& cyDesc,
                                       Data_t cy,
                                       Data_t workSpace,
                                       size_t workSpaceSize,
                                       Data_t reserveSpace,
                                       size_t reserveSpaceSize) const
{

	if (x == nullptr || w == nullptr || y == nullptr)
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}

    // TODO: DLOWELL put guards here.
    std::string network_config;
    std::vector<int> in_n;
    int in_h  = xDesc[0].GetLengths()[1]; // input vector size
    int hy_d  = hxDesc.GetLengths()[0];   // biNumLayers
    int hy_n  = hxDesc.GetLengths()[1];   // max batch size
    int hy_h  = hxDesc.GetLengths()[2];   // hidden size
    int out_h = yDesc[0].GetLengths()[1]; // output vector size

	if (in_h == 0 || hy_h == 0 || hy_n == 0 || hy_d == 0 || out_h == 0)
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}

    int batch_n = 0;
    for(int i = 0; i < seqLen; i++)
    {
        int batchval, inputvec, batchvalout, outputvec;
        std::tie(batchval, inputvec) = miopen::tien<2>(xDesc[i].GetLengths());
		std::tie(batchvalout, outputvec) = miopen::tien<2>(yDesc[i].GetLengths());
		if (batchval != batchvalout) {
			MIOPEN_THROW(miopenStatusBadParm);
		}
        in_n.push_back(batchval);
        batch_n += xDesc[i].GetLengths()[0];
    }

    int bacc, baccbi;
    int bi = dirMode ? 2 : 1;

    int wei_len = (bi * (in_h + hy_h + out_h) + (nLayers - 1) * bi * (bi + 1) * hy_h) * hy_h;
    if(biasMode)
    {
        wei_len += (bi * 2 + (nLayers - 1) * bi * (bi + 1)) * hy_h + bi * out_h;
    }

    int wei_shift_bias =
        ((in_h + hy_h + out_h) * bi + (bi * hy_h + hy_h) * bi * (nLayers - 1)) * hy_h;
    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * workspaceScale;
    int h_stride   = hy_h * bi;
    int out_stride = out_h;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;
	size_t rsv_h = reserveSpaceSize / hy_stride / sizeof(miopenFloat);
	size_t rsv_w = hy_stride;

    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("rnn gpu fwd \n");
		float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        for(int li = 0; li < nLayers; li++)
        {
            int hid_shift = li * batch_n * hy_h * bi;
            int hx_shift  = li * bi * hy_n * hy_h;

            // from input
            if(li == 0)
            {
                gg = CreateGemmGeometryRNN(batch_n,
                                           hy_h * bi,
                                           in_h,
                                           1,
                                           1,
                                           false,
                                           false,
                                           false,
                                           in_stride,
                                           wei_stride,
                                           hy_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, x, w, reserveSpace, false);
                gg.RunGemm(handle, x, w, reserveSpace, 0, 0, hid_shift);

				// Update time
				if (handle.IsProfilingEnabled()){
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}					

                if(biasMode)
                {

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
                }
            }
            else
            {
                int wei_shift =
                    bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
                int prelayer_shift = (li - 1) * batch_n * hy_h * bi;

                gg = CreateGemmGeometryRNN(batch_n,
                                           hy_h * bi,
                                           hy_h * bi,
                                           1,
                                           1,
                                           false,
                                           false,
                                           false,
                                           hy_stride,
                                           wei_stride,
                                           hy_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, workSpace, w, reserveSpace, false);
                gg.RunGemm(
                    handle, workSpace, w, reserveSpace, prelayer_shift, wei_shift, hid_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

                if(biasMode)
                {

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
                }
            }

            // from hidden state
            bacc   = 0;
            baccbi = batch_n;
            for(int ti = 0; ti < seqLen; ti++)
            {
                baccbi -= in_n[seqLen - 1 - ti];

                int wei_shift =
                    li == 0 ? (in_h * hy_h * bi)
                            : (bi * (in_h + hy_h) * hy_h +
                               (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_stride);

                if(ti == 0)
                {
					if (in_n[ti] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							h_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
						gg.RunGemm(handle,
							hx,
							w,
							reserveSpace,
							hx_shift,
							wei_shift,
							hid_shift + bacc * hy_stride);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}
                    
                    if(dirMode)
                    {
						if (in_n[seqLen - 1 - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								h_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
							gg.RunGemm(handle,
								hx,
								w,
								reserveSpace,
								hx_shift + hy_h,
								wei_shift + hy_h,
								hid_shift + baccbi * hy_stride + hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
                    }
                }
                else
                {
					if (in_n[ti] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							hy_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, workSpace, w, reserveSpace, false);
						gg.RunGemm(handle,
							workSpace,
							w,
							reserveSpace,
							hid_shift + (bacc - in_n[ti - 1]) * hy_stride,
							wei_shift,
							hid_shift + bacc * hy_stride);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}					                    

                    if(dirMode)
                    {
						if (in_n[seqLen - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - ti],
								hy_h,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								hy_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, workSpace, w, reserveSpace, false);
							gg.RunGemm(handle,
								workSpace,
								w,
								reserveSpace,
								hid_shift + (baccbi + in_n[seqLen - 1 - ti]) * hy_stride + hy_h,
								wei_shift + hy_h,
								hid_shift + baccbi * hy_stride + hy_h);

							// Update time
							if (handle.IsProfilingEnabled())
							{
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}                        
                    }
                }

                /*				if (!dirMode)
                                                {
                                                                int rsv_sz = in_n[ti] * hy_h;
                                                                std::vector<int> rsv_size(3, 1);
                                                                rsv_size.push_back(rsv_sz);

                                                                miopenTensorDescriptor_t rsvTensor;
                                                                miopenCreateTensorDescriptor(&rsvTensor);
                                                                SetTensor4d(rsvTensor, rsv_size);

                                                                float alpha = 1, beta = 0;
                                                                ActivationDescriptor activDesc;

                                                                if (rnnMode == miopenRNNRELU)
                                                                {
                                                                        activDesc = {
                 miopenActivationRELU, 1, 0, 1 };
                                                                }
                 else if (rnnMode == miopenRNNTANH)
                                                                {
                                                                        activDesc = {
                 miopenActivationTANH, 1, 1, 1 };
                                                                }

                                                                shared<Data_t> mem1 =
                 handle.CreateSubBuffer(reserveSpace, hid_shift + bacc * hy_stride, in_n[ti] *
                 hy_h);
                                                                shared<Data_t> mem2 =
                 handle.CreateSubBuffer(workSpace, hid_shift + bacc * hy_stride, in_n[ti] * hy_h);

                                                                activDesc.Forward(handle,
                                                                        &alpha,
                                                                        miopen::deref(rsvTensor),
                                                                mem1.get(),
                                                                        &beta,
                                                                        miopen::deref(rsvTensor),
                                                                        mem2.get());

                                                }
                                                else
                                                {
                                                        int rsv_sz = hy_h;
                                                        std::vector<int> rsv_size(3, 1);
                                                        rsv_size.push_back(rsv_sz);

                                                        miopenTensorDescriptor_t rsvTensor;
                                                        miopenCreateTensorDescriptor(&rsvTensor);
                                                        SetTensor4d(rsvTensor, rsv_size);

                                                        float alpha = 1, beta = 0;
                                                        ActivationDescriptor activDesc;

                                                        if (rnnMode == miopenRNNRELU)
                                                        {
                                                                activDesc = { miopenActivationRELU,
                 1, 0, 1 };
                                                        }
                                                        else if (rnnMode == miopenRNNTANH)
                                                        {
                                                                activDesc = { miopenActivationTANH,
                 1, 1, 1 };
                                                        }

                                                        for (int bs = 0; bs < in_n[ti]; bs++)
                                                        {
                                                                activDesc.Forward(handle,
                                                                        &alpha,
                                                                        miopen::deref(rsvTensor),
                                                                        handle.CreateSubBuffer(reserveSpace,
                 hid_shift + (bacc + bs) * hy_stride, hy_h),
                                                                        &beta,
                                                                        miopen::deref(rsvTensor),
                                                                        handle.CreateSubBuffer(workSpace,
                 hid_shift + (bacc + bs) * hy_stride, hy_h));
                                                        }

                                                        for (int bs = 0; bs < in_n[seqLen - 1 -
                 ti]; bs++)
                                                        {
                                                                activDesc.Forward(handle,
                                                                        &alpha,
                                                                        miopen::deref(rsvTensor),
                                                                        handle.CreateSubBuffer(reserveSpace,
                 hid_shift + (baccbi + bs) * hy_stride + hy_h, hy_h),
                                                                        &beta,
                                                                        miopen::deref(rsvTensor),
                                                                        handle.CreateSubBuffer(workSpace,
                 hid_shift + (baccbi + bs) * hy_stride + hy_h, hy_h));
                                                        }
                                                }*/

                std::vector<int> rsv_size(2, 1);
                rsv_size.push_back(in_n[ti]);
				rsv_size.push_back(hy_h);

				std::vector<int> rsv_stride;
				rsv_stride.push_back(rsv_h*rsv_w);
				rsv_stride.push_back(rsv_h*rsv_w);
				rsv_stride.push_back(rsv_w);
				rsv_stride.push_back(1);

                miopenTensorDescriptor_t rsvTensor;
                miopenCreateTensorDescriptor(&rsvTensor);
				miopenSetTensorDescriptor(rsvTensor, miopenFloat, 4, rsv_size, rsv_stride);

                float alpha = 1, beta = 0;
                ActivationDescriptor activDesc;

                if(rnnMode == miopenRNNRELU)
                {
                    activDesc = {miopenActivationRELU, 1, 0, 1};
                }
                else if(rnnMode == miopenRNNTANH)
                {
                    activDesc = {miopenActivationTANH, 1, 1, 1};
                }

				size_t offset = hid_shift + bacc * hy_stride;

                activDesc.Forward(handle,
                                  &alpha,
                                  miopen::deref(rsvTensor),
                                  reserveSpace,
                                  &beta,
                                  miopen::deref(rsvTensor),
                                  workSpace,
					offset,
					offset);
				
				if (bidirection)
				{
					rsv_size[2] = in_n[seqLength - 1 - ti];
					
					miopenCreateTensorDescriptor(&rsvTensor);
					miopenSetTensorDescriptor(rsvTensor, miopenFloat, 4, rsv_size, rsv_stride);
					
					offset = hid_shift + baccbi * hy_stride + hy_h;

					activDesc.Forward(handle,
						&alpha,
						miopen::deref(rsvTensor),
						reserveSpace,
						&beta,
						miopen::deref(rsvTensor),
						workSpace,
						offset,
						offset);
				}

				// Update time
				if (handle.IsProfilingEnabled())
				{
					time_0 = handle.GetKernelTime();
					handle.AccumKernelTime(time_0);
				}

                bacc += in_n[ti];
            }
        }

        // output
        int prelayer_shift = (nLayers - 1) * batch_n * hy_h * bi;
        int wei_shift = bi * (in_h + hy_h) * hy_h + (nLayers - 1) * bi * (bi * hy_h + hy_h) * hy_h;

        gg = CreateGemmGeometryRNN(batch_n,
                                   out_h,
                                   hy_h * bi,
                                   1,
                                   1,
                                   false,
                                   true,
                                   false,
                                   hy_stride,
                                   wei_stride,
                                   out_stride,
                                   false,
                                   network_config);
        gg.FindSolution(.003, handle, workSpace, w, y, false);
        gg.RunGemm(handle, workSpace, w, y, prelayer_shift, wei_shift, 0);

		// Update time
		if (handle.IsProfilingEnabled()) {
			time_gemm = handle.GetKernelTime();
			handle.AccumKernelTime(time_gemm);
		}

        if(biasMode)
        {

			if (handle.IsProfilingEnabled())
			{
				time_0 = handle.GetKernelTime();
				handle.AccumKernelTime(time_gemm + time_0);
			}
        }
#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
	else if (rnnMode == miopenLSTM)
	{

#if MIOPEN_USE_MIOPENGEMM
		printf("lstm gpu fwd \n");
		float time_gemm = 0, time_0 = 0;
		GemmGeometry gg;

		for (int li = 0; li < nLayers; li++)
		{
			int hid_shift = li * batch_n * hy_stride;
			int hx_shift = li * hy_n * h_stride;

			// from input
			if (li == 0)
			{
				gg = CreateGemmGeometryRNN(batch_n,
					hy_h * bi * 4,
					in_h,
					1,
					1,
					false,
					false,
					false,
					in_stride,
					wei_stride,
					hy_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, x, w, reserveSpace, false);
				gg.RunGemm(handle, x, w, reserveSpace, 0, 0, hid_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
			}
			else
			{
				int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
				int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * 5 * hy_h;

				gg = CreateGemmGeometryRNN(batch_n,
					hy_h * bi * 4,
					hy_h * bi,
					1,
					1,
					false,
					false,
					false,
					hy_stride,
					wei_stride,
					hy_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, reserveSpace, w, reserveSpace, false);
				gg.RunGemm(handle, reserveSpace, w, reserveSpace, prelayer_shift, wei_shift, hid_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
			}

			// from hidden state
			bacc = 0;
			baccbi = batch_n;
			for (int ti = 0; ti < seqLen; ti++)
			{
				baccbi -= in_n[seqLen - 1 - ti];
				int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

				if (ti == 0)
				{
					if (in_n[ti] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h * 4,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							h_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
						gg.RunGemm(handle, hx, w, reserveSpace, hx_shift, wei_shift, hid_shift + bacc * hy_stride);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}					

					if (dirMode)
					{
						if (in_n[seqLen - 1 - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h * 4,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								h_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
							gg.RunGemm(handle, hx, w, reserveSpace, hx_shift + hy_h, wei_shift + 4 * hy_h, hid_shift + baccbi * hy_stride + 4 * hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
				}
				else
				{
					if (in_n[ti] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h * 4,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							h_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, hy, w, reserveSpace, false);
						gg.RunGemm(handle, hy, w, reserveSpace, hx_shift, wei_shift, hid_shift + bacc * hy_stride);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}

					if (dirMode)
					{
						if (in_n[seqLen - 1 - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h * 4,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								h_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hy, w, reserveSpace, false);
							gg.RunGemm(handle, hy, w, reserveSpace, hx_shift + hy_h, wei_shift + 4 * hy_h, hid_shift + baccbi * hy_stride + 4 * hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
				}

				// update hidden status
/*				for (int bs = 0; bs < in_n[ti]; bs++)
				{

	            }

				if (dirMode)
				{
					for (int bs = 0; bs < in_n[seqLen - 1 - ti]; bs++)
					{

					}
				}
				*/
				bacc += in_n[ti];
			}
		}

		// output
		int prelayer_shift = (nLayers - 1) * batch_n * hy_stride + bi * 5 * hy_h;
		int wei_shift = (in_h + hy_h) * wei_stride + (nLayers - 1) * (bi * hy_h + hy_h) * wei_stride;

		gg = CreateGemmGeometryRNN(batch_n,
			out_h,
			hy_h * bi,
			1,
			1,
			false,
			true,
			false,
			hy_stride,
			h_stride,
			out_stride,
			false,
			network_config);
		gg.FindSolution(.003, handle, reserveSpace, w, y, false);
		gg.RunGemm(handle, reserveSpace, w, y, prelayer_shift, wei_shift, 0);

		// Update time
		if (handle.IsProfilingEnabled()) {
			time_gemm = handle.GetKernelTime();
			handle.AccumKernelTime(time_gemm);
		}

		if (biasMode)
		{

			if (handle.IsProfilingEnabled())
			{
				time_0 = handle.GetKernelTime();
				handle.AccumKernelTime(time_gemm + time_0);
			}
		}
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenGRU)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("gru gpu fwd \n");
		float time_gemm = 0, time_0 = 0;
		GemmGeometry gg;

		for (int li = 0; li < nLayers; li++)
		{
			int hid_shift = li * batch_n * hy_stride;
			int hx_shift = li * hy_n * h_stride;
			int wei_shift_bias_temp =
				wei_shift_bias + 2 * wei_stride + (li - 1) * (bi + 1) * wei_stride;
			
			// from input
			if (li == 0)
			{
				gg = CreateGemmGeometryRNN(batch_n,
					hy_h * bi * 3,
					in_h,
					1,
					1,
					false,
					false,
					false,
					in_stride,
					wei_stride,
					hy_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, x, w, reserveSpace, false);
				gg.RunGemm(handle, x, w, reserveSpace, 0, 0, hid_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
			}
			else
			{
				int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
				int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * 3 * hy_h;

				gg = CreateGemmGeometryRNN(batch_n,
					hy_h * bi * 3,
					hy_h * bi,
					1,
					1,
					false,
					false,
					false,
					hy_stride,
					wei_stride,
					hy_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, reserveSpace, w, reserveSpace, false);
				gg.RunGemm(handle, reserveSpace, w, reserveSpace, prelayer_shift, wei_shift, hid_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
			}

			// from hidden state
			bacc = 0;
			baccbi = batch_n;
			for (int ti = 0; ti < seqLen; ti++)
			{
				baccbi -= in_n[seqLen - 1 - ti];
				int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
				int pretime_shift =
					li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride + bi * 3 * hy_h;

				if (ti == 0)
				{
					if (in_n[ti] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h * 2,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							h_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
						gg.RunGemm(handle, hx, w, reserveSpace, hx_shift, wei_shift, hid_shift + bacc * hy_stride);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}

						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							h_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
						gg.RunGemm(handle, hx, w, reserveSpace, hx_shift, wei_shift + 2 * hy_h, hid_shift + bacc * hy_stride + bi * 3 * hy_h);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}

					if (dirMode)
					{
						if (in_n[seqLen - 1 - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h * 2,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								h_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
							gg.RunGemm(handle, hx, w, reserveSpace, hx_shift + hy_h, wei_shift + 3 * hy_h, hid_shift + baccbi * hy_stride + 3 * hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}

							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								h_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
							gg.RunGemm(handle, hx, w, reserveSpace, hx_shift + hy_h, wei_shift + 5 * hy_h, hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
				}
				else
				{
					if (in_n[ti] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h * 2,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							h_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, hy, w, reserveSpace, false);
						gg.RunGemm(handle, hy, w, reserveSpace, hx_shift, wei_shift, hid_shift + bacc * hy_stride);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}

						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							h_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, hy, w, reserveSpace, false);
						gg.RunGemm(handle, hy, w, reserveSpace, hx_shift, wei_shift + 2 * hy_h, hid_shift + bacc * hy_stride + bi * 3 * hy_h);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}					

					if (dirMode)
					{
						if (in_n[seqLen - 1 - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h * 2,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								h_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hy, w, reserveSpace, false);
							gg.RunGemm(handle, hy, w, reserveSpace, hx_shift + hy_h, wei_shift + 3 * hy_h, hid_shift + baccbi * hy_stride + 3 * hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}

							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								h_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hy, w, reserveSpace, false);
							gg.RunGemm(handle, hy, w, reserveSpace, hx_shift + hy_h, wei_shift + 5 * hy_h, hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
				}

				// update hidden status
/*				for (int bs = 0; bs < in_n[ti]; bs++)
				{
				}

				if (dirMode)
				{
					pretime_shift = li * batch_n * hy_stride +
						(baccbi + in_n[seqLen - 1 - ti]) * hy_stride + bi * 3 * hy_h +
						hy_h;

					for (int bs = 0; bs < in_n[seqLen - 1 - ti]; bs++)
					{

					}
				}*/

				bacc += in_n[ti];
			}
		}

		// output
		int prelayer_shift = (nLayers - 1) * batch_n * hy_stride + bi * 3 * hy_h;
		int wei_shift = (in_h + hy_h) * wei_stride + (nLayers - 1) * (bi * hy_h + hy_h) * wei_stride;

		gg = CreateGemmGeometryRNN(batch_n,
			out_h,
			hy_h * bi,
			1,
			1,
			false,
			true,
			false,
			hy_stride,
			h_stride,
			out_stride,
			false,
			network_config);
		gg.FindSolution(.003, handle, reserveSpace, w, y, false);
		gg.RunGemm(handle, reserveSpace, w, y, prelayer_shift, wei_shift, 0);

		// Update time
		if (handle.IsProfilingEnabled()) {
			time_gemm = handle.GetKernelTime();
			handle.AccumKernelTime(time_gemm);
		}

		if (biasMode)
		{
			int wei_shift_bias_temp =
				wei_shift_bias + 2 * wei_stride + (bi + 1) * (nLayers - 1) * wei_stride;


			if (handle.IsProfilingEnabled())
			{
				time_0 = handle.GetKernelTime();
				handle.AccumKernelTime(time_gemm + time_0);
			}
		}
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
    }

	// Suppress warning
	(void)cxDesc;
	(void)cyDesc;
	(void)hyDesc;
	(void)wDesc;
	(void)workSpaceSize;
};

void RNNDescriptor::RNNBackwardData(Handle& handle,
                                    const int seqLen,
                                    c_array_view<miopenTensorDescriptor_t> yDesc,
                                    ConstData_t y,
                                    c_array_view<miopenTensorDescriptor_t> dyDesc,
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
                                    c_array_view<miopenTensorDescriptor_t> dxDesc,
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

	if (dx == nullptr || w == nullptr || dy == nullptr)
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}

	// TODO: DLOWELL put guards here.
    std::string network_config;
    std::vector<int> in_n;
    int in_h  = dxDesc[0].GetLengths()[1];
    int hy_d  = dhxDesc.GetLengths()[0];
    int hy_n  = dhxDesc.GetLengths()[1];
    int hy_h  = dhxDesc.GetLengths()[2];
    int out_h = dyDesc[0].GetLengths()[1];

	if (in_h == 0 || hy_h == 0 || hy_n == 0 || hy_d == 0 || out_h == 0)
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}

    int batch_n = 0;
    for(int i = 0; i < seqLen; i++)
    {
		int batchval, inputvec, batchvalout, outputvec;
		std::tie(batchval, inputvec) = miopen::tien<2>(dxDesc[i].GetLengths());
		std::tie(batchvalout, outputvec) = miopen::tien<2>(dyDesc[i].GetLengths());
		if (batchval != batchvalout) {
			MIOPEN_THROW(miopenStatusBadParm);
		}
        in_n.push_back(batchval);
        batch_n += dxDesc[i].GetLengths()[0];
    }

    int bacc, baccbi;
    int bi = dirMode ? 2 : 1;

    int wei_len = (bi * (in_h + hy_h + out_h) + (nLayers - 1) * bi * (bi + 1) * hy_h) * hy_h;
    if(biasMode)
    {
        wei_len += (bi * 2 + (nLayers - 1) * bi * (bi + 1)) * hy_h + bi * out_h;
    }

    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * workspaceScale;
    int h_stride   = hy_h * bi;
    int out_stride = out_h;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;

    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
    {
#if MIOPEN_USE_MIOPENGEMM
        printf("rnn gpu bwd data \n");
		float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        for(int li = nLayers - 1; li >= 0; li--)
        {
            int wei_shift = bi * (in_h + hy_h) * hy_h + li * bi * (bi * hy_h + hy_h) * hy_h;
            int hid_shift = li * batch_n * hy_h * bi;
            int hx_shift  = li * bi * hy_n * hy_h;

            // feedback from output
            if(li == nLayers - 1)
            {

                gg = CreateGemmGeometryRNN(batch_n,
                                           hy_h * bi,
                                           out_h,
                                           1,
                                           1,
                                           false,
                                           false,
                                           false,
                                           out_stride,
                                           wei_stride,
                                           hy_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, dy, w, workSpace, false);
                gg.RunGemm(handle, dy, w, workSpace, 0, wei_shift, hid_shift);

				// Update time
				if (handle.IsProfilingEnabled())
				{
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
            }
            else
            {
                int prelayer_shift = (li + 1) * batch_n * hy_h * bi;

                gg = CreateGemmGeometryRNN(batch_n,
                                           hy_h * bi,
                                           hy_h * bi,
                                           1,
                                           1,
                                           false,
                                           true,
                                           false,
                                           hy_stride,
                                           wei_stride,
                                           hy_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
                gg.RunGemm(handle, workSpace, w, workSpace, prelayer_shift, wei_shift, hid_shift);

				// Update time
				if (handle.IsProfilingEnabled())
				{
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
            }

            // from hidden state
            bacc   = batch_n;
            baccbi = 0;
            for(int ti = seqLen - 1; ti >= 0; ti--)
            {
                bacc -= in_n[ti];
                wei_shift =
                    li == 0 ? (in_h * hy_stride)
                            : (bi * (in_h + hy_h) * hy_h +
                               (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_stride);

				if (in_n[ti] > 0)
				{
					gg = CreateGemmGeometryRNN(in_n[ti],
						hy_h,
						hy_h,
						1,
						0,
						false,
						true,
						false,
						hy_stride,
						wei_stride,
						hy_stride,
						false,
						network_config);
					gg.FindSolution(.003, handle, workSpace, w, dhx, false);
					gg.RunGemm(
						handle, workSpace, w, dhx, hid_shift + bacc * hy_stride, wei_shift, hx_shift);

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_gemm = handle.GetKernelTime();
						handle.AccumKernelTime(time_gemm);
					}
				}
                
                if(dirMode)
                {

					if (in_n[seqLen - 1 - ti] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
							hy_h,
							hy_h,
							1,
							0,
							false,
							true,
							false,
							hy_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, workSpace, w, dhx, false);
						gg.RunGemm(handle,
							workSpace,
							w,
							dhx,
							hid_shift + baccbi * hy_stride + hy_h,
							wei_shift + hy_h,
							hx_shift + hy_h);
						
						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}
                }

                baccbi += in_n[seqLen - 1 - ti];
            }
        }

        // dinput

        gg = CreateGemmGeometryRNN(batch_n,
                                   in_h,
                                   hy_h * bi,
                                   1,
                                   1,
                                   false,
                                   true,
                                   false,
                                   hy_stride,
                                   wei_stride,
                                   in_stride,
                                   false,
                                   network_config);
        gg.FindSolution(.003, handle, workSpace, w, dx, false);
        gg.RunGemm(handle, workSpace, w, dx, 0, 0, 0);

		// Update time
		if (handle.IsProfilingEnabled())
		{
			time_gemm = handle.GetKernelTime();
			handle.AccumKernelTime(time_gemm);
		}

#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
	else if (rnnMode == miopenLSTM)
	{

#if MIOPEN_USE_MIOPENGEMM
		printf("lstm gpu bwd data \n");
		float time_gemm = 0, time_0 = 0;
		GemmGeometry gg;

		for (int li = nLayers - 1; li >= 0; li--)
		{
			int wei_shift = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
			int hid_shift = li * batch_n * hy_stride;
			int hx_shift = li * hy_n * h_stride;

			if (li == nLayers - 1)
			{
				gg = CreateGemmGeometryRNN(batch_n,
					hy_h * bi,
					out_h,
					1,
					1,
					false,
					false,
					false,
					out_stride,
					wei_stride,
					hy_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, dy, w, workSpace, false);
				gg.RunGemm(handle, dy, w, workSpace, 0, wei_shift, hid_shift + bi * 5 * hy_h);

				// Update time
				if (handle.IsProfilingEnabled())
				{
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
			}
			else
			{
				int prelayer_shift = (li + 1) * batch_n * hy_stride;

				gg = CreateGemmGeometryRNN(batch_n,
					hy_h * bi,
					hy_h * bi * 4,
					1,
					1,
					false,
					true,
					false,
					hy_stride,
					wei_stride,
					hy_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
				gg.RunGemm(handle, workSpace, w, workSpace, prelayer_shift, wei_shift, hid_shift + bi * 5 * hy_h);

				// Update time
				if (handle.IsProfilingEnabled())
				{
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
			}

			// from hidden state
			bacc = batch_n;
			baccbi = 0;
			for (int ti = seqLen - 1; ti >= 0; ti--)
			{
				bacc -= in_n[ti];

				if (ti == seqLen - 1)
				{

				}
				else
				{
					int pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;
					int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

					if (in_n[ti + 1] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti + 1],
							hy_h,
							hy_h * 4,
							1,
							1,
							false,
							true,
							false,
							hy_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
						gg.RunGemm(
							handle, workSpace, w, workSpace, pretime_shift, weitime_shift, hid_shift + bacc * hy_stride + bi * 5 * hy_h);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}
					
					if (dirMode)
					{
						pretime_shift = li * batch_n * hy_stride +
							(baccbi - in_n[seqLen - 2 - ti]) * hy_stride + hy_h * 4;
						weitime_shift =
							in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride + hy_h * 4;
						
						if (in_n[seqLen - 1 - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h,
								hy_h * 4,
								1,
								1,
								false,
								true,
								false,
								hy_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
							gg.RunGemm(
								handle, workSpace, w, workSpace, pretime_shift, weitime_shift, hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
				}

				// update hidden status
/*				for (int bs = 0; bs < in_n[ti]; bs++)
				{

				}

				if (dirMode)
				{
					for (int bs = 0; bs < in_n[seqLen - 1 - ti]; bs++)
					{

					}
				}*/

				baccbi += in_n[seqLen - 1 - ti];
			}

			// dcx, dhx
			int pretime_shift = li * batch_n * hy_stride;
			int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

			if (in_n[0] > 0)
			{
				gg = CreateGemmGeometryRNN(in_n[0],
					hy_h,
					hy_h * 4,
					1,
					1,
					false,
					true,
					false,
					hy_stride,
					wei_stride,
					h_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, workSpace, w, dhx, false);
				gg.RunGemm(
					handle, workSpace, w, dhx, pretime_shift, weitime_shift, hx_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
			}			

			if (dirMode)
			{
				pretime_shift = li * batch_n * hy_stride + (batch_n - in_n[seqLen - 1]) * hy_stride;

				if (in_n[seqLen - 1] > 0)
				{
					gg = CreateGemmGeometryRNN(in_n[seqLen - 1],
						hy_h,
						hy_h * 4,
						1,
						1,
						false,
						true,
						false,
						hy_stride,
						wei_stride,
						h_stride,
						false,
						network_config);
					gg.FindSolution(.003, handle, workSpace, w, dhx, false);
					gg.RunGemm(
						handle, workSpace, w, dhx, pretime_shift + 4 * hy_h, weitime_shift + 4 * hy_h, hx_shift + hy_h);

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_gemm = handle.GetKernelTime();
						handle.AccumKernelTime(time_gemm);
					}
				}
			}
		}

		// dinput
		gg = CreateGemmGeometryRNN(batch_n,
			in_h,
			hy_h * bi * 4,
			1,
			1,
			false,
			true,
			false,
			hy_stride,
			wei_stride,
			in_stride,
			false,
			network_config);
		gg.FindSolution(.003, handle, workSpace, w, dx, false);
		gg.RunGemm(
			handle, workSpace, w, dx, 0, 0, 0);

		// Update time
		if (handle.IsProfilingEnabled()) {
			time_gemm = handle.GetKernelTime();
			handle.AccumKernelTime(time_gemm);
		}
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenGRU)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("gru gpu bwd data \n");
		float time_gemm = 0, time_0 = 0;
		GemmGeometry gg;

		for (int li = nLayers - 1; li >= 0; li--)
		{
			int wei_shift = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
			int hid_shift = li * batch_n * hy_stride;
			int hx_shift = li * hy_n * h_stride;
			int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

			if (li == nLayers - 1)
			{
				gg = CreateGemmGeometryRNN(batch_n,
					hy_h * bi,
					out_h,
					1,
					1,
					false,
					false,
					false,
					out_stride,
					h_stride,
					hy_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, dy, w, workSpace, false);
				gg.RunGemm(handle, dy, w, workSpace, 0, wei_shift, hid_shift + bi * 3 * hy_h);

				// Update time
				if (handle.IsProfilingEnabled())
				{
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
			}
			else
			{
				int prelayer_shift = (li + 1) * batch_n * hy_stride;

				gg = CreateGemmGeometryRNN(batch_n,
					hy_h * bi,
					hy_h * bi * 3,
					1,
					1,
					false,
					true,
					false,
					hy_stride,
					wei_stride,
					hy_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
				gg.RunGemm(handle, workSpace, w, workSpace, prelayer_shift, wei_shift, hid_shift + bi * 3 * hy_h);

				// Update time
				if (handle.IsProfilingEnabled())
				{
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
			}

			// from hidden state
			bacc = batch_n;
			baccbi = 0;
			for (int ti = seqLen - 1; ti >= 0; ti--)
			{
				bacc -= in_n[ti];

				if (ti == seqLen - 1)
				{

				}
				else
				{
					int pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;

					if (in_n[ti + 1] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti + 1],
							hy_h,
							hy_h * 2,
							1,
							1,
							false,
							true,
							false,
							hy_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
						gg.RunGemm(
							handle, workSpace, w, workSpace, pretime_shift, weitime_shift, hid_shift + bacc * hy_stride + bi * 3 * hy_h);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}

						gg = CreateGemmGeometryRNN(in_n[ti + 1],
							hy_h,
							hy_h,
							1,
							1,
							false,
							true,
							false,
							hy_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
						gg.RunGemm(
							handle, workSpace, w, workSpace, hid_shift + bacc * hy_stride + 2 * hy_h, weitime_shift + 2 * hy_h, hid_shift + bacc * hy_stride + bi * 3 * hy_h);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}

					if (dirMode)
					{
						pretime_shift = li * batch_n * hy_stride +
							(baccbi - in_n[seqLen - 2 - ti]) * hy_stride + hy_h * 3;

						if (in_n[seqLen - 1 - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h,
								hy_h * 2,
								1,
								1,
								false,
								true,
								false,
								hy_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
							gg.RunGemm(
								handle, workSpace, w, workSpace, pretime_shift, weitime_shift + hy_h * 3, hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}

							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h,
								hy_h,
								1,
								1,
								false,
								true,
								false,
								hy_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
							gg.RunGemm(
								handle, workSpace, w, workSpace, hid_shift + baccbi * hy_stride + 5 * hy_h, weitime_shift + 5 * hy_h, hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
				}

				if (ti == 0)
				{
					if (in_n[ti] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							h_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, hx, w, workSpace, false);
						gg.RunGemm(
							handle, hx, w, workSpace, hx_shift, weitime_shift + 2 * hy_h, hid_shift + bacc * hy_stride + hy_h);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}
				}
				else
				{
					if (in_n[ti] > 0)
					{
						gg = CreateGemmGeometryRNN(in_n[ti],
							hy_h,
							hy_h,
							1,
							1,
							false,
							false,
							false,
							hy_stride,
							wei_stride,
							hy_stride,
							false,
							network_config);
						gg.FindSolution(.003, handle, reserveSpace, w, workSpace, false);
						gg.RunGemm(
							handle, reserveSpace, w, workSpace, hid_shift + (bacc - in_n[ti - 1]) * hy_stride + bi * 3 * hy_h, weitime_shift + 2 * hy_h, hid_shift + bacc * hy_stride + hy_h);

						// Update time
						if (handle.IsProfilingEnabled()) {
							time_gemm = handle.GetKernelTime();
							handle.AccumKernelTime(time_gemm);
						}
					}
				}

				if (dirMode)
				{
					if (ti == 0)
					{
						if (in_n[seqLen - 1 - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
								hy_h,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								h_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hx, w, workSpace, false);
							gg.RunGemm(
								handle, hx, w, workSpace, hx_shift + hy_h, weitime_shift + 5 * hy_h, hid_shift + baccbi * hy_stride + 4 * hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
				    }
					else
					{
						if (in_n[seqLen - ti] > 0)
						{
							gg = CreateGemmGeometryRNN(in_n[seqLen - ti],
								hy_h,
								hy_h,
								1,
								1,
								false,
								false,
								false,
								hy_stride,
								wei_stride,
								hy_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, reserveSpace, w, workSpace, false);
							gg.RunGemm(
								handle, reserveSpace, w, workSpace, hid_shift + (baccbi + in_n[seqLen - 1 - ti]) * hy_stride + bi * 3 * hy_h + hy_h, weitime_shift + 5 * hy_h, hid_shift + baccbi * hy_stride + 4 * hy_h);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
				}

				baccbi += in_n[seqLen - 1 - ti];
			}

			// dhx
			int pretime_shift = li * batch_n * hy_stride;

			if (in_n[0] > 0)
			{
				gg = CreateGemmGeometryRNN(in_n[0],
					hy_h,
					hy_h * 2,
					1,
					1,
					false,
					true,
					false,
					hy_stride,
					wei_stride,
					h_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, workSpace, w, dhx, false);
				gg.RunGemm(
					handle, workSpace, w, dhx, pretime_shift, weitime_shift, hx_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}


				gg = CreateGemmGeometryRNN(in_n[0],
					hy_h,
					hy_h,
					1,
					1,
					false,
					true,
					false,
					h_stride,
					wei_stride,
					h_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, dcx, w, dhx, false);
				gg.RunGemm(
					handle, dcx, w, dhx, hx_shift, weitime_shift + 2 * hy_h, hx_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}
			}			

			if (dirMode)
			{
				pretime_shift = li * batch_n * hy_stride + (batch_n - in_n[seqLen - 1]) * hy_stride;

				if (in_n[seqLen - 1] > 0)
				{
					gg = CreateGemmGeometryRNN(in_n[seqLen - 1],
						hy_h,
						hy_h * 2,
						1,
						1,
						false,
						true,
						false,
						hy_stride,
						wei_stride,
						h_stride,
						false,
						network_config);
					gg.FindSolution(.003, handle, workSpace, w, dhx, false);
					gg.RunGemm(
						handle, workSpace, w, dhx, pretime_shift + 3 * hy_h, weitime_shift + 3 * hy_h, hx_shift + hy_h);

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_gemm = handle.GetKernelTime();
						handle.AccumKernelTime(time_gemm);
					}


					gg = CreateGemmGeometryRNN(in_n[seqLen - 1],
						hy_h,
						hy_h,
						1,
						1,
						false,
						true,
						false,
						h_stride,
						wei_stride,
						h_stride,
						false,
						network_config);
					gg.FindSolution(.003, handle, dcx, w, dhx, false);
					gg.RunGemm(
						handle, dcx, w, dhx, hx_shift, weitime_shift + 2 * hy_h, hx_shift);

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_gemm = handle.GetKernelTime();
						handle.AccumKernelTime(time_gemm);
					}
				}
			}
		}

		// dinput
		gg = CreateGemmGeometryRNN(batch_n,
			in_h,
			hy_h * bi * 3,
			1,
			1,
			false,
			true,
			false,
			hy_stride,
			wei_stride,
			in_stride,
			false,
			network_config);
		gg.FindSolution(.003, handle, workSpace, w, dx, false);
		gg.RunGemm(
			handle, workSpace, w, dx, 0, 0, 0);

		// Update time
		if (handle.IsProfilingEnabled()) {
			time_gemm = handle.GetKernelTime();
			handle.AccumKernelTime(time_gemm);
		}
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
    }
	
	// Suppress warning
	(void)y;
	(void)yDesc;
	(void)hxDesc;
	(void)cxDesc;
	(void)dcxDesc;
	(void)dcyDesc;
	(void)dhyDesc;
	(void)wDesc;
	(void)workSpaceSize;
	(void)reserveSpaceSize;
};

void RNNDescriptor::RNNBackwardWeights(Handle& handle,
                                       const int seqLen,
                                       c_array_view<miopenTensorDescriptor_t> xDesc,
                                       ConstData_t x,
                                       const TensorDescriptor& hxDesc,
                                       ConstData_t hx,
                                       c_array_view<miopenTensorDescriptor_t> dyDesc,
                                       ConstData_t dy,
                                       const TensorDescriptor& dwDesc,
                                       Data_t dw,
                                       ConstData_t workSpace,
                                       size_t workSpaceSize,
                                       ConstData_t reserveSpace,
                                       size_t reserveSpaceSize) const
{

	if (x == nullptr || dw == nullptr || dy == nullptr)
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}

	// TODO: DLOWELL put guards here.
    std::string network_config;
    std::vector<int> in_n;
    int in_h  = xDesc[0].GetLengths()[1];
    int hy_d  = hxDesc.GetLengths()[0];
    int hy_n  = hxDesc.GetLengths()[1];
    int hy_h  = hxDesc.GetLengths()[2];
    int out_h = dyDesc[0].GetLengths()[1];

	if (in_h == 0 || hy_h == 0 || hy_n == 0 || hy_d == 0 || out_h == 0)
	{
		MIOPEN_THROW(miopenStatusBadParm);
	}

    int batch_n = 0;
    for(int i = 0; i < seqLen; i++)
    {
		int batchval, inputvec, batchvalout, outputvec;
		std::tie(batchval, inputvec) = miopen::tien<2>(xDesc[i].GetLengths());
		std::tie(batchvalout, outputvec) = miopen::tien<2>(dyDesc[i].GetLengths());
		if (batchval != batchvalout) {
			MIOPEN_THROW(miopenStatusBadParm);
		}
        in_n.push_back(batchval);
        batch_n += xDesc[i].GetLengths()[0];
    }

    int bacc;
    int bi = dirMode ? 2 : 1;

    int wei_len = (bi * (in_h + hy_h + out_h) + (nLayers - 1) * bi * (bi + 1) * hy_h) * hy_h;
    if(biasMode)
    {
        wei_len += (bi * 2 + (nLayers - 1) * bi * (bi + 1)) * hy_h + bi * out_h;
    }

    int wei_shift_bias =
        ((in_h + hy_h + out_h) * bi + (bi * hy_h + hy_h) * bi * (nLayers - 1)) * hy_h;
    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * workspaceScale;
    int h_stride   = hy_h * bi;
    int out_stride = out_h;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;

    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("rnn gpu bwd weights \n");
		float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        int rsv_sz = batch_n * hy_d * hy_h;
        std::vector<int> rsv_size(3, 1);
        rsv_size.push_back(rsv_sz);

        miopenTensorDescriptor_t rsvTensor;
        miopenCreateTensorDescriptor(&rsvTensor);
        SetTensor4d(rsvTensor, rsv_size);

        float alpha = 1, beta = 0;
        ActivationDescriptor activDesc;

        if(rnnMode == miopenRNNRELU)
        {
            activDesc = {miopenActivationRELU, 1, 0, 1};
        }
        else if(rnnMode == miopenRNNTANH)
        {
            activDesc = {miopenActivationTANH, 1, 1, 1};
        }

        activDesc.Forward(handle,
                          &alpha,
                          miopen::deref(rsvTensor),
                          reserveSpace,
                          &beta,
                          miopen::deref(rsvTensor),
                          reserveSpace);

		// Update time
		if (handle.IsProfilingEnabled()){
			time_0 = handle.GetKernelTime();
			handle.AccumKernelTime(time_0);
		}

        for(int li = 0; li <= nLayers; li++)
        {
            // between layers
            if(li == 0)
            {

                gg = CreateGemmGeometryRNN(in_h,
                                           hy_h * bi,
                                           batch_n,
                                           1,
                                           1,
                                           true,
                                           false,
                                           false,
                                           in_stride,
                                           hy_stride,
                                           wei_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, x, workSpace, dw, false);
                gg.RunGemm(handle, x, workSpace, dw, 0, 0, 0);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

                if(biasMode)
                {

					// Update time
					if (handle.IsProfilingEnabled()){
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
                }
            }
            else if(li == nLayers)
            {
                int wei_shift =
                    bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
                int prelayer_shift = (li - 1) * bi * batch_n * hy_h;

                gg = CreateGemmGeometryRNN(out_h,
                                           hy_h * bi,
                                           batch_n,
                                           1,
                                           1,
                                           true,
                                           false,
                                           false,
                                           out_stride,
                                           hy_stride,
                                           wei_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, dy, reserveSpace, dw, false);
                gg.RunGemm(handle, dy, reserveSpace, dw, 0, prelayer_shift, wei_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{

					// Update time
					if (handle.IsProfilingEnabled()){
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
            }
            else
            {
                int prelayer_shift = (li - 1) * bi * batch_n * hy_h;
                int hid_shift      = li * bi * batch_n * hy_h;
                int wei_shift =
                    bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;

                gg = CreateGemmGeometryRNN(hy_h * bi,
                                           hy_h * bi,
                                           batch_n,
                                           1,
                                           1,
                                           true,
                                           false,
                                           false,
                                           hy_stride,
                                           hy_stride,
                                           wei_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, reserveSpace, workSpace, dw, false);
                gg.RunGemm(
                    handle, reserveSpace, workSpace, dw, prelayer_shift, hid_shift, wei_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{

					// Update time
					if (handle.IsProfilingEnabled()){
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
            }

            // between time
            if(li < nLayers)
            {
                bacc = 0;
                for(int ti = 0; ti < seqLen; ti++)
                {
                    int hid_shift = li * bi * batch_n * hy_h + bacc * hy_stride;
                    int hx_shift  = li * bi * hy_n * hy_h;
                    int wei_shift;
                    int pretime_shift;

                    wei_shift =
                        li == 0 ? (in_h * hy_stride) : (bi * (in_h + hy_h) * hy_h +
                                                        (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                                                        bi * hy_h * hy_stride);

                    if(ti == 0)
                    {
						if (in_n[ti] > 0)
						{
							gg = CreateGemmGeometryRNN(hy_h,
								hy_h,
								in_n[ti],
								1,
								1,
								true,
								false,
								false,
								h_stride,
								hy_stride,
								wei_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hx, workSpace, dw, false);
							gg.RunGemm(handle, hx, workSpace, dw, hx_shift, hid_shift, wei_shift);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
                    }
                    else
                    {
                        pretime_shift =
                            li * bi * batch_n * hy_h + (bacc - in_n[ti - 1]) * hy_stride;

						if (in_n[ti] > 0)
						{
							gg = CreateGemmGeometryRNN(hy_h,
								hy_h,
								in_n[ti],
								1,
								1,
								true,
								false,
								false,
								hy_stride,
								hy_stride,
								wei_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, reserveSpace, workSpace, dw, false);
							gg.RunGemm(handle,
								reserveSpace,
								workSpace,
								dw,
								pretime_shift,
								hid_shift,
								wei_shift);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
                    }

                    if(dirMode)
                    {
                        if(ti == seqLen - 1)
                        {
							if (in_n[ti] > 0)
							{
								gg = CreateGemmGeometryRNN(hy_h,
									hy_h,
									in_n[ti],
									1,
									1,
									true,
									false,
									false,
									h_stride,
									hy_stride,
									wei_stride,
									false,
									network_config);
								gg.FindSolution(.003, handle, hx, workSpace, dw, false);
								gg.RunGemm(handle,
									hx,
									workSpace,
									dw,
									hx_shift + hy_h,
									hid_shift + hy_h,
									wei_shift + hy_h);

								// Update time
								if (handle.IsProfilingEnabled()) {
									time_gemm = handle.GetKernelTime();
									handle.AccumKernelTime(time_gemm);
								}
							}
                        }
                        else
                        {
                            pretime_shift =
                                li * bi * batch_n * hy_h + (bacc + in_n[ti]) * hy_stride;

							if (in_n[ti + 1] > 0)
							{
								gg = CreateGemmGeometryRNN(hy_h,
									hy_h,
									in_n[ti + 1],
									1,
									1,
									true,
									false,
									false,
									hy_stride,
									hy_stride,
									wei_stride,
									false,
									network_config);
								gg.FindSolution(.003, handle, reserveSpace, workSpace, dw, false);
								gg.RunGemm(handle,
									reserveSpace,
									workSpace,
									dw,
									pretime_shift + hy_h,
									hid_shift + hy_h,
									wei_shift + hy_h);

								// Update time
								if (handle.IsProfilingEnabled()) {
									time_gemm = handle.GetKernelTime();
									handle.AccumKernelTime(time_gemm);
								}
							}
                        }
                    }

                    bacc += in_n[ti];
                }
            }
        }

#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenLSTM)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("lstm gpu bwd weights \n");
		float time_gemm = 0, time_0 = 0;
		GemmGeometry gg;

		for (int li = 0; li <= nLayers; li++)
		{
			// between layers
			if (li == 0)
			{
				gg = CreateGemmGeometryRNN(in_h, 
					hy_h * bi * 4,
					batch_n,
					1,
					1,
					true,
					false,
					false,
					in_stride,
					hy_stride,
					wei_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, x, workSpace, dw, false);
				gg.RunGemm(handle, x, workSpace, dw, 0, 0, 0);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
			}
			else if (li == nLayers)
			{
				int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
				int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * hy_h * 5;

				gg = CreateGemmGeometryRNN(out_h,
					hy_h * bi,
					batch_n,
					1,
					1,
					true,
					false,
					false,
					out_stride,
					hy_stride,
					h_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, dy, reserveSpace, dw, false);
				gg.RunGemm(handle, dy, reserveSpace, dw, 0, prelayer_shift, wei_shift);
				
				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
			}
			else
			{
				int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * hy_h * 5;
				int hid_shift = li * batch_n * hy_stride;
				int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

				gg = CreateGemmGeometryRNN(hy_h * bi,
					hy_h * bi * 4,
					batch_n,
					1,
					1,
					true,
					false,
					false,
					hy_stride,
					hy_stride,
					wei_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, reserveSpace, workSpace, dw, false);
				gg.RunGemm(
					handle, reserveSpace, workSpace, dw, prelayer_shift, hid_shift, wei_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
			}

			// between time
			if (li < nLayers)
			{
				bacc = 0;
				for (int ti = 0; ti < seqLen; ti++)
				{
					int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
					int hx_shift = li * hy_n * h_stride;
					int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
					int pretime_shift;

					// between time
					if (ti == 0)
					{
						if (in_n[ti] > 0)
						{
							gg = CreateGemmGeometryRNN(hy_h,
								hy_h * 4,
								in_n[ti],
								1,
								1,
								true,
								false,
								false,
								h_stride,
								hy_stride,
								wei_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hx, workSpace, dw, false);
							gg.RunGemm(handle, hx, workSpace, dw, hx_shift, hid_shift, wei_shift);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
					else
					{
						pretime_shift = li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride +
							bi * 5 * hy_h;

						if (in_n[ti] > 0)
						{
							gg = CreateGemmGeometryRNN(hy_h,
								hy_h * 4,
								in_n[ti],
								1,
								1,
								true,
								false,
								false,
								hy_stride,
								hy_stride,
								wei_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, reserveSpace, workSpace, dw, false);
							gg.RunGemm(handle, reserveSpace, workSpace, dw, pretime_shift, hid_shift, wei_shift);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
					
					if (dirMode)
					{
						if (ti == seqLen - 1)
						{
							if (in_n[ti] > 0)
							{
								gg = CreateGemmGeometryRNN(hy_h,
									hy_h * 4,
									in_n[ti],
									1,
									1,
									true,
									false,
									false,
									h_stride,
									hy_stride,
									wei_stride,
									false,
									network_config);
								gg.FindSolution(.003, handle, hx, workSpace, dw, false);
								gg.RunGemm(handle,
									hx,
									workSpace,
									dw,
									hx_shift + hy_h,
									hid_shift + 4 * hy_h,
									wei_shift + 4 * hy_h);

								// Update time
								if (handle.IsProfilingEnabled()) {
									time_gemm = handle.GetKernelTime();
									handle.AccumKernelTime(time_gemm);
								}
							}
						}
						else
						{
							pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride +
								bi * 5 * hy_h;

							if (in_n[ti + 1] > 0)
							{
								gg = CreateGemmGeometryRNN(hy_h,
									hy_h * 4,
									in_n[ti + 1],
									1,
									1,
									true,
									false,
									false,
									hy_stride,
									hy_stride,
									wei_stride,
									false,
									network_config);
								gg.FindSolution(.003, handle, reserveSpace, workSpace, dw, false);
								gg.RunGemm(handle,
									reserveSpace,
									workSpace,
									dw,
									pretime_shift + hy_h,
									hid_shift + 4 * hy_h,
									wei_shift + 4 * hy_h);

								// Update time
								if (handle.IsProfilingEnabled()) {
									time_gemm = handle.GetKernelTime();
									handle.AccumKernelTime(time_gemm);
								}
							}
						}
					}
					
					bacc += in_n[ti];
				}
			}
		}
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenGRU)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("gru gpu bwd weights \n");
		float time_gemm = 0, time_0 = 0;
		GemmGeometry gg;

		for (int li = 0; li <= nLayers; li++)
		{
			// between layers
			if (li == 0)
			{
				gg = CreateGemmGeometryRNN(in_h,
					hy_h * bi * 3,					
					batch_n,
					1,
					1,
					true,
					false,
					false,
					in_stride,
					hy_stride,
					wei_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, x, workSpace, dw, false);
				gg.RunGemm(handle, x, workSpace, dw, 0, 0, 0);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{

					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
			}
			else if (li == nLayers)
			{
				int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
				int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * hy_h * 3;

				gg = CreateGemmGeometryRNN(out_h,
					hy_h * bi,
					batch_n,
					1,
					1,
					true,
					false,
					false,
					out_stride,
					hy_stride,
					h_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, dy, reserveSpace, dw, false);
				gg.RunGemm(handle, dy, reserveSpace, dw, 0, prelayer_shift, wei_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{
					wei_shift = wei_shift_bias + 2 * wei_stride + (li - 1) * (bi + 1) * wei_stride;


					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
			}
			else
			{
				int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * hy_h * 3;
				int hid_shift = li * batch_n * hy_stride;
				int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

				gg = CreateGemmGeometryRNN(hy_h * bi,
					hy_h * bi * 3,
					batch_n,
					1,
					1,
					true,
					false,
					false,
					hy_stride,
					hy_stride,
					wei_stride,
					false,
					network_config);
				gg.FindSolution(.003, handle, reserveSpace, workSpace, dw, false);
				gg.RunGemm(
					handle, reserveSpace, workSpace, dw, prelayer_shift, hid_shift, wei_shift);

				// Update time
				if (handle.IsProfilingEnabled()) {
					time_gemm = handle.GetKernelTime();
					handle.AccumKernelTime(time_gemm);
				}

				if (biasMode)
				{
					wei_shift = wei_shift_bias + 2 * wei_stride + (li - 1) * (bi + 1) * wei_stride;


					// Update time
					if (handle.IsProfilingEnabled()) {
						time_0 = handle.GetKernelTime();
						handle.AccumKernelTime(time_0);
					}
				}
			}

			// between time
			if (li < nLayers)
			{
				bacc = 0;
				for (int ti = 0; ti < seqLen; ti++)
				{
					int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
					int hx_shift = li * hy_n * h_stride;
					int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
					int pretime_shift;


					if (ti == 0)
					{
						if (in_n[ti] > 0)
						{
							gg = CreateGemmGeometryRNN(hy_h,
								hy_h * 3,
								in_n[ti],
								1,
								1,
								true,
								false,
								false,
								h_stride,
								hy_stride,
								wei_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, hx, workSpace, dw, false);
							gg.RunGemm(handle, hx, workSpace, dw, hx_shift, hid_shift, wei_shift);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
					else
					{
						pretime_shift = li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride +
							bi * 3 * hy_h;

						if (in_n[ti] > 0)
						{
							gg = CreateGemmGeometryRNN(hy_h,
								hy_h * 3,
								in_n[ti],
								1,
								1,
								true,
								false,
								false,
								hy_stride,
								hy_stride,
								wei_stride,
								false,
								network_config);
							gg.FindSolution(.003, handle, reserveSpace, workSpace, dw, false);
							gg.RunGemm(handle, reserveSpace, workSpace, dw, pretime_shift, hid_shift, wei_shift);

							// Update time
							if (handle.IsProfilingEnabled()) {
								time_gemm = handle.GetKernelTime();
								handle.AccumKernelTime(time_gemm);
							}
						}
					}
					
					if (dirMode)
					{
						if (ti == seqLen - 1)
						{
							if (in_n[ti] > 0)
							{
								gg = CreateGemmGeometryRNN(hy_h,
									hy_h * 3,
									in_n[ti],
									1,
									1,
									true,
									false,
									false,
									h_stride,
									hy_stride,
									wei_stride,
									false,
									network_config);
								gg.FindSolution(.003, handle, hx, workSpace, dw, false);
								gg.RunGemm(handle,
									hx,
									workSpace,
									dw,
									hx_shift + hy_h,
									hid_shift + 3 * hy_h,
									wei_shift + 3 * hy_h);

								// Update time
								if (handle.IsProfilingEnabled()) {
									time_gemm = handle.GetKernelTime();
									handle.AccumKernelTime(time_gemm);
								}
							}
						}
						else
						{
							pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride +
								bi * 3 * hy_h;

							if (in_n[ti + 1] > 0)
							{
								gg = CreateGemmGeometryRNN(hy_h,
									hy_h * 3,
									in_n[ti + 1],
									1,
									1,
									true,
									false,
									false,
									hy_stride,
									hy_stride,
									wei_stride,
									false,
									network_config);
								gg.FindSolution(.003, handle, reserveSpace, workSpace, dw, false);
								gg.RunGemm(handle,
									reserveSpace,
									workSpace,
									dw,
									pretime_shift + hy_h,
									hid_shift + 3 * hy_h,
									wei_shift + 3 * hy_h);

								// Update time
								if (handle.IsProfilingEnabled()) {
									time_gemm = handle.GetKernelTime();
									handle.AccumKernelTime(time_gemm);
								}
							}
						}
					}

					bacc += in_n[ti];
				}

				if (biasMode)
				{
					int wei_shift = (li == 0)
						? (wei_shift_bias + wei_stride)
						: (wei_shift_bias + wei_stride + li * (bi + 1) * wei_stride);
					int hid_shift = li * batch_n * hy_stride;


				}
			}
		}
#else
		MIOPEN_THROW("GEMM is not supported");
#endif
    }

	// Suppress warning
	(void)dwDesc;
	(void)workSpaceSize;
	(void)reserveSpaceSize;
};

// TODO: LATER

void RNNDescriptor::ForwardRNNInferCell(Handle& handle,
                                        const TensorDescriptor& xDesc,
                                        ConstData_t x,
                                        const TensorDescriptor& hxDesc,
                                        ConstData_t hx,
                                        const TensorDescriptor& wDesc,
                                        ConstData_t w,
                                        const TensorDescriptor& yDesc,
                                        Data_t y,
                                        const TensorDescriptor& hyDesc,
                                        Data_t hy,
                                        Data_t workSpace,
                                        size_t workSpaceSize) const
{
}

void RNNDescriptor::ForwardRNNTrainCell(Handle& handle,
                                        const TensorDescriptor& xDesc,
                                        ConstData_t x,
                                        const TensorDescriptor& hxDesc,
                                        ConstData_t hx,
                                        const TensorDescriptor& wDesc,
                                        ConstData_t w,
                                        const TensorDescriptor& yDesc,
                                        Data_t y,
                                        const TensorDescriptor& hyDesc,
                                        Data_t hy,
                                        Data_t workSpace,
                                        size_t workSpaceSize,
                                        Data_t reserveSpace,
                                        size_t reserveSpaceSize) const
{
}

void RNNDescriptor::BackwardRNNDataCell(Handle& handle,
                                        const TensorDescriptor& yDesc,
                                        ConstData_t y,
                                        const TensorDescriptor& dyDesc,
                                        ConstData_t dy,
                                        const TensorDescriptor& dhyDesc,
                                        ConstData_t dhy,
                                        const TensorDescriptor& wDesc,
                                        ConstData_t w,
                                        const TensorDescriptor& hxDesc,
                                        ConstData_t hx,
                                        const TensorDescriptor& dxDesc,
                                        Data_t dx,
                                        const TensorDescriptor& dhxDesc,
                                        Data_t dhx,
                                        Data_t workSpace,
                                        size_t workSpaceSize,
                                        ConstData_t reserveSpace,
                                        size_t reserveSpaceSize) const
{
}

void RNNDescriptor::BackwardRNNWeightsCell(Handle& handle,
                                           const TensorDescriptor& xDesc,
                                           ConstData_t x,
                                           const TensorDescriptor& hxDesc,
                                           ConstData_t hx,
                                           const TensorDescriptor& yDesc,
                                           ConstData_t y,
                                           const TensorDescriptor& dwDesc,
                                           Data_t dw,
                                           ConstData_t workSpace,
                                           size_t workSpaceSize,
                                           ConstData_t reserveSpace,
                                           size_t reserveSpaceSize) const
{
}

} // namespace miopen
