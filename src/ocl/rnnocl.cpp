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
    (void)handle;
    (void)seqLen;
    (void)xDesc;
    (void)x;
    (void)hxDesc;
    (void)hx;
    (void)cxDesc;
    (void)cx;
    (void)wDesc;
    (void)w;
    (void)yDesc;
    (void)y;
    (void)hyDesc;
    (void)hy;
    (void)cyDesc;
    (void)cy;
    (void)workSpace;
    (void)workSpaceSize;
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

    if(x == nullptr || w == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    // TODO: DLOWELL put guards here.
    std::string network_config;
    std::vector<int> in_n;
    int in_h  = xDesc[0].GetLengths()[1]; // input vector size
    int hy_d  = hyDesc.GetLengths()[0];   // biNumLayers
    int hy_n  = hyDesc.GetLengths()[1];   // max batch size
    int hy_h  = hyDesc.GetLengths()[2];   // hidden size
    int out_h = yDesc[0].GetLengths()[1]; // output vector size

    if(in_h == 0 || hy_h == 0 || hy_n == 0 || hy_d == 0 || out_h == 0)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    int batch_n = 0;
    for(int i = 0; i < seqLen; i++)
    {
        int batchval, inputvec, batchvalout, outputvec;
        std::tie(batchval, inputvec)     = miopen::tien<2>(xDesc[i].GetLengths());
        std::tie(batchvalout, outputvec) = miopen::tien<2>(yDesc[i].GetLengths());
        if(batchval != batchvalout)
        {
            printf("Input batch length: %d, Output batch length: %d\n", batchval, batchvalout);
            MIOPEN_THROW(miopenStatusBadParm);
        }
        in_n.push_back(batchval);
        batch_n += batchval;
    }

    int bacc, baccbi;
    int bi = dirMode ? 2 : 1;
    if(out_h != (bi * hy_h))
    {
        printf("Output size doesn't match hidden state size!\n");
        MIOPEN_THROW(miopenStatusBadParm);
    }

    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * workspaceScale;
    int h_stride   = hy_h * bi;
    int out_stride = out_h;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;
    size_t rsv_h   = reserveSpaceSize / hy_stride / sizeof(miopenFloat);
    size_t rsv_w   = hy_stride;

    if(inputMode == miopenRNNskip)
    {
        if(in_h != hy_h)
        {
            printf("The input tensor size must equal to the hidden state size of the network in "
                   "SKIP_INPUT mode!\n");
            MIOPEN_THROW(miopenStatusBadParm);
        }
        in_h = 0;
    }

    size_t wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (nLayers - 1)) * wei_stride;

    float alpha0, alpha1, beta_t;

    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("run rnn gpu fwd \n");
        float time_gemm = 0, time_0 = 0;

        GemmGeometry gg;
		
		std::vector<int> sp_size(4, 1), sp_stride(4, 1), w_size(4, 1), w_stride(4, 1), x_size(4, 1), x_stride(4, 1), y_size(4, 1), y_stride(4, 1), hx_size(4, 1), hx_stride(4, 1);
		miopenTensorDescriptor_t sp_desc, w_desc, x_desc, y_desc, hx_desc;
		sp_stride[0] = batch_n * hy_stride;
		sp_stride[1] = batch_n * hy_stride;
		sp_stride[2] = hy_stride;
		w_stride[0] = wei_stride;
		w_stride[1] = wei_stride;
		w_stride[2] = wei_stride;
		x_stride[0] = batch_n * in_stride;
		x_stride[1] = batch_n * in_stride;
		x_stride[2] = in_stride;
		y_stride[0] = batch_n * out_stride;
		y_stride[1] = batch_n * out_stride;
		y_stride[2] = out_stride;
		hx_stride[0] = in_n[0] * h_stride;
		hx_stride[1] = in_n[0] * h_stride;
		hx_stride[2] = h_stride;

		float alpha = 1, beta = 0;
		ActivationDescriptor activDesc;
		size_t offset;

		if (rnnMode == miopenRNNRELU)
		{
			activDesc = { miopenActivationRELU, 1, 0, 1 };
		}
		else if (rnnMode == miopenRNNTANH)
		{
			activDesc = { miopenActivationTANH, 1, 1, 1 };
		}

        for(int li = 0; li < nLayers; li++)
        {
            int hid_shift = li * batch_n * hy_h * bi;
            int hx_shift  = li * bi * hy_n * hy_h;

            // from input
            if(li == 0)
            {
                if(inputMode == miopenRNNskip)
                {
                    x_size[2]    = batch_n;
                    x_size[3]    = hy_h;
                    sp_size[2]   = batch_n;
                    sp_size[3]   = hy_h;

                    miopenCreateTensorDescriptor(&x_desc);
                    miopenCreateTensorDescriptor(&sp_desc);
                    miopenSetTensorDescriptor(
						x_desc, miopenFloat, 4, x_size.data(), x_stride.data());
                    miopenSetTensorDescriptor(
						sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    CopyTensor(handle,
                               miopen::deref(x_desc),
                               x,
                               miopen::deref(sp_desc),
                               reserveSpace,
                               0,
                               0);
                    // Update time
                    profileSequence(handle, 0);

                    if(dirMode)
                    {
                        CopyTensor(handle,
                                   miopen::deref(x_desc),
                                   x,
                                   miopen::deref(sp_desc),
                                   reserveSpace,
                                   0,
                                   hy_h);

                        // Update time
                        profileSequence(handle, 1);
                    }

                    if(biasMode)
                    {
                        w_size[2]    = 1;
                        w_size[3]    = wei_stride;
                        sp_size[2]   = 1;
                        sp_size[3]   = wei_stride;
                        miopenCreateTensorDescriptor(&w_desc);
						miopenCreateTensorDescriptor(&sp_desc);
                        miopenSetTensorDescriptor(
							w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                        miopenSetTensorDescriptor(
							sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;
                        for(int bs = 0; bs < batch_n; bs++)
                        {
                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(w_desc),
                                     w,
                                     &alpha1,
                                     miopen::deref(w_desc),
                                     w,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     wei_shift_bias,
                                     wei_shift_bias,
                                     hid_shift + bs * hy_stride);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }
                else
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
                    if(li == 0)
                        profileSequence(handle, 0);
                    else
                        profileSequence(handle, 1);

                    if(biasMode)
                    {
                        w_size[2]   = 1;
                        w_size[3]   = wei_stride;
                        sp_size[2]   = 1;
                        sp_size[3]   = wei_stride;

                        miopenCreateTensorDescriptor(&w_desc);
                        miopenCreateTensorDescriptor(&sp_desc);
                        miopenSetTensorDescriptor(
							w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                        miopenSetTensorDescriptor(
							sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        for(int bs = 0; bs < batch_n; bs++)
                        {
                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(w_desc),
                                     w,
                                     &alpha1,
                                     miopen::deref(w_desc),
                                     w,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     wei_shift_bias,
                                     wei_shift_bias + wei_stride,
                                     hid_shift + bs * hy_stride);

                            // Update time
                            profileSequence(handle, 1);
                        }
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
                gg.FindSolution(.003, handle, reserveSpace, w, reserveSpace, false);
                gg.RunGemm(handle,
                           reserveSpace,
                           w,
                           reserveSpace,
                           prelayer_shift + nLayers * batch_n * hy_stride,
                           wei_shift,
                           hid_shift);

                // Update time
                if(li == 0)
                    profileSequence(handle, 0);
                else
                    profileSequence(handle, 1);

                if(biasMode)
                {
                    int wei_shift_bias_temp =
                        (inputMode == miopenRNNskip)
                            ? (wei_shift_bias + bi * hy_h + bi * (li - 1) * 2 * hy_h)
                            : (wei_shift_bias + bi * li * 2 * hy_h);

                    w_size[2]   = 1;
                    w_size[3]   = wei_stride;
                    sp_size[2]   = 1;
                    sp_size[3]   = wei_stride;

                    miopenCreateTensorDescriptor(&w_desc);
                    miopenCreateTensorDescriptor(&sp_desc);
                    miopenSetTensorDescriptor(
						w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                    miopenSetTensorDescriptor(
						sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(w_desc),
                                 w,
                                 &alpha1,
                                 miopen::deref(w_desc),
                                 w,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 wei_shift_bias_temp,
                                 wei_shift_bias_temp + wei_stride,
                                 hid_shift + bs * hy_stride);

                        // Update time
                        profileSequence(handle, 1);
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
                    if(in_n[ti] > 0)
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
                        profileSequence(handle, 1);
                    }

                    if(dirMode)
                    {
                        if(in_n[seqLen - 1 - ti] > 0)
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
                            profileSequence(handle, 1);
                        }
                    }
                }
                else
                {
                    if(in_n[ti] > 0)
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
                        gg.FindSolution(.003, handle, reserveSpace, w, reserveSpace, false);
                        gg.RunGemm(handle,
                                   reserveSpace,
                                   w,
                                   reserveSpace,
                                   hid_shift + (bacc - in_n[ti - 1]) * hy_stride +
                                       nLayers * batch_n * hy_stride,
                                   wei_shift,
                                   hid_shift + bacc * hy_stride);

                        // Update time
                        profileSequence(handle, 1);
                    }

                    if(dirMode)
                    {
                        if(in_n[seqLen - ti] > 0)
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

                            gg.FindSolution(.003, handle, reserveSpace, w, reserveSpace, false);
                            gg.RunGemm(handle,
                                       reserveSpace,
                                       w,
                                       reserveSpace,
                                       hid_shift + (baccbi + in_n[seqLen - 1 - ti]) * hy_stride +
                                           hy_h + nLayers * batch_n * hy_stride,
                                       wei_shift + hy_h,
                                       hid_shift + baccbi * hy_stride + hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }
				                
                if(in_n[ti] > 0)
                {
                    sp_size[2]   = in_n[ti];
                    sp_size[3]   = hy_h;

                    miopenCreateTensorDescriptor(&sp_desc);
                    miopenSetTensorDescriptor(
						sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    offset = hid_shift + bacc * hy_stride;

                    activDesc.Forward(handle,
                                      &alpha,
                                      miopen::deref(sp_desc),
                                      reserveSpace,
                                      &beta,
                                      miopen::deref(sp_desc),
                                      reserveSpace,
                                      offset,
                                      offset + nLayers * batch_n * hy_stride);

                    // Update time
                    profileSequence(handle, 1);
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        sp_size[2]   = in_n[seqLen - 1 - ti];
                        sp_size[3]   = hy_h;

                        miopenCreateTensorDescriptor(&sp_desc);
                        miopenSetTensorDescriptor(
							sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        offset = hid_shift + baccbi * hy_stride + hy_h;

                        activDesc.Forward(handle,
                                          &alpha,
                                          miopen::deref(sp_desc),
                                          reserveSpace,
                                          &beta,
                                          miopen::deref(sp_desc),
                                          reserveSpace,
                                          offset,
                                          offset + nLayers * batch_n * hy_stride);

                        // Update time
                        profileSequence(handle, 1);
                    }
                }

                bacc += in_n[ti];
            }

            // hy
            if(in_n[seqLen - 1] > 0)
            {
                sp_size[2]   = in_n[seqLen - 1];
                sp_size[3]   = hy_h;
                hx_size[2]   = in_n[seqLen - 1];
                hx_size[3]   = hy_h;

                miopenCreateTensorDescriptor(&sp_desc);
                miopenCreateTensorDescriptor(&hx_desc);
                miopenSetTensorDescriptor(
					sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
                miopenSetTensorDescriptor(
					hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                CopyTensor(handle,
                           miopen::deref(sp_desc),
                           reserveSpace,
                           miopen::deref(hx_desc),
                           hy,
                           hid_shift + (batch_n - in_n[seqLen - 1]) * hy_stride +
                               nLayers * batch_n * hy_stride,
                           hx_shift);
                // Update time
                profileSequence(handle, 1);
            }

            if(dirMode)
            {
                if(in_n[0] > 0)
                {
                    sp_size[2]   = in_n[0];
                    sp_size[3]   = hy_h;
                    hx_size[2]   = in_n[0];
                    hx_size[3]   = hy_h;

                    miopenCreateTensorDescriptor(&sp_desc);
                    miopenCreateTensorDescriptor(&hx_desc);
                    miopenSetTensorDescriptor(
						sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
                    miopenSetTensorDescriptor(
						hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                    CopyTensor(handle,
                               miopen::deref(sp_desc),
                               reserveSpace,
                               miopen::deref(hx_desc),
                               hy,
                               hid_shift + hy_h + nLayers * batch_n * hy_stride,
                               hx_shift + hy_h);
                    // Update time
                    profileSequence(handle, 1);
                }
            }
        }

        // output
        int prelayer_shift = (nLayers - 1) * batch_n * hy_stride;

        sp_size[2]   = batch_n;
        sp_size[3]   = hy_h * bi;
        y_size[2]   = batch_n;
        y_size[3]   = out_h;

        miopenCreateTensorDescriptor(&sp_desc);
        miopenCreateTensorDescriptor(&y_desc);
        miopenSetTensorDescriptor(sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
        miopenSetTensorDescriptor(y_desc, miopenFloat, 4, y_size.data(), y_stride.data());

        CopyTensor(handle,
                   miopen::deref(sp_desc),
                   reserveSpace,
                   miopen::deref(y_desc),
                   y,
                   prelayer_shift + nLayers * batch_n * hy_stride,
                   0);
        // Update time
        profileSequence(handle, 1);

#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenLSTM)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("run lstm gpu fwd \n");
        float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        for(int li = 0; li < nLayers; li++)
        {
            int hid_shift = li * batch_n * hy_stride;
            int hx_shift  = li * hy_n * h_stride;

            // from input
            if(li == 0)
            {
                if(inputMode == miopenRNNskip)
                {
                    std::vector<int> src_size(4, 1), src_stride(4, 1), dest_size(4, 1),
                        dest_stride(4, 1);
                    miopenTensorDescriptor_t srcTensor, destTensor;

                    src_size[2]    = batch_n;
                    src_size[3]    = hy_h;
                    src_stride[0]  = batch_n * in_stride;
                    src_stride[1]  = batch_n * in_stride;
                    src_stride[2]  = in_stride;
                    dest_size[2]   = batch_n;
                    dest_size[3]   = hy_h;
                    dest_stride[0] = batch_n * hy_stride;
                    dest_stride[1] = batch_n * hy_stride;
                    dest_stride[2] = hy_stride;
                    miopenCreateTensorDescriptor(&srcTensor);
                    miopenCreateTensorDescriptor(&destTensor);
                    miopenSetTensorDescriptor(
                        srcTensor, miopenFloat, 4, src_size.data(), src_stride.data());
                    miopenSetTensorDescriptor(
                        destTensor, miopenFloat, 4, dest_size.data(), dest_stride.data());

                    for(int gi = 0; gi < 4; gi++)
                    {
                        CopyTensor(handle,
                                   miopen::deref(srcTensor),
                                   x,
                                   miopen::deref(destTensor),
                                   reserveSpace,
                                   0,
                                   gi * hy_h);

                        // Update time
                        if((gi + li) == 0)
                            profileSequence(handle, 0);
                        else
                            profileSequence(handle, 1);

                        if(dirMode)
                        {
                            CopyTensor(handle,
                                       miopen::deref(srcTensor),
                                       x,
                                       miopen::deref(destTensor),
                                       reserveSpace,
                                       0,
                                       (gi + 4) * hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }

                    if(biasMode)
                    {
                        std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                        miopenTensorDescriptor_t Adesc, Cdesc;

                        a_size[2]   = 1;
                        a_size[3]   = wei_stride;
                        a_stride[0] = wei_stride;
                        a_stride[1] = wei_stride;
                        a_stride[2] = wei_stride;
                        c_size[2]   = 1;
                        c_size[3]   = wei_stride;
                        c_stride[0] = hy_stride;
                        c_stride[1] = hy_stride;
                        c_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenCreateTensorDescriptor(&Cdesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                        miopenSetTensorDescriptor(
                            Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        for(int bs = 0; bs < batch_n; bs++)
                        {
                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     w,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     w,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     reserveSpace,
                                     wei_shift_bias,
                                     wei_shift_bias,
                                     hid_shift + bs * hy_stride);
                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }
                else
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
                    if(li == 0)
                        profileSequence(handle, 0);
                    else
                        profileSequence(handle, 1);

                    if(biasMode)
                    {

                        std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                        miopenTensorDescriptor_t Adesc, Cdesc;

                        a_size[2]   = 1;
                        a_size[3]   = wei_stride;
                        a_stride[0] = wei_stride;
                        a_stride[1] = wei_stride;
                        a_stride[2] = wei_stride;
                        c_size[2]   = 1;
                        c_size[3]   = wei_stride;
                        c_stride[0] = hy_stride;
                        c_stride[1] = hy_stride;
                        c_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenCreateTensorDescriptor(&Cdesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                        miopenSetTensorDescriptor(
                            Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        for(int bs = 0; bs < batch_n; bs++)
                        {
                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     w,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     w,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     reserveSpace,
                                     wei_shift_bias,
                                     wei_shift_bias + wei_stride,
                                     hid_shift + bs * hy_stride);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }
            }
            else
            {
                int wei_shift =
                    (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
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
                gg.RunGemm(
                    handle, reserveSpace, w, reserveSpace, prelayer_shift, wei_shift, hid_shift);

                // Update time
                profileSequence(handle, 1);
                //                if(handle.IsProfilingEnabled())
                //                {
                //                    time_gemm = handle.GetKernelTime();
                //                    handle.AccumKernelTime(time_gemm);
                //                }

                if(biasMode)
                {
                    int wei_shift_bias_temp =
                        (inputMode == miopenRNNskip)
                            ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                            : (wei_shift_bias + li * 2 * wei_stride);

                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    a_size[2]   = 1;
                    a_size[3]   = wei_stride;
                    a_stride[0] = wei_stride;
                    a_stride[1] = wei_stride;
                    a_stride[2] = wei_stride;
                    c_size[2]   = 1;
                    c_size[3]   = wei_stride;
                    c_stride[0] = hy_stride;
                    c_stride[1] = hy_stride;
                    c_stride[2] = hy_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenCreateTensorDescriptor(&Cdesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                    miopenSetTensorDescriptor(
                        Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());
                    alpha0 = 1;

                    beta_t = 1;

                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        alpha1 = 1;
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 w,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 w,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 reserveSpace,
                                 wei_shift_bias_temp,
                                 wei_shift_bias_temp + wei_stride,
                                 hid_shift + bs * hy_stride);
                        // Update time
                        profileSequence(handle, 1);
                    }
                }
            }

            // from hidden state
            bacc   = 0;
            baccbi = batch_n;
            for(int ti = 0; ti < seqLen; ti++)
            {
                baccbi -= in_n[seqLen - 1 - ti];
                int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

                if(ti == 0)
                {
                    if(in_n[ti] > 0)
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
                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   reserveSpace,
                                   hx_shift,
                                   wei_shift,
                                   hid_shift + bacc * hy_stride);

                        // Update time
                        profileSequence(handle, 1);
                    }

                    if(dirMode)
                    {
                        if(in_n[seqLen - 1 - ti] > 0)
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
                            gg.RunGemm(handle,
                                       hx,
                                       w,
                                       reserveSpace,
                                       hx_shift + hy_h,
                                       wei_shift + 4 * hy_h,
                                       hid_shift + baccbi * hy_stride + 4 * hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }
                else
                {
                    if(in_n[ti] > 0)
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
                        gg.RunGemm(handle,
                                   hy,
                                   w,
                                   reserveSpace,
                                   hx_shift,
                                   wei_shift,
                                   hid_shift + bacc * hy_stride);

                        // Update time
                        profileSequence(handle, 1);
                    }

                    if(dirMode)
                    {
                        if(in_n[seqLen - 1 - ti] > 0)
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
                            gg.RunGemm(handle,
                                       hy,
                                       w,
                                       reserveSpace,
                                       hx_shift + hy_h,
                                       wei_shift + 4 * hy_h,
                                       hid_shift + baccbi * hy_stride + 4 * hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }

                // update hidden status
                float alpha = 1, beta = 0;
                ActivationDescriptor tanhDesc, sigDesc;
                size_t offset;

                sigDesc  = {miopenActivationLOGISTIC, 1, 0, 1};
                tanhDesc = {miopenActivationTANH, 1, 1, 1};

                std::vector<int> rsv_size(4, 1), rsv_stride(4, 1);
                miopenTensorDescriptor_t rsvTensor;

                if(in_n[ti] > 0)
                {
                    rsv_size[2]   = in_n[ti];
                    rsv_stride[0] = in_n[ti] * hy_stride;
                    rsv_stride[1] = in_n[ti] * hy_stride;
                    rsv_stride[2] = hy_stride;

                    // active gate i, f, o
                    rsv_size[3] = hy_h * 3;
                    miopenCreateTensorDescriptor(&rsvTensor);
                    miopenSetTensorDescriptor(
                        rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                    offset = hid_shift + bacc * hy_stride;

                    sigDesc.Forward(handle,
                                    &alpha,
                                    miopen::deref(rsvTensor),
                                    reserveSpace,
                                    &beta,
                                    miopen::deref(rsvTensor),
                                    reserveSpace,
                                    offset,
                                    offset + nLayers * batch_n * hy_stride);
                    // Update time
                    profileSequence(handle, 1);

                    // active gate c
                    rsv_size[3] = hy_h;
                    miopenCreateTensorDescriptor(&rsvTensor);
                    miopenSetTensorDescriptor(
                        rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                    offset = hid_shift + bacc * hy_stride + 3 * hy_h;

                    tanhDesc.Forward(handle,
                                     &alpha,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     offset,
                                     offset + nLayers * batch_n * hy_stride);
                    // Update time
                    profileSequence(handle, 1);

                    // update cell state
                    std::vector<int> a_size(4, 1), a_stride(4, 1);
                    miopenTensorDescriptor_t Adesc;

                    a_size[2]   = in_n[ti];
                    a_size[3]   = hy_h;
                    a_stride[0] = in_n[ti] * hy_stride;
                    a_stride[1] = in_n[ti] * hy_stride;
                    a_stride[2] = hy_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + 3 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 4 * hy_h);

                    // Update time
                    profileSequence(handle, 1);

                    if(ti == 0)
                    {
                        std::vector<int> b_size(4, 1), b_stride(4, 1);
                        miopenTensorDescriptor_t Bdesc;

                        b_size[2]   = in_n[ti];
                        b_size[3]   = hy_h;
                        b_stride[0] = in_n[ti] * h_stride;
                        b_stride[1] = in_n[ti] * h_stride;
                        b_stride[2] = h_stride;

                        miopenCreateTensorDescriptor(&Bdesc);
                        miopenSetTensorDescriptor(
                            Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Bdesc),
                                 cx,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 hid_shift + bacc * hy_stride + hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 4 * hy_h);
                        // Update time
                        profileSequence(handle, 1);
                    }
                    else
                    {
                        int prec_shift = li * batch_n * hy_stride +
                                         (bacc - in_n[ti - 1]) * hy_stride + bi * 4 * hy_h;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 hid_shift + bacc * hy_stride + hy_h +
                                     nLayers * batch_n * hy_stride,
                                 prec_shift,
                                 hid_shift + bacc * hy_stride + bi * 4 * hy_h);
                        // Update time
                        profileSequence(handle, 1);
                    }

                    // active cell state
                    rsv_size[3] = hy_h;
                    miopenCreateTensorDescriptor(&rsvTensor);
                    miopenSetTensorDescriptor(
                        rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                    offset = hid_shift + bacc * hy_stride + bi * 4 * hy_h;

                    tanhDesc.Forward(handle,
                                     &alpha,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     offset,
                                     offset + nLayers * batch_n * hy_stride);
                    // Update time
                    profileSequence(handle, 1);

                    // update hidden state
                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + 2 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 4 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 5 * hy_h);
                    // Update time
                    profileSequence(handle, 1);

                    // update cy, hy
                    std::vector<int> b_size(4, 1), b_stride(4, 1);
                    miopenTensorDescriptor_t Bdesc;

                    b_size[2]   = in_n[ti];
                    b_size[3]   = hy_h;
                    b_stride[0] = in_n[ti] * h_stride;
                    b_stride[1] = in_n[ti] * h_stride;
                    b_stride[2] = h_stride;

                    miopenCreateTensorDescriptor(&Bdesc);
                    miopenSetTensorDescriptor(
                        Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                    CopyTensor(handle,
                               miopen::deref(Adesc),
                               reserveSpace,
                               miopen::deref(Bdesc),
                               cy,
                               hid_shift + bacc * hy_stride + bi * 4 * hy_h,
                               hx_shift);

                    // Update time
                    profileSequence(handle, 1);

                    CopyTensor(handle,
                               miopen::deref(Adesc),
                               reserveSpace,
                               miopen::deref(Bdesc),
                               hy,
                               hid_shift + bacc * hy_stride + bi * 5 * hy_h,
                               hx_shift);
                    // Update time
                    profileSequence(handle, 1);
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        rsv_size[2]   = in_n[seqLen - 1 - ti];
                        rsv_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[2] = hy_stride;

                        // active gate i, f, o
                        rsv_size[3] = hy_h * 3;
                        miopenCreateTensorDescriptor(&rsvTensor);
                        miopenSetTensorDescriptor(
                            rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                        offset = hid_shift + baccbi * hy_stride + 4 * hy_h;

                        sigDesc.Forward(handle,
                                        &alpha,
                                        miopen::deref(rsvTensor),
                                        reserveSpace,
                                        &beta,
                                        miopen::deref(rsvTensor),
                                        reserveSpace,
                                        offset,
                                        offset + nLayers * batch_n * hy_stride);
                        // Update time
                        profileSequence(handle, 1);

                        // active gate c
                        rsv_size[3] = hy_h;
                        miopenCreateTensorDescriptor(&rsvTensor);
                        miopenSetTensorDescriptor(
                            rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                        offset = hid_shift + baccbi * hy_stride + 7 * hy_h;

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         offset,
                                         offset + nLayers * batch_n * hy_stride);
                        // Update time
                        profileSequence(handle, 1);

                        // update cell state
                        std::vector<int> a_size(4, 1), a_stride(4, 1);
                        miopenTensorDescriptor_t Adesc;

                        a_size[2]   = in_n[seqLen - 1 - ti];
                        a_size[3]   = hy_h;
                        a_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        a_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        a_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + 7 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);

                        // Update time
                        profileSequence(handle, 1);

                        if(ti == 0)
                        {
                            std::vector<int> b_size(4, 1), b_stride(4, 1);
                            miopenTensorDescriptor_t Bdesc;

                            b_size[2]   = in_n[seqLen - 1 - ti];
                            b_size[3]   = hy_h;
                            b_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                            b_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                            b_stride[2] = h_stride;

                            miopenCreateTensorDescriptor(&Bdesc);
                            miopenSetTensorDescriptor(
                                Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     &alpha1,
                                     miopen::deref(Bdesc),
                                     cx,
                                     &beta_t,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     hid_shift + baccbi * hy_stride + 5 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     hx_shift + hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                        else
                        {
                            if(in_n[seqLen - ti] > 0)
                            {
                                int prec_shift = li * batch_n * hy_stride +
                                                 (baccbi + in_n[seqLen - 1 - ti]) * hy_stride +
                                                 bi * 4 * hy_h + hy_h;

                                std::vector<int> b_size(4, 1), b_stride(4, 1);
                                miopenTensorDescriptor_t Bdesc;

                                b_size[2]   = in_n[seqLen - ti];
                                b_size[3]   = hy_h;
                                b_stride[0] = in_n[seqLen - ti] * hy_stride;
                                b_stride[1] = in_n[seqLen - ti] * hy_stride;
                                b_stride[2] = hy_stride;

                                miopenCreateTensorDescriptor(&Bdesc);
                                miopenSetTensorDescriptor(
                                    Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         miopen::deref(Bdesc),
                                         reserveSpace,
                                         &alpha1,
                                         miopen::deref(Bdesc),
                                         reserveSpace,
                                         &beta_t,
                                         miopen::deref(Bdesc),
                                         reserveSpace,
                                         hid_shift + baccbi * hy_stride + 5 * hy_h +
                                             nLayers * batch_n * hy_stride,
                                         prec_shift,
                                         hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);

                                // Update time
                                profileSequence(handle, 1);
                            }
                        }

                        // active cell state
                        rsv_size[3] = hy_h;
                        miopenCreateTensorDescriptor(&rsvTensor);
                        miopenSetTensorDescriptor(
                            rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                        offset = hid_shift + baccbi * hy_stride + (bi * 4 + 1) * hy_h;

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         offset,
                                         offset + nLayers * batch_n * hy_stride);
                        // Update time
                        profileSequence(handle, 1);

                        // update hidden state
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 6 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h);
                        // Update time
                        profileSequence(handle, 1);

                        // update cy, hy
                        std::vector<int> b_size(4, 1), b_stride(4, 1);
                        miopenTensorDescriptor_t Bdesc;

                        b_size[2]   = in_n[seqLen - 1 - ti];
                        b_size[3]   = hy_h;
                        b_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                        b_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                        b_stride[2] = h_stride;

                        miopenCreateTensorDescriptor(&Bdesc);
                        miopenSetTensorDescriptor(
                            Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                        CopyTensor(handle,
                                   miopen::deref(Adesc),
                                   reserveSpace,
                                   miopen::deref(Bdesc),
                                   cy,
                                   hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h,
                                   hx_shift + hy_h);
                        // Update time
                        profileSequence(handle, 1);

                        CopyTensor(handle,
                                   miopen::deref(Adesc),
                                   reserveSpace,
                                   miopen::deref(Bdesc),
                                   hy,
                                   hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h,
                                   hx_shift + hy_h);
                        // Update time
                        profileSequence(handle, 1);
                    }
                }

                bacc += in_n[ti];
            }

            // hy, cy clean
            if(in_n[0] - in_n[seqLen - 1] > 0)
            {
                std::vector<int> a_size(4, 1), a_stride(4, 1);
                miopenTensorDescriptor_t Adesc;

                a_size[2]   = in_n[0] - in_n[seqLen - 1];
                a_size[3]   = hy_h;
                a_stride[0] = (in_n[0] - in_n[seqLen - 1]) * h_stride;
                a_stride[1] = (in_n[0] - in_n[seqLen - 1]) * h_stride;
                a_stride[2] = h_stride;

                miopenCreateTensorDescriptor(&Adesc);
                miopenSetTensorDescriptor(Adesc, miopenFloat, 4, a_size.data(), a_stride.data());

                alpha0 = 0;
                alpha1 = 0;
                beta_t = 0;

                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         miopen::deref(Adesc),
                         hy,
                         &alpha1,
                         miopen::deref(Adesc),
                         hy,
                         &beta_t,
                         miopen::deref(Adesc),
                         hy,
                         hx_shift + in_n[seqLen - 1] * h_stride,
                         hx_shift + in_n[seqLen - 1] * h_stride,
                         hx_shift + in_n[seqLen - 1] * h_stride);
                // Update time
                profileSequence(handle, 1);

                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         miopen::deref(Adesc),
                         cy,
                         &alpha1,
                         miopen::deref(Adesc),
                         cy,
                         &beta_t,
                         miopen::deref(Adesc),
                         cy,
                         hx_shift + in_n[seqLen - 1] * h_stride,
                         hx_shift + in_n[seqLen - 1] * h_stride,
                         hx_shift + in_n[seqLen - 1] * h_stride);
                // Update time
                profileSequence(handle, 1);
            }
        } // end for nLayers

        // output
        int prelayer_shift = (nLayers - 1) * batch_n * hy_stride + bi * 5 * hy_h;

        std::vector<int> src_size(4, 1), src_stride(4, 1), dest_size(4, 1), dest_stride(4, 1);
        miopenTensorDescriptor_t srcTensor, destTensor;

        src_size[2]   = batch_n;
        src_size[3]   = hy_h * bi;
        src_stride[0] = batch_n * hy_stride;
        src_stride[1] = batch_n * hy_stride;
        src_stride[2] = hy_stride;

        dest_size[2]   = batch_n;
        dest_size[3]   = out_h;
        dest_stride[0] = batch_n * out_stride;
        dest_stride[1] = batch_n * out_stride;
        dest_stride[2] = out_stride;

        miopenCreateTensorDescriptor(&srcTensor);
        miopenCreateTensorDescriptor(&destTensor);
        miopenSetTensorDescriptor(srcTensor, miopenFloat, 4, src_size.data(), src_stride.data());
        miopenSetTensorDescriptor(destTensor, miopenFloat, 4, dest_size.data(), dest_stride.data());

        CopyTensor(handle,
                   miopen::deref(srcTensor),
                   reserveSpace,
                   miopen::deref(destTensor),
                   y,
                   prelayer_shift,
                   0);
        // Update time
        profileSequence(handle, 1);
#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenGRU)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("run gru gpu fwd \n");
        float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        for(int li = 0; li < nLayers; li++)
        {
            int hid_shift = li * batch_n * hy_stride;
            int hx_shift  = li * hy_n * h_stride;
            int wei_shift_bias_temp =
                inputMode == miopenRNNskip
                    ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                    : (wei_shift_bias + li * 2 * wei_stride);

            // from input
            if(li == 0)
            {
                if(inputMode == miopenRNNskip)
                {
                    std::vector<int> src_size(4, 1), src_stride(4, 1), dest_size(4, 1),
                        dest_stride(4, 1);
                    miopenTensorDescriptor_t srcTensor, destTensor;

                    src_size[2]    = batch_n;
                    src_size[3]    = hy_h;
                    src_stride[0]  = batch_n * in_stride;
                    src_stride[1]  = batch_n * in_stride;
                    src_stride[2]  = in_stride;
                    dest_size[2]   = batch_n;
                    dest_size[3]   = hy_h;
                    dest_stride[0] = batch_n * hy_stride;
                    dest_stride[1] = batch_n * hy_stride;
                    dest_stride[2] = hy_stride;
                    miopenCreateTensorDescriptor(&srcTensor);
                    miopenCreateTensorDescriptor(&destTensor);
                    miopenSetTensorDescriptor(
                        srcTensor, miopenFloat, 4, src_size.data(), src_stride.data());
                    miopenSetTensorDescriptor(
                        destTensor, miopenFloat, 4, dest_size.data(), dest_stride.data());

                    for(int gi = 0; gi < 3; gi++)
                    {
                        CopyTensor(handle,
                                   miopen::deref(srcTensor),
                                   x,
                                   miopen::deref(destTensor),
                                   reserveSpace,
                                   0,
                                   gi * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        if(dirMode)
                        {
                            CopyTensor(handle,
                                       miopen::deref(srcTensor),
                                       x,
                                       miopen::deref(destTensor),
                                       reserveSpace,
                                       0,
                                       (gi + 3) * hy_h);
                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }
                        }
                    }
                }
                else
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
                    profileSequence(handle, 0);
                }
            }
            else
            {
                int wei_shift =
                    (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
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
                gg.RunGemm(
                    handle, reserveSpace, w, reserveSpace, prelayer_shift, wei_shift, hid_shift);

                // Update time
                profileSequence(handle, 1);
            }

            // from hidden state
            bacc   = 0;
            baccbi = batch_n;
            for(int ti = 0; ti < seqLen; ti++)
            {
                baccbi -= in_n[seqLen - 1 - ti];
                int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
                int pretime_shift;
                if(ti > 0)
                {
                    pretime_shift = li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride +
                                    bi * 3 * hy_h;
                }

                if(ti == 0)
                {
                    if(in_n[ti] > 0)
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
                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   reserveSpace,
                                   hx_shift,
                                   wei_shift,
                                   hid_shift + bacc * hy_stride);

                        // Update time
                        profileSequence(handle, 1);

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
                                   wei_shift + 2 * hy_h,
                                   hid_shift + bacc * hy_stride + bi * 3 * hy_h);

                        // Update time
                        profileSequence(handle, 1);
                    }

                    if(dirMode)
                    {
                        if(in_n[seqLen - 1 - ti] > 0)
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
                            gg.RunGemm(handle,
                                       hx,
                                       w,
                                       reserveSpace,
                                       hx_shift + hy_h,
                                       wei_shift + 3 * hy_h,
                                       hid_shift + baccbi * hy_stride + 3 * hy_h);

                            // Update time
                            profileSequence(handle, 1);

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
                                       wei_shift + 5 * hy_h,
                                       hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }
                else
                {
                    if(in_n[ti] > 0)
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
                        gg.RunGemm(handle,
                                   hy,
                                   w,
                                   reserveSpace,
                                   hx_shift,
                                   wei_shift,
                                   hid_shift + bacc * hy_stride);

                        // Update time
                        profileSequence(handle, 1);

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
                        gg.RunGemm(handle,
                                   hy,
                                   w,
                                   reserveSpace,
                                   hx_shift,
                                   wei_shift + 2 * hy_h,
                                   hid_shift + bacc * hy_stride + bi * 3 * hy_h);

                        // Update time
                        profileSequence(handle, 1);
                    }

                    if(dirMode)
                    {
                        if(in_n[seqLen - 1 - ti] > 0)
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
                            gg.RunGemm(handle,
                                       hy,
                                       w,
                                       reserveSpace,
                                       hx_shift + hy_h,
                                       wei_shift + 3 * hy_h,
                                       hid_shift + baccbi * hy_stride + 3 * hy_h);

                            // Update time
                            profileSequence(handle, 1);

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
                            gg.RunGemm(handle,
                                       hy,
                                       w,
                                       reserveSpace,
                                       hx_shift + hy_h,
                                       wei_shift + 5 * hy_h,
                                       hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }

                // update hidden status
                float alpha = 1, beta = 0;
                ActivationDescriptor tanhDesc, sigDesc;
                size_t offset;

                sigDesc  = {miopenActivationLOGISTIC, 1, 0, 1};
                tanhDesc = {miopenActivationTANH, 1, 1, 1};

                std::vector<int> rsv_size(4, 1), rsv_stride(4, 1);
                miopenTensorDescriptor_t rsvTensor;

                if(in_n[ti] > 0)
                {
                    // apply bias
                    if(biasMode)
                    {
                        if(li == 0 && inputMode == miopenRNNskip)
                        {
                            std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1),
                                c_stride(4, 1);
                            miopenTensorDescriptor_t Adesc, Cdesc;

                            a_size[2]   = 1;
                            a_size[3]   = 2 * hy_h;
                            a_stride[0] = wei_stride;
                            a_stride[1] = wei_stride;
                            a_stride[2] = wei_stride;
                            c_size[2]   = 1;
                            c_size[3]   = 2 * hy_h;
                            c_stride[0] = hy_stride;
                            c_stride[1] = hy_stride;
                            c_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Adesc);
                            miopenCreateTensorDescriptor(&Cdesc);
                            miopenSetTensorDescriptor(
                                Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                            miopenSetTensorDescriptor(
                                Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                            alpha0 = 1;
                            alpha1 = 0;
                            beta_t = 1;

                            for(int bs = 0; bs < in_n[ti]; bs++)
                            {
                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         miopen::deref(Adesc),
                                         w,
                                         &alpha1,
                                         miopen::deref(Adesc),
                                         w,
                                         &beta_t,
                                         miopen::deref(Cdesc),
                                         reserveSpace,
                                         wei_shift_bias,
                                         wei_shift_bias,
                                         hid_shift + (bacc + bs) * hy_stride);
                                // Update time
                                if(handle.IsProfilingEnabled())
                                {
                                    time_0 = handle.GetKernelTime();
                                    handle.AccumKernelTime(time_0);
                                }
                            }

                            //
                            a_size[2]   = 1;
                            a_size[3]   = hy_h;
                            a_stride[0] = wei_stride;
                            a_stride[1] = wei_stride;
                            a_stride[2] = wei_stride;
                            c_size[2]   = 1;
                            c_size[3]   = hy_h;
                            c_stride[0] = hy_stride;
                            c_stride[1] = hy_stride;
                            c_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Adesc);
                            miopenCreateTensorDescriptor(&Cdesc);
                            miopenSetTensorDescriptor(
                                Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                            miopenSetTensorDescriptor(
                                Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                            for(int bs = 0; bs < in_n[ti]; bs++)
                            {
                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         miopen::deref(Adesc),
                                         w,
                                         &alpha1,
                                         miopen::deref(Adesc),
                                         w,
                                         &beta_t,
                                         miopen::deref(Cdesc),
                                         reserveSpace,
                                         wei_shift_bias + 2 * hy_h,
                                         wei_shift_bias + 2 * hy_h,
                                         hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h);
                                // Update time
                                if(handle.IsProfilingEnabled())
                                {
                                    time_0 = handle.GetKernelTime();
                                    handle.AccumKernelTime(time_0);
                                }
                            }
                        }
                        else
                        {
                            std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1),
                                c_stride(4, 1);
                            miopenTensorDescriptor_t Adesc, Cdesc;

                            a_size[2]   = 1;
                            a_size[3]   = 3 * hy_h;
                            a_stride[0] = wei_stride;
                            a_stride[1] = wei_stride;
                            a_stride[2] = wei_stride;
                            c_size[2]   = 1;
                            c_size[3]   = 3 * hy_h;
                            c_stride[0] = hy_stride;
                            c_stride[1] = hy_stride;
                            c_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Adesc);
                            miopenCreateTensorDescriptor(&Cdesc);
                            miopenSetTensorDescriptor(
                                Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                            miopenSetTensorDescriptor(
                                Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                            alpha0 = 1;
                            alpha1 = 0;
                            beta_t = 1;

                            for(int bs = 0; bs < in_n[ti]; bs++)
                            {
                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         miopen::deref(Adesc),
                                         w,
                                         &alpha1,
                                         miopen::deref(Adesc),
                                         w,
                                         &beta_t,
                                         miopen::deref(Cdesc),
                                         reserveSpace,
                                         wei_shift_bias_temp,
                                         wei_shift_bias_temp,
                                         hid_shift + (bacc + bs) * hy_stride);

                                // Update time
                                profileSequence(handle, 1);
                            }

                            //
                            a_size[2]   = 1;
                            a_size[3]   = 2 * hy_h;
                            a_stride[0] = wei_stride;
                            a_stride[1] = wei_stride;
                            a_stride[2] = wei_stride;
                            c_size[2]   = 1;
                            c_size[3]   = 2 * hy_h;
                            c_stride[0] = hy_stride;
                            c_stride[1] = hy_stride;
                            c_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Adesc);
                            miopenCreateTensorDescriptor(&Cdesc);
                            miopenSetTensorDescriptor(
                                Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                            miopenSetTensorDescriptor(
                                Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                            for(int bs = 0; bs < in_n[ti]; bs++)
                            {
                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         miopen::deref(Adesc),
                                         w,
                                         &alpha1,
                                         miopen::deref(Adesc),
                                         w,
                                         &beta_t,
                                         miopen::deref(Cdesc),
                                         reserveSpace,
                                         wei_shift_bias_temp + wei_stride,
                                         wei_shift_bias_temp + wei_stride,
                                         hid_shift + (bacc + bs) * hy_stride);
                                // Update time
                                if(handle.IsProfilingEnabled())
                                {
                                    time_0 = handle.GetKernelTime();
                                    handle.AccumKernelTime(time_0);
                                }
                            }

                            //
                            a_size[2]   = 1;
                            a_size[3]   = hy_h;
                            a_stride[0] = wei_stride;
                            a_stride[1] = wei_stride;
                            a_stride[2] = wei_stride;
                            c_size[2]   = 1;
                            c_size[3]   = hy_h;
                            c_stride[0] = hy_stride;
                            c_stride[1] = hy_stride;
                            c_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Adesc);
                            miopenCreateTensorDescriptor(&Cdesc);
                            miopenSetTensorDescriptor(
                                Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                            miopenSetTensorDescriptor(
                                Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                            for(int bs = 0; bs < in_n[ti]; bs++)
                            {
                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         miopen::deref(Adesc),
                                         w,
                                         &alpha1,
                                         miopen::deref(Adesc),
                                         w,
                                         &beta_t,
                                         miopen::deref(Cdesc),
                                         reserveSpace,
                                         wei_shift_bias_temp + wei_stride + 2 * hy_h,
                                         wei_shift_bias_temp + wei_stride + 2 * hy_h,
                                         hid_shift + (bacc + bs) * hy_stride + bi * 3 * hy_h);
                                // Update time
                                if(handle.IsProfilingEnabled())
                                {
                                    time_0 = handle.GetKernelTime();
                                    handle.AccumKernelTime(time_0);
                                }
                            }
                        }
                    }

                    // active z, r gate
                    rsv_size[2]   = in_n[ti];
                    rsv_stride[0] = in_n[ti] * hy_stride;
                    rsv_stride[1] = in_n[ti] * hy_stride;
                    rsv_stride[2] = hy_stride;

                    rsv_size[3] = 2 * hy_h;
                    miopenCreateTensorDescriptor(&rsvTensor);
                    miopenSetTensorDescriptor(
                        rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                    offset = hid_shift + bacc * hy_stride;

                    sigDesc.Forward(handle,
                                    &alpha,
                                    miopen::deref(rsvTensor),
                                    reserveSpace,
                                    &beta,
                                    miopen::deref(rsvTensor),
                                    reserveSpace,
                                    offset,
                                    offset + nLayers * batch_n * hy_stride);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    // calculate c gate
                    std::vector<int> a_size(4, 1), a_stride(4, 1);
                    miopenTensorDescriptor_t Adesc;

                    a_size[2]   = in_n[ti];
                    a_size[3]   = hy_h;
                    a_stride[0] = in_n[ti] * hy_stride;
                    a_stride[1] = in_n[ti] * hy_stride;
                    a_stride[2] = hy_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + hy_h + nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + 2 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    // Update time
                    profileSequence(handle, 1);

                    // active c gate
                    rsv_size[2]   = in_n[ti];
                    rsv_stride[0] = in_n[ti] * hy_stride;
                    rsv_stride[1] = in_n[ti] * hy_stride;
                    rsv_stride[2] = hy_stride;

                    rsv_size[3] = hy_h;
                    miopenCreateTensorDescriptor(&rsvTensor);
                    miopenSetTensorDescriptor(
                        rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                    tanhDesc.Forward(handle,
                                     &alpha,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     offset + 2 * hy_h,
                                     offset + 2 * hy_h + nLayers * batch_n * hy_stride);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    // calculate hidden state
                    alpha0 = -1;
                    alpha1 = 1;
                    beta_t = 0;
                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + 2 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + 2 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    // Update time
                    profileSequence(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;
                    if(ti == 0)
                    {
                        std::vector<int> b_size(4, 1), b_stride(4, 1);
                        miopenTensorDescriptor_t Bdesc;

                        b_size[2]   = in_n[ti];
                        b_size[3]   = hy_h;
                        b_stride[0] = in_n[ti] * h_stride;
                        b_stride[1] = in_n[ti] * h_stride;
                        b_stride[2] = h_stride;

                        miopenCreateTensorDescriptor(&Bdesc);
                        miopenSetTensorDescriptor(
                            Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Bdesc),
                                 hx,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 hid_shift + bacc * hy_stride + nLayers * batch_n * hy_stride,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    }
                    else
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 hid_shift + bacc * hy_stride + nLayers * batch_n * hy_stride,
                                 pretime_shift,
                                 hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    }
                    // Update time
                    profileSequence(handle, 1);

                    // update hy
                    std::vector<int> b_size(4, 1), b_stride(4, 1);
                    miopenTensorDescriptor_t Bdesc;

                    b_size[2]   = in_n[ti];
                    b_size[3]   = hy_h;
                    b_stride[0] = in_n[ti] * h_stride;
                    b_stride[1] = in_n[ti] * h_stride;
                    b_stride[2] = h_stride;

                    miopenCreateTensorDescriptor(&Bdesc);
                    miopenSetTensorDescriptor(
                        Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                    CopyTensor(handle,
                               miopen::deref(Adesc),
                               reserveSpace,
                               miopen::deref(Bdesc),
                               hy,
                               hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                               hx_shift);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }
                }

                if(dirMode)
                {
                    pretime_shift = li * batch_n * hy_stride +
                                    (baccbi + in_n[seqLen - 1 - ti]) * hy_stride + bi * 3 * hy_h +
                                    hy_h;

                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        // apply bias
                        if(biasMode)
                        {
                            if(li == 0 && inputMode == miopenRNNskip)
                            {
                                std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1),
                                    c_stride(4, 1);
                                miopenTensorDescriptor_t Adesc, Cdesc;

                                a_size[2]   = 1;
                                a_size[3]   = 2 * hy_h;
                                a_stride[0] = wei_stride;
                                a_stride[1] = wei_stride;
                                a_stride[2] = wei_stride;
                                c_size[2]   = 1;
                                c_size[3]   = 2 * hy_h;
                                c_stride[0] = hy_stride;
                                c_stride[1] = hy_stride;
                                c_stride[2] = hy_stride;

                                miopenCreateTensorDescriptor(&Adesc);
                                miopenCreateTensorDescriptor(&Cdesc);
                                miopenSetTensorDescriptor(
                                    Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                                miopenSetTensorDescriptor(
                                    Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                                alpha0 = 1;
                                alpha1 = 0;
                                beta_t = 1;

                                for(int bs = 0; bs < in_n[seqLen - 1 - ti]; bs++)
                                {
                                    OpTensor(handle,
                                             miopenTensorOpAdd,
                                             &alpha0,
                                             miopen::deref(Adesc),
                                             w,
                                             &alpha1,
                                             miopen::deref(Adesc),
                                             w,
                                             &beta_t,
                                             miopen::deref(Cdesc),
                                             reserveSpace,
                                             wei_shift_bias + 3 * hy_h,
                                             wei_shift_bias + 3 * hy_h,
                                             hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h);
                                    // Update time
                                    if(handle.IsProfilingEnabled())
                                    {
                                        time_0 = handle.GetKernelTime();
                                        handle.AccumKernelTime(time_0);
                                    }
                                }

                                //
                                a_size[2]   = 1;
                                a_size[3]   = hy_h;
                                a_stride[0] = wei_stride;
                                a_stride[1] = wei_stride;
                                a_stride[2] = wei_stride;
                                c_size[2]   = 1;
                                c_size[3]   = hy_h;
                                c_stride[0] = hy_stride;
                                c_stride[1] = hy_stride;
                                c_stride[2] = hy_stride;

                                miopenCreateTensorDescriptor(&Adesc);
                                miopenCreateTensorDescriptor(&Cdesc);
                                miopenSetTensorDescriptor(
                                    Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                                miopenSetTensorDescriptor(
                                    Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                                for(int bs = 0; bs < in_n[seqLen - 1 - ti]; bs++)
                                {
                                    OpTensor(handle,
                                             miopenTensorOpAdd,
                                             &alpha0,
                                             miopen::deref(Adesc),
                                             w,
                                             &alpha1,
                                             miopen::deref(Adesc),
                                             w,
                                             &beta_t,
                                             miopen::deref(Cdesc),
                                             reserveSpace,
                                             wei_shift_bias + 5 * hy_h,
                                             wei_shift_bias + 5 * hy_h,
                                             hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                                 hy_h);
                                    // Update time
                                    if(handle.IsProfilingEnabled())
                                    {
                                        time_0 = handle.GetKernelTime();
                                        handle.AccumKernelTime(time_0);
                                    }
                                }
                            }
                            else
                            {
                                std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1),
                                    c_stride(4, 1);
                                miopenTensorDescriptor_t Adesc, Cdesc;

                                a_size[2]   = 1;
                                a_size[3]   = 3 * hy_h;
                                a_stride[0] = wei_stride;
                                a_stride[1] = wei_stride;
                                a_stride[2] = wei_stride;
                                c_size[2]   = 1;
                                c_size[3]   = 3 * hy_h;
                                c_stride[0] = hy_stride;
                                c_stride[1] = hy_stride;
                                c_stride[2] = hy_stride;

                                miopenCreateTensorDescriptor(&Adesc);
                                miopenCreateTensorDescriptor(&Cdesc);
                                miopenSetTensorDescriptor(
                                    Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                                miopenSetTensorDescriptor(
                                    Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                                alpha0 = 1;
                                alpha1 = 0;
                                beta_t = 1;

                                for(int bs = 0; bs < in_n[seqLen - 1 - ti]; bs++)
                                {
                                    OpTensor(handle,
                                             miopenTensorOpAdd,
                                             &alpha0,
                                             miopen::deref(Adesc),
                                             w,
                                             &alpha1,
                                             miopen::deref(Adesc),
                                             w,
                                             &beta_t,
                                             miopen::deref(Cdesc),
                                             reserveSpace,
                                             wei_shift_bias_temp + 3 * hy_h,
                                             wei_shift_bias_temp + 3 * hy_h,
                                             hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h);
                                    // Update time
                                    if(handle.IsProfilingEnabled())
                                    {
                                        time_0 = handle.GetKernelTime();
                                        handle.AccumKernelTime(time_0);
                                    }
                                }

                                //
                                a_size[2]   = 1;
                                a_size[3]   = 2 * hy_h;
                                a_stride[0] = wei_stride;
                                a_stride[1] = wei_stride;
                                a_stride[2] = wei_stride;
                                c_size[2]   = 1;
                                c_size[3]   = 2 * hy_h;
                                c_stride[0] = hy_stride;
                                c_stride[1] = hy_stride;
                                c_stride[2] = hy_stride;

                                miopenCreateTensorDescriptor(&Adesc);
                                miopenCreateTensorDescriptor(&Cdesc);
                                miopenSetTensorDescriptor(
                                    Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                                miopenSetTensorDescriptor(
                                    Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                                for(int bs = 0; bs < in_n[seqLen - 1 - ti]; bs++)
                                {
                                    OpTensor(handle,
                                             miopenTensorOpAdd,
                                             &alpha0,
                                             miopen::deref(Adesc),
                                             w,
                                             &alpha1,
                                             miopen::deref(Adesc),
                                             w,
                                             &beta_t,
                                             miopen::deref(Cdesc),
                                             reserveSpace,
                                             wei_shift_bias_temp + wei_stride + 3 * hy_h,
                                             wei_shift_bias_temp + wei_stride + 3 * hy_h,
                                             hid_shift + (baccbi + bs) * hy_stride + 3 * hy_h);
                                    // Update time
                                    if(handle.IsProfilingEnabled())
                                    {
                                        time_0 = handle.GetKernelTime();
                                        handle.AccumKernelTime(time_0);
                                    }
                                }

                                //
                                a_size[2]   = 1;
                                a_size[3]   = hy_h;
                                a_stride[0] = wei_stride;
                                a_stride[1] = wei_stride;
                                a_stride[2] = wei_stride;
                                c_size[2]   = 1;
                                c_size[3]   = hy_h;
                                c_stride[0] = hy_stride;
                                c_stride[1] = hy_stride;
                                c_stride[2] = hy_stride;

                                miopenCreateTensorDescriptor(&Adesc);
                                miopenCreateTensorDescriptor(&Cdesc);
                                miopenSetTensorDescriptor(
                                    Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                                miopenSetTensorDescriptor(
                                    Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                                for(int bs = 0; bs < in_n[seqLen - 1 - ti]; bs++)
                                {
                                    OpTensor(handle,
                                             miopenTensorOpAdd,
                                             &alpha0,
                                             miopen::deref(Adesc),
                                             w,
                                             &alpha1,
                                             miopen::deref(Adesc),
                                             w,
                                             &beta_t,
                                             miopen::deref(Cdesc),
                                             reserveSpace,
                                             wei_shift_bias_temp + wei_stride + 5 * hy_h,
                                             wei_shift_bias_temp + wei_stride + 5 * hy_h,
                                             hid_shift + (baccbi + bs) * hy_stride + bi * 3 * hy_h +
                                                 hy_h);
                                    // Update time
                                    if(handle.IsProfilingEnabled())
                                    {
                                        time_0 = handle.GetKernelTime();
                                        handle.AccumKernelTime(time_0);
                                    }
                                }
                            }
                        }

                        // active z, r gate
                        rsv_size[2]   = in_n[seqLen - 1 - ti];
                        rsv_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[2] = hy_stride;

                        rsv_size[3] = 2 * hy_h;
                        miopenCreateTensorDescriptor(&rsvTensor);
                        miopenSetTensorDescriptor(
                            rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                        offset = hid_shift + baccbi * hy_stride;

                        sigDesc.Forward(handle,
                                        &alpha,
                                        miopen::deref(rsvTensor),
                                        reserveSpace,
                                        &beta,
                                        miopen::deref(rsvTensor),
                                        reserveSpace,
                                        offset + 3 * hy_h,
                                        offset + 3 * hy_h + nLayers * batch_n * hy_stride);
                        // Update time
                        profileSequence(handle, 1);

                        // calculate c gate
                        std::vector<int> a_size(4, 1), a_stride(4, 1);
                        miopenTensorDescriptor_t Adesc;

                        a_size[2]   = in_n[seqLen - 1 - ti];
                        a_size[3]   = hy_h;
                        a_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        a_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        a_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h);
                        // Update time
                        profileSequence(handle, 1);

                        // active c gate
                        rsv_size[2]   = in_n[seqLen - 1 - ti];
                        rsv_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[2] = hy_stride;

                        rsv_size[3] = hy_h;
                        miopenCreateTensorDescriptor(&rsvTensor);
                        miopenSetTensorDescriptor(
                            rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         offset + 5 * hy_h,
                                         offset + 5 * hy_h + nLayers * batch_n * hy_stride);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        // calculate hidden state
                        alpha0 = -1;
                        alpha1 = 1;
                        beta_t = 0;
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 3 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;
                        if(ti == 0)
                        {
                            std::vector<int> b_size(4, 1), b_stride(4, 1);
                            miopenTensorDescriptor_t Bdesc;

                            b_size[2]   = in_n[seqLen - 1 - ti];
                            b_size[3]   = hy_h;
                            b_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                            b_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                            b_stride[2] = h_stride;

                            miopenCreateTensorDescriptor(&Bdesc);
                            miopenSetTensorDescriptor(
                                Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     &alpha1,
                                     miopen::deref(Bdesc),
                                     hx,
                                     &beta_t,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     hx_shift + hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     &beta_t,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     pretime_shift,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        }
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        // update hy
                        std::vector<int> b_size(4, 1), b_stride(4, 1);
                        miopenTensorDescriptor_t Bdesc;

                        b_size[2]   = in_n[seqLen - 1 - ti];
                        b_size[3]   = hy_h;
                        b_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                        b_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                        b_stride[2] = h_stride;

                        miopenCreateTensorDescriptor(&Bdesc);
                        miopenSetTensorDescriptor(
                            Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                        CopyTensor(handle,
                                   miopen::deref(Adesc),
                                   reserveSpace,
                                   miopen::deref(Bdesc),
                                   hy,
                                   hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                   hx_shift + hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }
                }

                bacc += in_n[ti];
            }

            // hy, cy clean
            if(in_n[0] - in_n[seqLen - 1] > 0)
            {
                std::vector<int> a_size(4, 1), a_stride(4, 1);
                miopenTensorDescriptor_t Adesc;

                a_size[2]   = in_n[0] - in_n[seqLen - 1];
                a_size[3]   = hy_h;
                a_stride[0] = (in_n[0] - in_n[seqLen - 1]) * h_stride;
                a_stride[1] = (in_n[0] - in_n[seqLen - 1]) * h_stride;
                a_stride[2] = h_stride;

                miopenCreateTensorDescriptor(&Adesc);
                miopenSetTensorDescriptor(Adesc, miopenFloat, 4, a_size.data(), a_stride.data());

                alpha0 = 0;
                alpha1 = 0;
                beta_t = 0;

                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         miopen::deref(Adesc),
                         hy,
                         &alpha1,
                         miopen::deref(Adesc),
                         hy,
                         &beta_t,
                         miopen::deref(Adesc),
                         hy,
                         hx_shift + in_n[seqLen - 1] * h_stride,
                         hx_shift + in_n[seqLen - 1] * h_stride,
                         hx_shift + in_n[seqLen - 1] * h_stride);
                // Update time
                profileSequence(handle, 1);
            }
        }

        // output
        int prelayer_shift = (nLayers - 1) * batch_n * hy_stride + bi * 3 * hy_h;

        std::vector<int> src_size(4, 1), src_stride(4, 1), dest_size(4, 1), dest_stride(4, 1);
        miopenTensorDescriptor_t srcTensor, destTensor;

        src_size[2]   = batch_n;
        src_size[3]   = hy_h * bi;
        src_stride[0] = batch_n * hy_stride;
        src_stride[1] = batch_n * hy_stride;
        src_stride[2] = hy_stride;

        dest_size[2]   = batch_n;
        dest_size[3]   = out_h;
        dest_stride[0] = batch_n * out_stride;
        dest_stride[1] = batch_n * out_stride;
        dest_stride[2] = out_stride;

        miopenCreateTensorDescriptor(&srcTensor);
        miopenCreateTensorDescriptor(&destTensor);
        miopenSetTensorDescriptor(srcTensor, miopenFloat, 4, src_size.data(), src_stride.data());
        miopenSetTensorDescriptor(destTensor, miopenFloat, 4, dest_size.data(), dest_stride.data());

        CopyTensor(handle,
                   miopen::deref(srcTensor),
                   reserveSpace,
                   miopen::deref(destTensor),
                   y,
                   prelayer_shift,
                   0);
        // Update time
        profileSequence(handle, 1);
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

    if(dx == nullptr || w == nullptr || dy == nullptr)
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

    if(in_h == 0 || hy_h == 0 || hy_n == 0 || hy_d == 0 || out_h == 0)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    int batch_n = 0;
    for(int i = 0; i < seqLen; i++)
    {
        int batchval, inputvec, batchvalout, outputvec;
        std::tie(batchval, inputvec)     = miopen::tien<2>(dxDesc[i].GetLengths());
        std::tie(batchvalout, outputvec) = miopen::tien<2>(dyDesc[i].GetLengths());
        if(batchval != batchvalout)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        in_n.push_back(batchval);
        batch_n += dxDesc[i].GetLengths()[0];
    }

    int bacc, baccbi;
    int bi = dirMode ? 2 : 1;

    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * workspaceScale;
    int h_stride   = hy_h * bi;
    int out_stride = out_h;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;

    if(inputMode == miopenRNNskip)
    {
        if(in_h != hy_h)
        {
            printf("The input tensor size must equal to the hidden state size of the network in "
                   "SKIP_INPUT mode!\n");
            MIOPEN_THROW(miopenStatusBadParm);
        }
        in_h = 0;
    }

    float alpha0, alpha1, beta_t;

    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
    {
#if MIOPEN_USE_MIOPENGEMM
        printf("run rnn gpu bwd data \n");
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
                std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                miopenTensorDescriptor_t Adesc, Cdesc;

                a_size[2]   = batch_n;
                a_size[3]   = out_h;
                a_stride[0] = batch_n * out_stride;
                a_stride[1] = batch_n * out_stride;
                a_stride[2] = out_stride;
                c_size[2]   = batch_n;
                c_size[3]   = hy_h * bi;
                c_stride[0] = batch_n * hy_stride;
                c_stride[1] = batch_n * hy_stride;
                c_stride[2] = hy_stride;

                miopenCreateTensorDescriptor(&Adesc);
                miopenCreateTensorDescriptor(&Cdesc);
                miopenSetTensorDescriptor(Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                miopenSetTensorDescriptor(Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                alpha0 = 1;
                alpha1 = 0;
                beta_t = 1;

                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         miopen::deref(Adesc),
                         dy,
                         &alpha1,
                         miopen::deref(Adesc),
                         dy,
                         &beta_t,
                         miopen::deref(Cdesc),
                         workSpace,
                         0,
                         0,
                         hid_shift);

                // Update time
                profileSequence(handle, 1);
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
                if(handle.IsProfilingEnabled())
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

                // from post state
                std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                miopenTensorDescriptor_t Adesc, Cdesc;

                alpha0 = 1;
                alpha1 = 0;
                beta_t = 1;

                // activation
                float alpha = 1, beta = 0;
                ActivationDescriptor activDesc;
                size_t offset;

                if(rnnMode == miopenRNNRELU)
                {
                    activDesc = {miopenActivationRELU, 1, 0, 1};
                }
                else if(rnnMode == miopenRNNTANH)
                {
                    activDesc = {miopenActivationTANH, 1, 1, 1};
                }

                std::vector<int> rsv_size(4, 1), rsv_stride(4, 1);
                miopenTensorDescriptor_t rsvTensor;

                if(in_n[ti] > 0)
                {
                    // from post state
                    a_size[2]   = in_n[ti];
                    a_size[3]   = hy_h;
                    a_stride[0] = in_n[ti] * h_stride;
                    a_stride[1] = in_n[ti] * h_stride;
                    a_stride[2] = h_stride;
                    c_size[2]   = in_n[ti];
                    c_size[3]   = hy_h;
                    c_stride[0] = in_n[ti] * hy_stride;
                    c_stride[1] = in_n[ti] * hy_stride;
                    c_stride[2] = hy_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenCreateTensorDescriptor(&Cdesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                    miopenSetTensorDescriptor(
                        Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                    if(ti == seqLen - 1)
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 dhy,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 dhy,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 workSpace,
                                 hx_shift,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride);
                    }
                    else
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 dhx,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 dhx,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 workSpace,
                                 hx_shift,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride);
                    }

                    // activation
                    offset        = hid_shift + bacc * hy_stride;
                    rsv_size[2]   = in_n[ti];
                    rsv_size[3]   = hy_h;
                    rsv_stride[0] = in_n[ti] * hy_stride;
                    rsv_stride[1] = in_n[ti] * hy_stride;
                    rsv_stride[2] = hy_stride;

                    miopenCreateTensorDescriptor(&rsvTensor);
                    miopenSetTensorDescriptor(
                        rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                    activDesc.Backward(handle,
                                       &alpha,
                                       miopen::deref(rsvTensor),
                                       reserveSpace,
                                       miopen::deref(rsvTensor),
                                       workSpace,
                                       miopen::deref(rsvTensor),
                                       reserveSpace,
                                       &beta,
                                       miopen::deref(rsvTensor),
                                       workSpace,
                                       offset + nLayers * batch_n * hy_stride,
                                       offset,
                                       offset,
                                       offset);

                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

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
                                               h_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, workSpace, w, dhx, false);
                    gg.RunGemm(handle,
                               workSpace,
                               w,
                               dhx,
                               hid_shift + bacc * hy_stride,
                               wei_shift,
                               hx_shift);

                    // Update time
                    profileSequence(handle, 1);
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        // from post state
                        a_size[2]   = in_n[seqLen - 1 - ti];
                        a_size[3]   = hy_h;
                        a_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                        a_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                        a_stride[2] = h_stride;
                        c_size[2]   = in_n[seqLen - 1 - ti];
                        c_size[3]   = hy_h;
                        c_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        c_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        c_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenCreateTensorDescriptor(&Cdesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                        miopenSetTensorDescriptor(
                            Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                        if(ti == seqLen - 1)
                        {
                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     dhy,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     dhy,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     workSpace,
                                     hx_shift + hy_h,
                                     hx_shift + hy_h,
                                     hid_shift + baccbi * hy_stride + hy_h);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     dhx,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     dhx,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     workSpace,
                                     hx_shift + hy_h,
                                     hx_shift + hy_h,
                                     hid_shift + baccbi * hy_stride + hy_h);
                        }

                        // activation
                        offset        = hid_shift + baccbi * hy_stride + hy_h;
                        rsv_size[2]   = in_n[seqLen - 1 - ti];
                        rsv_size[3]   = hy_h;
                        rsv_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&rsvTensor);
                        miopenSetTensorDescriptor(
                            rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                        activDesc.Backward(handle,
                                           &alpha,
                                           miopen::deref(rsvTensor),
                                           reserveSpace,
                                           miopen::deref(rsvTensor),
                                           workSpace,
                                           miopen::deref(rsvTensor),
                                           reserveSpace,
                                           &beta,
                                           miopen::deref(rsvTensor),
                                           workSpace,
                                           offset + nLayers * batch_n * hy_stride,
                                           offset,
                                           offset,
                                           offset);

                        // Update time
                        profileSequence(handle, 1);

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
                                                   h_stride,
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
                        profileSequence(handle, 1);
                    }
                }

                baccbi += in_n[seqLen - 1 - ti];
            }
        }

        // dinput
        if(inputMode == miopenRNNskip)
        {
            std::vector<int> src_size(4, 1), src_stride(4, 1), dest_size(4, 1), dest_stride(4, 1);
            miopenTensorDescriptor_t srcTensor, destTensor;

            src_size[2]    = batch_n;
            src_size[3]    = hy_h;
            src_stride[0]  = batch_n * hy_stride;
            src_stride[1]  = batch_n * hy_stride;
            src_stride[2]  = hy_stride;
            dest_size[2]   = batch_n;
            dest_size[3]   = hy_h;
            dest_stride[0] = batch_n * in_stride;
            dest_stride[1] = batch_n * in_stride;
            dest_stride[2] = in_stride;

            miopenCreateTensorDescriptor(&srcTensor);
            miopenCreateTensorDescriptor(&destTensor);
            miopenSetTensorDescriptor(
                srcTensor, miopenFloat, 4, src_size.data(), src_stride.data());
            miopenSetTensorDescriptor(
                destTensor, miopenFloat, 4, dest_size.data(), dest_stride.data());

            CopyTensor(
                handle, miopen::deref(srcTensor), workSpace, miopen::deref(destTensor), dx, 0, 0);

            if(dirMode)
            {
                alpha0 = 1;
                alpha1 = 0;
                beta_t = 1;

                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         miopen::deref(srcTensor),
                         workSpace,
                         &alpha1,
                         miopen::deref(srcTensor),
                         workSpace,
                         &beta_t,
                         miopen::deref(destTensor),
                         dx,
                         hy_h,
                         hy_h,
                         0);
            }
        }
        else
        {
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
            if(handle.IsProfilingEnabled())
            {
                time_gemm = handle.GetKernelTime();
                handle.AccumKernelTime(time_gemm);
            }
        }

#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenLSTM)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("run lstm gpu bwd data \n");
        float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        for(int li = nLayers - 1; li >= 0; li--)
        {
            int wei_shift = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int hid_shift = li * batch_n * hy_stride;
            int hx_shift  = li * hy_n * h_stride;

            if(li == nLayers - 1)
            {
                std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                miopenTensorDescriptor_t Adesc, Cdesc;

                a_size[2]   = batch_n;
                a_size[3]   = out_h;
                a_stride[0] = batch_n * out_stride;
                a_stride[1] = batch_n * out_stride;
                a_stride[2] = out_stride;
                c_size[2]   = batch_n;
                c_size[3]   = hy_h * bi;
                c_stride[0] = batch_n * hy_stride;
                c_stride[1] = batch_n * hy_stride;
                c_stride[2] = hy_stride;

                miopenCreateTensorDescriptor(&Adesc);
                miopenCreateTensorDescriptor(&Cdesc);
                miopenSetTensorDescriptor(Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                miopenSetTensorDescriptor(Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                alpha0 = 1;
                alpha1 = 0;
                beta_t = 1;

                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         miopen::deref(Adesc),
                         dy,
                         &alpha1,
                         miopen::deref(Adesc),
                         dy,
                         &beta_t,
                         miopen::deref(Cdesc),
                         workSpace,
                         0,
                         0,
                         hid_shift + bi * 5 * hy_h);

                // Update time
                profileSequence(handle, 1);
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
                gg.RunGemm(handle,
                           workSpace,
                           w,
                           workSpace,
                           prelayer_shift,
                           wei_shift,
                           hid_shift + bi * 5 * hy_h);

                // Update time
                if(handle.IsProfilingEnabled())
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

                float alpha = 1, beta = 0;
                ActivationDescriptor tanhDesc, sigDesc;
                size_t offset;

                sigDesc  = {miopenActivationLOGISTIC, 1, 0, 1};
                tanhDesc = {miopenActivationTANH, 1, 1, 1};

                std::vector<int> rsv_size(4, 1), rsv_stride(4, 1);
                miopenTensorDescriptor_t rsvTensor;

                // from post state
                if(ti == seqLen - 1)
                {
                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    if(in_n[ti] > 0)
                    {
                        a_size[2]   = in_n[ti];
                        a_size[3]   = hy_h;
                        a_stride[0] = in_n[ti] * h_stride;
                        a_stride[1] = in_n[ti] * h_stride;
                        a_stride[2] = h_stride;
                        c_size[2]   = in_n[ti];
                        c_size[3]   = hy_h;
                        c_stride[0] = in_n[ti] * hy_stride;
                        c_stride[1] = in_n[ti] * hy_stride;
                        c_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenCreateTensorDescriptor(&Cdesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                        miopenSetTensorDescriptor(
                            Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 dhy,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 dhy,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 workSpace,
                                 hx_shift,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 5 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }

                    if(dirMode)
                    {
                        if(in_n[seqLen - 1 - ti] > 0)
                        {
                            a_size[2]   = in_n[seqLen - 1 - ti];
                            a_size[3]   = hy_h;
                            a_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                            a_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                            a_stride[2] = h_stride;
                            c_size[2]   = in_n[seqLen - 1 - ti];
                            c_size[3]   = hy_h;
                            c_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                            c_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                            c_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Adesc);
                            miopenCreateTensorDescriptor(&Cdesc);
                            miopenSetTensorDescriptor(
                                Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                            miopenSetTensorDescriptor(
                                Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     dhy,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     dhy,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     workSpace,
                                     hx_shift + hy_h,
                                     hx_shift + hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h);
                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }
                        }
                    }
                }
                else
                {
                    int pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;
                    int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

                    if(in_n[ti + 1] > 0)
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
                        gg.RunGemm(handle,
                                   workSpace,
                                   w,
                                   workSpace,
                                   pretime_shift,
                                   weitime_shift,
                                   hid_shift + bacc * hy_stride + bi * 5 * hy_h);

                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_gemm = handle.GetKernelTime();
                            handle.AccumKernelTime(time_gemm);
                        }
                    }

                    if(dirMode)
                    {
                        pretime_shift = li * batch_n * hy_stride +
                                        (baccbi - in_n[seqLen - 2 - ti]) * hy_stride + hy_h * 4;
                        weitime_shift =
                            in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride + hy_h * 4;

                        if(in_n[seqLen - 1 - ti] > 0)
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
                            gg.RunGemm(handle,
                                       workSpace,
                                       w,
                                       workSpace,
                                       pretime_shift,
                                       weitime_shift,
                                       hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h);

                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_gemm = handle.GetKernelTime();
                                handle.AccumKernelTime(time_gemm);
                            }
                        }
                    }
                }

                // update hidden status
                if(in_n[ti] > 0)
                {
                    offset        = hid_shift + bacc * hy_stride;
                    rsv_size[2]   = in_n[ti];
                    rsv_size[3]   = hy_h;
                    rsv_stride[0] = in_n[ti] * hy_stride;
                    rsv_stride[1] = in_n[ti] * hy_stride;
                    rsv_stride[2] = hy_stride;

                    miopenCreateTensorDescriptor(&rsvTensor);
                    miopenSetTensorDescriptor(
                        rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                    // update cell state
                    tanhDesc.Backward(handle,
                                      &alpha,
                                      miopen::deref(rsvTensor),
                                      reserveSpace,
                                      miopen::deref(rsvTensor),
                                      workSpace,
                                      miopen::deref(rsvTensor),
                                      reserveSpace,
                                      &beta,
                                      miopen::deref(rsvTensor),
                                      workSpace,
                                      offset + bi * 4 * hy_h + nLayers * batch_n * hy_stride,
                                      offset + bi * 5 * hy_h,
                                      offset + bi * 4 * hy_h,
                                      offset + bi * 4 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(rsvTensor),
                             workSpace,
                             &alpha1,
                             miopen::deref(rsvTensor),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(rsvTensor),
                             workSpace,
                             offset + bi * 4 * hy_h,
                             offset + 2 * hy_h + nLayers * batch_n * hy_stride,
                             offset + bi * 4 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    if(ti == seqLen - 1)
                    {
                        std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                        miopenTensorDescriptor_t Adesc, Cdesc;

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        a_size[2]   = in_n[ti];
                        a_size[3]   = hy_h;
                        a_stride[0] = in_n[ti] * h_stride;
                        a_stride[1] = in_n[ti] * h_stride;
                        a_stride[2] = h_stride;
                        c_size[2]   = in_n[ti];
                        c_size[3]   = hy_h;
                        c_stride[0] = in_n[ti] * hy_stride;
                        c_stride[1] = in_n[ti] * hy_stride;
                        c_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenCreateTensorDescriptor(&Cdesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                        miopenSetTensorDescriptor(
                            Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 dcy,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 dcy,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 workSpace,
                                 hx_shift,
                                 hx_shift,
                                 offset + bi * 4 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }
                    else
                    {
                        int pretime_shift =
                            li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;

                        std::vector<int> b_size(4, 1), b_stride(4, 1);
                        miopenTensorDescriptor_t Bdesc;

                        b_size[2]   = in_n[ti + 1];
                        b_size[3]   = hy_h;
                        b_stride[0] = in_n[ti + 1] * hy_stride;
                        b_stride[1] = in_n[ti + 1] * hy_stride;
                        b_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Bdesc);
                        miopenSetTensorDescriptor(
                            Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Bdesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Bdesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Bdesc),
                                 workSpace,
                                 pretime_shift + bi * 4 * hy_h,
                                 pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 4 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }

                    // update forget gate
                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     offset + hy_h + nLayers * batch_n * hy_stride,
                                     offset + bi * 4 * hy_h,
                                     offset + hy_h,
                                     offset + hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    if(ti == 0)
                    {
                        std::vector<int> b_size(4, 1), b_stride(4, 1);
                        miopenTensorDescriptor_t Bdesc;

                        b_size[2]   = in_n[ti];
                        b_size[3]   = hy_h;
                        b_stride[0] = in_n[ti] * h_stride;
                        b_stride[1] = in_n[ti] * h_stride;
                        b_stride[2] = h_stride;

                        miopenCreateTensorDescriptor(&Bdesc);
                        miopenSetTensorDescriptor(
                            Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Bdesc),
                                 cx,
                                 &beta_t,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 offset + hy_h,
                                 hx_shift,
                                 offset + hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }
                    else
                    {
                        int pretime_shift =
                            li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride;

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(rsvTensor),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 offset + hy_h,
                                 pretime_shift + bi * 4 * hy_h,
                                 offset + hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }

                    // update input gate
                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     offset + nLayers * batch_n * hy_stride,
                                     offset + bi * 4 * hy_h,
                                     offset,
                                     offset);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(rsvTensor),
                             workSpace,
                             &alpha1,
                             miopen::deref(rsvTensor),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(rsvTensor),
                             workSpace,
                             offset,
                             offset + 3 * hy_h + nLayers * batch_n * hy_stride,
                             offset);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    // update output gate
                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     offset + 2 * hy_h + nLayers * batch_n * hy_stride,
                                     offset + bi * 5 * hy_h,
                                     offset + 2 * hy_h,
                                     offset + 2 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(rsvTensor),
                             workSpace,
                             &alpha1,
                             miopen::deref(rsvTensor),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(rsvTensor),
                             workSpace,
                             offset + 2 * hy_h,
                             offset + bi * 4 * hy_h + nLayers * batch_n * hy_stride,
                             offset + 2 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    // update c gate
                    tanhDesc.Backward(handle,
                                      &alpha,
                                      miopen::deref(rsvTensor),
                                      reserveSpace,
                                      miopen::deref(rsvTensor),
                                      workSpace,
                                      miopen::deref(rsvTensor),
                                      reserveSpace,
                                      &beta,
                                      miopen::deref(rsvTensor),
                                      workSpace,
                                      offset + 3 * hy_h + nLayers * batch_n * hy_stride,
                                      offset + bi * 4 * hy_h,
                                      offset + 3 * hy_h,
                                      offset + 3 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(rsvTensor),
                             workSpace,
                             &alpha1,
                             miopen::deref(rsvTensor),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(rsvTensor),
                             workSpace,
                             offset + 3 * hy_h,
                             offset + nLayers * batch_n * hy_stride,
                             offset + 3 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        offset        = hid_shift + baccbi * hy_stride;
                        rsv_size[2]   = in_n[seqLen - 1 - ti];
                        rsv_size[3]   = hy_h;
                        rsv_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&rsvTensor);
                        miopenSetTensorDescriptor(
                            rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                        // update cell state
                        tanhDesc.Backward(handle,
                                          &alpha,
                                          miopen::deref(rsvTensor),
                                          reserveSpace,
                                          miopen::deref(rsvTensor),
                                          workSpace,
                                          miopen::deref(rsvTensor),
                                          reserveSpace,
                                          &beta,
                                          miopen::deref(rsvTensor),
                                          workSpace,
                                          offset + bi * 4 * hy_h + hy_h +
                                              nLayers * batch_n * hy_stride,
                                          offset + bi * 5 * hy_h + hy_h,
                                          offset + bi * 4 * hy_h + hy_h,
                                          offset + bi * 4 * hy_h + hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(rsvTensor),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 offset + bi * 4 * hy_h + hy_h,
                                 offset + 6 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 4 * hy_h + hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        if(ti == seqLen - 1)
                        {
                            std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1),
                                c_stride(4, 1);
                            miopenTensorDescriptor_t Adesc, Cdesc;

                            alpha0 = 1;
                            alpha1 = 0;
                            beta_t = 1;

                            a_size[2]   = in_n[seqLen - 1 - ti];
                            a_size[3]   = hy_h;
                            a_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                            a_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                            a_stride[2] = h_stride;
                            c_size[2]   = in_n[seqLen - 1 - ti];
                            c_size[3]   = hy_h;
                            c_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                            c_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                            c_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Adesc);
                            miopenCreateTensorDescriptor(&Cdesc);
                            miopenSetTensorDescriptor(
                                Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                            miopenSetTensorDescriptor(
                                Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     dcy,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     dcy,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     workSpace,
                                     hx_shift + hy_h,
                                     hx_shift + hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);
                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }
                        }
                        else
                        {
                            int pretime_shift = li * batch_n * hy_stride +
                                                (baccbi - in_n[seqLen - 2 - ti]) * hy_stride;

                            std::vector<int> b_size(4, 1), b_stride(4, 1);
                            miopenTensorDescriptor_t Bdesc;

                            b_size[2]   = in_n[seqLen - 1 - ti];
                            b_size[3]   = hy_h;
                            b_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                            b_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                            b_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Bdesc);
                            miopenSetTensorDescriptor(
                                Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 1;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(Bdesc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(Bdesc),
                                     reserveSpace,
                                     &beta_t,
                                     miopen::deref(Bdesc),
                                     workSpace,
                                     pretime_shift + bi * 4 * hy_h + hy_h,
                                     pretime_shift + 5 * hy_h + nLayers * batch_n * hy_stride,
                                     offset + bi * 4 * hy_h + hy_h);
                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }
                        }

                        // update forget gate
                        if(ti == 0)
                        {
                            sigDesc.Backward(handle,
                                             &alpha,
                                             miopen::deref(rsvTensor),
                                             reserveSpace,
                                             miopen::deref(rsvTensor),
                                             workSpace,
                                             miopen::deref(rsvTensor),
                                             reserveSpace,
                                             &beta,
                                             miopen::deref(rsvTensor),
                                             workSpace,
                                             offset + 5 * hy_h + nLayers * batch_n * hy_stride,
                                             offset + bi * 4 * hy_h + hy_h,
                                             offset + 5 * hy_h,
                                             offset + 5 * hy_h);
                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }

                            std::vector<int> b_size(4, 1), b_stride(4, 1);
                            miopenTensorDescriptor_t Bdesc;

                            b_size[2]   = in_n[seqLen - 1 - ti];
                            b_size[3]   = hy_h;
                            b_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                            b_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                            b_stride[2] = h_stride;

                            miopenCreateTensorDescriptor(&Bdesc);
                            miopenSetTensorDescriptor(
                                Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(Bdesc),
                                     cx,
                                     &beta_t,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     offset + 5 * hy_h,
                                     hx_shift + hy_h,
                                     offset + 5 * hy_h);
                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }
                        }
                        else
                        {
                            if(in_n[seqLen - ti] > 0)
                            {
                                int pretime_shift = li * batch_n * hy_stride +
                                                    (baccbi + in_n[seqLen - 1 - ti]) * hy_stride;

                                std::vector<int> b_size(4, 1), b_stride(4, 1);
                                miopenTensorDescriptor_t Bdesc;

                                b_size[2]   = in_n[seqLen - ti];
                                b_size[3]   = hy_h;
                                b_stride[0] = in_n[seqLen - ti] * hy_stride;
                                b_stride[1] = in_n[seqLen - ti] * hy_stride;
                                b_stride[2] = hy_stride;

                                miopenCreateTensorDescriptor(&Bdesc);
                                miopenSetTensorDescriptor(
                                    Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                                alpha0 = 1;
                                alpha1 = 1;
                                beta_t = 0;

                                sigDesc.Backward(handle,
                                                 &alpha,
                                                 miopen::deref(Bdesc),
                                                 reserveSpace,
                                                 miopen::deref(Bdesc),
                                                 workSpace,
                                                 miopen::deref(Bdesc),
                                                 reserveSpace,
                                                 &beta,
                                                 miopen::deref(Bdesc),
                                                 workSpace,
                                                 offset + 5 * hy_h + nLayers * batch_n * hy_stride,
                                                 offset + bi * 4 * hy_h + hy_h,
                                                 offset + 5 * hy_h,
                                                 offset + 5 * hy_h);
                                // Update time
                                if(handle.IsProfilingEnabled())
                                {
                                    time_0 = handle.GetKernelTime();
                                    handle.AccumKernelTime(time_0);
                                }

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         miopen::deref(Bdesc),
                                         workSpace,
                                         &alpha1,
                                         miopen::deref(Bdesc),
                                         reserveSpace,
                                         &beta_t,
                                         miopen::deref(Bdesc),
                                         workSpace,
                                         offset + 5 * hy_h,
                                         pretime_shift + bi * 4 * hy_h + hy_h,
                                         offset + 5 * hy_h);
                                // Update time
                                if(handle.IsProfilingEnabled())
                                {
                                    time_0 = handle.GetKernelTime();
                                    handle.AccumKernelTime(time_0);
                                }
                            }
                        }

                        // update input gate
                        sigDesc.Backward(handle,
                                         &alpha,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         miopen::deref(rsvTensor),
                                         workSpace,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(rsvTensor),
                                         workSpace,
                                         offset + 4 * hy_h + nLayers * batch_n * hy_stride,
                                         offset + bi * 4 * hy_h + hy_h,
                                         offset + 4 * hy_h,
                                         offset + 4 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(rsvTensor),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 offset + 4 * hy_h,
                                 offset + 7 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 4 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        // update output gate
                        sigDesc.Backward(handle,
                                         &alpha,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         miopen::deref(rsvTensor),
                                         workSpace,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(rsvTensor),
                                         workSpace,
                                         offset + 6 * hy_h + nLayers * batch_n * hy_stride,
                                         offset + bi * 5 * hy_h + hy_h,
                                         offset + 6 * hy_h,
                                         offset + 6 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(rsvTensor),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 offset + 6 * hy_h,
                                 offset + bi * 4 * hy_h + hy_h + nLayers * batch_n * hy_stride,
                                 offset + 6 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        // update c gate
                        tanhDesc.Backward(handle,
                                          &alpha,
                                          miopen::deref(rsvTensor),
                                          reserveSpace,
                                          miopen::deref(rsvTensor),
                                          workSpace,
                                          miopen::deref(rsvTensor),
                                          reserveSpace,
                                          &beta,
                                          miopen::deref(rsvTensor),
                                          workSpace,
                                          offset + 7 * hy_h + nLayers * batch_n * hy_stride,
                                          offset + bi * 4 * hy_h + hy_h,
                                          offset + 7 * hy_h,
                                          offset + 7 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(rsvTensor),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(rsvTensor),
                                 workSpace,
                                 offset + 7 * hy_h,
                                 offset + 4 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 7 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }
                }

                baccbi += in_n[seqLen - 1 - ti];
            }

            // dcx, dhx
            int pretime_shift = li * batch_n * hy_stride;
            int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(in_n[0] > 0)
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
                gg.RunGemm(handle, workSpace, w, dhx, pretime_shift, weitime_shift, hx_shift);

                // Update time
                if(handle.IsProfilingEnabled())
                {
                    time_gemm = handle.GetKernelTime();
                    handle.AccumKernelTime(time_gemm);
                }

                std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                miopenTensorDescriptor_t Adesc, Cdesc;

                a_size[2]   = in_n[0];
                a_size[3]   = hy_h;
                a_stride[0] = in_n[0] * hy_stride;
                a_stride[1] = in_n[0] * hy_stride;
                a_stride[2] = hy_stride;
                c_size[2]   = in_n[0];
                c_size[3]   = hy_h;
                c_stride[0] = in_n[0] * h_stride;
                c_stride[1] = in_n[0] * h_stride;
                c_stride[2] = h_stride;

                miopenCreateTensorDescriptor(&Adesc);
                miopenCreateTensorDescriptor(&Cdesc);
                miopenSetTensorDescriptor(Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                miopenSetTensorDescriptor(Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                alpha0 = 1;
                alpha1 = 1;
                beta_t = 1;

                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         miopen::deref(Adesc),
                         workSpace,
                         &alpha1,
                         miopen::deref(Adesc),
                         reserveSpace,
                         &beta_t,
                         miopen::deref(Cdesc),
                         dcx,
                         pretime_shift + bi * 4 * hy_h,
                         pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                         hx_shift);
                // Update time
                if(handle.IsProfilingEnabled())
                {
                    time_0 = handle.GetKernelTime();
                    handle.AccumKernelTime(time_0);
                }
            }

            if(dirMode)
            {
                pretime_shift = li * batch_n * hy_stride + (batch_n - in_n[seqLen - 1]) * hy_stride;

                if(in_n[seqLen - 1] > 0)
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
                    gg.RunGemm(handle,
                               workSpace,
                               w,
                               dhx,
                               pretime_shift + 4 * hy_h,
                               weitime_shift + 4 * hy_h,
                               hx_shift + hy_h);

                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_gemm = handle.GetKernelTime();
                        handle.AccumKernelTime(time_gemm);
                    }

                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    a_size[2]   = in_n[seqLen - 1];
                    a_size[3]   = hy_h;
                    a_stride[0] = in_n[seqLen - 1] * hy_stride;
                    a_stride[1] = in_n[seqLen - 1] * hy_stride;
                    a_stride[2] = hy_stride;
                    c_size[2]   = in_n[seqLen - 1];
                    c_size[3]   = hy_h;
                    c_stride[0] = in_n[seqLen - 1] * h_stride;
                    c_stride[1] = in_n[seqLen - 1] * h_stride;
                    c_stride[2] = h_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenCreateTensorDescriptor(&Cdesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                    miopenSetTensorDescriptor(
                        Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             workSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(Cdesc),
                             dcx,
                             pretime_shift + bi * 4 * hy_h + hy_h,
                             pretime_shift + 5 * hy_h + nLayers * batch_n * hy_stride,
                             hx_shift + hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }
                }
            }
        }

        // dinput
        if(inputMode == miopenRNNskip)
        {
            std::vector<int> src_size(4, 1), src_stride(4, 1), dest_size(4, 1), dest_stride(4, 1);
            miopenTensorDescriptor_t srcTensor, destTensor;

            src_size[2]    = batch_n;
            src_size[3]    = hy_h;
            src_stride[0]  = batch_n * hy_stride;
            src_stride[1]  = batch_n * hy_stride;
            src_stride[2]  = hy_stride;
            dest_size[2]   = batch_n;
            dest_size[3]   = hy_h;
            dest_stride[0] = batch_n * in_stride;
            dest_stride[1] = batch_n * in_stride;
            dest_stride[2] = in_stride;
            miopenCreateTensorDescriptor(&srcTensor);
            miopenCreateTensorDescriptor(&destTensor);
            miopenSetTensorDescriptor(
                srcTensor, miopenFloat, 4, src_size.data(), src_stride.data());
            miopenSetTensorDescriptor(
                destTensor, miopenFloat, 4, dest_size.data(), dest_stride.data());

            alpha0 = 1;
            alpha1 = 0;
            beta_t = 1;

            for(int gi = 0; gi < 4; gi++)
            {
                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         miopen::deref(srcTensor),
                         workSpace,
                         &alpha1,
                         miopen::deref(srcTensor),
                         workSpace,
                         &beta_t,
                         miopen::deref(destTensor),
                         dx,
                         gi * hy_h,
                         gi * hy_h,
                         0);
                // Update time
                if(handle.IsProfilingEnabled())
                {
                    time_0 = handle.GetKernelTime();
                    handle.AccumKernelTime(time_0);
                }

                if(dirMode)
                {
                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(srcTensor),
                             workSpace,
                             &alpha1,
                             miopen::deref(srcTensor),
                             workSpace,
                             &beta_t,
                             miopen::deref(destTensor),
                             dx,
                             (gi + 4) * hy_h,
                             (gi + 4) * hy_h,
                             0);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }
                }
            }
        }
        else
        {
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
            gg.RunGemm(handle, workSpace, w, dx, 0, 0, 0);

            // Update time
            if(handle.IsProfilingEnabled())
            {
                time_gemm = handle.GetKernelTime();
                handle.AccumKernelTime(time_gemm);
            }
        }
#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenGRU)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("run gru gpu bwd data \n");
        float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        for(int li = nLayers - 1; li >= 0; li--)
        {
            int wei_shift     = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int hid_shift     = li * batch_n * hy_stride;
            int hx_shift      = li * hy_n * h_stride;
            int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(li == nLayers - 1)
            {
                std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                miopenTensorDescriptor_t Adesc, Cdesc;

                a_size[2]   = batch_n;
                a_size[3]   = out_h;
                a_stride[0] = batch_n * out_stride;
                a_stride[1] = batch_n * out_stride;
                a_stride[2] = out_stride;
                c_size[2]   = batch_n;
                c_size[3]   = hy_h * bi;
                c_stride[0] = batch_n * hy_stride;
                c_stride[1] = batch_n * hy_stride;
                c_stride[2] = hy_stride;

                miopenCreateTensorDescriptor(&Adesc);
                miopenCreateTensorDescriptor(&Cdesc);
                miopenSetTensorDescriptor(Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                miopenSetTensorDescriptor(Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                alpha0 = 1;
                alpha1 = 0;
                beta_t = 1;

                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         miopen::deref(Adesc),
                         dy,
                         &alpha1,
                         miopen::deref(Adesc),
                         dy,
                         &beta_t,
                         miopen::deref(Cdesc),
                         workSpace,
                         0,
                         0,
                         hid_shift + bi * 3 * hy_h);

                // Update time
                profileSequence(handle, 1);
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
                gg.RunGemm(handle,
                           workSpace,
                           w,
                           workSpace,
                           prelayer_shift,
                           wei_shift,
                           hid_shift + bi * 3 * hy_h);

                // Update time
                if(handle.IsProfilingEnabled())
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

                float alpha = 1, beta = 0;
                ActivationDescriptor tanhDesc, sigDesc;
                size_t offset;

                sigDesc  = {miopenActivationLOGISTIC, 1, 0, 1};
                tanhDesc = {miopenActivationTANH, 1, 1, 1};

                std::vector<int> rsv_size(4, 1), rsv_stride(4, 1);
                miopenTensorDescriptor_t rsvTensor;

                if(ti == seqLen - 1)
                {
                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    if(in_n[ti] > 0)
                    {
                        a_size[2]   = in_n[ti];
                        a_size[3]   = hy_h;
                        a_stride[0] = in_n[ti] * h_stride;
                        a_stride[1] = in_n[ti] * h_stride;
                        a_stride[2] = h_stride;
                        c_size[2]   = in_n[ti];
                        c_size[3]   = hy_h;
                        c_stride[0] = in_n[ti] * hy_stride;
                        c_stride[1] = in_n[ti] * hy_stride;
                        c_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenCreateTensorDescriptor(&Cdesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                        miopenSetTensorDescriptor(
                            Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 dhy,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 dhy,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 workSpace,
                                 hx_shift,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }

                    if(dirMode)
                    {
                        if(in_n[seqLen - 1 - ti] > 0)
                        {
                            a_size[2]   = in_n[seqLen - 1 - ti];
                            a_size[3]   = hy_h;
                            a_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                            a_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                            a_stride[2] = h_stride;
                            c_size[2]   = in_n[seqLen - 1 - ti];
                            c_size[3]   = hy_h;
                            c_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                            c_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                            c_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Adesc);
                            miopenCreateTensorDescriptor(&Cdesc);
                            miopenSetTensorDescriptor(
                                Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                            miopenSetTensorDescriptor(
                                Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                            alpha0 = 1;
                            alpha1 = 0;
                            beta_t = 1;

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     dhy,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     dhy,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     workSpace,
                                     hx_shift + hy_h,
                                     hx_shift + hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }
                        }
                    }
                }
                else
                {
                    int pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;

                    if(in_n[ti + 1] > 0)
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
                        gg.RunGemm(handle,
                                   workSpace,
                                   w,
                                   workSpace,
                                   pretime_shift,
                                   weitime_shift,
                                   hid_shift + bacc * hy_stride + bi * 3 * hy_h);

                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_gemm = handle.GetKernelTime();
                            handle.AccumKernelTime(time_gemm);
                        }

                        std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                        miopenTensorDescriptor_t Adesc, Cdesc;

                        a_size[2]   = in_n[ti + 1];
                        a_size[3]   = hy_h;
                        a_stride[0] = in_n[ti + 1] * hy_stride;
                        a_stride[1] = in_n[ti + 1] * hy_stride;
                        a_stride[2] = hy_stride;
                        c_size[2]   = in_n[ti + 1];
                        c_size[3]   = hy_h;
                        c_stride[0] = in_n[ti + 1] * hy_stride;
                        c_stride[1] = in_n[ti + 1] * hy_stride;
                        c_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenCreateTensorDescriptor(&Cdesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                        miopenSetTensorDescriptor(
                            Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 workSpace,
                                 pretime_shift + bi * 3 * hy_h,
                                 pretime_shift + nLayers * batch_n * hy_stride,
                                 hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                        // Update time
                        profileSequence(handle, 1);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 workSpace,
                                 pretime_shift + 2 * hy_h,
                                 pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                                 hid_shift + bacc * hy_stride + 2 * hy_h);
                        // Update time
                        profileSequence(handle, 1);

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
                        gg.RunGemm(handle,
                                   workSpace,
                                   w,
                                   workSpace,
                                   hid_shift + bacc * hy_stride + 2 * hy_h,
                                   weitime_shift + 2 * hy_h,
                                   hid_shift + bacc * hy_stride + bi * 3 * hy_h);

                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_gemm = handle.GetKernelTime();
                            handle.AccumKernelTime(time_gemm);
                        }
                    }

                    if(dirMode)
                    {
                        pretime_shift = li * batch_n * hy_stride +
                                        (baccbi - in_n[seqLen - 2 - ti]) * hy_stride + hy_h * 3;

                        if(in_n[seqLen - 1 - ti] > 0)
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
                            gg.RunGemm(handle,
                                       workSpace,
                                       w,
                                       workSpace,
                                       pretime_shift,
                                       weitime_shift + hy_h * 3,
                                       hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_gemm = handle.GetKernelTime();
                                handle.AccumKernelTime(time_gemm);
                            }

                            std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1),
                                c_stride(4, 1);
                            miopenTensorDescriptor_t Adesc, Cdesc;

                            a_size[2]   = in_n[seqLen - 1 - ti];
                            a_size[3]   = hy_h;
                            a_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                            a_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                            a_stride[2] = hy_stride;
                            c_size[2]   = in_n[seqLen - 1 - ti];
                            c_size[3]   = hy_h;
                            c_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                            c_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                            c_stride[2] = hy_stride;

                            miopenCreateTensorDescriptor(&Adesc);
                            miopenCreateTensorDescriptor(&Cdesc);
                            miopenSetTensorDescriptor(
                                Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                            miopenSetTensorDescriptor(
                                Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 1;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     workSpace,
                                     pretime_shift + 3 * hy_h + hy_h,
                                     pretime_shift + nLayers * batch_n * hy_stride,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                            // Update time
                            profileSequence(handle, 1);

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     workSpace,
                                     pretime_shift + 2 * hy_h,
                                     pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                                     hid_shift + baccbi * hy_stride + 5 * hy_h);

                            // Update time
                            profileSequence(handle, 1);

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
                            gg.RunGemm(handle,
                                       workSpace,
                                       w,
                                       workSpace,
                                       hid_shift + baccbi * hy_stride + 5 * hy_h,
                                       weitime_shift + 5 * hy_h,
                                       hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }

                // update z, r, c gate
                if(in_n[ti] > 0)
                {
                    offset        = hid_shift + bacc * hy_stride;
                    rsv_size[2]   = in_n[ti];
                    rsv_size[3]   = hy_h;
                    rsv_stride[0] = in_n[ti] * hy_stride;
                    rsv_stride[1] = in_n[ti] * hy_stride;
                    rsv_stride[2] = hy_stride;

                    miopenCreateTensorDescriptor(&rsvTensor);
                    miopenSetTensorDescriptor(
                        rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                    std::vector<int> a_size(4, 1), a_stride(4, 1);
                    miopenTensorDescriptor_t Adesc;

                    a_size[2]   = in_n[ti];
                    a_size[3]   = hy_h;
                    a_stride[0] = in_n[ti] * hy_stride;
                    a_stride[1] = in_n[ti] * hy_stride;
                    a_stride[2] = hy_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());

                    // c gate
                    alpha0 = 1;
                    alpha1 = -1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             workSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             workSpace,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + 2 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(Adesc),
                             workSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             workSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             workSpace,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + 2 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    tanhDesc.Backward(handle,
                                      &alpha,
                                      miopen::deref(rsvTensor),
                                      reserveSpace,
                                      miopen::deref(rsvTensor),
                                      workSpace,
                                      miopen::deref(rsvTensor),
                                      reserveSpace,
                                      &beta,
                                      miopen::deref(rsvTensor),
                                      workSpace,
                                      offset + 2 * hy_h + nLayers * batch_n * hy_stride,
                                      offset + 2 * hy_h,
                                      offset + 2 * hy_h,
                                      offset + 2 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    // r gate
                    if(ti == 0)
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
                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   workSpace,
                                   hx_shift,
                                   weitime_shift + 2 * hy_h,
                                   hid_shift + bacc * hy_stride + hy_h);

                        // Update time
                        profileSequence(handle, 1);
                    }
                    else
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
                        gg.RunGemm(handle,
                                   reserveSpace,
                                   w,
                                   workSpace,
                                   hid_shift + (bacc - in_n[ti - 1]) * hy_stride + bi * 3 * hy_h,
                                   weitime_shift + 2 * hy_h,
                                   hid_shift + bacc * hy_stride + hy_h);

                        // Update time
                        profileSequence(handle, 1);
                    }

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             workSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             workSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             workSpace,
                             hid_shift + bacc * hy_stride + 2 * hy_h,
                             hid_shift + bacc * hy_stride + hy_h,
                             hid_shift + bacc * hy_stride + hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     offset + hy_h + nLayers * batch_n * hy_stride,
                                     offset + hy_h,
                                     offset + hy_h,
                                     offset + hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    // z gate
                    alpha0 = 1;
                    alpha1 = -1;
                    beta_t = 0;

                    if(ti == 0)
                    {
                        std::vector<int> b_size(4, 1), b_stride(4, 1);
                        miopenTensorDescriptor_t Bdesc;

                        b_size[2]   = in_n[ti];
                        b_size[3]   = hy_h;
                        b_stride[0] = in_n[ti] * h_stride;
                        b_stride[1] = in_n[ti] * h_stride;
                        b_stride[2] = h_stride;

                        miopenCreateTensorDescriptor(&Bdesc);
                        miopenSetTensorDescriptor(
                            Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Bdesc),
                                 hx,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + 2 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + bacc * hy_stride);
                    }
                    else
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 hid_shift + (bacc - in_n[ti - 1]) * hy_stride + bi * 3 * hy_h,
                                 hid_shift + bacc * hy_stride + 2 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + bacc * hy_stride);
                    }
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             workSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             workSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             workSpace,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride,
                             hid_shift + bacc * hy_stride);
                    // Update time
                    profileSequence(handle, 1);

                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     miopen::deref(rsvTensor),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(rsvTensor),
                                     workSpace,
                                     offset + nLayers * batch_n * hy_stride,
                                     offset,
                                     offset,
                                     offset);
                    // Update time
                    //                    if(gi==3) profileSequence(handle, 2);
                    //                    else profileSequence(handle, 1);
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        offset        = hid_shift + baccbi * hy_stride;
                        rsv_size[2]   = in_n[seqLen - 1 - ti];
                        rsv_size[3]   = hy_h;
                        rsv_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        rsv_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&rsvTensor);
                        miopenSetTensorDescriptor(
                            rsvTensor, miopenFloat, 4, rsv_size.data(), rsv_stride.data());

                        std::vector<int> a_size(4, 1), a_stride(4, 1);
                        miopenTensorDescriptor_t Adesc;

                        a_size[2]   = in_n[seqLen - 1 - ti];
                        a_size[3]   = hy_h;
                        a_stride[0] = in_n[seqLen - 1 - ti] * hy_stride;
                        a_stride[1] = in_n[seqLen - 1 - ti] * hy_stride;
                        a_stride[2] = hy_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());

                        // c gate
                        alpha0 = 1;
                        alpha1 = -1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + 3 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h);
                        // Update time
                        profileSequence(handle, 1);

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h);
                        // Update time
                        profileSequence(handle, 1);

                        tanhDesc.Backward(handle,
                                          &alpha,
                                          miopen::deref(rsvTensor),
                                          reserveSpace,
                                          miopen::deref(rsvTensor),
                                          workSpace,
                                          miopen::deref(rsvTensor),
                                          reserveSpace,
                                          &beta,
                                          miopen::deref(rsvTensor),
                                          workSpace,
                                          offset + 5 * hy_h + nLayers * batch_n * hy_stride,
                                          offset + 5 * hy_h,
                                          offset + 5 * hy_h,
                                          offset + 5 * hy_h);
                        // Update time
                        profileSequence(handle, 1);

                        // r gate
                        if(ti == 0)
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
                            gg.RunGemm(handle,
                                       hx,
                                       w,
                                       workSpace,
                                       hx_shift + hy_h,
                                       weitime_shift + 5 * hy_h,
                                       hid_shift + baccbi * hy_stride + 4 * hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                        else
                        {
                            if(in_n[seqLen - ti] > 0)
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
                                gg.RunGemm(handle,
                                           reserveSpace,
                                           w,
                                           workSpace,
                                           hid_shift +
                                               (baccbi + in_n[seqLen - 1 - ti]) * hy_stride +
                                               bi * 3 * hy_h + hy_h,
                                           weitime_shift + 5 * hy_h,
                                           hid_shift + baccbi * hy_stride + 4 * hy_h);

                                // Update time
                                if(handle.IsProfilingEnabled())
                                {
                                    time_gemm = handle.GetKernelTime();
                                    handle.AccumKernelTime(time_gemm);
                                }
                            }
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h);
                        // Update time
                        profileSequence(handle, 1);

                        sigDesc.Backward(handle,
                                         &alpha,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         miopen::deref(rsvTensor),
                                         workSpace,
                                         miopen::deref(rsvTensor),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(rsvTensor),
                                         workSpace,
                                         offset + 4 * hy_h + nLayers * batch_n * hy_stride,
                                         offset + 4 * hy_h,
                                         offset + 4 * hy_h,
                                         offset + 4 * hy_h);
                        // Update time
                        profileSequence(handle, 1);

                        // z gate
                        if(ti == 0)
                        {
                            alpha0 = 1;
                            alpha1 = -1;
                            beta_t = 0;

                            std::vector<int> b_size(4, 1), b_stride(4, 1);
                            miopenTensorDescriptor_t Bdesc;

                            b_size[2]   = in_n[seqLen - 1 - ti];
                            b_size[3]   = hy_h;
                            b_stride[0] = in_n[seqLen - 1 - ti] * h_stride;
                            b_stride[1] = in_n[seqLen - 1 - ti] * h_stride;
                            b_stride[2] = h_stride;

                            miopenCreateTensorDescriptor(&Bdesc);
                            miopenSetTensorDescriptor(
                                Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(Bdesc),
                                     hx,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     reserveSpace,
                                     &beta_t,
                                     miopen::deref(Adesc),
                                     workSpace,
                                     hx_shift + hy_h,
                                     hid_shift + baccbi * hy_stride + 5 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h);
                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     workSpace,
                                     &beta_t,
                                     miopen::deref(Adesc),
                                     workSpace,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h);
                            // Update time
                            profileSequence(handle, 1);

                            sigDesc.Backward(handle,
                                             &alpha,
                                             miopen::deref(rsvTensor),
                                             reserveSpace,
                                             miopen::deref(rsvTensor),
                                             workSpace,
                                             miopen::deref(rsvTensor),
                                             reserveSpace,
                                             &beta,
                                             miopen::deref(rsvTensor),
                                             workSpace,
                                             offset + 3 * hy_h + nLayers * batch_n * hy_stride,
                                             offset + 3 * hy_h,
                                             offset + 3 * hy_h,
                                             offset + 3 * hy_h);
                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }
                        }
                        else
                        {
                            if(in_n[seqLen - ti] > 0)
                            {
                                std::vector<int> b_size(4, 1), b_stride(4, 1);
                                miopenTensorDescriptor_t Bdesc;

                                b_size[2]   = in_n[seqLen - ti];
                                b_size[3]   = hy_h;
                                b_stride[0] = in_n[seqLen - ti] * hy_stride;
                                b_stride[1] = in_n[seqLen - ti] * hy_stride;
                                b_stride[2] = hy_stride;

                                miopenCreateTensorDescriptor(&Bdesc);
                                miopenSetTensorDescriptor(
                                    Bdesc, miopenFloat, 4, b_size.data(), b_stride.data());

                                alpha0 = 1;
                                alpha1 = -1;
                                beta_t = 0;

                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         miopen::deref(Bdesc),
                                         reserveSpace,
                                         &alpha1,
                                         miopen::deref(Bdesc),
                                         reserveSpace,
                                         &beta_t,
                                         miopen::deref(Bdesc),
                                         workSpace,
                                         hid_shift + (baccbi + in_n[seqLen - 1 - ti]) * hy_stride +
                                             bi * 3 * hy_h + hy_h,
                                         hid_shift + baccbi * hy_stride + 5 * hy_h +
                                             nLayers * batch_n * hy_stride,
                                         hid_shift + baccbi * hy_stride + 3 * hy_h);
                                // Update time
                                if(handle.IsProfilingEnabled())
                                {
                                    time_0 = handle.GetKernelTime();
                                    handle.AccumKernelTime(time_0);
                                }

                                alpha0 = 1;
                                alpha1 = 1;
                                beta_t = 0;

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         miopen::deref(Bdesc),
                                         workSpace,
                                         &alpha1,
                                         miopen::deref(Bdesc),
                                         workSpace,
                                         &beta_t,
                                         miopen::deref(Bdesc),
                                         workSpace,
                                         hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                         hid_shift + baccbi * hy_stride + 3 * hy_h,
                                         hid_shift + baccbi * hy_stride + 3 * hy_h);
                                // Update time
                                if(handle.IsProfilingEnabled())
                                {
                                    time_0 = handle.GetKernelTime();
                                    handle.AccumKernelTime(time_0);
                                }

                                sigDesc.Backward(handle,
                                                 &alpha,
                                                 miopen::deref(Bdesc),
                                                 reserveSpace,
                                                 miopen::deref(Bdesc),
                                                 workSpace,
                                                 miopen::deref(Bdesc),
                                                 reserveSpace,
                                                 &beta,
                                                 miopen::deref(Bdesc),
                                                 workSpace,
                                                 offset + 3 * hy_h + nLayers * batch_n * hy_stride,
                                                 offset + 3 * hy_h,
                                                 offset + 3 * hy_h,
                                                 offset + 3 * hy_h);
                                // Update time
                                profileSequence(handle, 1);
                            }
                        }
                    }
                }

                baccbi += in_n[seqLen - 1 - ti];
            }

            // dhx
            int pretime_shift;

            if(in_n[0] > 0)
            {
                pretime_shift = li * batch_n * hy_stride;

                std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                miopenTensorDescriptor_t Adesc, Cdesc;

                a_size[2]   = in_n[0];
                a_size[3]   = hy_h;
                a_stride[0] = in_n[0] * hy_stride;
                a_stride[1] = in_n[0] * hy_stride;
                a_stride[2] = hy_stride;
                c_size[2]   = in_n[0];
                c_size[3]   = hy_h;
                c_stride[0] = in_n[0] * h_stride;
                c_stride[1] = in_n[0] * h_stride;
                c_stride[2] = h_stride;

                miopenCreateTensorDescriptor(&Adesc);
                miopenCreateTensorDescriptor(&Cdesc);
                miopenSetTensorDescriptor(Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                miopenSetTensorDescriptor(Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                alpha0 = 1;
                alpha1 = 1;
                beta_t = 0;

                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         miopen::deref(Adesc),
                         workSpace,
                         &alpha1,
                         miopen::deref(Adesc),
                         reserveSpace,
                         &beta_t,
                         miopen::deref(Adesc),
                         reserveSpace,
                         pretime_shift + 2 * hy_h,
                         pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                         pretime_shift + bi * 3 * hy_h + nLayers * batch_n * hy_stride);
                // Update time
                if(handle.IsProfilingEnabled())
                {
                    time_0 = handle.GetKernelTime();
                    handle.AccumKernelTime(time_0);
                }

                gg = CreateGemmGeometryRNN(in_n[0],
                                           hy_h,
                                           hy_h,
                                           1,
                                           0,
                                           false,
                                           true,
                                           false,
                                           hy_stride,
                                           wei_stride,
                                           h_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, reserveSpace, w, dhx, false);
                gg.RunGemm(handle,
                           reserveSpace,
                           w,
                           dhx,
                           pretime_shift + bi * 3 * hy_h + nLayers * batch_n * hy_stride,
                           weitime_shift + 2 * hy_h,
                           hx_shift);

                // Update time
                if(handle.IsProfilingEnabled())
                {
                    time_gemm = handle.GetKernelTime();
                    handle.AccumKernelTime(time_gemm);
                }

                alpha0 = 1;
                alpha1 = 1;
                beta_t = 1;

                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         miopen::deref(Adesc),
                         workSpace,
                         &alpha1,
                         miopen::deref(Adesc),
                         reserveSpace,
                         &beta_t,
                         miopen::deref(Cdesc),
                         dhx,
                         pretime_shift + bi * 3 * hy_h,
                         pretime_shift + nLayers * batch_n * hy_stride,
                         hx_shift);
                // Update time
                profileSequence(handle, 1);

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
                gg.RunGemm(handle, workSpace, w, dhx, pretime_shift, weitime_shift, hx_shift);

                // Update time
                profileSequence(handle, 1);
            }

            if(dirMode)
            {
                if(in_n[seqLen - 1] > 0)
                {
                    pretime_shift =
                        li * batch_n * hy_stride + (batch_n - in_n[seqLen - 1]) * hy_stride;

                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    a_size[2]   = in_n[seqLen - 1];
                    a_size[3]   = hy_h;
                    a_stride[0] = in_n[seqLen - 1] * hy_stride;
                    a_stride[1] = in_n[seqLen - 1] * hy_stride;
                    a_stride[2] = hy_stride;
                    c_size[2]   = in_n[seqLen - 1];
                    c_size[3]   = hy_h;
                    c_stride[0] = in_n[seqLen - 1] * h_stride;
                    c_stride[1] = in_n[seqLen - 1] * h_stride;
                    c_stride[2] = h_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenCreateTensorDescriptor(&Cdesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                    miopenSetTensorDescriptor(
                        Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             workSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             reserveSpace,
                             pretime_shift + 5 * hy_h,
                             pretime_shift + 4 * hy_h + nLayers * batch_n * hy_stride,
                             pretime_shift + bi * 3 * hy_h + hy_h + nLayers * batch_n * hy_stride);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

                    gg = CreateGemmGeometryRNN(in_n[seqLen - 1],
                                               hy_h,
                                               hy_h,
                                               1,
                                               0,
                                               false,
                                               true,
                                               false,
                                               hy_stride,
                                               wei_stride,
                                               h_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, reserveSpace, w, dhx, false);
                    gg.RunGemm(handle,
                               reserveSpace,
                               w,
                               dhx,
                               pretime_shift + bi * 3 * hy_h + hy_h + nLayers * batch_n * hy_stride,
                               weitime_shift + 5 * hy_h,
                               hx_shift + hy_h);

                    // Update time
                    profileSequence(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             workSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(Cdesc),
                             dhx,
                             pretime_shift + bi * 3 * hy_h + hy_h,
                             pretime_shift + 3 * hy_h + nLayers * batch_n * hy_stride,
                             hx_shift + hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }

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
                    gg.RunGemm(handle,
                               workSpace,
                               w,
                               dhx,
                               pretime_shift + 3 * hy_h,
                               weitime_shift + 3 * hy_h,
                               hx_shift + hy_h);

                    // Update time
                    profileSequence(handle, 1);
                }
            }
        }

        // dinput
        if(inputMode == miopenRNNskip)
        {
            std::vector<int> src_size(4, 1), src_stride(4, 1), dest_size(4, 1), dest_stride(4, 1);
            miopenTensorDescriptor_t srcTensor, destTensor;

            src_size[2]    = batch_n;
            src_size[3]    = hy_h;
            src_stride[0]  = batch_n * hy_stride;
            src_stride[1]  = batch_n * hy_stride;
            src_stride[2]  = hy_stride;
            dest_size[2]   = batch_n;
            dest_size[3]   = hy_h;
            dest_stride[0] = batch_n * in_stride;
            dest_stride[1] = batch_n * in_stride;
            dest_stride[2] = in_stride;
            miopenCreateTensorDescriptor(&srcTensor);
            miopenCreateTensorDescriptor(&destTensor);
            miopenSetTensorDescriptor(
                srcTensor, miopenFloat, 4, src_size.data(), src_stride.data());
            miopenSetTensorDescriptor(
                destTensor, miopenFloat, 4, dest_size.data(), dest_stride.data());

            alpha0 = 1;
            alpha1 = 0;
            beta_t = 1;

            for(int gi = 0; gi < 3; gi++)
            {
                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         miopen::deref(srcTensor),
                         workSpace,
                         &alpha1,
                         miopen::deref(srcTensor),
                         workSpace,
                         &beta_t,
                         miopen::deref(destTensor),
                         dx,
                         gi * hy_h,
                         gi * hy_h,
                         0);
                // Update time
                if(handle.IsProfilingEnabled())
                {
                    time_0 = handle.GetKernelTime();
                    handle.AccumKernelTime(time_0);
                }

                if(dirMode)
                {
                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(srcTensor),
                             workSpace,
                             &alpha1,
                             miopen::deref(srcTensor),
                             workSpace,
                             &beta_t,
                             miopen::deref(destTensor),
                             dx,
                             (gi + 3) * hy_h,
                             (gi + 3) * hy_h,
                             0);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }
                }
            }
        }
        else
        {
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
            gg.RunGemm(handle, workSpace, w, dx, 0, 0, 0);

            // Update time
            profileSequence(handle, 2);
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

    if(x == nullptr || dw == nullptr || dy == nullptr)
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

    if(in_h == 0 || hy_h == 0 || hy_n == 0 || hy_d == 0 || out_h == 0)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    int batch_n = 0;
    for(int i = 0; i < seqLen; i++)
    {
        int batchval, inputvec, batchvalout, outputvec;
        std::tie(batchval, inputvec)     = miopen::tien<2>(xDesc[i].GetLengths());
        std::tie(batchvalout, outputvec) = miopen::tien<2>(dyDesc[i].GetLengths());
        if(batchval != batchvalout)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        in_n.push_back(batchval);
        batch_n += xDesc[i].GetLengths()[0];
    }

    int bacc;
    int bi = dirMode ? 2 : 1;

    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * workspaceScale;
    int h_stride   = hy_h * bi;
    int out_stride = out_h;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;

    if(inputMode == miopenRNNskip)
    {
        if(in_h != hy_h)
        {
            printf("The input tensor size must equal to the hidden state size of the network in "
                   "SKIP_INPUT mode!\n");
            MIOPEN_THROW(miopenStatusBadParm);
        }
        in_h = 0;
    }

    size_t wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (nLayers - 1)) * wei_stride;

    float alpha0, alpha1, beta_t;

    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("run rnn gpu bwd weights \n");
        float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        for(int li = 0; li < nLayers; li++)
        {
            // between layers
            if(li == 0)
            {
                if(inputMode == miopenRNNlinear)
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
                    profileSequence(handle, 0);
                }

                if(biasMode)
                {
                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    a_size[2]   = 1;
                    a_size[3]   = wei_stride;
                    a_stride[0] = hy_stride;
                    a_stride[1] = hy_stride;
                    a_stride[2] = hy_stride;
                    c_size[2]   = 1;
                    c_size[3]   = wei_stride;
                    c_stride[0] = wei_stride;
                    c_stride[1] = wei_stride;
                    c_stride[2] = wei_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenCreateTensorDescriptor(&Cdesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                    miopenSetTensorDescriptor(
                        Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 dw,
                                 bs * hy_stride,
                                 bs * hy_stride,
                                 wei_shift_bias);

                        // Update time
                        if((inputMode != miopenRNNlinear) && bs == 0)
                            profileSequence(handle, 0);
                        else
                            profileSequence(handle, 1);
                    }

                    if(inputMode == miopenRNNlinear)
                    {
                        CopyTensor(handle,
                                   miopen::deref(Cdesc),
                                   dw,
                                   miopen::deref(Cdesc),
                                   dw,
                                   wei_shift_bias,
                                   wei_shift_bias + wei_stride);
                        // Update time
                        profileSequence(handle, 1);
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
                gg.RunGemm(handle,
                           reserveSpace,
                           workSpace,
                           dw,
                           prelayer_shift + nLayers * batch_n * hy_stride,
                           hid_shift,
                           wei_shift);

                // Update time
                profileSequence(handle, 1);

                if(biasMode)
                {
                    wei_shift = (inputMode == miopenRNNskip)
                                    ? (wei_shift_bias + bi * hy_h + (li - 1) * bi * 2 * hy_h)
                                    : (wei_shift_bias + li * bi * 2 * hy_h);

                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    a_size[2]   = 1;
                    a_size[3]   = wei_stride;
                    a_stride[0] = hy_stride;
                    a_stride[1] = hy_stride;
                    a_stride[2] = hy_stride;
                    c_size[2]   = 1;
                    c_size[3]   = wei_stride;
                    c_stride[0] = wei_stride;
                    c_stride[1] = wei_stride;
                    c_stride[2] = wei_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenCreateTensorDescriptor(&Cdesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                    miopenSetTensorDescriptor(
                        Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 dw,
                                 hid_shift + bs * hy_stride,
                                 hid_shift + bs * hy_stride,
                                 wei_shift);

                        // Update time
                        profileSequence(handle, 1);
                    }

                    CopyTensor(handle,
                               miopen::deref(Cdesc),
                               dw,
                               miopen::deref(Cdesc),
                               dw,
                               wei_shift,
                               wei_shift + wei_stride);

                    // Update time
                    profileSequence(handle, 1);
                }
            }

            // between time
            bacc = 0;
            for(int ti = 0; ti < seqLen; ti++)
            {
                int hid_shift = li * bi * batch_n * hy_h + bacc * hy_stride;
                int hx_shift  = li * bi * hy_n * hy_h;
                int wei_shift;
                int pretime_shift;

                wei_shift =
                    li == 0 ? (in_h * hy_stride)
                            : (bi * (in_h + hy_h) * hy_h +
                               (li - 1) * bi * (bi * hy_h + hy_h) * hy_h + bi * hy_h * hy_stride);

                if(ti == 0)
                {
                    if(in_n[ti] > 0)
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
                        profileSequence(handle, 1);
                    }
                }
                else
                {
                    pretime_shift = li * bi * batch_n * hy_h + (bacc - in_n[ti - 1]) * hy_stride;

                    if(in_n[ti] > 0)
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
                                   pretime_shift + nLayers * batch_n * hy_stride,
                                   hid_shift,
                                   wei_shift);

                        // Update time
                        profileSequence(handle, 1);
                    }
                }

                if(dirMode)
                {
                    if(ti == seqLen - 1)
                    {
                        if(in_n[ti] > 0)
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
                            profileSequence(handle, 1);
                        }
                    }
                    else
                    {
                        pretime_shift = li * bi * batch_n * hy_h + (bacc + in_n[ti]) * hy_stride;

                        if(in_n[ti + 1] > 0)
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
                                       pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                                       hid_shift + hy_h,
                                       wei_shift + hy_h);

                            // Update time
                            profileSequence(handle, 1);
                        }
                    }
                }
                bacc += in_n[ti];
            }
        }

#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenLSTM)
    {

#if MIOPEN_USE_MIOPENGEMM
        printf("run lstm gpu bwd weights \n");
        float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        for(int li = 0; li < nLayers; li++)
        {
            // between layers
            if(li == 0)
            {
                if(inputMode == miopenRNNlinear)
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
                    profileSequence(handle, 0);
                }

                if(biasMode)
                {
                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    a_size[2]   = 1;
                    a_size[3]   = wei_stride;
                    a_stride[0] = hy_stride;
                    a_stride[1] = hy_stride;
                    a_stride[2] = hy_stride;
                    c_size[2]   = 1;
                    c_size[3]   = wei_stride;
                    c_stride[0] = wei_stride;
                    c_stride[1] = wei_stride;
                    c_stride[2] = wei_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenCreateTensorDescriptor(&Cdesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                    miopenSetTensorDescriptor(
                        Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 dw,
                                 bs * hy_stride,
                                 bs * hy_stride,
                                 wei_shift_bias);

                        // Update time
                        if((inputMode != miopenRNNlinear) && bs == 0)
                            profileSequence(handle, 0);
                        else
                            profileSequence(handle, 1);
                    }

                    if(inputMode == miopenRNNlinear)
                    {
                        CopyTensor(handle,
                                   miopen::deref(Cdesc),
                                   dw,
                                   miopen::deref(Cdesc),
                                   dw,
                                   wei_shift_bias,
                                   wei_shift_bias + wei_stride);

                        // Update time
                        profileSequence(handle, 1);
                    }
                }
            }
            else
            {
                int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * hy_h * 5;
                int hid_shift      = li * batch_n * hy_stride;
                int wei_shift =
                    (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

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
                profileSequence(handle, 1);

                if(biasMode)
                {
                    wei_shift = (inputMode == miopenRNNskip)
                                    ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                    : (wei_shift_bias + li * 2 * wei_stride);

                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    a_size[2]   = 1;
                    a_size[3]   = wei_stride;
                    a_stride[0] = hy_stride;
                    a_stride[1] = hy_stride;
                    a_stride[2] = hy_stride;
                    c_size[2]   = 1;
                    c_size[3]   = wei_stride;
                    c_stride[0] = wei_stride;
                    c_stride[1] = wei_stride;
                    c_stride[2] = wei_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenCreateTensorDescriptor(&Cdesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                    miopenSetTensorDescriptor(
                        Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 dw,
                                 hid_shift + bs * hy_stride,
                                 hid_shift + bs * hy_stride,
                                 wei_shift);

                        // Update time
                        profileSequence(handle, 1);
                    }

                    CopyTensor(handle,
                               miopen::deref(Cdesc),
                               dw,
                               miopen::deref(Cdesc),
                               dw,
                               wei_shift,
                               wei_shift + wei_stride);

                    // Update time
                    profileSequence(handle, 1);
                }
            }

            // between time
            bacc = 0;
            for(int ti = 0; ti < seqLen; ti++)
            {
                int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
                int hx_shift  = li * hy_n * h_stride;
                int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
                int pretime_shift;

                // between time
                if(ti == 0)
                {
                    if(in_n[ti] > 0)
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
                        profileSequence(handle, 1);
                    }
                }
                else
                {
                    pretime_shift = li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride +
                                    bi * 5 * hy_h;

                    if(in_n[ti] > 0)
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
                        gg.RunGemm(handle,
                                   reserveSpace,
                                   workSpace,
                                   dw,
                                   pretime_shift,
                                   hid_shift,
                                   wei_shift);

                        // Update time
                        if(dirMode)
                            profileSequence(handle, 1);
                        else if((li == nLayers) && (ti == seqLen - 1))
                            profileSequence(handle, 2);
                        else
                            profileSequence(handle, 1);
                    }
                }

                if(dirMode)
                {
                    if(ti == seqLen - 1)
                    {
                        if(in_n[ti] > 0)
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
                            profileSequence(handle, 2);
                        }
                    }
                    else
                    {
                        pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride +
                                        bi * 5 * hy_h;

                        if(in_n[ti + 1] > 0)
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
                            profileSequence(handle, 1);
                        }
                    }
                }

                bacc += in_n[ti];
            }
        }
#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(rnnMode == miopenGRU)
    {

#if MIOPEN_USE_MIOPENGEMM

        printf("run gru gpu bwd weights \n");
        float time_gemm = 0, time_0 = 0;
        GemmGeometry gg;

        for(int li = 0; li < nLayers; li++)
        {
            // between layers
            if(li == 0)
            {
                if(inputMode == miopenRNNlinear)
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
                    profileSequence(handle, 0);

                    if(biasMode)
                    {
                        std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                        miopenTensorDescriptor_t Adesc, Cdesc;

                        a_size[2]   = 1;
                        a_size[3]   = wei_stride;
                        a_stride[0] = hy_stride;
                        a_stride[1] = hy_stride;
                        a_stride[2] = hy_stride;
                        c_size[2]   = 1;
                        c_size[3]   = wei_stride;
                        c_stride[0] = wei_stride;
                        c_stride[1] = wei_stride;
                        c_stride[2] = wei_stride;

                        miopenCreateTensorDescriptor(&Adesc);
                        miopenCreateTensorDescriptor(&Cdesc);
                        miopenSetTensorDescriptor(
                            Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                        miopenSetTensorDescriptor(
                            Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        for(int bs = 0; bs < batch_n; bs++)
                        {
                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(Adesc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(Adesc),
                                     workSpace,
                                     &beta_t,
                                     miopen::deref(Cdesc),
                                     dw,
                                     bs * hy_stride,
                                     bs * hy_stride,
                                     wei_shift_bias);

                            // Update time
                            if(handle.IsProfilingEnabled())
                            {
                                time_0 = handle.GetKernelTime();
                                handle.AccumKernelTime(time_0);
                            }
                        }
                    }
                }
            }
            else
            {
                int prelayer_shift = (li - 1) * batch_n * hy_stride + bi * hy_h * 3;
                int hid_shift      = li * batch_n * hy_stride;
                int wei_shift =
                    (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

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
                profileSequence(handle, 1);

                if(biasMode)
                {
                    wei_shift = (inputMode == miopenRNNskip)
                                    ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                    : (wei_shift_bias + li * 2 * wei_stride);

                    std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                    miopenTensorDescriptor_t Adesc, Cdesc;

                    a_size[2]   = 1;
                    a_size[3]   = wei_stride;
                    a_stride[0] = hy_stride;
                    a_stride[1] = hy_stride;
                    a_stride[2] = hy_stride;
                    c_size[2]   = 1;
                    c_size[3]   = wei_stride;
                    c_stride[0] = wei_stride;
                    c_stride[1] = wei_stride;
                    c_stride[2] = wei_stride;

                    miopenCreateTensorDescriptor(&Adesc);
                    miopenCreateTensorDescriptor(&Cdesc);
                    miopenSetTensorDescriptor(
                        Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                    miopenSetTensorDescriptor(
                        Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(Cdesc),
                                 dw,
                                 hid_shift + bs * hy_stride,
                                 hid_shift + bs * hy_stride,
                                 wei_shift);

                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }
                }
            }

            // between time
            bacc = 0;
            for(int ti = 0; ti < seqLen; ti++)
            {
                int hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
                int hx_shift  = li * hy_n * h_stride;
                int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
                int pretime_shift;

                std::vector<int> a_size(4, 1), a_stride(4, 1);
                miopenTensorDescriptor_t Adesc;

                a_size[2]   = in_n[ti];
                a_size[3]   = hy_h;
                a_stride[0] = in_n[ti] * hy_stride;
                a_stride[1] = in_n[ti] * hy_stride;
                a_stride[2] = hy_stride;

                miopenCreateTensorDescriptor(&Adesc);
                miopenSetTensorDescriptor(Adesc, miopenFloat, 4, a_size.data(), a_stride.data());

                alpha0 = 1;
                alpha1 = 1;
                beta_t = 0;

                if(in_n[ti] > 0)
                {
                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(Adesc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             workSpace,
                             &beta_t,
                             miopen::deref(Adesc),
                             workSpace,
                             hid_shift + hy_h + nLayers * batch_n * hy_stride,
                             hid_shift + 2 * hy_h,
                             hid_shift + 2 * hy_h);
                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }
                }

                if(ti == 0)
                {
                    if(in_n[ti] > 0)
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
                        profileSequence(handle, 1);
                    }
                }
                else
                {
                    pretime_shift = li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride +
                                    bi * 3 * hy_h;

                    if(in_n[ti] > 0)
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
                        gg.RunGemm(handle,
                                   reserveSpace,
                                   workSpace,
                                   dw,
                                   pretime_shift,
                                   hid_shift,
                                   wei_shift);

                        // Update time
                        profileSequence(handle, 1);
                    }
                }

                if(dirMode)
                {
                    if(in_n[ti] > 0)
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(Adesc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(Adesc),
                                 workSpace,
                                 hid_shift + 4 * hy_h + nLayers * batch_n * hy_stride,
                                 hid_shift + 5 * hy_h,
                                 hid_shift + 5 * hy_h);
                        // Update time
                        if(handle.IsProfilingEnabled())
                        {
                            time_0 = handle.GetKernelTime();
                            handle.AccumKernelTime(time_0);
                        }
                    }

                    if(ti == seqLen - 1)
                    {
                        if(in_n[ti] > 0)
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
                            profileSequence(handle, 1);
                        }
                    }
                    else
                    {
                        pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride +
                                        bi * 3 * hy_h;

                        if(in_n[ti + 1] > 0)
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
                            profileSequence(handle, 1);
                        }
                    }
                }

                bacc += in_n[ti];
            }

            if(biasMode)
            {
                int wei_shift;
                int hid_shift   = li * batch_n * hy_stride;
                int in_bias_val = inputMode == miopenRNNskip ? 0 : wei_stride;

                wei_shift = (li == 0) ? (wei_shift_bias + in_bias_val)
                                      : (wei_shift_bias + in_bias_val + li * 2 * wei_stride);

                std::vector<int> a_size(4, 1), a_stride(4, 1), c_size(4, 1), c_stride(4, 1);
                miopenTensorDescriptor_t Adesc, Cdesc;

                a_size[2]   = 1;
                a_size[3]   = wei_stride;
                a_stride[0] = hy_stride;
                a_stride[1] = hy_stride;
                a_stride[2] = hy_stride;
                c_size[2]   = 1;
                c_size[3]   = wei_stride;
                c_stride[0] = wei_stride;
                c_stride[1] = wei_stride;
                c_stride[2] = wei_stride;

                miopenCreateTensorDescriptor(&Adesc);
                miopenCreateTensorDescriptor(&Cdesc);
                miopenSetTensorDescriptor(Adesc, miopenFloat, 4, a_size.data(), a_stride.data());
                miopenSetTensorDescriptor(Cdesc, miopenFloat, 4, c_size.data(), c_stride.data());

                alpha0 = 1;
                alpha1 = 0;
                beta_t = 1;

                for(int bs = 0; bs < batch_n; bs++)
                {
                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(Adesc),
                             workSpace,
                             &alpha1,
                             miopen::deref(Adesc),
                             workSpace,
                             &beta_t,
                             miopen::deref(Cdesc),
                             dw,
                             hid_shift + bs * hy_stride,
                             hid_shift + bs * hy_stride,
                             wei_shift);

                    // Update time
                    if(handle.IsProfilingEnabled())
                    {
                        time_0 = handle.GetKernelTime();
                        handle.AccumKernelTime(time_0);
                    }
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
