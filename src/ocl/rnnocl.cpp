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

#define MIO_RNN_OCL_DEBUG 1

namespace miopen {

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
    int out_stride = out_h;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

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
    size_t offset;
    float alpha0, alpha1, beta_t;
    float alpha = 1, beta = 0;

    std::vector<int> sp_size(4, 1), sp_stride(4, 1), w_size(4, 1), w_stride(4, 1), x_size(4, 1),
        x_stride(4, 1), y_size(4, 1), y_stride(4, 1), hx_size(4, 1), hx_stride(4, 1);
    miopenTensorDescriptor_t sp_desc, w_desc, x_desc, y_desc, hx_desc;
    miopenCreateTensorDescriptor(&sp_desc);
    miopenCreateTensorDescriptor(&w_desc);
    miopenCreateTensorDescriptor(&x_desc);
    miopenCreateTensorDescriptor(&y_desc);
    miopenCreateTensorDescriptor(&hx_desc);

    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = batch_n * hy_stride;
    sp_stride[2] = hy_stride;
    w_stride[0]  = wei_stride;
    w_stride[1]  = wei_stride;
    w_stride[2]  = wei_stride;
    x_stride[0]  = batch_n * in_stride;
    x_stride[1]  = batch_n * in_stride;
    x_stride[2]  = in_stride;
    y_stride[0]  = batch_n * out_stride;
    y_stride[1]  = batch_n * out_stride;
    y_stride[2]  = out_stride;
    hx_stride[0] = in_n[0] * uni_stride;
    hx_stride[1] = in_n[0] * uni_stride;
    hx_stride[2] = uni_stride;

#if MIOPEN_USE_MIOPENGEMM
    GemmGeometry gg;
    int hid_shift, hx_shift, wei_shift_bias_temp, wei_shift, prelayer_shift;
    int wei_len, wei_len_t, hid_off;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        printf("run rnn gpu inference \n");
        wei_len   = hy_h;
        wei_len_t = hy_h;
        hid_off   = 0;
        break;
    case miopenLSTM:
        printf("run lstm gpu inference \n");
        wei_len   = hy_h * 4;
        wei_len_t = hy_h * 4;
        hid_off   = bi * hy_h * 5;
        break;
    case miopenGRU:
        printf("run gru gpu inference \n");
        wei_len   = hy_h * 3;
        wei_len_t = hy_h * 2;
        hid_off   = bi * hy_h * 3;
        break;
    }

    ActivationDescriptor tanhDesc, sigDesc, activDesc;
    sigDesc  = {miopenActivationLOGISTIC, 1, 0, 1};
    tanhDesc = {miopenActivationTANH, 1, 1, 1};
    if(rnnMode == miopenRNNRELU)
    {
        activDesc = {miopenActivationRELU, 1, 0, 1};
    }
    else if(rnnMode == miopenRNNTANH)
    {
        activDesc = {miopenActivationTANH, 1, 1, 1};
    }

    for(int li = 0; li < nLayers; li++)
    {
        hid_shift           = li * batch_n * hy_stride;
        hx_shift            = li * hy_n * bi_stride;
        wei_shift_bias_temp = inputMode == miopenRNNskip
                                  ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                  : (wei_shift_bias + li * 2 * wei_stride);

        // from input
        if(li == 0)
        {
            if(inputMode == miopenRNNskip)
            {
                x_size[2]  = batch_n;
                x_size[3]  = hy_h;
                sp_size[2] = batch_n;
                sp_size[3] = hy_h;
                miopenSetTensorDescriptor(x_desc, miopenFloat, 4, x_size.data(), x_stride.data());
                miopenSetTensorDescriptor(
                    sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
                {
                    CopyTensor(handle,
                               miopen::deref(x_desc),
                               x,
                               miopen::deref(sp_desc),
                               workSpace,
                               0,
                               gi * hy_h);
                    // Update time
                    profileRNNkernels(handle, (gi == 0) ? 0 : 1);
                }
            }
            else
            {
                gg = CreateGemmGeometryRNN(batch_n,
                                           wei_len * bi,
                                           in_h,
                                           1,
                                           1,
                                           false,
                                           true,
                                           false,
                                           in_stride,
                                           in_stride,
                                           hy_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, x, w, workSpace, false);
                gg.RunGemm(handle, x, w, workSpace, 0, 0, hid_shift);

                // Update time
                profileRNNkernels(handle, 0);
            }
        }
        else
        {
            wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            prelayer_shift = (li - 1) * batch_n * hy_stride + hid_off;

            gg = CreateGemmGeometryRNN(batch_n,
                                       wei_len * bi,
                                       hy_h * bi,
                                       1,
                                       1,
                                       false,
                                       true,
                                       false,
                                       hy_stride,
                                       bi_stride,
                                       hy_stride,
                                       false,
                                       network_config);
            gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
            gg.RunGemm(handle, workSpace, w, workSpace, prelayer_shift, wei_shift, hid_shift);

            // Update time
            profileRNNkernels(handle, 1);
        }

        if(biasMode && rnnMode != miopenGRU)
        {
            int wn = 2;
            if(inputMode == miopenRNNskip && li == 0)
            {
                wei_shift_bias_temp = wei_shift_bias;
                wn                  = 1;
            }

            w_size[2]  = 1;
            w_size[3]  = wei_stride;
            sp_size[2] = batch_n;
            sp_size[3] = wei_stride;
            miopenSetTensorDescriptor(w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
            miopenSetTensorDescriptor(sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
            alpha0 = 1;
            alpha1 = 1;
            beta_t = 0;

            for(int bs = 0; bs < wn; bs++)
            {
                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         miopen::deref(sp_desc),
                         workSpace,
                         &alpha1,
                         miopen::deref(w_desc),
                         w,
                         &beta_t,
                         miopen::deref(sp_desc),
                         workSpace,
                         hid_shift,
                         wei_shift_bias_temp + bs * wei_stride,
                         hid_shift);
                // Update time
                profileRNNkernels(handle, 1);
            }
        }

        // from hidden state
        bacc   = 0;
        baccbi = batch_n;
        for(int ti = 0; ti < seqLen; ti++)
        {
            baccbi -= in_n[seqLen - 1 - ti];
            wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(ti == 0)
            {
                if(in_n[ti] > 0)
                {
                    gg = CreateGemmGeometryRNN(in_n[ti],
                                               wei_len_t,
                                               hy_h,
                                               1,
                                               1,
                                               false,
                                               true,
                                               false,
                                               uni_stride,
                                               uni_stride,
                                               hy_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, hx, w, workSpace, false);
                    gg.RunGemm(handle,
                               hx,
                               w,
                               workSpace,
                               hx_shift,
                               wei_shift,
                               hid_shift + bacc * hy_stride);

                    // Update time
                    profileRNNkernels(handle, 1);

                    if(rnnMode == miopenGRU)
                    {
                        gg = CreateGemmGeometryRNN(in_n[ti],
                                                   hy_h,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   uni_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, hx, w, workSpace, false);
                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   workSpace,
                                   hx_shift,
                                   wei_shift + 2 * hy_h * uni_stride,
                                   hid_shift + bacc * hy_stride + bi * 3 * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                   wei_len_t,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   uni_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, hx, w, workSpace, false);
                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   workSpace,
                                   hx_shift + hy_n * hy_h,
                                   wei_shift + wei_len * uni_stride,
                                   hid_shift + baccbi * hy_stride + wei_len);

                        // Update time
                        profileRNNkernels(handle, 1);

                        if(rnnMode == miopenGRU)
                        {
                            gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                       hy_h,
                                                       hy_h,
                                                       1,
                                                       1,
                                                       false,
                                                       true,
                                                       false,
                                                       uni_stride,
                                                       uni_stride,
                                                       hy_stride,
                                                       false,
                                                       network_config);
                            gg.FindSolution(.003, handle, hx, w, workSpace, false);
                            gg.RunGemm(handle,
                                       hx,
                                       w,
                                       workSpace,
                                       hx_shift + hy_n * hy_h,
                                       wei_shift + 5 * hy_h * uni_stride,
                                       hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                    }
                }
            }
            else
            {
                if(in_n[ti] > 0)
                {
                    gg = CreateGemmGeometryRNN(in_n[ti],
                                               wei_len_t,
                                               hy_h,
                                               1,
                                               1,
                                               false,
                                               true,
                                               false,
                                               uni_stride,
                                               uni_stride,
                                               hy_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, hy, w, workSpace, false);
                    gg.RunGemm(handle,
                               hy,
                               w,
                               workSpace,
                               hx_shift,
                               wei_shift,
                               hid_shift + bacc * hy_stride);

                    // Update time
                    profileRNNkernels(handle, 1);

                    if(rnnMode == miopenGRU)
                    {
                        gg = CreateGemmGeometryRNN(in_n[ti],
                                                   hy_h,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   uni_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, hy, w, workSpace, false);
                        gg.RunGemm(handle,
                                   hy,
                                   w,
                                   workSpace,
                                   hx_shift,
                                   wei_shift + 2 * hy_h * uni_stride,
                                   hid_shift + bacc * hy_stride + bi * 3 * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                   wei_len_t,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   uni_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);

                        gg.FindSolution(.003, handle, hy, w, workSpace, false);
                        gg.RunGemm(handle,
                                   hy,
                                   w,
                                   workSpace,
                                   hx_shift + hy_n * hy_h,
                                   wei_shift + wei_len * uni_stride,
                                   hid_shift + baccbi * hy_stride + wei_len);

                        // Update time
                        profileRNNkernels(handle, 1);

                        if(rnnMode == miopenGRU)
                        {
                            gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                       hy_h,
                                                       hy_h,
                                                       1,
                                                       1,
                                                       false,
                                                       true,
                                                       false,
                                                       uni_stride,
                                                       uni_stride,
                                                       hy_stride,
                                                       false,
                                                       network_config);
                            gg.FindSolution(.003, handle, hy, w, workSpace, false);
                            gg.RunGemm(handle,
                                       hy,
                                       w,
                                       workSpace,
                                       hx_shift + hy_n * hy_h,
                                       wei_shift + 5 * hy_h * uni_stride,
                                       hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                    }
                }
            }

            // update hidden status
            if(in_n[ti] > 0)
            {
                if(rnnMode == miopenGRU && biasMode)
                {
                    // apply bias
                    int wn = 1;
                    if(inputMode == miopenRNNskip && li == 0)
                    {
                        wei_shift_bias_temp = wei_shift_bias;
                        wn                  = 0;
                    }

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    if(!(li == 0 && inputMode == miopenRNNskip))
                    {
                        w_size[2]  = 1;
                        w_size[3]  = 3 * hy_h;
                        sp_size[2] = in_n[ti];
                        sp_size[3] = 3 * hy_h;
                        miopenSetTensorDescriptor(
                            w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(w_desc),
                                 w,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + bacc * hy_stride,
                                 wei_shift_bias_temp,
                                 hid_shift + bacc * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }

                    //
                    w_size[2]  = 1;
                    w_size[3]  = 2 * hy_h;
                    sp_size[2] = in_n[ti];
                    sp_size[3] = 2 * hy_h;
                    miopenSetTensorDescriptor(
                        w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(w_desc),
                             w,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride,
                             wei_shift_bias_temp + wn * wei_stride,
                             hid_shift + bacc * hy_stride);
                    // Update time
                    profileRNNkernels(handle, 1);

                    //
                    w_size[2]  = 1;
                    w_size[3]  = hy_h;
                    sp_size[2] = in_n[ti];
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(w_desc),
                             w,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             wei_shift_bias_temp + wn * wei_stride + 2 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);
                }

                hx_size[2] = in_n[ti];
                hx_size[3] = hy_h;
                miopenSetTensorDescriptor(
                    hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                sp_size[2] = in_n[ti];
                if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                {
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    offset = hid_shift + bacc * hy_stride;

                    activDesc.Forward(handle,
                                      &alpha,
                                      miopen::deref(sp_desc),
                                      workSpace,
                                      &beta,
                                      miopen::deref(sp_desc),
                                      workSpace,
                                      offset,
                                      offset);

                    // Update time
                    profileRNNkernels(handle, 1);
                }
                else if(rnnMode == miopenLSTM)
                {
                    // active gate i, f, o
                    sp_size[3] = hy_h * 3;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    offset = hid_shift + bacc * hy_stride;

                    sigDesc.Forward(handle,
                                    &alpha,
                                    miopen::deref(sp_desc),
                                    workSpace,
                                    &beta,
                                    miopen::deref(sp_desc),
                                    workSpace,
                                    offset,
                                    offset);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // active gate c
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    offset = hid_shift + bacc * hy_stride + 3 * hy_h;

                    tanhDesc.Forward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     offset,
                                     offset);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // update cell state
                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride,
                             hid_shift + bacc * hy_stride + 3 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 4 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    if(ti == 0)
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 cx,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + bacc * hy_stride + hy_h,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 4 * hy_h);
                    }
                    else
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 cy,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + bacc * hy_stride + hy_h,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 4 * hy_h);
                    }
                    // Update time
                    profileRNNkernels(handle, 1);

                    // update cy
                    CopyTensor(handle,
                               miopen::deref(sp_desc),
                               workSpace,
                               miopen::deref(hx_desc),
                               cy,
                               hid_shift + bacc * hy_stride + bi * 4 * hy_h,
                               hx_shift);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // active cell state
                    offset = hid_shift + bacc * hy_stride + bi * 4 * hy_h;

                    tanhDesc.Forward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     offset,
                                     offset);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // update hidden state
                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride + 2 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 4 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 5 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);
                }
                else if(rnnMode == miopenGRU)
                {
                    // active z, r gate
                    sp_size[3] = 2 * hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    offset = hid_shift + bacc * hy_stride;

                    sigDesc.Forward(handle,
                                    &alpha,
                                    miopen::deref(sp_desc),
                                    workSpace,
                                    &beta,
                                    miopen::deref(sp_desc),
                                    workSpace,
                                    offset,
                                    offset);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // calculate c gate
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride + hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + 2 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // active c gate
                    tanhDesc.Forward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     offset + 2 * hy_h,
                                     offset + 2 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // calculate hidden state
                    alpha0 = -1;
                    alpha1 = 1;
                    beta_t = 0;
                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride,
                             hid_shift + bacc * hy_stride + 2 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride + 2 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;
                    if(ti == 0)
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 hx,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + bacc * hy_stride,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    }
                    else
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 hy,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + bacc * hy_stride,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    }
                    // Update time
                    profileRNNkernels(handle, 1);
                }

                // update hy
                CopyTensor(handle,
                           miopen::deref(sp_desc),
                           workSpace,
                           miopen::deref(hx_desc),
                           hy,
                           hid_shift + bacc * hy_stride + hid_off,
                           hx_shift);
                // Update time
                profileRNNkernels(handle, 1);
            }

            if(dirMode)
            {
                if(in_n[seqLen - 1 - ti] > 0)
                {
                    if(rnnMode == miopenGRU && biasMode)
                    {
                        // apply bias
                        int wn = 1;
                        if(inputMode == miopenRNNskip && li == 0)
                        {
                            wei_shift_bias_temp = wei_shift_bias;
                            wn                  = 0;
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        if(!(li == 0 && inputMode == miopenRNNskip))
                        {
                            w_size[2]  = 1;
                            w_size[3]  = 3 * hy_h;
                            sp_size[2] = in_n[seqLen - 1 - ti];
                            sp_size[3] = 3 * hy_h;
                            miopenSetTensorDescriptor(
                                w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                            miopenSetTensorDescriptor(
                                sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(w_desc),
                                     w,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h,
                                     wei_shift_bias_temp + 3 * hy_h,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);
                        }

                        //
                        w_size[2]  = 1;
                        w_size[3]  = 2 * hy_h;
                        sp_size[2] = in_n[seqLen - 1 - ti];
                        sp_size[3] = 2 * hy_h;
                        miopenSetTensorDescriptor(
                            w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(w_desc),
                                 w,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + 3 * hy_h,
                                 wei_shift_bias_temp + wn * wei_stride + 3 * hy_h,
                                 hid_shift + baccbi * hy_stride + 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        //
                        w_size[2]  = 1;
                        w_size[3]  = hy_h;
                        sp_size[2] = in_n[seqLen - 1 - ti];
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(w_desc),
                                 w,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 wei_shift_bias_temp + wn * wei_stride + 5 * hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }

                    hx_size[2] = in_n[seqLen - 1 - ti];
                    hx_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                    sp_size[2] = in_n[seqLen - 1 - ti];
                    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                    {
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        offset = hid_shift + baccbi * hy_stride + hy_h;

                        activDesc.Forward(handle,
                                          &alpha,
                                          miopen::deref(sp_desc),
                                          workSpace,
                                          &beta,
                                          miopen::deref(sp_desc),
                                          workSpace,
                                          offset,
                                          offset);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else if(rnnMode == miopenLSTM)
                    {
                        // active gate i, f, o
                        sp_size[3] = hy_h * 3;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        offset = hid_shift + baccbi * hy_stride + 4 * hy_h;

                        sigDesc.Forward(handle,
                                        &alpha,
                                        miopen::deref(sp_desc),
                                        workSpace,
                                        &beta,
                                        miopen::deref(sp_desc),
                                        workSpace,
                                        offset,
                                        offset);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // active gate c
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        offset = hid_shift + baccbi * hy_stride + 7 * hy_h;

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         &beta,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         offset,
                                         offset);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // update cell state
                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h,
                                 hid_shift + baccbi * hy_stride + 7 * hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        if(ti == 0)
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     cx,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     hid_shift + baccbi * hy_stride + 5 * hy_h,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     cy,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     hid_shift + baccbi * hy_stride + 5 * hy_h,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);
                        }
                        // Update time
                        profileRNNkernels(handle, 1);

                        // update cy
                        CopyTensor(handle,
                                   miopen::deref(sp_desc),
                                   workSpace,
                                   miopen::deref(hx_desc),
                                   cy,
                                   hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h,
                                   hx_shift + hy_n * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // active cell state
                        offset = hid_shift + baccbi * hy_stride + (bi * 4 + 1) * hy_h;

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         &beta,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         offset,
                                         offset);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // update hidden state
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + 6 * hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else if(rnnMode == miopenGRU)
                    {
                        // active z, r gate
                        sp_size[3] = 2 * hy_h;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        offset = hid_shift + baccbi * hy_stride;

                        sigDesc.Forward(handle,
                                        &alpha,
                                        miopen::deref(sp_desc),
                                        workSpace,
                                        &beta,
                                        miopen::deref(sp_desc),
                                        workSpace,
                                        offset + 3 * hy_h,
                                        offset + 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // calculate c gate
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // active c gate
                        tanhDesc.Forward(handle,
                                         &alpha,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         &beta,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         offset + 5 * hy_h,
                                         offset + 5 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // calculate hidden state
                        alpha0 = -1;
                        alpha1 = 1;
                        beta_t = 0;
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + 3 * hy_h,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        if(ti == 0)
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     hx,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     hy,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        }
                        // Update time
                        profileRNNkernels(handle, 1);
                    }

                    // update hy
                    CopyTensor(handle,
                               miopen::deref(sp_desc),
                               workSpace,
                               miopen::deref(hx_desc),
                               hy,
                               hid_shift + baccbi * hy_stride + hid_off + hy_h,
                               hx_shift + hy_n * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);
                }
            }

            bacc += in_n[ti];
        }

        // hy, cy clean
        if(in_n[0] - in_n[seqLen - 1] > 0)
        {
            hx_size[2] = in_n[0] - in_n[seqLen - 1];
            hx_size[3] = hy_h;
            miopenSetTensorDescriptor(hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

            alpha0 = 0;
            alpha1 = 0;
            beta_t = 0;

            OpTensor(handle,
                     miopenTensorOpMul,
                     &alpha0,
                     miopen::deref(hx_desc),
                     hy,
                     &alpha1,
                     miopen::deref(hx_desc),
                     hy,
                     &beta_t,
                     miopen::deref(hx_desc),
                     hy,
                     hx_shift + in_n[seqLen - 1] * uni_stride,
                     hx_shift + in_n[seqLen - 1] * uni_stride,
                     hx_shift + in_n[seqLen - 1] * uni_stride);
            // Update time
            profileRNNkernels(handle, 1);

            if(rnnMode == miopenLSTM)
            {
                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         miopen::deref(hx_desc),
                         cy,
                         &alpha1,
                         miopen::deref(hx_desc),
                         cy,
                         &beta_t,
                         miopen::deref(hx_desc),
                         cy,
                         hx_shift + in_n[seqLen - 1] * uni_stride,
                         hx_shift + in_n[seqLen - 1] * uni_stride,
                         hx_shift + in_n[seqLen - 1] * uni_stride);
                // Update time
                profileRNNkernels(handle, 1);
            }
        }
    }

    // output
    prelayer_shift = (nLayers - 1) * batch_n * hy_stride + hid_off;

    sp_size[2] = batch_n;
    sp_size[3] = hy_h * bi;
    y_size[2]  = batch_n;
    y_size[3]  = out_h;
    miopenSetTensorDescriptor(sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
    miopenSetTensorDescriptor(y_desc, miopenFloat, 4, y_size.data(), y_stride.data());

    CopyTensor(
        handle, miopen::deref(sp_desc), workSpace, miopen::deref(y_desc), y, prelayer_shift, 0);
    // Update time
    profileRNNkernels(handle, 2);

#else
    MIOPEN_THROW("GEMM is not supported");
#endif

    // Suppress warning
    (void)cxDesc;
    (void)cyDesc;
    (void)hxDesc;
    (void)hyDesc;
    (void)wDesc;
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
    int out_stride = out_h;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

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
    size_t offset;
    float alpha0, alpha1, beta_t;
    float alpha = 1, beta = 0;

    std::vector<int> sp_size(4, 1), sp_stride(4, 1), w_size(4, 1), w_stride(4, 1), x_size(4, 1),
        x_stride(4, 1), y_size(4, 1), y_stride(4, 1), hx_size(4, 1), hx_stride(4, 1);
    miopenTensorDescriptor_t sp_desc, w_desc, x_desc, y_desc, hx_desc;
    miopenCreateTensorDescriptor(&sp_desc);
    miopenCreateTensorDescriptor(&w_desc);
    miopenCreateTensorDescriptor(&x_desc);
    miopenCreateTensorDescriptor(&y_desc);
    miopenCreateTensorDescriptor(&hx_desc);

    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = batch_n * hy_stride;
    sp_stride[2] = hy_stride;
    w_stride[0]  = wei_stride;
    w_stride[1]  = wei_stride;
    w_stride[2]  = wei_stride;
    x_stride[0]  = batch_n * in_stride;
    x_stride[1]  = batch_n * in_stride;
    x_stride[2]  = in_stride;
    y_stride[0]  = batch_n * out_stride;
    y_stride[1]  = batch_n * out_stride;
    y_stride[2]  = out_stride;
    hx_stride[0] = in_n[0] * uni_stride;
    hx_stride[1] = in_n[0] * uni_stride;
    hx_stride[2] = uni_stride;

#if MIOPEN_USE_MIOPENGEMM
    GemmGeometry gg;
    int hid_shift, hx_shift, wei_shift_bias_temp, wei_shift, prelayer_shift;
    int wei_len, wei_len_t, hid_off;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        printf("run rnn gpu fwd \n");
        wei_len   = hy_h;
        wei_len_t = hy_h;
        hid_off   = nLayers * batch_n * hy_stride;
        break;
    case miopenLSTM:
        printf("run lstm gpu fwd \n");
        wei_len   = hy_h * 4;
        wei_len_t = hy_h * 4;
        hid_off   = bi * hy_h * 5;
        break;
    case miopenGRU:
        printf("run gru gpu fwd \n");
        wei_len   = hy_h * 3;
        wei_len_t = hy_h * 2;
        hid_off   = bi * hy_h * 3;
        break;
    }

    ActivationDescriptor tanhDesc, sigDesc, activDesc;
    sigDesc  = {miopenActivationLOGISTIC, 1, 0, 1};
    tanhDesc = {miopenActivationTANH, 1, 1, 1};
    if(rnnMode == miopenRNNRELU)
    {
        activDesc = {miopenActivationRELU, 1, 0, 1};
    }
    else if(rnnMode == miopenRNNTANH)
    {
        activDesc = {miopenActivationTANH, 1, 1, 1};
    }

    for(int li = 0; li < nLayers; li++)
    {
        hid_shift           = li * batch_n * hy_stride;
        hx_shift            = li * hy_n * bi_stride;
        wei_shift_bias_temp = inputMode == miopenRNNskip
                                  ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                  : (wei_shift_bias + li * 2 * wei_stride);

        // from input
        if(li == 0)
        {
            if(inputMode == miopenRNNskip)
            {
                x_size[2]  = batch_n;
                x_size[3]  = hy_h;
                sp_size[2] = batch_n;
                sp_size[3] = hy_h;
                miopenSetTensorDescriptor(x_desc, miopenFloat, 4, x_size.data(), x_stride.data());
                miopenSetTensorDescriptor(
                    sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
                {
                    CopyTensor(handle,
                               miopen::deref(x_desc),
                               x,
                               miopen::deref(sp_desc),
                               reserveSpace,
                               0,
                               gi * hy_h);
                    // Update time
                    profileRNNkernels(handle, (gi == 0) ? 0 : 1);
                }
            }
            else
            {
                gg = CreateGemmGeometryRNN(batch_n,
                                           wei_len * bi,
                                           in_h,
                                           1,
                                           1,
                                           false,
                                           true,
                                           false,
                                           in_stride,
                                           in_stride,
                                           hy_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, x, w, reserveSpace, false);
                gg.RunGemm(handle, x, w, reserveSpace, 0, 0, hid_shift);

                // Update time
                profileRNNkernels(handle, 0);
            }
        }
        else
        {
            wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            prelayer_shift = (li - 1) * batch_n * hy_stride + hid_off;

            gg = CreateGemmGeometryRNN(batch_n,
                                       wei_len * bi,
                                       hy_h * bi,
                                       1,
                                       1,
                                       false,
                                       true,
                                       false,
                                       hy_stride,
                                       bi_stride,
                                       hy_stride,
                                       false,
                                       network_config);
            gg.FindSolution(.003, handle, reserveSpace, w, reserveSpace, false);
            gg.RunGemm(handle, reserveSpace, w, reserveSpace, prelayer_shift, wei_shift, hid_shift);

            // Update time
            profileRNNkernels(handle, 1);
        }

        if(biasMode && rnnMode != miopenGRU)
        {
            int wn = 2;
            if(inputMode == miopenRNNskip && li == 0)
            {
                wei_shift_bias_temp = wei_shift_bias;
                wn                  = 1;
            }

            w_size[2]  = 1;
            w_size[3]  = wei_stride;
            sp_size[2] = batch_n;
            sp_size[3] = wei_stride;
            miopenSetTensorDescriptor(w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
            miopenSetTensorDescriptor(sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
            alpha0 = 1;
            alpha1 = 1;
            beta_t = 0;

            for(int bs = 0; bs < wn; bs++)
            {
                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         miopen::deref(sp_desc),
                         reserveSpace,
                         &alpha1,
                         miopen::deref(w_desc),
                         w,
                         &beta_t,
                         miopen::deref(sp_desc),
                         reserveSpace,
                         hid_shift,
                         wei_shift_bias_temp + bs * wei_stride,
                         hid_shift);
                // Update time
                profileRNNkernels(handle, 1);
            }
        }

        // from hidden state
        bacc   = 0;
        baccbi = batch_n;
        for(int ti = 0; ti < seqLen; ti++)
        {
            baccbi -= in_n[seqLen - 1 - ti];
            wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(ti == 0)
            {
                if(in_n[ti] > 0)
                {
                    gg = CreateGemmGeometryRNN(in_n[ti],
                                               wei_len_t,
                                               hy_h,
                                               1,
                                               1,
                                               false,
                                               true,
                                               false,
                                               uni_stride,
                                               uni_stride,
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
                    profileRNNkernels(handle, 1);

                    if(rnnMode == miopenGRU)
                    {
                        gg = CreateGemmGeometryRNN(in_n[ti],
                                                   hy_h,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   uni_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   reserveSpace,
                                   hx_shift,
                                   wei_shift + 2 * hy_h * uni_stride,
                                   hid_shift + bacc * hy_stride + bi * 3 * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                   wei_len_t,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   uni_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   reserveSpace,
                                   hx_shift + hy_n * hy_h,
                                   wei_shift + wei_len * uni_stride,
                                   hid_shift + baccbi * hy_stride + wei_len);

                        // Update time
                        profileRNNkernels(handle, 1);

                        if(rnnMode == miopenGRU)
                        {
                            gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                       hy_h,
                                                       hy_h,
                                                       1,
                                                       1,
                                                       false,
                                                       true,
                                                       false,
                                                       uni_stride,
                                                       uni_stride,
                                                       hy_stride,
                                                       false,
                                                       network_config);
                            gg.FindSolution(.003, handle, hx, w, reserveSpace, false);
                            gg.RunGemm(handle,
                                       hx,
                                       w,
                                       reserveSpace,
                                       hx_shift + hy_n * hy_h,
                                       wei_shift + 5 * hy_h * uni_stride,
                                       hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                    }
                }
            }
            else
            {
                if(in_n[ti] > 0)
                {
                    gg = CreateGemmGeometryRNN(in_n[ti],
                                               wei_len_t,
                                               hy_h,
                                               1,
                                               1,
                                               false,
                                               true,
                                               false,
                                               uni_stride,
                                               uni_stride,
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
                    profileRNNkernels(handle, 1);

                    if(rnnMode == miopenGRU)
                    {
                        gg = CreateGemmGeometryRNN(in_n[ti],
                                                   hy_h,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   uni_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, hy, w, reserveSpace, false);
                        gg.RunGemm(handle,
                                   hy,
                                   w,
                                   reserveSpace,
                                   hx_shift,
                                   wei_shift + 2 * hy_h * uni_stride,
                                   hid_shift + bacc * hy_stride + bi * 3 * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                   wei_len_t,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   uni_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);

                        gg.FindSolution(.003, handle, hy, w, reserveSpace, false);
                        gg.RunGemm(handle,
                                   hy,
                                   w,
                                   reserveSpace,
                                   hx_shift + hy_n * hy_h,
                                   wei_shift + wei_len * uni_stride,
                                   hid_shift + baccbi * hy_stride + wei_len);

                        // Update time
                        profileRNNkernels(handle, 1);

                        if(rnnMode == miopenGRU)
                        {
                            gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                       hy_h,
                                                       hy_h,
                                                       1,
                                                       1,
                                                       false,
                                                       true,
                                                       false,
                                                       uni_stride,
                                                       uni_stride,
                                                       hy_stride,
                                                       false,
                                                       network_config);
                            gg.FindSolution(.003, handle, hy, w, reserveSpace, false);
                            gg.RunGemm(handle,
                                       hy,
                                       w,
                                       reserveSpace,
                                       hx_shift + hy_n * hy_h,
                                       wei_shift + 5 * hy_h * uni_stride,
                                       hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                    }
                }
            }

            // update hidden status
            if(in_n[ti] > 0)
            {
                if(rnnMode == miopenGRU && biasMode)
                {
                    // apply bias
                    int wn = 1;
                    if(inputMode == miopenRNNskip && li == 0)
                    {
                        wei_shift_bias_temp = wei_shift_bias;
                        wn                  = 0;
                    }

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    if(!(li == 0 && inputMode == miopenRNNskip))
                    {
                        w_size[2]  = 1;
                        w_size[3]  = 3 * hy_h;
                        sp_size[2] = in_n[ti];
                        sp_size[3] = 3 * hy_h;
                        miopenSetTensorDescriptor(
                            w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(w_desc),
                                 w,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + bacc * hy_stride,
                                 wei_shift_bias_temp,
                                 hid_shift + bacc * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }

                    //
                    w_size[2]  = 1;
                    w_size[3]  = 2 * hy_h;
                    sp_size[2] = in_n[ti];
                    sp_size[3] = 2 * hy_h;
                    miopenSetTensorDescriptor(
                        w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(w_desc),
                             w,
                             &beta_t,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride,
                             wei_shift_bias_temp + wn * wei_stride,
                             hid_shift + bacc * hy_stride);
                    // Update time
                    profileRNNkernels(handle, 1);

                    //
                    w_size[2]  = 1;
                    w_size[3]  = hy_h;
                    sp_size[2] = in_n[ti];
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(w_desc),
                             w,
                             &beta_t,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             wei_shift_bias_temp + wn * wei_stride + 2 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);
                }

                hx_size[2] = in_n[ti];
                hx_size[3] = hy_h;
                miopenSetTensorDescriptor(
                    hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                sp_size[2] = in_n[ti];
                if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                {
                    sp_size[3] = hy_h;
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
                    profileRNNkernels(handle, 1);
                }
                else if(rnnMode == miopenLSTM)
                {
                    // active gate i, f, o
                    sp_size[3] = hy_h * 3;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    offset = hid_shift + bacc * hy_stride;

                    sigDesc.Forward(handle,
                                    &alpha,
                                    miopen::deref(sp_desc),
                                    reserveSpace,
                                    &beta,
                                    miopen::deref(sp_desc),
                                    reserveSpace,
                                    offset,
                                    offset + nLayers * batch_n * hy_stride);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // active gate c
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    offset = hid_shift + bacc * hy_stride + 3 * hy_h;

                    tanhDesc.Forward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     offset,
                                     offset + nLayers * batch_n * hy_stride);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // update cell state
                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + 3 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 4 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    if(ti == 0)
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 cx,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + bacc * hy_stride + hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 cy,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + bacc * hy_stride + hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 4 * hy_h);
                    }
                    // Update time
                    profileRNNkernels(handle, 1);

                    // update cy
                    CopyTensor(handle,
                               miopen::deref(sp_desc),
                               reserveSpace,
                               miopen::deref(hx_desc),
                               cy,
                               hid_shift + bacc * hy_stride + bi * 4 * hy_h,
                               hx_shift);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // active cell state
                    offset = hid_shift + bacc * hy_stride + bi * 4 * hy_h;

                    tanhDesc.Forward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     offset,
                                     offset + nLayers * batch_n * hy_stride);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // update hidden state
                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + 2 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 4 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 5 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);
                }
                else if(rnnMode == miopenGRU)
                {
                    // active z, r gate
                    sp_size[3] = 2 * hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    offset = hid_shift + bacc * hy_stride;

                    sigDesc.Forward(handle,
                                    &alpha,
                                    miopen::deref(sp_desc),
                                    reserveSpace,
                                    &beta,
                                    miopen::deref(sp_desc),
                                    reserveSpace,
                                    offset,
                                    offset + nLayers * batch_n * hy_stride);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // calculate c gate
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + hy_h + nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + 2 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // active c gate
                    tanhDesc.Forward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     offset + 2 * hy_h,
                                     offset + 2 * hy_h + nLayers * batch_n * hy_stride);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // calculate hidden state
                    alpha0 = -1;
                    alpha1 = 1;
                    beta_t = 0;
                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + 2 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             hid_shift + bacc * hy_stride + 2 * hy_h +
                                 nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;
                    if(ti == 0)
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 hx,
                                 &beta_t,
                                 miopen::deref(sp_desc),
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
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 hy,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + bacc * hy_stride + nLayers * batch_n * hy_stride,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                    }
                    // Update time
                    profileRNNkernels(handle, 1);
                }

                // update hy
                CopyTensor(handle,
                           miopen::deref(sp_desc),
                           reserveSpace,
                           miopen::deref(hx_desc),
                           hy,
                           hid_shift + bacc * hy_stride + hid_off,
                           hx_shift);
                // Update time
                profileRNNkernels(handle, 1);
            }

            if(dirMode)
            {
                if(in_n[seqLen - 1 - ti] > 0)
                {
                    if(rnnMode == miopenGRU && biasMode)
                    {
                        // apply bias
                        int wn = 1;
                        if(inputMode == miopenRNNskip && li == 0)
                        {
                            wei_shift_bias_temp = wei_shift_bias;
                            wn                  = 0;
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        if(!(li == 0 && inputMode == miopenRNNskip))
                        {
                            w_size[2]  = 1;
                            w_size[3]  = 3 * hy_h;
                            sp_size[2] = in_n[seqLen - 1 - ti];
                            sp_size[3] = 3 * hy_h;
                            miopenSetTensorDescriptor(
                                w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                            miopenSetTensorDescriptor(
                                sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &alpha1,
                                     miopen::deref(w_desc),
                                     w,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h,
                                     wei_shift_bias_temp + 3 * hy_h,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);
                        }

                        //
                        w_size[2]  = 1;
                        w_size[3]  = 2 * hy_h;
                        sp_size[2] = in_n[seqLen - 1 - ti];
                        sp_size[3] = 2 * hy_h;
                        miopenSetTensorDescriptor(
                            w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(w_desc),
                                 w,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 3 * hy_h,
                                 wei_shift_bias_temp + wn * wei_stride + 3 * hy_h,
                                 hid_shift + baccbi * hy_stride + 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        //
                        w_size[2]  = 1;
                        w_size[3]  = hy_h;
                        sp_size[2] = in_n[seqLen - 1 - ti];
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            w_desc, miopenFloat, 4, w_size.data(), w_stride.data());
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(w_desc),
                                 w,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 wei_shift_bias_temp + wn * wei_stride + 5 * hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }

                    hx_size[2] = in_n[seqLen - 1 - ti];
                    hx_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                    sp_size[2] = in_n[seqLen - 1 - ti];
                    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                    {
                        sp_size[3] = hy_h;
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
                        profileRNNkernels(handle, 1);
                    }
                    else if(rnnMode == miopenLSTM)
                    {
                        // active gate i, f, o
                        sp_size[3] = hy_h * 3;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        offset = hid_shift + baccbi * hy_stride + 4 * hy_h;

                        sigDesc.Forward(handle,
                                        &alpha,
                                        miopen::deref(sp_desc),
                                        reserveSpace,
                                        &beta,
                                        miopen::deref(sp_desc),
                                        reserveSpace,
                                        offset,
                                        offset + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // active gate c
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        offset = hid_shift + baccbi * hy_stride + 7 * hy_h;

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         offset,
                                         offset + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // update cell state
                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + 7 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        if(ti == 0)
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     cx,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     hid_shift + baccbi * hy_stride + 5 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     cy,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     hid_shift + baccbi * hy_stride + 5 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);
                        }

                        // update cy
                        CopyTensor(handle,
                                   miopen::deref(sp_desc),
                                   reserveSpace,
                                   miopen::deref(hx_desc),
                                   cy,
                                   hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h,
                                   hx_shift + hy_n * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // active cell state
                        offset = hid_shift + baccbi * hy_stride + (bi * 4 + 1) * hy_h;

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         offset,
                                         offset + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // update hidden state
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 6 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else if(rnnMode == miopenGRU)
                    {
                        // active z, r gate
                        sp_size[3] = 2 * hy_h;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        offset = hid_shift + baccbi * hy_stride;

                        sigDesc.Forward(handle,
                                        &alpha,
                                        miopen::deref(sp_desc),
                                        reserveSpace,
                                        &beta,
                                        miopen::deref(sp_desc),
                                        reserveSpace,
                                        offset + 3 * hy_h,
                                        offset + 3 * hy_h + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // calculate c gate
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // active c gate
                        tanhDesc.Forward(handle,
                                         &alpha,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         offset + 5 * hy_h,
                                         offset + 5 * hy_h + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // calculate hidden state
                        alpha0 = -1;
                        alpha1 = 1;
                        beta_t = 0;
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 3 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;
                        if(ti == 0)
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     hx,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     hy,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                        }
                        // Update time
                        profileRNNkernels(handle, 1);
                    }

                    // update hy
                    CopyTensor(handle,
                               miopen::deref(sp_desc),
                               reserveSpace,
                               miopen::deref(hx_desc),
                               hy,
                               hid_shift + baccbi * hy_stride + hid_off + hy_h,
                               hx_shift + hy_n * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);
                }
            }

            bacc += in_n[ti];
        }

        // hy, cy clean
        if(in_n[0] - in_n[seqLen - 1] > 0)
        {
            hx_size[2] = in_n[0] - in_n[seqLen - 1];
            hx_size[3] = hy_h;
            miopenSetTensorDescriptor(hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

            alpha0 = 0;
            alpha1 = 0;
            beta_t = 0;

            OpTensor(handle,
                     miopenTensorOpMul,
                     &alpha0,
                     miopen::deref(hx_desc),
                     hy,
                     &alpha1,
                     miopen::deref(hx_desc),
                     hy,
                     &beta_t,
                     miopen::deref(hx_desc),
                     hy,
                     hx_shift + in_n[seqLen - 1] * uni_stride,
                     hx_shift + in_n[seqLen - 1] * uni_stride,
                     hx_shift + in_n[seqLen - 1] * uni_stride);
            // Update time
            profileRNNkernels(handle, 1);

            if(rnnMode == miopenLSTM)
            {
                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         miopen::deref(hx_desc),
                         cy,
                         &alpha1,
                         miopen::deref(hx_desc),
                         cy,
                         &beta_t,
                         miopen::deref(hx_desc),
                         cy,
                         hx_shift + in_n[seqLen - 1] * uni_stride,
                         hx_shift + in_n[seqLen - 1] * uni_stride,
                         hx_shift + in_n[seqLen - 1] * uni_stride);
                // Update time
                profileRNNkernels(handle, 1);
            }
        }
    }

    // output
    prelayer_shift = (nLayers - 1) * batch_n * hy_stride + hid_off;

    sp_size[2] = batch_n;
    sp_size[3] = hy_h * bi;
    y_size[2]  = batch_n;
    y_size[3]  = out_h;
    miopenSetTensorDescriptor(sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
    miopenSetTensorDescriptor(y_desc, miopenFloat, 4, y_size.data(), y_stride.data());

    CopyTensor(
        handle, miopen::deref(sp_desc), reserveSpace, miopen::deref(y_desc), y, prelayer_shift, 0);
    // Update time
    profileRNNkernels(handle, 2);

#else
    MIOPEN_THROW("GEMM is not supported");
#endif

    // Suppress warning
    (void)cxDesc;
    (void)cyDesc;
    (void)hxDesc;
    (void)hyDesc;
    (void)wDesc;
    (void)workSpace;
    (void)workSpaceSize;
    (void)reserveSpaceSize;
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
                                    Data_t reserveSpace,
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
    int out_stride = out_h;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

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

    size_t offset;
    float alpha0, alpha1, beta_t;
    float alpha = 1, beta = 0;

    std::vector<int> sp_size(4, 1), sp_stride(4, 1), x_size(4, 1), x_stride(4, 1), y_size(4, 1),
        y_stride(4, 1), hx_size(4, 1), hx_stride(4, 1);
    miopenTensorDescriptor_t sp_desc, x_desc, y_desc, hx_desc;
    miopenCreateTensorDescriptor(&sp_desc);
    miopenCreateTensorDescriptor(&x_desc);
    miopenCreateTensorDescriptor(&y_desc);
    miopenCreateTensorDescriptor(&hx_desc);
    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = batch_n * hy_stride;
    sp_stride[2] = hy_stride;
    x_stride[0]  = batch_n * in_stride;
    x_stride[1]  = batch_n * in_stride;
    x_stride[2]  = in_stride;
    y_stride[0]  = batch_n * out_stride;
    y_stride[1]  = batch_n * out_stride;
    y_stride[2]  = out_stride;
    hx_stride[0] = in_n[0] * uni_stride;
    hx_stride[1] = in_n[0] * uni_stride;
    hx_stride[2] = uni_stride;

#if MIOPEN_USE_MIOPENGEMM
    GemmGeometry gg;
    int hid_shift, hx_shift, weitime_shift, wei_shift, prelayer_shift, pretime_shift;
    int wei_len, wei_len_t, dhd_off;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        printf("run rnn gpu bwd data \n");
        wei_len   = hy_h;
        wei_len_t = hy_h;
        dhd_off   = 0;
        break;
    case miopenLSTM:
        printf("run lstm gpu bwd data \n");
        wei_len   = hy_h * 4;
        wei_len_t = hy_h * 4;
        dhd_off   = bi * hy_h * 5;
        break;
    case miopenGRU:
        printf("run gru gpu bwd data \n");
        wei_len   = hy_h * 3;
        wei_len_t = hy_h * 2;
        dhd_off   = bi * hy_h * 3;
        break;
    }

    ActivationDescriptor tanhDesc, sigDesc, activDesc;
    sigDesc  = {miopenActivationLOGISTIC, 1, 0, 1};
    tanhDesc = {miopenActivationTANH, 1, 1, 1};
    if(rnnMode == miopenRNNRELU)
    {
        activDesc = {miopenActivationRELU, 1, 0, 1};
    }
    else if(rnnMode == miopenRNNTANH)
    {
        activDesc = {miopenActivationTANH, 1, 1, 1};
    }

    for(int li = nLayers - 1; li >= 0; li--)
    {
        wei_shift     = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
        hid_shift     = li * batch_n * hy_stride;
        hx_shift      = li * hy_n * bi_stride;
        weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

        // feedback from output
        if(li == nLayers - 1)
        {
            y_size[2]  = batch_n;
            y_size[3]  = out_h;
            sp_size[2] = batch_n;
            sp_size[3] = hy_h * bi;
            miopenSetTensorDescriptor(y_desc, miopenFloat, 4, y_size.data(), y_stride.data());
            miopenSetTensorDescriptor(sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

            alpha0 = 1;
            alpha1 = 0;
            beta_t = 1;

            OpTensor(handle,
                     miopenTensorOpAdd,
                     &alpha0,
                     miopen::deref(y_desc),
                     dy,
                     &alpha1,
                     miopen::deref(y_desc),
                     dy,
                     &beta_t,
                     miopen::deref(sp_desc),
                     workSpace,
                     0,
                     0,
                     hid_shift + dhd_off);
            // Update time
            profileRNNkernels(handle, 0); // start timing
        }
        else
        {
            prelayer_shift = (li + 1) * batch_n * hy_stride;

            gg = CreateGemmGeometryRNN(batch_n,
                                       hy_h * bi,
                                       wei_len * bi,
                                       1,
                                       1,
                                       false,
                                       false,
                                       false,
                                       hy_stride,
                                       bi_stride,
                                       hy_stride,
                                       false,
                                       network_config);
            gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
            gg.RunGemm(
                handle, workSpace, w, workSpace, prelayer_shift, wei_shift, hid_shift + dhd_off);

            // Update time
            profileRNNkernels(handle, 1);
        }

        // from hidden state
        bacc   = batch_n;
        baccbi = 0;
        for(int ti = seqLen - 1; ti >= 0; ti--)
        {
            bacc -= in_n[ti];

            alpha0 = 1;
            alpha1 = 0;
            beta_t = 1;

            // from post state
            if(ti == seqLen - 1)
            {
                if(in_n[ti] > 0)
                {
                    hx_size[2] = in_n[ti];
                    hx_size[3] = hy_h;
                    sp_size[2] = in_n[ti];
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(hx_desc),
                             dhy,
                             &alpha1,
                             miopen::deref(hx_desc),
                             dhy,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hx_shift,
                             hx_shift,
                             hid_shift + bacc * hy_stride + dhd_off);
                    // Update time
                    profileRNNkernels(handle, 1);
                }

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        hx_size[2] = in_n[seqLen - 1 - ti];
                        hx_size[3] = hy_h;
                        sp_size[2] = in_n[seqLen - 1 - ti];
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(hx_desc),
                                 dhy,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 dhy,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hx_shift + hy_n * hy_h,
                                 hx_shift + hy_n * hy_h,
                                 hid_shift + baccbi * hy_stride + dhd_off + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }
            }
            else
            {
                pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;

                if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                {
                    if(in_n[ti] > 0)
                    {
                        hx_size[2] = in_n[ti];
                        hx_size[3] = hy_h;
                        sp_size[2] = in_n[ti];
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(hx_desc),
                                 dhx,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 dhx,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hx_shift,
                                 hx_shift,
                                 hid_shift + bacc * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }
                else if(rnnMode == miopenLSTM || rnnMode == miopenGRU)
                {
                    if(in_n[ti + 1] > 0)
                    {
                        gg = CreateGemmGeometryRNN(in_n[ti + 1],
                                                   hy_h,
                                                   wei_len_t,
                                                   1,
                                                   1,
                                                   false,
                                                   false,
                                                   false,
                                                   hy_stride,
                                                   uni_stride,
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
                                   hid_shift + bacc * hy_stride + dhd_off);

                        // Update time
                        profileRNNkernels(handle, 1);

                        if(rnnMode == miopenGRU)
                        {
                            sp_size[2] = in_n[ti + 1];
                            sp_size[3] = hy_h;
                            miopenSetTensorDescriptor(
                                sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 1;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     pretime_shift + bi * 3 * hy_h,
                                     pretime_shift + nLayers * batch_n * hy_stride,
                                     hid_shift + bacc * hy_stride + bi * 3 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     pretime_shift + 2 * hy_h,
                                     pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                                     hid_shift + bacc * hy_stride + 2 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);

                            gg = CreateGemmGeometryRNN(in_n[ti + 1],
                                                       hy_h,
                                                       hy_h,
                                                       1,
                                                       1,
                                                       false,
                                                       false,
                                                       false,
                                                       hy_stride,
                                                       uni_stride,
                                                       hy_stride,
                                                       false,
                                                       network_config);
                            gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
                            gg.RunGemm(handle,
                                       workSpace,
                                       w,
                                       workSpace,
                                       hid_shift + bacc * hy_stride + 2 * hy_h,
                                       weitime_shift + 2 * hy_h * uni_stride,
                                       hid_shift + bacc * hy_stride + bi * 3 * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                    }
                }

                alpha0 = 1;
                alpha1 = 0;
                beta_t = 1;

                if(dirMode)
                {
                    if(in_n[seqLen - 1 - ti] > 0)
                    {
                        pretime_shift = li * batch_n * hy_stride +
                                        (baccbi - in_n[seqLen - 2 - ti]) * hy_stride + wei_len;

                        if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                        {
                            hx_size[2] = in_n[seqLen - 1 - ti];
                            hx_size[3] = hy_h;
                            sp_size[2] = in_n[seqLen - 1 - ti];
                            sp_size[3] = hy_h;
                            miopenSetTensorDescriptor(
                                hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());
                            miopenSetTensorDescriptor(
                                sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(hx_desc),
                                     dhx,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     dhx,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     hx_shift + hy_n * hy_h,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                        else if(rnnMode == miopenLSTM || rnnMode == miopenGRU)
                        {
                            gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                       hy_h,
                                                       wei_len_t,
                                                       1,
                                                       1,
                                                       false,
                                                       false,
                                                       false,
                                                       hy_stride,
                                                       uni_stride,
                                                       hy_stride,
                                                       false,
                                                       network_config);
                            gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
                            gg.RunGemm(handle,
                                       workSpace,
                                       w,
                                       workSpace,
                                       pretime_shift,
                                       weitime_shift + wei_len * uni_stride,
                                       hid_shift + baccbi * hy_stride + dhd_off + hy_h);

                            // Update time
                            profileRNNkernels(handle, 1);

                            if(rnnMode == miopenGRU)
                            {
                                sp_size[2] = in_n[seqLen - 1 - ti];
                                sp_size[3] = hy_h;
                                miopenSetTensorDescriptor(
                                    sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                                alpha0 = 1;
                                alpha1 = 1;
                                beta_t = 1;

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         &alpha1,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta_t,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         pretime_shift + 3 * hy_h + hy_h,
                                         pretime_shift + nLayers * batch_n * hy_stride,
                                         hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);
                                // Update time
                                profileRNNkernels(handle, 1);

                                alpha0 = 1;
                                alpha1 = 1;
                                beta_t = 0;

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         &alpha1,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta_t,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         pretime_shift + 2 * hy_h,
                                         pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                                         hid_shift + baccbi * hy_stride + 5 * hy_h);

                                // Update time
                                profileRNNkernels(handle, 1);

                                gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                           hy_h,
                                                           hy_h,
                                                           1,
                                                           1,
                                                           false,
                                                           false,
                                                           false,
                                                           hy_stride,
                                                           uni_stride,
                                                           hy_stride,
                                                           false,
                                                           network_config);
                                gg.FindSolution(.003, handle, workSpace, w, workSpace, false);
                                gg.RunGemm(handle,
                                           workSpace,
                                           w,
                                           workSpace,
                                           hid_shift + baccbi * hy_stride + 5 * hy_h,
                                           weitime_shift + 5 * hy_h * uni_stride,
                                           hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h);

                                // Update time
                                profileRNNkernels(handle, 1);
                            }
                        }
                    }
                }
            }

            // update hidden status
            if(in_n[ti] > 0)
            {
                offset     = hid_shift + bacc * hy_stride;
                sp_size[2] = in_n[ti];
                sp_size[3] = hy_h;
                miopenSetTensorDescriptor(
                    sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                {
                    // activation
                    activDesc.Backward(handle,
                                       &alpha,
                                       miopen::deref(sp_desc),
                                       reserveSpace,
                                       miopen::deref(sp_desc),
                                       workSpace,
                                       miopen::deref(sp_desc),
                                       reserveSpace,
                                       &beta,
                                       miopen::deref(sp_desc),
                                       workSpace,
                                       offset + nLayers * batch_n * hy_stride,
                                       offset,
                                       offset,
                                       offset);
                    // Update time
                    profileRNNkernels(handle, 1);

                    gg = CreateGemmGeometryRNN(in_n[ti],
                                               hy_h,
                                               hy_h,
                                               1,
                                               0,
                                               false,
                                               false,
                                               false,
                                               hy_stride,
                                               uni_stride,
                                               uni_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, workSpace, w, dhx, false);
                    gg.RunGemm(handle,
                               workSpace,
                               w,
                               dhx,
                               hid_shift + bacc * hy_stride,
                               weitime_shift,
                               hx_shift);

                    // Update time
                    profileRNNkernels(handle, 1);
                }
                else if(rnnMode == miopenLSTM)
                {
                    // update cell state
                    tanhDesc.Backward(handle,
                                      &alpha,
                                      miopen::deref(sp_desc),
                                      reserveSpace,
                                      miopen::deref(sp_desc),
                                      workSpace,
                                      miopen::deref(sp_desc),
                                      reserveSpace,
                                      &beta,
                                      miopen::deref(sp_desc),
                                      workSpace,
                                      offset + bi * 4 * hy_h + nLayers * batch_n * hy_stride,
                                      offset + bi * 5 * hy_h,
                                      offset + bi * 4 * hy_h,
                                      offset + bi * 4 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             offset + bi * 4 * hy_h,
                             offset + 2 * hy_h + nLayers * batch_n * hy_stride,
                             offset + bi * 4 * hy_h);

                    // Update time
                    profileRNNkernels(handle, 1);

                    if(ti == seqLen - 1)
                    {
                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        hx_size[2] = in_n[ti];
                        hx_size[3] = hy_h;
                        sp_size[2] = in_n[ti];
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(hx_desc),
                                 dcy,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 dcy,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hx_shift,
                                 hx_shift,
                                 offset + bi * 4 * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else
                    {
                        pretime_shift = li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride;

                        sp_size[2] = in_n[ti + 1];
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 pretime_shift + bi * 4 * hy_h,
                                 pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 4 * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }

                    // update forget gate
                    sp_size[2] = in_n[ti];
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     offset + hy_h + nLayers * batch_n * hy_stride,
                                     offset + bi * 4 * hy_h,
                                     offset + hy_h,
                                     offset + hy_h);

                    // Update time
                    profileRNNkernels(handle, 1);

                    if(ti == 0)
                    {
                        hx_size[2] = in_n[ti];
                        hx_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(hx_desc),
                                 cx,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 offset + hy_h,
                                 hx_shift,
                                 offset + hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else
                    {
                        pretime_shift =
                            li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride;

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 offset + hy_h,
                                 pretime_shift + bi * 4 * hy_h,
                                 offset + hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }

                    // update input gate
                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     offset + nLayers * batch_n * hy_stride,
                                     offset + bi * 4 * hy_h,
                                     offset,
                                     offset);
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             offset,
                             offset + 3 * hy_h + nLayers * batch_n * hy_stride,
                             offset);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // update output gate
                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     offset + 2 * hy_h + nLayers * batch_n * hy_stride,
                                     offset + bi * 5 * hy_h,
                                     offset + 2 * hy_h,
                                     offset + 2 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             offset + 2 * hy_h,
                             offset + bi * 4 * hy_h + nLayers * batch_n * hy_stride,
                             offset + 2 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // update c gate
                    tanhDesc.Backward(handle,
                                      &alpha,
                                      miopen::deref(sp_desc),
                                      reserveSpace,
                                      miopen::deref(sp_desc),
                                      workSpace,
                                      miopen::deref(sp_desc),
                                      reserveSpace,
                                      &beta,
                                      miopen::deref(sp_desc),
                                      workSpace,
                                      offset + 3 * hy_h + nLayers * batch_n * hy_stride,
                                      offset + bi * 4 * hy_h,
                                      offset + 3 * hy_h,
                                      offset + 3 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             offset + 3 * hy_h,
                             offset + nLayers * batch_n * hy_stride,
                             offset + 3 * hy_h);

                    // Update time
                    profileRNNkernels(handle, 1);
                }
                else if(rnnMode == miopenGRU)
                {
                    // c gate
                    alpha0 = 1;
                    alpha1 = -1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + nLayers * batch_n * hy_stride,
                             hid_shift + bacc * hy_stride + 2 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride + 2 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    tanhDesc.Backward(handle,
                                      &alpha,
                                      miopen::deref(sp_desc),
                                      reserveSpace,
                                      miopen::deref(sp_desc),
                                      workSpace,
                                      miopen::deref(sp_desc),
                                      reserveSpace,
                                      &beta,
                                      miopen::deref(sp_desc),
                                      workSpace,
                                      offset + 2 * hy_h + nLayers * batch_n * hy_stride,
                                      offset + 2 * hy_h,
                                      offset + 2 * hy_h,
                                      offset + 2 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // r gate
                    if(ti == 0)
                    {
                        gg = CreateGemmGeometryRNN(in_n[ti],
                                                   hy_h,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   uni_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, hx, w, workSpace, false);
                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   workSpace,
                                   hx_shift,
                                   weitime_shift + 2 * hy_h * uni_stride,
                                   hid_shift + bacc * hy_stride + hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else
                    {
                        gg = CreateGemmGeometryRNN(in_n[ti],
                                                   hy_h,
                                                   hy_h,
                                                   1,
                                                   1,
                                                   false,
                                                   true,
                                                   false,
                                                   hy_stride,
                                                   uni_stride,
                                                   hy_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, reserveSpace, w, workSpace, false);
                        gg.RunGemm(handle,
                                   reserveSpace,
                                   w,
                                   workSpace,
                                   hid_shift + (bacc - in_n[ti - 1]) * hy_stride + bi * 3 * hy_h,
                                   weitime_shift + 2 * hy_h * uni_stride,
                                   hid_shift + bacc * hy_stride + hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride + 2 * hy_h,
                             hid_shift + bacc * hy_stride + hy_h,
                             hid_shift + bacc * hy_stride + hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     offset + hy_h + nLayers * batch_n * hy_stride,
                                     offset + hy_h,
                                     offset + hy_h,
                                     offset + hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);

                    // z gate
                    alpha0 = 1;
                    alpha1 = -1;
                    beta_t = 0;

                    if(ti == 0)
                    {
                        hx_size[2] = in_n[ti];
                        hx_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(hx_desc),
                                 hx,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
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
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + (bacc - in_n[ti - 1]) * hy_stride + bi * 3 * hy_h,
                                 hid_shift + bacc * hy_stride + 2 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + bacc * hy_stride);
                    }
                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + bacc * hy_stride + bi * 3 * hy_h,
                             hid_shift + bacc * hy_stride,
                             hid_shift + bacc * hy_stride);
                    // Update time
                    profileRNNkernels(handle, 1);

                    sigDesc.Backward(handle,
                                     &alpha,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     offset + nLayers * batch_n * hy_stride,
                                     offset,
                                     offset,
                                     offset);
                    // Update time
                    profileRNNkernels(handle, 1);
                }
            }

            if(dirMode)
            {
                if(in_n[seqLen - 1 - ti] > 0)
                {
                    offset     = hid_shift + baccbi * hy_stride;
                    sp_size[2] = in_n[seqLen - 1 - ti];
                    sp_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                    {
                        // activation
                        activDesc.Backward(handle,
                                           &alpha,
                                           miopen::deref(sp_desc),
                                           reserveSpace,
                                           miopen::deref(sp_desc),
                                           workSpace,
                                           miopen::deref(sp_desc),
                                           reserveSpace,
                                           &beta,
                                           miopen::deref(sp_desc),
                                           workSpace,
                                           offset + hy_h + nLayers * batch_n * hy_stride,
                                           offset + hy_h,
                                           offset + hy_h,
                                           offset + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                   hy_h,
                                                   hy_h,
                                                   1,
                                                   0,
                                                   false,
                                                   false,
                                                   false,
                                                   hy_stride,
                                                   uni_stride,
                                                   uni_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, workSpace, w, dhx, false);
                        gg.RunGemm(handle,
                                   workSpace,
                                   w,
                                   dhx,
                                   hid_shift + baccbi * hy_stride + hy_h,
                                   weitime_shift + wei_len * uni_stride,
                                   hx_shift + hy_n * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else if(rnnMode == miopenLSTM)
                    {
                        // update cell state
                        tanhDesc.Backward(handle,
                                          &alpha,
                                          miopen::deref(sp_desc),
                                          reserveSpace,
                                          miopen::deref(sp_desc),
                                          workSpace,
                                          miopen::deref(sp_desc),
                                          reserveSpace,
                                          &beta,
                                          miopen::deref(sp_desc),
                                          workSpace,
                                          offset + bi * 4 * hy_h + hy_h +
                                              nLayers * batch_n * hy_stride,
                                          offset + bi * 5 * hy_h + hy_h,
                                          offset + bi * 4 * hy_h + hy_h,
                                          offset + bi * 4 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 offset + bi * 4 * hy_h + hy_h,
                                 offset + 6 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 4 * hy_h + hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        if(ti == seqLen - 1)
                        {
                            alpha0 = 1;
                            alpha1 = 0;
                            beta_t = 1;

                            hx_size[2] = in_n[seqLen - 1 - ti];
                            hx_size[3] = hy_h;
                            miopenSetTensorDescriptor(
                                hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(hx_desc),
                                     dcy,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     dcy,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     hx_shift + hy_n * hy_h,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + bi * 4 * hy_h + hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                        else
                        {
                            pretime_shift = li * batch_n * hy_stride +
                                            (baccbi - in_n[seqLen - 2 - ti]) * hy_stride;

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 1;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     pretime_shift + bi * 4 * hy_h + hy_h,
                                     pretime_shift + 5 * hy_h + nLayers * batch_n * hy_stride,
                                     offset + bi * 4 * hy_h + hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);
                        }

                        // update forget gate
                        if(ti == 0)
                        {
                            sigDesc.Backward(handle,
                                             &alpha,
                                             miopen::deref(sp_desc),
                                             reserveSpace,
                                             miopen::deref(sp_desc),
                                             workSpace,
                                             miopen::deref(sp_desc),
                                             reserveSpace,
                                             &beta,
                                             miopen::deref(sp_desc),
                                             workSpace,
                                             offset + 5 * hy_h + nLayers * batch_n * hy_stride,
                                             offset + bi * 4 * hy_h + hy_h,
                                             offset + 5 * hy_h,
                                             offset + 5 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);

                            hx_size[2] = in_n[seqLen - 1 - ti];
                            hx_size[3] = hy_h;
                            miopenSetTensorDescriptor(
                                hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(hx_desc),
                                     cx,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     offset + 5 * hy_h,
                                     hx_shift + hy_n * hy_h,
                                     offset + 5 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                        else
                        {
                            if(in_n[seqLen - ti] > 0)
                            {
                                pretime_shift = li * batch_n * hy_stride +
                                                (baccbi + in_n[seqLen - 1 - ti]) * hy_stride;

                                sp_size[2] = in_n[seqLen - ti];
                                sp_size[3] = hy_h;
                                miopenSetTensorDescriptor(
                                    sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                                alpha0 = 1;
                                alpha1 = 1;
                                beta_t = 0;

                                sigDesc.Backward(handle,
                                                 &alpha,
                                                 miopen::deref(sp_desc),
                                                 reserveSpace,
                                                 miopen::deref(sp_desc),
                                                 workSpace,
                                                 miopen::deref(sp_desc),
                                                 reserveSpace,
                                                 &beta,
                                                 miopen::deref(sp_desc),
                                                 workSpace,
                                                 offset + 5 * hy_h + nLayers * batch_n * hy_stride,
                                                 offset + bi * 4 * hy_h + hy_h,
                                                 offset + 5 * hy_h,
                                                 offset + 5 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1);

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         &alpha1,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta_t,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         offset + 5 * hy_h,
                                         pretime_shift + bi * 4 * hy_h + hy_h,
                                         offset + 5 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1);
                            }
                        }

                        // update input gate
                        sp_size[2] = in_n[seqLen - 1 - ti];
                        sp_size[3] = hy_h;
                        miopenSetTensorDescriptor(
                            sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                        sigDesc.Backward(handle,
                                         &alpha,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         offset + 4 * hy_h + nLayers * batch_n * hy_stride,
                                         offset + bi * 4 * hy_h + hy_h,
                                         offset + 4 * hy_h,
                                         offset + 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 offset + 4 * hy_h,
                                 offset + 7 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // update output gate
                        sigDesc.Backward(handle,
                                         &alpha,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         offset + 6 * hy_h + nLayers * batch_n * hy_stride,
                                         offset + bi * 5 * hy_h + hy_h,
                                         offset + 6 * hy_h,
                                         offset + 6 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 offset + 6 * hy_h,
                                 offset + bi * 4 * hy_h + hy_h + nLayers * batch_n * hy_stride,
                                 offset + 6 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // update c gate
                        tanhDesc.Backward(handle,
                                          &alpha,
                                          miopen::deref(sp_desc),
                                          reserveSpace,
                                          miopen::deref(sp_desc),
                                          workSpace,
                                          miopen::deref(sp_desc),
                                          reserveSpace,
                                          &beta,
                                          miopen::deref(sp_desc),
                                          workSpace,
                                          offset + 7 * hy_h + nLayers * batch_n * hy_stride,
                                          offset + bi * 4 * hy_h + hy_h,
                                          offset + 7 * hy_h,
                                          offset + 7 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 offset + 7 * hy_h,
                                 offset + 4 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 7 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else if(rnnMode == miopenGRU)
                    {
                        // c gate
                        alpha0 = 1;
                        alpha1 = -1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + 3 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        tanhDesc.Backward(handle,
                                          &alpha,
                                          miopen::deref(sp_desc),
                                          reserveSpace,
                                          miopen::deref(sp_desc),
                                          workSpace,
                                          miopen::deref(sp_desc),
                                          reserveSpace,
                                          &beta,
                                          miopen::deref(sp_desc),
                                          workSpace,
                                          offset + 5 * hy_h + nLayers * batch_n * hy_stride,
                                          offset + 5 * hy_h,
                                          offset + 5 * hy_h,
                                          offset + 5 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // r gate
                        if(ti == 0)
                        {
                            gg = CreateGemmGeometryRNN(in_n[seqLen - 1 - ti],
                                                       hy_h,
                                                       hy_h,
                                                       1,
                                                       1,
                                                       false,
                                                       true,
                                                       false,
                                                       uni_stride,
                                                       uni_stride,
                                                       hy_stride,
                                                       false,
                                                       network_config);
                            gg.FindSolution(.003, handle, hx, w, workSpace, false);
                            gg.RunGemm(handle,
                                       hx,
                                       w,
                                       workSpace,
                                       hx_shift + hy_n * hy_h,
                                       weitime_shift + 5 * hy_h * uni_stride,
                                       hid_shift + baccbi * hy_stride + 4 * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1);
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
                                                           true,
                                                           false,
                                                           hy_stride,
                                                           uni_stride,
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
                                           weitime_shift + 5 * hy_h * uni_stride,
                                           hid_shift + baccbi * hy_stride + 4 * hy_h);

                                // Update time
                                profileRNNkernels(handle, 1);
                            }
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + baccbi * hy_stride + 5 * hy_h,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h,
                                 hid_shift + baccbi * hy_stride + 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        sigDesc.Backward(handle,
                                         &alpha,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         offset + 4 * hy_h + nLayers * batch_n * hy_stride,
                                         offset + 4 * hy_h,
                                         offset + 4 * hy_h,
                                         offset + 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        // z gate
                        if(ti == 0)
                        {
                            alpha0 = 1;
                            alpha1 = -1;
                            beta_t = 0;

                            hx_size[2] = in_n[seqLen - 1 - ti];
                            hx_size[3] = hy_h;
                            miopenSetTensorDescriptor(
                                hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     miopen::deref(hx_desc),
                                     hx,
                                     &alpha1,
                                     miopen::deref(sp_desc),
                                     reserveSpace,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     hx_shift + hy_n * hy_h,
                                     hid_shift + baccbi * hy_stride + 5 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &alpha1,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     &beta_t,
                                     miopen::deref(sp_desc),
                                     workSpace,
                                     hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h,
                                     hid_shift + baccbi * hy_stride + 3 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);

                            sigDesc.Backward(handle,
                                             &alpha,
                                             miopen::deref(sp_desc),
                                             reserveSpace,
                                             miopen::deref(sp_desc),
                                             workSpace,
                                             miopen::deref(sp_desc),
                                             reserveSpace,
                                             &beta,
                                             miopen::deref(sp_desc),
                                             workSpace,
                                             offset + 3 * hy_h + nLayers * batch_n * hy_stride,
                                             offset + 3 * hy_h,
                                             offset + 3 * hy_h,
                                             offset + 3 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1);
                        }
                        else
                        {
                            if(in_n[seqLen - ti] > 0)
                            {
                                sp_size[2] = in_n[seqLen - ti];
                                sp_size[3] = hy_h;
                                miopenSetTensorDescriptor(
                                    sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                                alpha0 = 1;
                                alpha1 = -1;
                                beta_t = 0;

                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &alpha1,
                                         miopen::deref(sp_desc),
                                         reserveSpace,
                                         &beta_t,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         hid_shift + (baccbi + in_n[seqLen - 1 - ti]) * hy_stride +
                                             bi * 3 * hy_h + hy_h,
                                         hid_shift + baccbi * hy_stride + 5 * hy_h +
                                             nLayers * batch_n * hy_stride,
                                         hid_shift + baccbi * hy_stride + 3 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1);

                                alpha0 = 1;
                                alpha1 = 1;
                                beta_t = 0;

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         &alpha1,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         &beta_t,
                                         miopen::deref(sp_desc),
                                         workSpace,
                                         hid_shift + baccbi * hy_stride + bi * 3 * hy_h + hy_h,
                                         hid_shift + baccbi * hy_stride + 3 * hy_h,
                                         hid_shift + baccbi * hy_stride + 3 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1);

                                sigDesc.Backward(handle,
                                                 &alpha,
                                                 miopen::deref(sp_desc),
                                                 reserveSpace,
                                                 miopen::deref(sp_desc),
                                                 workSpace,
                                                 miopen::deref(sp_desc),
                                                 reserveSpace,
                                                 &beta,
                                                 miopen::deref(sp_desc),
                                                 workSpace,
                                                 offset + 3 * hy_h + nLayers * batch_n * hy_stride,
                                                 offset + 3 * hy_h,
                                                 offset + 3 * hy_h,
                                                 offset + 3 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1);
                            }
                        }
                    }
                }
            }

            baccbi += in_n[seqLen - 1 - ti];
        }

        // dcx, dhx
        if(rnnMode == miopenLSTM || rnnMode == miopenGRU)
        {
            if(in_n[0] > 0)
            {
                pretime_shift = li * batch_n * hy_stride;

                sp_size[2] = in_n[0];
                sp_size[3] = hy_h;
                hx_size[2] = in_n[0];
                hx_size[3] = hy_h;
                miopenSetTensorDescriptor(
                    sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
                miopenSetTensorDescriptor(
                    hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                if(rnnMode == miopenLSTM)
                {
                    gg = CreateGemmGeometryRNN(in_n[0],
                                               hy_h,
                                               hy_h * 4,
                                               1,
                                               1,
                                               false,
                                               false,
                                               false,
                                               hy_stride,
                                               uni_stride,
                                               uni_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, workSpace, w, dhx, false);
                    gg.RunGemm(handle, workSpace, w, dhx, pretime_shift, weitime_shift, hx_shift);

                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(hx_desc),
                             dcx,
                             pretime_shift + bi * 4 * hy_h,
                             pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                             hx_shift);
                    // Update time
                    profileRNNkernels(handle, 1);
                }
                else if(rnnMode == miopenGRU)
                {
                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 0;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             pretime_shift + 2 * hy_h,
                             pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                             pretime_shift + bi * 3 * hy_h + nLayers * batch_n * hy_stride);
                    // Update time
                    profileRNNkernels(handle, 1);

                    gg = CreateGemmGeometryRNN(in_n[0],
                                               hy_h,
                                               hy_h,
                                               1,
                                               0,
                                               false,
                                               false,
                                               false,
                                               hy_stride,
                                               uni_stride,
                                               uni_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, reserveSpace, w, dhx, false);
                    gg.RunGemm(handle,
                               reserveSpace,
                               w,
                               dhx,
                               pretime_shift + bi * 3 * hy_h + nLayers * batch_n * hy_stride,
                               weitime_shift + 2 * hy_h * uni_stride,
                               hx_shift);

                    // Update time
                    profileRNNkernels(handle, 1);

                    alpha0 = 1;
                    alpha1 = 1;
                    beta_t = 1;

                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &beta_t,
                             miopen::deref(hx_desc),
                             dhx,
                             pretime_shift + bi * 3 * hy_h,
                             pretime_shift + nLayers * batch_n * hy_stride,
                             hx_shift);
                    // Update time
                    profileRNNkernels(handle, 1);

                    gg = CreateGemmGeometryRNN(in_n[0],
                                               hy_h,
                                               hy_h * 2,
                                               1,
                                               1,
                                               false,
                                               false,
                                               false,
                                               hy_stride,
                                               uni_stride,
                                               uni_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, workSpace, w, dhx, false);
                    gg.RunGemm(handle, workSpace, w, dhx, pretime_shift, weitime_shift, hx_shift);

                    // Update time
                    profileRNNkernels(handle, 1);
                }
            }

            if(dirMode)
            {
                if(in_n[seqLen - 1] > 0)
                {
                    pretime_shift =
                        li * batch_n * hy_stride + (batch_n - in_n[seqLen - 1]) * hy_stride;

                    sp_size[2] = in_n[seqLen - 1];
                    sp_size[3] = hy_h;
                    hx_size[2] = in_n[seqLen - 1];
                    hx_size[3] = hy_h;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
                    miopenSetTensorDescriptor(
                        hx_desc, miopenFloat, 4, hx_size.data(), hx_stride.data());

                    if(rnnMode == miopenLSTM)
                    {
                        gg = CreateGemmGeometryRNN(in_n[seqLen - 1],
                                                   hy_h,
                                                   hy_h * 4,
                                                   1,
                                                   1,
                                                   false,
                                                   false,
                                                   false,
                                                   hy_stride,
                                                   uni_stride,
                                                   uni_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, workSpace, w, dhx, false);
                        gg.RunGemm(handle,
                                   workSpace,
                                   w,
                                   dhx,
                                   pretime_shift + 4 * hy_h,
                                   weitime_shift + 4 * hy_h * uni_stride,
                                   hx_shift + hy_n * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(hx_desc),
                                 dcx,
                                 pretime_shift + bi * 4 * hy_h + hy_h,
                                 pretime_shift + 5 * hy_h + nLayers * batch_n * hy_stride,
                                 hx_shift + hy_n * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                    else if(rnnMode == miopenGRU)
                    {
                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 pretime_shift + 5 * hy_h,
                                 pretime_shift + 4 * hy_h + nLayers * batch_n * hy_stride,
                                 pretime_shift + bi * 3 * hy_h + hy_h +
                                     nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1);

                        gg = CreateGemmGeometryRNN(in_n[seqLen - 1],
                                                   hy_h,
                                                   hy_h,
                                                   1,
                                                   0,
                                                   false,
                                                   false,
                                                   false,
                                                   hy_stride,
                                                   uni_stride,
                                                   uni_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, reserveSpace, w, dhx, false);
                        gg.RunGemm(handle,
                                   reserveSpace,
                                   w,
                                   dhx,
                                   pretime_shift + bi * 3 * hy_h + hy_h +
                                       nLayers * batch_n * hy_stride,
                                   weitime_shift + 5 * hy_h * uni_stride,
                                   hx_shift + hy_n * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &beta_t,
                                 miopen::deref(hx_desc),
                                 dhx,
                                 pretime_shift + bi * 3 * hy_h + hy_h,
                                 pretime_shift + 3 * hy_h + nLayers * batch_n * hy_stride,
                                 hx_shift + hy_n * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);

                        gg = CreateGemmGeometryRNN(in_n[seqLen - 1],
                                                   hy_h,
                                                   hy_h * 2,
                                                   1,
                                                   1,
                                                   false,
                                                   false,
                                                   false,
                                                   hy_stride,
                                                   uni_stride,
                                                   uni_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, workSpace, w, dhx, false);
                        gg.RunGemm(handle,
                                   workSpace,
                                   w,
                                   dhx,
                                   pretime_shift + 3 * hy_h,
                                   weitime_shift + 3 * hy_h * uni_stride,
                                   hx_shift + hy_n * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }
            }
        }
    }

    // dinput
    if(inputMode == miopenRNNskip)
    {
        sp_size[2] = batch_n;
        sp_size[3] = hy_h;
        x_size[2]  = batch_n;
        x_size[3]  = hy_h;
        miopenSetTensorDescriptor(sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
        miopenSetTensorDescriptor(x_desc, miopenFloat, 4, x_size.data(), x_stride.data());

        alpha0 = 1;
        alpha1 = 0;
        beta_t = 1;

        for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
        {
            OpTensor(handle,
                     miopenTensorOpAdd,
                     &alpha0,
                     miopen::deref(sp_desc),
                     workSpace,
                     &alpha1,
                     miopen::deref(sp_desc),
                     workSpace,
                     &beta_t,
                     miopen::deref(x_desc),
                     dx,
                     gi * hy_h,
                     gi * hy_h,
                     0);
            // Update time
            profileRNNkernels(handle, (gi == nHiddenTensorsPerLayer * bi - 1) ? 2 : 1);
        }
    }
    else
    {
        gg = CreateGemmGeometryRNN(batch_n,
                                   in_h,
                                   wei_len * bi,
                                   1,
                                   1,
                                   false,
                                   false,
                                   false,
                                   hy_stride,
                                   in_stride,
                                   in_stride,
                                   false,
                                   network_config);
        gg.FindSolution(.003, handle, workSpace, w, dx, false);
        gg.RunGemm(handle, workSpace, w, dx, 0, 0, 0);

        // Update time
        profileRNNkernels(handle, 1);
    }
#else
    MIOPEN_THROW("GEMM is not supported");
#endif

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
                                       Data_t workSpace,
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
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

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

    std::vector<int> sp_size(4, 1), sp_stride(4, 1), w_size(4, 1), w_stride(4, 1);
    miopenTensorDescriptor_t sp_desc, w_desc;
    miopenCreateTensorDescriptor(&sp_desc);
    miopenCreateTensorDescriptor(&w_desc);
    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = batch_n * hy_stride;
    sp_stride[2] = hy_stride;
    w_stride[0]  = wei_stride;
    w_stride[1]  = wei_stride;
    w_stride[2]  = wei_stride;

#if MIOPEN_USE_MIOPENGEMM
    GemmGeometry gg;
    int hid_shift, hx_shift, wei_shift, prelayer_shift, pretime_shift;
    int wei_len, hid_off;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        printf("run rnn gpu bwd weights \n");
        wei_len = hy_h;
        hid_off = nLayers * batch_n * hy_stride;
        break;
    case miopenLSTM:
        printf("run lstm gpu bwd weights \n");
        wei_len = hy_h * 4;
        hid_off = bi * hy_h * 5;
        break;
    case miopenGRU:
        printf("run gru gpu bwd weights \n");
        wei_len = hy_h * 3;
        hid_off = bi * hy_h * 3;
        break;
    }

    for(int li = 0; li < nLayers; li++)
    {
        // between layers
        if(li == 0)
        {
            if(inputMode == miopenRNNlinear)
            {
                gg = CreateGemmGeometryRNN(wei_len * bi,
                                           in_h,
                                           batch_n,
                                           1,
                                           1,
                                           true,
                                           false,
                                           false,
                                           hy_stride,
                                           in_stride,
                                           in_stride,
                                           false,
                                           network_config);
                gg.FindSolution(.003, handle, workSpace, x, dw, false);
                gg.RunGemm(handle, workSpace, x, dw, 0, 0, 0);

                // Update time
                profileRNNkernels(handle, 0);
            }

            if(biasMode)
            {
                if(inputMode == miopenRNNskip && rnnMode == miopenGRU)
                    ;
                else
                {
                    sp_size[2] = 1;
                    sp_size[3] = wei_stride;
                    w_size[2]  = 1;
                    w_size[3]  = wei_stride;
                    miopenSetTensorDescriptor(
                        sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
                    miopenSetTensorDescriptor(
                        w_desc, miopenFloat, 4, w_size.data(), w_stride.data());

                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(w_desc),
                                 dw,
                                 bs * hy_stride,
                                 bs * hy_stride,
                                 wei_shift_bias);

                        // Update time
                        if((inputMode != miopenRNNlinear) && bs == 0)
                            profileRNNkernels(handle, 0);
                        else
                            profileRNNkernels(handle, 1);
                    }

                    if(inputMode == miopenRNNlinear && rnnMode != miopenGRU)
                    {
                        CopyTensor(handle,
                                   miopen::deref(w_desc),
                                   dw,
                                   miopen::deref(w_desc),
                                   dw,
                                   wei_shift_bias,
                                   wei_shift_bias + wei_stride);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }
            }
        }
        else
        {
            prelayer_shift = (li - 1) * batch_n * hy_stride + hid_off;
            hid_shift      = li * batch_n * hy_stride;
            wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

            gg = CreateGemmGeometryRNN(wei_len * bi,
                                       hy_h * bi,
                                       batch_n,
                                       1,
                                       1,
                                       true,
                                       false,
                                       false,
                                       hy_stride,
                                       hy_stride,
                                       bi_stride,
                                       false,
                                       network_config);
            gg.FindSolution(.003, handle, workSpace, reserveSpace, dw, false);
            gg.RunGemm(handle, workSpace, reserveSpace, dw, hid_shift, prelayer_shift, wei_shift);

            // Update time
            profileRNNkernels(handle, 1);

            if(biasMode)
            {
                wei_shift = (inputMode == miopenRNNskip)
                                ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                : (wei_shift_bias + li * 2 * wei_stride);

                sp_size[2] = 1;
                sp_size[3] = wei_stride;
                w_size[2]  = 1;
                w_size[3]  = wei_stride;
                miopenSetTensorDescriptor(
                    sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
                miopenSetTensorDescriptor(w_desc, miopenFloat, 4, w_size.data(), w_stride.data());

                alpha0 = 1;
                alpha1 = 0;
                beta_t = 1;

                for(int bs = 0; bs < batch_n; bs++)
                {
                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             miopen::deref(sp_desc),
                             workSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(w_desc),
                             dw,
                             hid_shift + bs * hy_stride,
                             hid_shift + bs * hy_stride,
                             wei_shift);

                    // Update time
                    profileRNNkernels(handle, 1);
                }

                if(rnnMode != miopenGRU)
                {
                    CopyTensor(handle,
                               miopen::deref(w_desc),
                               dw,
                               miopen::deref(w_desc),
                               dw,
                               wei_shift,
                               wei_shift + wei_stride);
                    // Update time
                    profileRNNkernels(handle, 1);
                }
            }
        }

        // between time
        bacc = 0;
        for(int ti = 0; ti < seqLen; ti++)
        {
            hid_shift = li * batch_n * hy_stride + bacc * hy_stride;
            hx_shift  = li * hy_n * bi_stride;
            wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(rnnMode == miopenGRU)
            {
                sp_size[2] = in_n[ti];
                sp_size[3] = hy_h;
                miopenSetTensorDescriptor(
                    sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());

                alpha0 = 1;
                alpha1 = 1;
                beta_t = 0;

                if(in_n[ti] > 0)
                {
                    OpTensor(handle,
                             miopenTensorOpMul,
                             &alpha0,
                             miopen::deref(sp_desc),
                             reserveSpace,
                             &alpha1,
                             miopen::deref(sp_desc),
                             workSpace,
                             &beta_t,
                             miopen::deref(sp_desc),
                             workSpace,
                             hid_shift + hy_h + nLayers * batch_n * hy_stride,
                             hid_shift + 2 * hy_h,
                             hid_shift + 2 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1);
                }
            }

            if(ti == 0)
            {
                if(in_n[ti] > 0)
                {
                    gg = CreateGemmGeometryRNN(wei_len,
                                               hy_h,
                                               in_n[ti],
                                               1,
                                               1,
                                               true,
                                               false,
                                               false,
                                               hy_stride,
                                               uni_stride,
                                               uni_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, workSpace, hx, dw, false);
                    gg.RunGemm(handle, workSpace, hx, dw, hid_shift, hx_shift, wei_shift);

                    // Update time
                    profileRNNkernels(handle, 1);
                }
            }
            else
            {
                pretime_shift =
                    li * batch_n * hy_stride + (bacc - in_n[ti - 1]) * hy_stride + hid_off;

                if(in_n[ti] > 0)
                {
                    gg = CreateGemmGeometryRNN(wei_len,
                                               hy_h,
                                               in_n[ti],
                                               1,
                                               1,
                                               true,
                                               false,
                                               false,
                                               hy_stride,
                                               hy_stride,
                                               uni_stride,
                                               false,
                                               network_config);
                    gg.FindSolution(.003, handle, workSpace, reserveSpace, dw, false);
                    gg.RunGemm(
                        handle, workSpace, reserveSpace, dw, hid_shift, pretime_shift, wei_shift);

                    // Update time
                    if(dirMode)
                        profileRNNkernels(handle, 1);
                    else if((li == nLayers) && (ti == seqLen - 1))
                        profileRNNkernels(handle, 2);
                    else
                        profileRNNkernels(handle, 1);
                }
            }

            if(dirMode)
            {
                if(rnnMode == miopenGRU)
                {
                    if(in_n[ti] > 0)
                    {
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 miopen::deref(sp_desc),
                                 reserveSpace,
                                 &alpha1,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 &beta_t,
                                 miopen::deref(sp_desc),
                                 workSpace,
                                 hid_shift + 4 * hy_h + nLayers * batch_n * hy_stride,
                                 hid_shift + 5 * hy_h,
                                 hid_shift + 5 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }

                if(ti == seqLen - 1)
                {
                    if(in_n[ti] > 0)
                    {
                        gg = CreateGemmGeometryRNN(wei_len,
                                                   hy_h,
                                                   in_n[ti],
                                                   1,
                                                   1,
                                                   true,
                                                   false,
                                                   false,
                                                   hy_stride,
                                                   uni_stride,
                                                   uni_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, workSpace, hx, dw, false);
                        gg.RunGemm(handle,
                                   workSpace,
                                   hx,
                                   dw,
                                   hid_shift + wei_len,
                                   hx_shift + hy_n * hy_h,
                                   wei_shift + wei_len * uni_stride);

                        // Update time
                        if(rnnMode != miopenGRU)
                            profileRNNkernels(handle, 2);
                        else
                            profileRNNkernels(handle, 1);
                    }
                }
                else
                {
                    pretime_shift =
                        li * batch_n * hy_stride + (bacc + in_n[ti]) * hy_stride + hid_off;

                    if(in_n[ti + 1] > 0)
                    {
                        gg = CreateGemmGeometryRNN(wei_len,
                                                   hy_h,
                                                   in_n[ti + 1],
                                                   1,
                                                   1,
                                                   true,
                                                   false,
                                                   false,
                                                   hy_stride,
                                                   hy_stride,
                                                   uni_stride,
                                                   false,
                                                   network_config);
                        gg.FindSolution(.003, handle, workSpace, reserveSpace, dw, false);
                        gg.RunGemm(handle,
                                   workSpace,
                                   reserveSpace,
                                   dw,
                                   hid_shift + wei_len,
                                   pretime_shift + hy_h,
                                   wei_shift + wei_len * uni_stride);

                        // Update time
                        profileRNNkernels(handle, 1);
                    }
                }
            }
            bacc += in_n[ti];
        }

        if(rnnMode == miopenGRU && biasMode)
        {
            int in_bias_val = inputMode == miopenRNNskip ? 0 : wei_stride;

            hid_shift = li * batch_n * hy_stride;
            wei_shift = (li == 0) ? (wei_shift_bias + in_bias_val)
                                  : (wei_shift_bias + in_bias_val + li * 2 * wei_stride);

            sp_size[2] = 1;
            sp_size[3] = wei_stride;
            w_size[2]  = 1;
            w_size[3]  = wei_stride;
            miopenSetTensorDescriptor(sp_desc, miopenFloat, 4, sp_size.data(), sp_stride.data());
            miopenSetTensorDescriptor(w_desc, miopenFloat, 4, w_size.data(), w_stride.data());

            alpha0 = 1;
            alpha1 = 0;
            beta_t = 1;

            for(int bs = 0; bs < batch_n; bs++)
            {
                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         miopen::deref(sp_desc),
                         workSpace,
                         &alpha1,
                         miopen::deref(sp_desc),
                         workSpace,
                         &beta_t,
                         miopen::deref(w_desc),
                         dw,
                         hid_shift + bs * hy_stride,
                         hid_shift + bs * hy_stride,
                         wei_shift);

                // Update time
                profileRNNkernels(handle, (bs == batch_n - 1) ? 2 : 1);
            }
        }
    }
#else
    MIOPEN_THROW("GEMM is not supported");
#endif

    // Suppress warning
    (void)dwDesc;
    (void)workSpaceSize;
    (void)reserveSpaceSize;
};

// TODO: LATER
/*
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
 */

} // namespace miopen
