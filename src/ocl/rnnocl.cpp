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

#include <miopen/activ.hpp>
#include <miopen/rnn.hpp>
#include <miopen/env.hpp>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <vector>
#include <numeric>
#include <algorithm>

#if MIOPEN_USE_MIOPENGEMM
#include <miopen/gemm.hpp>
#endif

//#define MIO_RNN_OCL_DEBUG 1
#define MIO_RNN_FINDSOL_TIMEOUT 0.003

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
    if(hxDesc.GetSize() != cxDesc.GetSize() || hxDesc.GetSize() != hyDesc.GetSize() ||
       hxDesc.GetSize() != cyDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(workSpaceSize < GetWorkspaceSize(handle, seqLen, xDesc))
    {
        MIOPEN_THROW("Workspace is required");
    }

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
            MIOPEN_THROW(miopenStatusBadParm,
                         "Input batch length: " + std::to_string(batchval) +
                             ", Output batch length: " + std::to_string(batchvalout));
        }
        if(i == 0)
        {
            if(batchval == 0)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Input batch is ZERO!");
            }
        }
        else
        {
            if(batchval > in_n.back())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Incorrect input batch size at time " + std::to_string(i) +
                                 "! Batch size must not ascend!");
            }
        }
        in_n.push_back(batchval);
        batch_n += batchval;
    }

    int bi = dirMode ? 2 : 1;
    if(out_h != (bi * hy_h))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Output size doesn't match hidden state size!");
    }

    float ctime    = 0.;
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
            MIOPEN_THROW(miopenStatusBadParm,
                         "The input tensor size must equal to the hidden "
                         "state size of the network in SKIP_INPUT mode!");
        }
        in_h = 0;
    }

    size_t wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (nLayers - 1)) * wei_stride;
    size_t offset;
    float alpha0, alpha1, beta_t;
    float alpha = 1, beta = 0;

    std::vector<int> sp_size(3, 1), sp_stride(3, 1), w_size(3, 1), w_stride(3, 1), x_size(3, 1),
        x_stride(3, 1), y_size(3, 1), y_stride(3, 1), hx_size(3, 1), hx_stride(3, 1);
    miopen::TensorDescriptor sp_desc, w_desc, x_desc, y_desc, hx_desc;

    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = hy_stride;
    w_stride[0]  = wei_stride;
    w_stride[1]  = wei_stride;
    x_stride[0]  = batch_n * in_stride;
    x_stride[1]  = in_stride;
    y_stride[0]  = batch_n * out_stride;
    y_stride[1]  = out_stride;
    hx_stride[0] = in_n[0] * uni_stride;
    hx_stride[1] = uni_stride;

#if MIOPEN_USE_MIOPENGEMM

    int wei_shift, prelayer_shift;
    int wei_len   = 0;
    int wei_len_t = 0;
    int hid_off   = 0;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        // printf("run rnn gpu inference \n");
        wei_len   = hy_h;
        wei_len_t = hy_h;
        hid_off   = 0;
        break;
    case miopenLSTM:
        // printf("run lstm gpu inference \n");
        wei_len   = hy_h * 4;
        wei_len_t = hy_h * 4;
        hid_off   = bi * hy_h * 5;
        break;
    case miopenGRU:
        // printf("run gru gpu inference \n");
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
        int hid_shift           = li * batch_n * hy_stride;
        int hx_shift            = li * hy_n * bi_stride;
        int wei_shift_bias_temp = inputMode == miopenRNNskip
                                      ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                      : (wei_shift_bias + li * 2 * wei_stride);

        // from input
        if(li == 0)
        {
            if(inputMode == miopenRNNskip)
            {
                x_size[1]  = batch_n;
                x_size[2]  = hy_h;
                sp_size[1] = batch_n;
                sp_size[2] = hy_h;
                x_desc = miopen::TensorDescriptor(miopenFloat, x_size.data(), x_stride.data(), 3);
                sp_desc =
                    miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

                for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
                {
                    CopyTensor(handle, x_desc, x, sp_desc, workSpace, 0, gi * hy_h);
                    // Update time
                    profileRNNkernels(handle, (gi == 0) ? 0 : 1, ctime);
                }
            }
            else
            {

                auto gg = ScanGemmGeometryRNN(handle,
                                              x,
                                              w,
                                              workSpace,
                                              batch_n,
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
                                              network_config,
                                              MIO_RNN_FINDSOL_TIMEOUT);

                gg.RunGemm(handle, x, w, workSpace, 0, 0, hid_shift);

                // Update time
                profileRNNkernels(handle, 0, ctime);
            }
        }
        else
        {
            wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            prelayer_shift = (li - 1) * batch_n * hy_stride + hid_off;

            auto gg = ScanGemmGeometryRNN(handle,
                                          workSpace,
                                          w,
                                          workSpace,
                                          batch_n,
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
                                          network_config,
                                          MIO_RNN_FINDSOL_TIMEOUT);

            gg.RunGemm(handle, workSpace, w, workSpace, prelayer_shift, wei_shift, hid_shift);

            // Update time
            profileRNNkernels(handle, 1, ctime);
        }

        if(biasMode)
        {
            int wn = rnnMode == miopenGRU ? 1 : 2;
            if(inputMode == miopenRNNskip && li == 0)
            {
                wei_shift_bias_temp = wei_shift_bias;
                wn                  = rnnMode == miopenGRU ? 0 : 1;
            }

            alpha0 = 1;
            alpha1 = 1;
            beta_t = 0;

            w_size[1]  = 1;
            w_size[2]  = wei_stride;
            sp_size[1] = batch_n;
            sp_size[2] = wei_stride;
            w_desc     = miopen::TensorDescriptor(miopenFloat, w_size.data(), w_stride.data(), 3);
            sp_desc    = miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

            for(int bs = 0; bs < wn; bs++)
            {
                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         sp_desc,
                         workSpace,
                         &alpha1,
                         w_desc,
                         w,
                         &beta_t,
                         sp_desc,
                         workSpace,
                         hid_shift,
                         wei_shift_bias_temp + bs * wei_stride,
                         hid_shift);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }

            if(rnnMode == miopenGRU)
            {
                for(int bs = 0; bs < bi; bs++)
                {
                    w_size[2]  = 2 * hy_h;
                    sp_size[2] = 2 * hy_h;
                    w_desc =
                        miopen::TensorDescriptor(miopenFloat, w_size.data(), w_stride.data(), 3);
                    sp_desc =
                        miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             sp_desc,
                             workSpace,
                             &alpha1,
                             w_desc,
                             w,
                             &beta_t,
                             sp_desc,
                             workSpace,
                             hid_shift + bs * 3 * hy_h,
                             wei_shift_bias_temp + wn * wei_stride + bs * 3 * hy_h,
                             hid_shift + bs * 3 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1, ctime);

                    w_size[2]  = hy_h;
                    sp_size[2] = hy_h;
                    w_desc =
                        miopen::TensorDescriptor(miopenFloat, w_size.data(), w_stride.data(), 3);
                    sp_desc =
                        miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             sp_desc,
                             workSpace,
                             &alpha1,
                             w_desc,
                             w,
                             &beta_t,
                             sp_desc,
                             workSpace,
                             hid_shift + bi * 3 * hy_h + bs * hy_h,
                             wei_shift_bias_temp + wn * wei_stride + 2 * hy_h + bs * 3 * hy_h,
                             hid_shift + bi * 3 * hy_h + bs * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1, ctime);
                }
            }
        }

        // from hidden state
        int bacc   = 0;
        int baccbi = batch_n;
        for(int ti = 0; ti < seqLen; ti++)
        {
            baccbi -= in_n[seqLen - 1 - ti];
            wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            for(int ri = 0; ri < bi; ri++)
            {
                int cur_time  = ri == 0 ? ti : seqLen - 1 - ti;
                int cur_batch = ri == 0 ? bacc : baccbi;
                offset        = hid_shift + cur_batch * hy_stride;

                if(in_n[cur_time] > 0)
                {
                    if(ti == 0)
                    {

                        auto gg = ScanGemmGeometryRNN(handle,
                                                      hx,
                                                      w,
                                                      workSpace,
                                                      in_n.at(cur_time),
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
                                                      network_config,
                                                      MIO_RNN_FINDSOL_TIMEOUT);

                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   workSpace,
                                   hx_shift + ri * hy_n * hy_h,
                                   wei_shift + ri * wei_len * uni_stride,
                                   offset + ri * wei_len);

                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(rnnMode == miopenGRU)
                        {

                            auto gg2 = ScanGemmGeometryRNN(handle,
                                                           hx,
                                                           w,
                                                           workSpace,
                                                           in_n.at(cur_time),
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
                                                           network_config,
                                                           MIO_RNN_FINDSOL_TIMEOUT);
                            gg2.RunGemm(handle,
                                        hx,
                                        w,
                                        workSpace,
                                        hx_shift + ri * hy_n * hy_h,
                                        wei_shift + 2 * hy_h * uni_stride +
                                            ri * 3 * hy_h * uni_stride,
                                        offset + bi * 3 * hy_h + ri * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }
                    else
                    {

                        auto gg = ScanGemmGeometryRNN(handle,
                                                      hy,
                                                      w,
                                                      workSpace,
                                                      in_n.at(cur_time),
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
                                                      network_config,
                                                      MIO_RNN_FINDSOL_TIMEOUT);

                        gg.RunGemm(handle,
                                   hy,
                                   w,
                                   workSpace,
                                   hx_shift + ri * hy_n * hy_h,
                                   wei_shift + ri * wei_len * uni_stride,
                                   offset + ri * wei_len);

                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(rnnMode == miopenGRU)
                        {

                            auto gg2 = ScanGemmGeometryRNN(handle,
                                                           hy,
                                                           w,
                                                           workSpace,
                                                           in_n.at(cur_time),
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
                                                           network_config,
                                                           MIO_RNN_FINDSOL_TIMEOUT);

                            gg2.RunGemm(handle,
                                        hy,
                                        w,
                                        workSpace,
                                        hx_shift + ri * hy_n * hy_h,
                                        wei_shift + 2 * hy_h * uni_stride +
                                            ri * 3 * hy_h * uni_stride,
                                        offset + bi * 3 * hy_h + ri * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }

                    // update hidden status
                    hx_size[1] = in_n[cur_time];
                    hx_size[2] = hy_h;
                    hx_desc =
                        miopen::TensorDescriptor(miopenFloat, hx_size.data(), hx_stride.data(), 3);

                    sp_size[1] = in_n[cur_time];
                    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                    {
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        activDesc.Forward(handle,
                                          &alpha,
                                          sp_desc,
                                          workSpace,
                                          &beta,
                                          sp_desc,
                                          workSpace,
                                          offset + ri * hy_h,
                                          offset + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenLSTM)
                    {
                        // active gate i, f, o
                        sp_size[2] = hy_h * 3;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        sigDesc.Forward(handle,
                                        &alpha,
                                        sp_desc,
                                        workSpace,
                                        &beta,
                                        sp_desc,
                                        workSpace,
                                        offset + ri * 4 * hy_h,
                                        offset + ri * 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // active gate c
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         workSpace,
                                         &beta,
                                         sp_desc,
                                         workSpace,
                                         offset + 3 * hy_h + ri * 4 * hy_h,
                                         offset + 3 * hy_h + ri * 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update cell state
                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 workSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + ri * 4 * hy_h,
                                 offset + 3 * hy_h + ri * 4 * hy_h,
                                 offset + bi * 4 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(ti == 0)
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     workSpace,
                                     &alpha1,
                                     hx_desc,
                                     cx,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     offset + hy_h + ri * 4 * hy_h,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + bi * 4 * hy_h + ri * hy_h);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     workSpace,
                                     &alpha1,
                                     hx_desc,
                                     cy,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     offset + hy_h + ri * 4 * hy_h,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + bi * 4 * hy_h + ri * hy_h);
                        }
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update cy
                        CopyTensor(handle,
                                   sp_desc,
                                   workSpace,
                                   hx_desc,
                                   cy,
                                   offset + bi * 4 * hy_h + ri * hy_h,
                                   hx_shift + ri * hy_n * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // active cell state
                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         workSpace,
                                         &beta,
                                         sp_desc,
                                         workSpace,
                                         offset + bi * 4 * hy_h + ri * hy_h,
                                         offset + bi * 4 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update hidden state
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 workSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + 2 * hy_h + ri * 4 * hy_h,
                                 offset + bi * 4 * hy_h + ri * hy_h,
                                 offset + bi * 5 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenGRU)
                    {
                        // active z, r gate
                        sp_size[2] = 2 * hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        sigDesc.Forward(handle,
                                        &alpha,
                                        sp_desc,
                                        workSpace,
                                        &beta,
                                        sp_desc,
                                        workSpace,
                                        offset + ri * 3 * hy_h,
                                        offset + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // calculate c gate
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 workSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + hy_h + ri * 3 * hy_h,
                                 offset + bi * 3 * hy_h + ri * hy_h,
                                 offset + 2 * hy_h + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // active c gate
                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         workSpace,
                                         &beta,
                                         sp_desc,
                                         workSpace,
                                         offset + 2 * hy_h + ri * 3 * hy_h,
                                         offset + 2 * hy_h + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // calculate hidden state
                        alpha0 = -1;
                        alpha1 = 1;
                        beta_t = 0;
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 workSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + ri * 3 * hy_h,
                                 offset + 2 * hy_h + ri * 3 * hy_h,
                                 offset + bi * 3 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 workSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + 2 * hy_h + ri * 3 * hy_h,
                                 offset + bi * 3 * hy_h + ri * hy_h,
                                 offset + bi * 3 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;
                        if(ti == 0)
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     workSpace,
                                     &alpha1,
                                     hx_desc,
                                     hx,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     offset + ri * 3 * hy_h,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + bi * 3 * hy_h + ri * hy_h);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     workSpace,
                                     &alpha1,
                                     hx_desc,
                                     hy,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     offset + ri * 3 * hy_h,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + bi * 3 * hy_h + ri * hy_h);
                        }
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }

                    // update hy
                    CopyTensor(handle,
                               sp_desc,
                               workSpace,
                               hx_desc,
                               hy,
                               offset + hid_off + ri * hy_h,
                               hx_shift + ri * hy_n * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1, ctime);
                }
            }

            bacc += in_n[ti];
        }

        // hy, cy clean
        if(in_n[0] - in_n[seqLen - 1] > 0)
        {
            hx_size[1] = in_n[0] - in_n[seqLen - 1];
            hx_size[2] = hy_h;
            hx_desc    = miopen::TensorDescriptor(miopenFloat, hx_size.data(), hx_stride.data(), 3);

            alpha0 = 0;
            alpha1 = 0;
            beta_t = 0;

            OpTensor(handle,
                     miopenTensorOpMul,
                     &alpha0,
                     hx_desc,
                     hy,
                     &alpha1,
                     hx_desc,
                     hy,
                     &beta_t,
                     hx_desc,
                     hy,
                     hx_shift + in_n[seqLen - 1] * uni_stride,
                     hx_shift + in_n[seqLen - 1] * uni_stride,
                     hx_shift + in_n[seqLen - 1] * uni_stride);
            // Update time
            profileRNNkernels(handle, 1, ctime);

            if(rnnMode == miopenLSTM)
            {
                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         hx_desc,
                         cy,
                         &alpha1,
                         hx_desc,
                         cy,
                         &beta_t,
                         hx_desc,
                         cy,
                         hx_shift + in_n[seqLen - 1] * uni_stride,
                         hx_shift + in_n[seqLen - 1] * uni_stride,
                         hx_shift + in_n[seqLen - 1] * uni_stride);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
        }
    }

    // output
    prelayer_shift = (nLayers - 1) * batch_n * hy_stride + hid_off;

    sp_size[1] = batch_n;
    sp_size[2] = hy_h * bi;
    y_size[1]  = batch_n;
    y_size[2]  = out_h;
    y_desc     = miopen::TensorDescriptor(miopenFloat, y_size.data(), y_stride.data(), 3);
    sp_desc    = miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

    CopyTensor(handle, sp_desc, workSpace, y_desc, y, prelayer_shift, 0);
    // Update time
    profileRNNkernels(handle, 2, ctime);

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
    if(hxDesc.GetSize() != cxDesc.GetSize() || hxDesc.GetSize() != hyDesc.GetSize() ||
       hxDesc.GetSize() != cyDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(workSpaceSize < GetWorkspaceSize(handle, seqLen, xDesc))
    {
        MIOPEN_THROW("Workspace is required");
    }
    if(reserveSpaceSize < GetReserveSize(handle, seqLen, xDesc))
    {
        MIOPEN_THROW("Reservespace is required");
    }

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
            MIOPEN_THROW(miopenStatusBadParm,
                         "Input batch length: " + std::to_string(batchval) +
                             ", Output batch length: " + std::to_string(batchvalout));
        }
        if(i == 0)
        {
            if(batchval == 0)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Input batch is ZERO!");
            }
        }
        else
        {
            if(batchval > in_n.back())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Incorrect input batch size at time " + std::to_string(i) +
                                 "! Batch size must not ascend!");
            }
        }
        in_n.push_back(batchval);
        batch_n += batchval;
    }

    int bi = dirMode ? 2 : 1;
    if(out_h != (bi * hy_h))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Output size doesn't match hidden state size!");
    }

    float ctime    = 0.;
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
            MIOPEN_THROW(miopenStatusBadParm,
                         "The input tensor size must equal to the hidden "
                         "state size of the network in SKIP_INPUT mode!");
        }
        in_h = 0;
    }

    size_t wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (nLayers - 1)) * wei_stride;
    size_t offset;
    float alpha0, alpha1, beta_t;
    float alpha = 1, beta = 0;

    std::vector<int> sp_size(3, 1), sp_stride(3, 1), w_size(3, 1), w_stride(3, 1), x_size(3, 1),
        x_stride(3, 1), y_size(3, 1), y_stride(3, 1), hx_size(3, 1), hx_stride(3, 1);
    miopen::TensorDescriptor sp_desc, w_desc, x_desc, y_desc, hx_desc;

    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = hy_stride;
    w_stride[0]  = wei_stride;
    w_stride[1]  = wei_stride;
    x_stride[0]  = batch_n * in_stride;
    x_stride[1]  = in_stride;
    y_stride[0]  = batch_n * out_stride;
    y_stride[1]  = out_stride;
    hx_stride[0] = in_n[0] * uni_stride;
    hx_stride[1] = uni_stride;

#if MIOPEN_USE_MIOPENGEMM

    int wei_shift, prelayer_shift;
    int wei_len   = 0;
    int wei_len_t = 0;
    int hid_off   = 0;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        // printf("run rnn gpu fwd \n");
        wei_len   = hy_h;
        wei_len_t = hy_h;
        hid_off   = nLayers * batch_n * hy_stride;
        break;
    case miopenLSTM:
        // printf("run lstm gpu fwd \n");
        wei_len   = hy_h * 4;
        wei_len_t = hy_h * 4;
        hid_off   = bi * hy_h * 5;
        break;
    case miopenGRU:
        // printf("run gru gpu fwd \n");
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
        int hid_shift           = li * batch_n * hy_stride;
        int hx_shift            = li * hy_n * bi_stride;
        int wei_shift_bias_temp = inputMode == miopenRNNskip
                                      ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                      : (wei_shift_bias + li * 2 * wei_stride);

        // from input
        if(li == 0)
        {
            if(inputMode == miopenRNNskip)
            {
                x_size[1]  = batch_n;
                x_size[2]  = hy_h;
                sp_size[1] = batch_n;
                sp_size[2] = hy_h;
                x_desc = miopen::TensorDescriptor(miopenFloat, x_size.data(), x_stride.data(), 3);
                sp_desc =
                    miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

                for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
                {
                    CopyTensor(handle, x_desc, x, sp_desc, reserveSpace, 0, gi * hy_h);
                    // Update time
                    profileRNNkernels(handle, (gi == 0) ? 0 : 1, ctime);
                }
            }
            else
            {
                auto gg = ScanGemmGeometryRNN(handle,
                                              x,
                                              w,
                                              reserveSpace,
                                              batch_n,
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
                                              network_config,
                                              MIO_RNN_FINDSOL_TIMEOUT);

                gg.RunGemm(handle, x, w, reserveSpace, 0, 0, hid_shift);

                // Update time
                profileRNNkernels(handle, 0, ctime);
            }
        }
        else
        {
            wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            prelayer_shift = (li - 1) * batch_n * hy_stride + hid_off;

            auto gg = ScanGemmGeometryRNN(handle,
                                          reserveSpace,
                                          w,
                                          reserveSpace,
                                          batch_n,
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
                                          network_config,
                                          MIO_RNN_FINDSOL_TIMEOUT);

            gg.RunGemm(handle, reserveSpace, w, reserveSpace, prelayer_shift, wei_shift, hid_shift);

            // Update time
            profileRNNkernels(handle, 1, ctime);
        }

        if(biasMode)
        {
            int wn = rnnMode == miopenGRU ? 1 : 2;
            if(inputMode == miopenRNNskip && li == 0)
            {
                wei_shift_bias_temp = wei_shift_bias;
                wn                  = rnnMode == miopenGRU ? 0 : 1;
            }

            alpha0 = 1;
            alpha1 = 1;
            beta_t = 0;

            w_size[1]  = 1;
            w_size[2]  = wei_stride;
            sp_size[1] = batch_n;
            sp_size[2] = wei_stride;
            w_desc     = miopen::TensorDescriptor(miopenFloat, w_size.data(), w_stride.data(), 3);
            sp_desc    = miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

            for(int bs = 0; bs < wn; bs++)
            {
                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         sp_desc,
                         reserveSpace,
                         &alpha1,
                         w_desc,
                         w,
                         &beta_t,
                         sp_desc,
                         reserveSpace,
                         hid_shift,
                         wei_shift_bias_temp + bs * wei_stride,
                         hid_shift);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }

            if(rnnMode == miopenGRU)
            {
                for(int bs = 0; bs < bi; bs++)
                {
                    w_size[2]  = 2 * hy_h;
                    sp_size[2] = 2 * hy_h;
                    w_desc =
                        miopen::TensorDescriptor(miopenFloat, w_size.data(), w_stride.data(), 3);
                    sp_desc =
                        miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             sp_desc,
                             reserveSpace,
                             &alpha1,
                             w_desc,
                             w,
                             &beta_t,
                             sp_desc,
                             reserveSpace,
                             hid_shift + bs * 3 * hy_h,
                             wei_shift_bias_temp + wn * wei_stride + bs * 3 * hy_h,
                             hid_shift + bs * 3 * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1, ctime);

                    w_size[2]  = hy_h;
                    sp_size[2] = hy_h;
                    w_desc =
                        miopen::TensorDescriptor(miopenFloat, w_size.data(), w_stride.data(), 3);
                    sp_desc =
                        miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             sp_desc,
                             reserveSpace,
                             &alpha1,
                             w_desc,
                             w,
                             &beta_t,
                             sp_desc,
                             reserveSpace,
                             hid_shift + bi * 3 * hy_h + bs * hy_h,
                             wei_shift_bias_temp + wn * wei_stride + 2 * hy_h + bs * 3 * hy_h,
                             hid_shift + bi * 3 * hy_h + bs * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1, ctime);
                }
            }
        }

        // from hidden state
        int bacc   = 0;
        int baccbi = batch_n;
        for(int ti = 0; ti < seqLen; ti++)
        {
            baccbi -= in_n[seqLen - 1 - ti];
            wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            for(int ri = 0; ri < bi; ri++)
            {
                int cur_time  = ri == 0 ? ti : seqLen - 1 - ti;
                int cur_batch = ri == 0 ? bacc : baccbi;
                offset        = hid_shift + cur_batch * hy_stride;

                if(in_n[cur_time] > 0)
                {
                    if(ti == 0)
                    {
                        auto gg = ScanGemmGeometryRNN(handle,
                                                      hx,
                                                      w,
                                                      reserveSpace,
                                                      in_n.at(cur_time),
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
                                                      network_config,
                                                      MIO_RNN_FINDSOL_TIMEOUT);

                        gg.RunGemm(handle,
                                   hx,
                                   w,
                                   reserveSpace,
                                   hx_shift + ri * hy_n * hy_h,
                                   wei_shift + ri * wei_len * uni_stride,
                                   offset + ri * wei_len);

                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(rnnMode == miopenGRU)
                        {

                            auto gg2 = ScanGemmGeometryRNN(handle,
                                                           hx,
                                                           w,
                                                           reserveSpace,
                                                           in_n.at(cur_time),
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
                                                           network_config,
                                                           MIO_RNN_FINDSOL_TIMEOUT);

                            gg2.RunGemm(handle,
                                        hx,
                                        w,
                                        reserveSpace,
                                        hx_shift + ri * hy_n * hy_h,
                                        wei_shift + 2 * hy_h * uni_stride +
                                            ri * 3 * hy_h * uni_stride,
                                        offset + bi * 3 * hy_h + ri * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }
                    else
                    {

                        auto gg = ScanGemmGeometryRNN(handle,
                                                      hx,
                                                      w,
                                                      reserveSpace,
                                                      in_n.at(cur_time),
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
                                                      network_config,
                                                      MIO_RNN_FINDSOL_TIMEOUT);

                        gg.RunGemm(handle,
                                   hy,
                                   w,
                                   reserveSpace,
                                   hx_shift + ri * hy_n * hy_h,
                                   wei_shift + ri * wei_len * uni_stride,
                                   offset + ri * wei_len);

                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(rnnMode == miopenGRU)
                        {

                            auto gg2 = ScanGemmGeometryRNN(handle,
                                                           hy,
                                                           w,
                                                           reserveSpace,
                                                           in_n.at(cur_time),
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
                                                           network_config,
                                                           MIO_RNN_FINDSOL_TIMEOUT);

                            gg2.RunGemm(handle,
                                        hy,
                                        w,
                                        reserveSpace,
                                        hx_shift + ri * hy_n * hy_h,
                                        wei_shift + 2 * hy_h * uni_stride +
                                            ri * 3 * hy_h * uni_stride,
                                        offset + bi * 3 * hy_h + ri * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }

                    // update hidden status
                    hx_size[1] = in_n[cur_time];
                    hx_size[2] = hy_h;
                    hx_desc =
                        miopen::TensorDescriptor(miopenFloat, hx_size.data(), hx_stride.data(), 3);

                    sp_size[1] = in_n[cur_time];
                    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                    {
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        activDesc.Forward(handle,
                                          &alpha,
                                          sp_desc,
                                          reserveSpace,
                                          &beta,
                                          sp_desc,
                                          reserveSpace,
                                          offset + ri * hy_h,
                                          offset + ri * hy_h + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenLSTM)
                    {
                        // active gate i, f, o
                        sp_size[2] = hy_h * 3;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        sigDesc.Forward(handle,
                                        &alpha,
                                        sp_desc,
                                        reserveSpace,
                                        &beta,
                                        sp_desc,
                                        reserveSpace,
                                        offset + ri * 4 * hy_h,
                                        offset + ri * 4 * hy_h + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // active gate c
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         reserveSpace,
                                         &beta,
                                         sp_desc,
                                         reserveSpace,
                                         offset + 3 * hy_h + ri * 4 * hy_h,
                                         offset + 3 * hy_h + ri * 4 * hy_h +
                                             nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update cell state
                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 reserveSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 reserveSpace,
                                 offset + ri * 4 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 3 * hy_h + ri * 4 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 4 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(ti == 0)
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     reserveSpace,
                                     &alpha1,
                                     hx_desc,
                                     cx,
                                     &beta_t,
                                     sp_desc,
                                     reserveSpace,
                                     offset + hy_h + ri * 4 * hy_h + nLayers * batch_n * hy_stride,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + bi * 4 * hy_h + ri * hy_h);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     reserveSpace,
                                     &alpha1,
                                     hx_desc,
                                     cy,
                                     &beta_t,
                                     sp_desc,
                                     reserveSpace,
                                     offset + hy_h + ri * 4 * hy_h + nLayers * batch_n * hy_stride,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + bi * 4 * hy_h + ri * hy_h);
                        }
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update cy
                        CopyTensor(handle,
                                   sp_desc,
                                   reserveSpace,
                                   hx_desc,
                                   cy,
                                   offset + bi * 4 * hy_h + ri * hy_h,
                                   hx_shift + ri * hy_n * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // active cell state
                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         reserveSpace,
                                         &beta,
                                         sp_desc,
                                         reserveSpace,
                                         offset + bi * 4 * hy_h + ri * hy_h,
                                         offset + bi * 4 * hy_h + ri * hy_h +
                                             nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update hidden state
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 reserveSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 reserveSpace,
                                 offset + 2 * hy_h + ri * 4 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 4 * hy_h + ri * hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 5 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenGRU)
                    {
                        // active z, r gate
                        sp_size[2] = 2 * hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        sigDesc.Forward(handle,
                                        &alpha,
                                        sp_desc,
                                        reserveSpace,
                                        &beta,
                                        sp_desc,
                                        reserveSpace,
                                        offset + ri * 3 * hy_h,
                                        offset + ri * 3 * hy_h + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // calculate c gate
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 reserveSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 reserveSpace,
                                 offset + hy_h + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 3 * hy_h + ri * hy_h,
                                 offset + 2 * hy_h + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // active c gate
                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         reserveSpace,
                                         &beta,
                                         sp_desc,
                                         reserveSpace,
                                         offset + 2 * hy_h + ri * 3 * hy_h,
                                         offset + 2 * hy_h + ri * 3 * hy_h +
                                             nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // calculate hidden state
                        alpha0 = -1;
                        alpha1 = 1;
                        beta_t = 0;
                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 reserveSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 reserveSpace,
                                 offset + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 2 * hy_h + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 3 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 sp_desc,
                                 reserveSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 reserveSpace,
                                 offset + 2 * hy_h + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 3 * hy_h + ri * hy_h,
                                 offset + bi * 3 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;
                        if(ti == 0)
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     reserveSpace,
                                     &alpha1,
                                     hx_desc,
                                     hx,
                                     &beta_t,
                                     sp_desc,
                                     reserveSpace,
                                     offset + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + bi * 3 * hy_h + ri * hy_h);
                        }
                        else
                        {
                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     reserveSpace,
                                     &alpha1,
                                     hx_desc,
                                     hy,
                                     &beta_t,
                                     sp_desc,
                                     reserveSpace,
                                     offset + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + bi * 3 * hy_h + ri * hy_h);
                        }
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }

                    // update hy
                    CopyTensor(handle,
                               sp_desc,
                               reserveSpace,
                               hx_desc,
                               hy,
                               offset + hid_off + ri * hy_h,
                               hx_shift + ri * hy_n * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1, ctime);
                }
            }

            bacc += in_n[ti];
        }

        // hy, cy clean
        if(in_n[0] - in_n[seqLen - 1] > 0)
        {
            hx_size[1] = in_n[0] - in_n[seqLen - 1];
            hx_size[2] = hy_h;
            hx_desc    = miopen::TensorDescriptor(miopenFloat, hx_size.data(), hx_stride.data(), 3);

            alpha0 = 0;
            alpha1 = 0;
            beta_t = 0;

            OpTensor(handle,
                     miopenTensorOpMul,
                     &alpha0,
                     hx_desc,
                     hy,
                     &alpha1,
                     hx_desc,
                     hy,
                     &beta_t,
                     hx_desc,
                     hy,
                     hx_shift + in_n[seqLen - 1] * uni_stride,
                     hx_shift + in_n[seqLen - 1] * uni_stride,
                     hx_shift + in_n[seqLen - 1] * uni_stride);
            // Update time
            profileRNNkernels(handle, 1, ctime);

            if(rnnMode == miopenLSTM)
            {
                OpTensor(handle,
                         miopenTensorOpMul,
                         &alpha0,
                         hx_desc,
                         cy,
                         &alpha1,
                         hx_desc,
                         cy,
                         &beta_t,
                         hx_desc,
                         cy,
                         hx_shift + in_n[seqLen - 1] * uni_stride,
                         hx_shift + in_n[seqLen - 1] * uni_stride,
                         hx_shift + in_n[seqLen - 1] * uni_stride);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
        }
    }

    // output
    prelayer_shift = (nLayers - 1) * batch_n * hy_stride + hid_off;

    sp_size[1] = batch_n;
    sp_size[2] = hy_h * bi;
    y_size[1]  = batch_n;
    y_size[2]  = out_h;
    y_desc     = miopen::TensorDescriptor(miopenFloat, y_size.data(), y_stride.data(), 3);
    sp_desc    = miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

    CopyTensor(handle, sp_desc, reserveSpace, y_desc, y, prelayer_shift, 0);
    // Update time
    profileRNNkernels(handle, 2, ctime);

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
    if(dhyDesc.GetSize() != dcyDesc.GetSize() || dhyDesc.GetSize() != hxDesc.GetSize() ||
       dhyDesc.GetSize() != cxDesc.GetSize() || dhyDesc.GetSize() != dhxDesc.GetSize() ||
       dhyDesc.GetSize() != dcxDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(workSpaceSize < GetWorkspaceSize(handle, seqLen, dxDesc))
    {
        MIOPEN_THROW("Workspace is required");
    }
    if(reserveSpaceSize < GetReserveSize(handle, seqLen, dxDesc))
    {
        MIOPEN_THROW("Reservespace is required");
    }

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
        if(i == 0)
        {
            if(batchval == 0)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Input batch is ZERO!");
            }
        }
        else
        {
            if(batchval > in_n.back())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Incorrect input batch size at time " + std::to_string(i) +
                                 "! Batch size must not ascend!");
            }
        }
        in_n.push_back(batchval);
        batch_n += dxDesc[i].GetLengths()[0];
    }

    int bi = dirMode ? 2 : 1;
    if(out_h != (bi * hy_h))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Output size doesn't match hidden state size!");
    }

    float ctime    = 0.;
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
            MIOPEN_THROW(miopenStatusBadParm,
                         "The input tensor size must equal to the hidden "
                         "state size of the network in SKIP_INPUT mode!");
        }
        in_h = 0;
    }

    size_t offset;
    float alpha0, alpha1, beta_t;
    float alpha = 1, beta = 0;

    std::vector<int> sp_size(3, 1), sp_stride(3, 1), x_size(3, 1), x_stride(3, 1), y_size(3, 1),
        y_stride(3, 1), hx_size(3, 1), hx_stride(3, 1);
    miopen::TensorDescriptor sp_desc, x_desc, y_desc, hx_desc;

    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = hy_stride;
    x_stride[0]  = batch_n * in_stride;
    x_stride[1]  = in_stride;
    y_stride[0]  = batch_n * out_stride;
    y_stride[1]  = out_stride;
    hx_stride[0] = in_n[0] * uni_stride;
    hx_stride[1] = uni_stride;

#if MIOPEN_USE_MIOPENGEMM

    int prelayer_shift, pretime_shift, cur_time, cur_batch;
    int wei_len    = 0;
    int wei_len_t  = 0;
    int dhd_off    = 0;
    int use_time   = 0;
    int pre_batch  = 0;
    int use_time2  = 0;
    int pre_batch2 = 0;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        // printf("run rnn gpu bwd data \n");
        wei_len   = hy_h;
        wei_len_t = hy_h;
        dhd_off   = 0;
        break;
    case miopenLSTM:
        // printf("run lstm gpu bwd data \n");
        wei_len   = hy_h * 4;
        wei_len_t = hy_h * 4;
        dhd_off   = bi * hy_h * 5;
        break;
    case miopenGRU:
        // printf("run gru gpu bwd data \n");
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
        int wei_shift     = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
        int hid_shift     = li * batch_n * hy_stride;
        int hx_shift      = li * hy_n * bi_stride;
        int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

        // feedback from output
        if(li == nLayers - 1)
        {
            y_size[1]  = batch_n;
            y_size[2]  = out_h;
            sp_size[1] = batch_n;
            sp_size[2] = hy_h * bi;
            y_desc     = miopen::TensorDescriptor(miopenFloat, y_size.data(), y_stride.data(), 3);
            sp_desc    = miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

            alpha0 = 1;
            alpha1 = 0;
            beta_t = 1;

            CopyTensor(handle, y_desc, dy, sp_desc, workSpace, 0, hid_shift + dhd_off);
            // Update time
            profileRNNkernels(handle, 0, ctime); // start timing
        }
        else
        {
            prelayer_shift = (li + 1) * batch_n * hy_stride;

            auto gg = ScanGemmGeometryRNN(handle,
                                          workSpace,
                                          w,
                                          workSpace,
                                          batch_n,
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
                                          network_config,
                                          MIO_RNN_FINDSOL_TIMEOUT);
            gg.RunGemm(
                handle, workSpace, w, workSpace, prelayer_shift, wei_shift, hid_shift + dhd_off);

            // Update time
            profileRNNkernels(handle, 1, ctime);
        }

        // from hidden state
        int bacc   = batch_n;
        int baccbi = 0;
        for(int ti = seqLen - 1; ti >= 0; ti--)
        {
            bacc -= in_n[ti];

            // from post state
            for(int ri = 0; ri < bi; ri++)
            {
                cur_time  = ri == 0 ? ti : seqLen - 1 - ti;
                cur_batch = ri == 0 ? bacc : baccbi;
                offset    = hid_shift + cur_batch * hy_stride;
                if(ti < seqLen - 1)
                {
                    use_time  = ri == 0 ? ti + 1 : seqLen - 1 - ti;
                    pre_batch = ri == 0 ? bacc + in_n[ti] : baccbi - in_n[seqLen - 2 - ti];
                }
                if(ti > 0)
                {
                    use_time2  = ri == 0 ? ti : seqLen - ti;
                    pre_batch2 = ri == 0 ? bacc - in_n[ti - 1] : baccbi + in_n[seqLen - 1 - ti];
                }

                if(in_n[cur_time] > 0)
                {
                    alpha0 = 1;
                    alpha1 = 0;
                    beta_t = 1;

                    if(ti == seqLen - 1)
                    {
                        hx_size[1] = in_n[cur_time];
                        hx_size[2] = hy_h;
                        sp_size[1] = in_n[cur_time];
                        sp_size[2] = hy_h;
                        hx_desc    = miopen::TensorDescriptor(
                            miopenFloat, hx_size.data(), hx_stride.data(), 3);
                        sp_desc = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 hx_desc,
                                 dhy,
                                 &alpha1,
                                 hx_desc,
                                 dhy,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 hx_shift + ri * hy_n * hy_h,
                                 hx_shift + ri * hy_n * hy_h,
                                 offset + dhd_off + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else
                    {
                        pretime_shift =
                            li * batch_n * hy_stride + pre_batch * hy_stride + ri * wei_len;

                        if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                        {
                            hx_size[1] = in_n[cur_time];
                            hx_size[2] = hy_h;
                            sp_size[1] = in_n[cur_time];
                            sp_size[2] = hy_h;
                            hx_desc    = miopen::TensorDescriptor(
                                miopenFloat, hx_size.data(), hx_stride.data(), 3);
                            sp_desc = miopen::TensorDescriptor(
                                miopenFloat, sp_size.data(), sp_stride.data(), 3);

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     hx_desc,
                                     dhx,
                                     &alpha1,
                                     hx_desc,
                                     dhx,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     hx_shift + ri * hy_n * hy_h,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + ri * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                        else if(rnnMode == miopenLSTM || rnnMode == miopenGRU)
                        {
                            if(in_n[use_time] > 0)
                            {

                                auto gg = ScanGemmGeometryRNN(handle,
                                                              workSpace,
                                                              w,
                                                              workSpace,
                                                              in_n.at(use_time),
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
                                                              network_config,
                                                              MIO_RNN_FINDSOL_TIMEOUT);

                                gg.RunGemm(handle,
                                           workSpace,
                                           w,
                                           workSpace,
                                           pretime_shift,
                                           weitime_shift + ri * wei_len * uni_stride,
                                           offset + dhd_off + ri * hy_h);

                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                if(rnnMode == miopenGRU)
                                {
                                    sp_size[1] = in_n[use_time];
                                    sp_size[2] = hy_h;
                                    sp_desc    = miopen::TensorDescriptor(
                                        miopenFloat, sp_size.data(), sp_stride.data(), 3);

                                    alpha0 = 1;
                                    alpha1 = 1;
                                    beta_t = 1;

                                    OpTensor(handle,
                                             miopenTensorOpMul,
                                             &alpha0,
                                             sp_desc,
                                             workSpace,
                                             &alpha1,
                                             sp_desc,
                                             reserveSpace,
                                             &beta_t,
                                             sp_desc,
                                             workSpace,
                                             pretime_shift + bi * 3 * hy_h - ri * 2 * hy_h,
                                             pretime_shift + nLayers * batch_n * hy_stride,
                                             offset + bi * 3 * hy_h + ri * hy_h);
                                    // Update time
                                    profileRNNkernels(handle, 1, ctime);

                                    alpha0 = 1;
                                    alpha1 = 1;
                                    beta_t = 0;

                                    OpTensor(handle,
                                             miopenTensorOpMul,
                                             &alpha0,
                                             sp_desc,
                                             workSpace,
                                             &alpha1,
                                             sp_desc,
                                             reserveSpace,
                                             &beta_t,
                                             sp_desc,
                                             workSpace,
                                             pretime_shift + 2 * hy_h,
                                             pretime_shift + hy_h + nLayers * batch_n * hy_stride,
                                             offset + 2 * hy_h + ri * 3 * hy_h);
                                    // Update time
                                    profileRNNkernels(handle, 1, ctime);

                                    auto gg2 = ScanGemmGeometryRNN(handle,
                                                                   workSpace,
                                                                   w,
                                                                   workSpace,
                                                                   in_n.at(use_time),
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
                                                                   network_config,
                                                                   MIO_RNN_FINDSOL_TIMEOUT);

                                    gg2.RunGemm(handle,
                                                workSpace,
                                                w,
                                                workSpace,
                                                offset + 2 * hy_h + ri * 3 * hy_h,
                                                weitime_shift + 2 * hy_h * uni_stride +
                                                    ri * 3 * hy_h * uni_stride,
                                                offset + bi * 3 * hy_h + ri * hy_h);

                                    // Update time
                                    profileRNNkernels(handle, 1, ctime);
                                }
                            }
                        }
                    }

                    // update hidden status
                    sp_size[1] = in_n[cur_time];
                    sp_size[2] = hy_h;
                    sp_desc =
                        miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

                    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                    {
                        // activation
                        activDesc.Backward(handle,
                                           &alpha,
                                           sp_desc,
                                           reserveSpace,
                                           sp_desc,
                                           workSpace,
                                           sp_desc,
                                           reserveSpace,
                                           &beta,
                                           sp_desc,
                                           workSpace,
                                           offset + ri * hy_h + nLayers * batch_n * hy_stride,
                                           offset + ri * hy_h,
                                           offset + ri * hy_h,
                                           offset + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        auto gg = ScanGemmGeometryRNN(handle,
                                                      workSpace,
                                                      w,
                                                      dhx,
                                                      in_n.at(cur_time),
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
                                                      network_config,
                                                      MIO_RNN_FINDSOL_TIMEOUT);

                        gg.RunGemm(handle,
                                   workSpace,
                                   w,
                                   dhx,
                                   offset + ri * hy_h,
                                   weitime_shift + ri * wei_len * uni_stride,
                                   hx_shift + ri * hy_n * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenLSTM)
                    {
                        // update cell state
                        tanhDesc.Backward(handle,
                                          &alpha,
                                          sp_desc,
                                          reserveSpace,
                                          sp_desc,
                                          workSpace,
                                          sp_desc,
                                          reserveSpace,
                                          &beta,
                                          sp_desc,
                                          workSpace,
                                          offset + bi * 4 * hy_h + ri * hy_h +
                                              nLayers * batch_n * hy_stride,
                                          offset + bi * 5 * hy_h + ri * hy_h,
                                          offset + bi * 4 * hy_h + ri * hy_h,
                                          offset + bi * 4 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + bi * 4 * hy_h + ri * hy_h,
                                 offset + 2 * hy_h + ri * 4 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + bi * 4 * hy_h + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(ti == seqLen - 1)
                        {
                            alpha0 = 1;
                            alpha1 = 0;
                            beta_t = 1;

                            hx_size[1] = in_n[cur_time];
                            hx_size[2] = hy_h;
                            sp_size[1] = in_n[cur_time];
                            sp_size[2] = hy_h;
                            hx_desc    = miopen::TensorDescriptor(
                                miopenFloat, hx_size.data(), hx_stride.data(), 3);
                            sp_desc = miopen::TensorDescriptor(
                                miopenFloat, sp_size.data(), sp_stride.data(), 3);

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     hx_desc,
                                     dcy,
                                     &alpha1,
                                     hx_desc,
                                     dcy,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     hx_shift + ri * hy_n * hy_h,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + bi * 4 * hy_h + ri * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                        else
                        {
                            pretime_shift = li * batch_n * hy_stride + pre_batch * hy_stride;

                            sp_size[1] = in_n[use_time];
                            sp_size[2] = hy_h;
                            sp_desc    = miopen::TensorDescriptor(
                                miopenFloat, sp_size.data(), sp_stride.data(), 3);

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 1;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     workSpace,
                                     &alpha1,
                                     sp_desc,
                                     reserveSpace,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     pretime_shift + bi * 4 * hy_h + ri * hy_h,
                                     pretime_shift + hy_h + ri * 4 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     offset + bi * 4 * hy_h + ri * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }

                        // update forget gate
                        sp_size[1] = in_n[cur_time];
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        if(ti == 0)
                        {
                            sigDesc.Backward(handle,
                                             &alpha,
                                             sp_desc,
                                             reserveSpace,
                                             sp_desc,
                                             workSpace,
                                             sp_desc,
                                             reserveSpace,
                                             &beta,
                                             sp_desc,
                                             workSpace,
                                             offset + hy_h + ri * 4 * hy_h +
                                                 nLayers * batch_n * hy_stride,
                                             offset + bi * 4 * hy_h + ri * hy_h,
                                             offset + hy_h + ri * 4 * hy_h,
                                             offset + hy_h + ri * 4 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);

                            hx_size[1] = in_n[cur_time];
                            hx_size[2] = hy_h;
                            hx_desc    = miopen::TensorDescriptor(
                                miopenFloat, hx_size.data(), hx_stride.data(), 3);

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     workSpace,
                                     &alpha1,
                                     hx_desc,
                                     cx,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     offset + hy_h + ri * 4 * hy_h,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + hy_h + ri * 4 * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                        else
                        {
                            if(in_n[use_time2] > 0)
                            {
                                pretime_shift = li * batch_n * hy_stride + pre_batch2 * hy_stride;

                                sp_size[1] = in_n[use_time2];
                                sp_size[2] = hy_h;
                                sp_desc    = miopen::TensorDescriptor(
                                    miopenFloat, sp_size.data(), sp_stride.data(), 3);

                                alpha0 = 1;
                                alpha1 = 1;
                                beta_t = 0;

                                sigDesc.Backward(handle,
                                                 &alpha,
                                                 sp_desc,
                                                 reserveSpace,
                                                 sp_desc,
                                                 workSpace,
                                                 sp_desc,
                                                 reserveSpace,
                                                 &beta,
                                                 sp_desc,
                                                 workSpace,
                                                 offset + hy_h + ri * 4 * hy_h +
                                                     nLayers * batch_n * hy_stride,
                                                 offset + bi * 4 * hy_h + ri * hy_h,
                                                 offset + hy_h + ri * 4 * hy_h,
                                                 offset + hy_h + ri * 4 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         sp_desc,
                                         workSpace,
                                         &alpha1,
                                         sp_desc,
                                         reserveSpace,
                                         &beta_t,
                                         sp_desc,
                                         workSpace,
                                         offset + hy_h + ri * 4 * hy_h,
                                         pretime_shift + bi * 4 * hy_h + ri * hy_h,
                                         offset + hy_h + ri * 4 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }

                        // update input gate
                        sp_size[1] = in_n[cur_time];
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        sigDesc.Backward(handle,
                                         &alpha,
                                         sp_desc,
                                         reserveSpace,
                                         sp_desc,
                                         workSpace,
                                         sp_desc,
                                         reserveSpace,
                                         &beta,
                                         sp_desc,
                                         workSpace,
                                         offset + ri * 4 * hy_h + nLayers * batch_n * hy_stride,
                                         offset + bi * 4 * hy_h + ri * hy_h,
                                         offset + ri * 4 * hy_h,
                                         offset + ri * 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + ri * 4 * hy_h,
                                 offset + 3 * hy_h + ri * 4 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + ri * 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update output gate
                        sigDesc.Backward(handle,
                                         &alpha,
                                         sp_desc,
                                         reserveSpace,
                                         sp_desc,
                                         workSpace,
                                         sp_desc,
                                         reserveSpace,
                                         &beta,
                                         sp_desc,
                                         workSpace,
                                         offset + 2 * hy_h + ri * 4 * hy_h +
                                             nLayers * batch_n * hy_stride,
                                         offset + bi * 5 * hy_h + ri * hy_h,
                                         offset + 2 * hy_h + ri * 4 * hy_h,
                                         offset + 2 * hy_h + ri * 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + 2 * hy_h + ri * 4 * hy_h,
                                 offset + bi * 4 * hy_h + ri * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 2 * hy_h + ri * 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update c gate
                        tanhDesc.Backward(handle,
                                          &alpha,
                                          sp_desc,
                                          reserveSpace,
                                          sp_desc,
                                          workSpace,
                                          sp_desc,
                                          reserveSpace,
                                          &beta,
                                          sp_desc,
                                          workSpace,
                                          offset + 3 * hy_h + ri * 4 * hy_h +
                                              nLayers * batch_n * hy_stride,
                                          offset + bi * 4 * hy_h + ri * hy_h,
                                          offset + 3 * hy_h + ri * 4 * hy_h,
                                          offset + 3 * hy_h + ri * 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + 3 * hy_h + ri * 4 * hy_h,
                                 offset + ri * 4 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 3 * hy_h + ri * 4 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
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
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + bi * 3 * hy_h + ri * hy_h,
                                 offset + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 2 * hy_h + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 0;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 workSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + bi * 3 * hy_h + ri * hy_h,
                                 offset + bi * 3 * hy_h + ri * hy_h,
                                 offset + 2 * hy_h + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        tanhDesc.Backward(handle,
                                          &alpha,
                                          sp_desc,
                                          reserveSpace,
                                          sp_desc,
                                          workSpace,
                                          sp_desc,
                                          reserveSpace,
                                          &beta,
                                          sp_desc,
                                          workSpace,
                                          offset + 2 * hy_h + ri * 3 * hy_h +
                                              nLayers * batch_n * hy_stride,
                                          offset + 2 * hy_h + ri * 3 * hy_h,
                                          offset + 2 * hy_h + ri * 3 * hy_h,
                                          offset + 2 * hy_h + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // r gate
                        if(ti == 0)
                        {

                            auto gg = ScanGemmGeometryRNN(handle,
                                                          hx,
                                                          w,
                                                          workSpace,
                                                          in_n.at(cur_time),
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
                                                          network_config,
                                                          MIO_RNN_FINDSOL_TIMEOUT);

                            gg.RunGemm(handle,
                                       hx,
                                       w,
                                       workSpace,
                                       hx_shift + ri * hy_n * hy_h,
                                       weitime_shift + 2 * hy_h * uni_stride +
                                           ri * 3 * hy_h * uni_stride,
                                       offset + hy_h + ri * 3 * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                        else
                        {
                            if(in_n[use_time2] > 0)
                            {

                                auto gg = ScanGemmGeometryRNN(handle,
                                                              reserveSpace,
                                                              w,
                                                              workSpace,
                                                              in_n.at(use_time2),
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
                                                              network_config,
                                                              MIO_RNN_FINDSOL_TIMEOUT);

                                gg.RunGemm(handle,
                                           reserveSpace,
                                           w,
                                           workSpace,
                                           hid_shift + pre_batch2 * hy_stride + bi * 3 * hy_h +
                                               ri * hy_h,
                                           weitime_shift + 2 * hy_h * uni_stride +
                                               ri * 3 * hy_h * uni_stride,
                                           offset + hy_h + ri * 3 * hy_h);

                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 workSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 offset + 2 * hy_h + ri * 3 * hy_h,
                                 offset + hy_h + ri * 3 * hy_h,
                                 offset + hy_h + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        sigDesc.Backward(handle,
                                         &alpha,
                                         sp_desc,
                                         reserveSpace,
                                         sp_desc,
                                         workSpace,
                                         sp_desc,
                                         reserveSpace,
                                         &beta,
                                         sp_desc,
                                         workSpace,
                                         offset + hy_h + ri * 3 * hy_h +
                                             nLayers * batch_n * hy_stride,
                                         offset + hy_h + ri * 3 * hy_h,
                                         offset + hy_h + ri * 3 * hy_h,
                                         offset + hy_h + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // z gate
                        alpha0 = 1;
                        alpha1 = -1;
                        beta_t = 0;

                        if(ti == 0)
                        {
                            hx_size[1] = in_n[cur_time];
                            hx_size[2] = hy_h;
                            hx_desc    = miopen::TensorDescriptor(
                                miopenFloat, hx_size.data(), hx_stride.data(), 3);

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     hx_desc,
                                     hx,
                                     &alpha1,
                                     sp_desc,
                                     reserveSpace,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + 2 * hy_h + ri * 3 * hy_h +
                                         nLayers * batch_n * hy_stride,
                                     offset + ri * 3 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);

                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            OpTensor(handle,
                                     miopenTensorOpMul,
                                     &alpha0,
                                     sp_desc,
                                     workSpace,
                                     &alpha1,
                                     sp_desc,
                                     workSpace,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     offset + bi * 3 * hy_h + ri * hy_h,
                                     offset + ri * 3 * hy_h,
                                     offset + ri * 3 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);

                            sigDesc.Backward(handle,
                                             &alpha,
                                             sp_desc,
                                             reserveSpace,
                                             sp_desc,
                                             workSpace,
                                             sp_desc,
                                             reserveSpace,
                                             &beta,
                                             sp_desc,
                                             workSpace,
                                             offset + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                             offset + ri * 3 * hy_h,
                                             offset + ri * 3 * hy_h,
                                             offset + ri * 3 * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                        else
                        {
                            if(in_n[use_time2] > 0)
                            {
                                sp_size[1] = in_n[use_time2];
                                sp_size[2] = hy_h;
                                sp_desc    = miopen::TensorDescriptor(
                                    miopenFloat, sp_size.data(), sp_stride.data(), 3);

                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         sp_desc,
                                         reserveSpace,
                                         &alpha1,
                                         sp_desc,
                                         reserveSpace,
                                         &beta_t,
                                         sp_desc,
                                         workSpace,
                                         hid_shift + pre_batch2 * hy_stride + bi * 3 * hy_h +
                                             ri * hy_h,
                                         offset + 2 * hy_h + ri * 3 * hy_h +
                                             nLayers * batch_n * hy_stride,
                                         offset + ri * 3 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                alpha0 = 1;
                                alpha1 = 1;
                                beta_t = 0;

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         sp_desc,
                                         workSpace,
                                         &alpha1,
                                         sp_desc,
                                         workSpace,
                                         &beta_t,
                                         sp_desc,
                                         workSpace,
                                         offset + bi * 3 * hy_h + ri * hy_h,
                                         offset + ri * 3 * hy_h,
                                         offset + ri * 3 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                sigDesc.Backward(handle,
                                                 &alpha,
                                                 sp_desc,
                                                 reserveSpace,
                                                 sp_desc,
                                                 workSpace,
                                                 sp_desc,
                                                 reserveSpace,
                                                 &beta,
                                                 sp_desc,
                                                 workSpace,
                                                 offset + ri * 3 * hy_h +
                                                     nLayers * batch_n * hy_stride,
                                                 offset + ri * 3 * hy_h,
                                                 offset + ri * 3 * hy_h,
                                                 offset + ri * 3 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
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
            for(int ri = 0; ri < bi; ri++)
            {
                cur_time  = ri == 0 ? 0 : seqLen - 1;
                cur_batch = ri == 0 ? 0 : batch_n - in_n[seqLen - 1];

                if(in_n[cur_time] > 0)
                {
                    pretime_shift = li * batch_n * hy_stride + cur_batch * hy_stride;

                    sp_size[1] = in_n[cur_time];
                    sp_size[2] = hy_h;
                    hx_size[1] = in_n[cur_time];
                    hx_size[2] = hy_h;
                    hx_desc =
                        miopen::TensorDescriptor(miopenFloat, hx_size.data(), hx_stride.data(), 3);
                    sp_desc =
                        miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

                    if(rnnMode == miopenLSTM)
                    {

                        auto gg = ScanGemmGeometryRNN(handle,
                                                      workSpace,
                                                      w,
                                                      dhx,
                                                      in_n.at(cur_time),
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
                                                      network_config,
                                                      MIO_RNN_FINDSOL_TIMEOUT);

                        gg.RunGemm(handle,
                                   workSpace,
                                   w,
                                   dhx,
                                   pretime_shift + ri * 4 * hy_h,
                                   weitime_shift + ri * 4 * hy_h * uni_stride,
                                   hx_shift + ri * hy_n * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 hx_desc,
                                 dcx,
                                 pretime_shift + bi * 4 * hy_h + ri * hy_h,
                                 pretime_shift + hy_h + ri * 4 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 hx_shift + ri * hy_n * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenGRU)
                    {
                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 sp_desc,
                                 reserveSpace,
                                 pretime_shift + 2 * hy_h + ri * 3 * hy_h,
                                 pretime_shift + hy_h + ri * 3 * hy_h +
                                     nLayers * batch_n * hy_stride,
                                 pretime_shift + bi * 3 * hy_h + ri * hy_h +
                                     nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        auto gg = ScanGemmGeometryRNN(handle,
                                                      reserveSpace,
                                                      w,
                                                      dhx,
                                                      in_n.at(cur_time),
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
                                                      network_config,
                                                      MIO_RNN_FINDSOL_TIMEOUT);

                        gg.RunGemm(handle,
                                   reserveSpace,
                                   w,
                                   dhx,
                                   pretime_shift + bi * 3 * hy_h + ri * hy_h +
                                       nLayers * batch_n * hy_stride,
                                   weitime_shift + 2 * hy_h * uni_stride +
                                       ri * 3 * hy_h * uni_stride,
                                   hx_shift + ri * hy_n * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 sp_desc,
                                 reserveSpace,
                                 &beta_t,
                                 hx_desc,
                                 dhx,
                                 pretime_shift + bi * 3 * hy_h + ri * hy_h,
                                 pretime_shift + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                 hx_shift + ri * hy_n * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        auto gg2 = ScanGemmGeometryRNN(handle,
                                                       workSpace,
                                                       w,
                                                       dhx,
                                                       in_n.at(cur_time),
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
                                                       network_config,
                                                       MIO_RNN_FINDSOL_TIMEOUT);

                        gg2.RunGemm(handle,
                                    workSpace,
                                    w,
                                    dhx,
                                    pretime_shift + ri * 3 * hy_h,
                                    weitime_shift + ri * 3 * hy_h * uni_stride,
                                    hx_shift + ri * hy_n * hy_h);

                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                }
            }
        }
    }

    // dinput
    if(inputMode == miopenRNNskip)
    {
        sp_size[1] = batch_n;
        sp_size[2] = hy_h;
        x_size[1]  = batch_n;
        x_size[2]  = hy_h;
        x_desc     = miopen::TensorDescriptor(miopenFloat, x_size.data(), x_stride.data(), 3);
        sp_desc    = miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

        alpha0 = 1;
        alpha1 = 0;
        beta_t = 1;

        for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
        {
            OpTensor(handle,
                     miopenTensorOpAdd,
                     &alpha0,
                     sp_desc,
                     workSpace,
                     &alpha1,
                     sp_desc,
                     workSpace,
                     &beta_t,
                     x_desc,
                     dx,
                     gi * hy_h,
                     gi * hy_h,
                     0);
            // Update time
            profileRNNkernels(handle, (gi == nHiddenTensorsPerLayer * bi - 1) ? 2 : 1, ctime);
        }
    }
    else
    {

        auto gg = ScanGemmGeometryRNN(handle,
                                      workSpace,
                                      w,
                                      dx,
                                      batch_n,
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
                                      network_config,
                                      MIO_RNN_FINDSOL_TIMEOUT);

        gg.RunGemm(handle, workSpace, w, dx, 0, 0, 0);

        // Update time
        profileRNNkernels(handle, 2, ctime);
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
    if(workSpaceSize < GetWorkspaceSize(handle, seqLen, xDesc))
    {
        MIOPEN_THROW("Workspace is required");
    }
    if(reserveSpaceSize < GetReserveSize(handle, seqLen, xDesc))
    {
        MIOPEN_THROW("Reservespace is required");
    }

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
        if(i == 0)
        {
            if(batchval == 0)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Input batch is ZERO!");
            }
        }
        else
        {
            if(batchval > in_n.back())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Incorrect input batch size at time " + std::to_string(i) +
                                 "! Batch size must not ascend!");
            }
        }
        in_n.push_back(batchval);
        batch_n += xDesc[i].GetLengths()[0];
    }

    int bi = dirMode ? 2 : 1;
    if(out_h != (bi * hy_h))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Output size doesn't match hidden state size!");
    }

    float ctime    = 0.;
    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * workspaceScale;
    int wei_stride = hy_h * bi * nHiddenTensorsPerLayer;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    if(inputMode == miopenRNNskip)
    {
        if(in_h != hy_h)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "The input tensor size must equal to the hidden "
                         "state size of the network in SKIP_INPUT mode!");
        }
        in_h = 0;
    }

    size_t wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (nLayers - 1)) * wei_stride;

    float alpha0, alpha1, beta_t;

    std::vector<int> sp_size(3, 1), sp_stride(3, 1), w_size(3, 1), w_stride(3, 1);
    miopen::TensorDescriptor sp_desc, w_desc;

    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = hy_stride;
    w_stride[0]  = wei_stride;
    w_stride[1]  = wei_stride;

#if MIOPEN_USE_MIOPENGEMM

    int wei_len   = 0;
    int hid_off   = 0;
    int use_time  = 0;
    int pre_batch = 0;
    int time_mark = 0;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        // printf("run rnn gpu bwd weights \n");
        wei_len = hy_h;
        hid_off = nLayers * batch_n * hy_stride;
        break;
    case miopenLSTM:
        // printf("run lstm gpu bwd weights \n");
        wei_len = hy_h * 4;
        hid_off = bi * hy_h * 5;
        break;
    case miopenGRU:
        // printf("run gru gpu bwd weights \n");
        wei_len = hy_h * 3;
        hid_off = bi * hy_h * 3;
        break;
    }

    for(int li = 0; li < nLayers; li++)
    {
        int hid_shift = li * batch_n * hy_stride;
        int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

        // between layers
        if(li == 0)
        {
            if(inputMode == miopenRNNlinear)
            {

                auto gg = ScanGemmGeometryRNN(handle,
                                              workSpace,
                                              x,
                                              dw,
                                              wei_len * bi,
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
                                              network_config,
                                              MIO_RNN_FINDSOL_TIMEOUT);

                gg.RunGemm(handle, workSpace, x, dw, 0, 0, 0);

                // Update time
                profileRNNkernels(handle, std::min(time_mark++, 1), ctime);
            }
        }
        else
        {
            int prelayer_shift = (li - 1) * batch_n * hy_stride + hid_off;

            auto gg = ScanGemmGeometryRNN(handle,
                                          workSpace,
                                          reserveSpace,
                                          dw,
                                          wei_len * bi,
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
                                          network_config,
                                          MIO_RNN_FINDSOL_TIMEOUT);

            gg.RunGemm(handle, workSpace, reserveSpace, dw, hid_shift, prelayer_shift, wei_shift);

            // Update time
            profileRNNkernels(handle, std::min(time_mark++, 1), ctime);
        }

        if(biasMode)
        {
            if(li == 0 && inputMode == miopenRNNskip && rnnMode == miopenGRU)
                ;
            else
            {
                if(li == 0)
                {
                    wei_shift = wei_shift_bias;
                }
                else
                {
                    wei_shift = (inputMode == miopenRNNskip)
                                    ? (wei_shift_bias + wei_stride + (li - 1) * 2 * wei_stride)
                                    : (wei_shift_bias + li * 2 * wei_stride);
                }

                sp_size[1] = 1;
                sp_size[2] = wei_stride;
                w_size[1]  = 1;
                w_size[2]  = wei_stride;
                w_desc = miopen::TensorDescriptor(miopenFloat, w_size.data(), w_stride.data(), 3);
                sp_desc =
                    miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

                alpha0 = 1;
                alpha1 = 0;
                beta_t = 1;

                for(int bs = 0; bs < batch_n; bs++)
                {
                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             sp_desc,
                             workSpace,
                             &alpha1,
                             sp_desc,
                             workSpace,
                             &beta_t,
                             w_desc,
                             dw,
                             hid_shift + bs * hy_stride,
                             hid_shift + bs * hy_stride,
                             wei_shift);

                    // Update time
                    profileRNNkernels(handle, std::min(time_mark++, 1), ctime);
                }

                if(rnnMode != miopenGRU && (!(li == 0 && inputMode == miopenRNNskip)))
                {
                    CopyTensor(handle, w_desc, dw, w_desc, dw, wei_shift, wei_shift + wei_stride);
                    // Update time
                    profileRNNkernels(handle, std::min(time_mark++, 1), ctime);
                }
            }
        }

        // between time
        int bacc   = 0;
        int baccbi = batch_n;
        for(int ti = 0; ti < seqLen; ti++)
        {
            baccbi -= in_n[seqLen - 1 - ti];

            int hx_shift = li * hy_n * bi_stride;
            wei_shift    = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            for(int ri = 0; ri < bi; ri++)
            {
                hid_shift = ri == 0 ? (li * batch_n * hy_stride + bacc * hy_stride)
                                    : (li * batch_n * hy_stride + baccbi * hy_stride);
                int cur_time = ri == 0 ? ti : seqLen - 1 - ti;
                if(ti > 0)
                {
                    pre_batch = ri == 0 ? bacc - in_n[ti - 1] : baccbi + in_n[seqLen - 1 - ti];
                    use_time  = ri == 0 ? ti : seqLen - ti;
                }

                if(in_n[cur_time] > 0)
                {
                    if(rnnMode == miopenGRU)
                    {
                        if(ri == 0)
                        {
                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;
                        }

                        sp_size[1] = in_n[cur_time];
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            miopenFloat, sp_size.data(), sp_stride.data(), 3);

                        OpTensor(handle,
                                 miopenTensorOpMul,
                                 &alpha0,
                                 sp_desc,
                                 reserveSpace,
                                 &alpha1,
                                 sp_desc,
                                 workSpace,
                                 &beta_t,
                                 sp_desc,
                                 workSpace,
                                 hid_shift + hy_h + ri * 3 * hy_h + nLayers * batch_n * hy_stride,
                                 hid_shift + 2 * hy_h + ri * 3 * hy_h,
                                 hid_shift + 2 * hy_h + ri * 3 * hy_h);
                        // Update time
                        profileRNNkernels(handle, std::min(time_mark++, 1), ctime);
                    }

                    if(ti == 0)
                    {

                        auto gg = ScanGemmGeometryRNN(handle,
                                                      workSpace,
                                                      hx,
                                                      dw,
                                                      wei_len,
                                                      hy_h,
                                                      in_n.at(cur_time),
                                                      1,
                                                      1,
                                                      true,
                                                      false,
                                                      false,
                                                      hy_stride,
                                                      uni_stride,
                                                      uni_stride,
                                                      false,
                                                      network_config,
                                                      MIO_RNN_FINDSOL_TIMEOUT);

                        gg.RunGemm(handle,
                                   workSpace,
                                   hx,
                                   dw,
                                   hid_shift + ri * wei_len,
                                   hx_shift + ri * hy_n * hy_h,
                                   wei_shift + ri * wei_len * uni_stride);

                        // Update time
                        if(li == nLayers - 1 && ti == seqLen - 1 && ri == bi - 1 &&
                           !(rnnMode == miopenGRU && biasMode))
                            profileRNNkernels(handle, 2, ctime);
                        else
                            profileRNNkernels(handle, std::min(time_mark++, 1), ctime);
                    }
                    else
                    {
                        int pretime_shift =
                            li * batch_n * hy_stride + pre_batch * hy_stride + hid_off;

                        if(in_n[use_time] > 0)
                        {

                            auto gg = ScanGemmGeometryRNN(handle,
                                                          workSpace,
                                                          reserveSpace,
                                                          dw,
                                                          wei_len,
                                                          hy_h,
                                                          in_n.at(use_time),
                                                          1,
                                                          1,
                                                          true,
                                                          false,
                                                          false,
                                                          hy_stride,
                                                          hy_stride,
                                                          uni_stride,
                                                          false,
                                                          network_config,
                                                          MIO_RNN_FINDSOL_TIMEOUT);

                            gg.RunGemm(handle,
                                       workSpace,
                                       reserveSpace,
                                       dw,
                                       hid_shift + ri * wei_len,
                                       pretime_shift + ri * hy_h,
                                       wei_shift + ri * wei_len * uni_stride);

                            // Update time
                            if(li == nLayers - 1 && ti == seqLen - 1 && ri == bi - 1 &&
                               !(rnnMode == miopenGRU && biasMode))
                                profileRNNkernels(handle, 2, ctime);
                            else
                                profileRNNkernels(handle, 1, ctime);
                        }
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

            sp_size[1] = 1;
            sp_size[2] = wei_stride;
            w_size[1]  = 1;
            w_size[2]  = wei_stride;
            w_desc     = miopen::TensorDescriptor(miopenFloat, w_size.data(), w_stride.data(), 3);
            sp_desc    = miopen::TensorDescriptor(miopenFloat, sp_size.data(), sp_stride.data(), 3);

            alpha0 = 1;
            alpha1 = 0;
            beta_t = 1;

            for(int bs = 0; bs < batch_n; bs++)
            {
                OpTensor(handle,
                         miopenTensorOpAdd,
                         &alpha0,
                         sp_desc,
                         workSpace,
                         &alpha1,
                         sp_desc,
                         workSpace,
                         &beta_t,
                         w_desc,
                         dw,
                         hid_shift + bs * hy_stride,
                         hid_shift + bs * hy_stride,
                         wei_shift);

                // Update time
                if(li == nLayers - 1 && bs == batch_n - 1)
                    profileRNNkernels(handle, 2, ctime);
                else
                    profileRNNkernels(handle, 1, ctime);
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

} // namespace miopen
