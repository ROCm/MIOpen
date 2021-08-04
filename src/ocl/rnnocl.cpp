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
#include <miopen/rnn_util.hpp>

#include <miopen/activ.hpp>
#include <miopen/env.hpp>
#include <miopen/gemm_v2.hpp>
#include <miopen/logger.hpp>

#include <vector>
#include <numeric>
#include <algorithm>

namespace miopen {

// Assuming sequence length is set to > 0 otherwise throw exception.
void RNNDescriptor::RNNForwardInference(Handle& handle,
                                        const int seqLen,
                                        c_array_view<const miopenTensorDescriptor_t> xDesc,
                                        ConstData_t x,
                                        const TensorDescriptor& hxDesc,
                                        ConstData_t hx,
                                        const TensorDescriptor& cxDesc,
                                        ConstData_t cx,
                                        const TensorDescriptor& wDesc,
                                        ConstData_t w,
                                        c_array_view<const miopenTensorDescriptor_t> yDesc,
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

    if(in_h <= 0 || hy_h <= 0 || hy_n <= 0 || hy_d <= 0 || out_h <= 0 || seqLen <= 0)
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
            if(batchval <= 0)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Input batch is ZERO!");
            }
        }
        else
        {
            if(batchval > in_n.back() || batchval < 0)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Incorrect input batch size at time " + std::to_string(i) +
                                 "! Batch size must not ascend!");
            }
        }
        in_n.push_back(batchval);
        batch_n += batchval;
    }

    int bi = dirMode != 0u ? 2 : 1;
    if(out_h != (bi * hy_h))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Output size doesn't match hidden state size!");
    }

    float ctime    = 0.;
    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * static_cast<int>(workspaceScale);
    int out_stride = out_h;
    int wei_stride = hy_h * bi * static_cast<int>(nHiddenTensorsPerLayer);
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

    sp_size[2]   = workSpaceSize / GetTypeSize(wDesc.GetType());
    sp_stride[0] = sp_size[2];
    sp_stride[1] = sp_size[2];
    sp_desc      = miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
    SetTensor(handle, sp_desc, workSpace, &beta);
    // Update time
    profileRNNkernels(handle, 0, ctime);
    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = hy_stride;
    sp_size[2]   = 1;
    w_stride[0]  = wei_stride;
    w_stride[1]  = wei_stride;
    x_stride[0]  = batch_n * in_stride;
    x_stride[1]  = in_stride;
    y_stride[0]  = batch_n * out_stride;
    y_stride[1]  = out_stride;
    if(hy != nullptr || (rnnMode == miopenLSTM && cy != nullptr))
    {
        hx_size[2]   = hy_d * hy_n * hy_h;
        hx_stride[0] = hx_size[2];
        hx_stride[1] = hx_size[2];
        hx_desc = miopen::TensorDescriptor(wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);
        if(hy != nullptr)
        {
            SetTensor(handle, hx_desc, hy, &beta);
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }
        if(rnnMode == miopenLSTM && cy != nullptr)
        {
            SetTensor(handle, hx_desc, cy, &beta);
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }
    }
    hx_stride[0] = in_n.at(0) * uni_stride;
    hx_stride[1] = uni_stride;

#if MIOPEN_USE_GEMM

    int wei_shift, prelayer_shift;
    int wei_len = 0;
    int hid_off = 0;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        // printf("run rnn gpu inference \n");
        wei_len = hy_h;
        hid_off = 0;
        break;
    case miopenLSTM:
        // printf("run lstm gpu inference \n");
        wei_len = hy_h * 4;
        hid_off = bi * hy_h * 5;
        break;
    case miopenGRU:
        // printf("run gru gpu inference \n");
        wei_len = hy_h * 3;
        hid_off = bi * hy_h * 3;
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
        int wei_shift_bias_temp = static_cast<int>(wei_shift_bias) + li * 2 * wei_stride;

        // from input
        if(li == 0)
        {
            if(inputMode == miopenRNNskip)
            {
                x_size[1]  = batch_n;
                x_size[2]  = hy_h;
                sp_size[1] = batch_n;
                sp_size[2] = hy_h;
                x_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), x_size.data(), x_stride.data(), 3);
                sp_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
                {
                    CopyTensor(handle, x_desc, x, sp_desc, workSpace, 0, gi * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1, ctime);
                }
            }
            else
            {
                miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                                  false,
                                                                  true,
                                                                  batch_n,
                                                                  wei_len * bi,
                                                                  in_h,
                                                                  in_stride,
                                                                  in_stride,
                                                                  hy_stride,
                                                                  1, // batch count
                                                                  0, // Stride A
                                                                  0, // Stride B
                                                                  0, // Stride C
                                                                  1, // alpha
                                                                  1, // beta
                                                                  xDesc[0].GetType()};

                miopenStatus_t gemm_status = CallGemm(handle,
                                                      gemm_desc,
                                                      x,
                                                      0,
                                                      w,
                                                      0,
                                                      workSpace,
                                                      hid_shift,
                                                      nullptr,
                                                      GemmBackend_t::miopengemm);

                if(gemm_status != miopenStatusSuccess)
                {
                    if(gemm_status == miopenStatusNotImplemented)
                    {
                        MIOPEN_LOG_E("GEMM not implemented");
                    }
                    else
                    {
                        MIOPEN_LOG_E("GEMM failed");
                    }
                }
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
        }
        else
        {
            wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            prelayer_shift = (li - 1) * batch_n * hy_stride + hid_off;

            miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                              false,
                                                              true,
                                                              batch_n,
                                                              wei_len * bi,
                                                              hy_h * bi,
                                                              hy_stride,
                                                              bi_stride,
                                                              hy_stride,
                                                              1, // batch count
                                                              0, // Stride A
                                                              0, // Stride B
                                                              0, // Stride C
                                                              1, // alpha
                                                              1, // beta
                                                              xDesc[0].GetType()};
            miopenStatus_t gemm_status       = CallGemm(handle,
                                                  gemm_desc,
                                                  workSpace,
                                                  prelayer_shift,
                                                  w,
                                                  wei_shift,
                                                  workSpace,
                                                  hid_shift,
                                                  nullptr,
                                                  GemmBackend_t::miopengemm);

            if(gemm_status != miopenStatusSuccess)
            {
                if(gemm_status == miopenStatusNotImplemented)
                {
                    MIOPEN_LOG_E("GEMM not implemented");
                }
                else
                {
                    MIOPEN_LOG_E("GEMM failed");
                }
            }
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }

        if(biasMode != 0u)
        {
            alpha0 = 1;
            alpha1 = 1;
            beta_t = 0;

            w_size[1]  = 1;
            w_size[2]  = wei_stride;
            sp_size[1] = batch_n;
            sp_size[2] = wei_stride;
            w_desc = miopen::TensorDescriptor(wDesc.GetType(), w_size.data(), w_stride.data(), 3);
            sp_desc =
                miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                     wei_shift_bias_temp,
                     hid_shift);
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }

        if(rnnMode == miopenGRU)
        {
            sp_size[1] = batch_n;
            sp_size[2] = hy_h;
            sp_desc =
                miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

            alpha0 = 0;
            alpha1 = 0;
            beta_t = 0;
            for(int bs = 0; bs < bi; bs++)
            {
                CopyTensor(handle,
                           sp_desc,
                           workSpace,
                           sp_desc,
                           workSpace,
                           hid_shift + bs * wei_len + 2 * hy_h,
                           hid_shift + hid_off + bs * hy_h);
                // Update time
                profileRNNkernels(handle, 1, ctime);

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
                         hid_shift + bs * wei_len + 2 * hy_h,
                         hid_shift + bs * wei_len + 2 * hy_h,
                         hid_shift + bs * wei_len + 2 * hy_h);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
        }

        if(biasMode != 0u)
        {
            wei_shift_bias_temp += wei_stride;

            alpha0 = 1;
            alpha1 = 1;
            beta_t = 0;

            if(hx != nullptr)
            {
                sp_size[1] = batch_n;
                sp_size[2] = wei_stride;
                sp_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                         wei_shift_bias_temp,
                         hid_shift);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
            else
            {
                sp_size[1] = batch_n - in_n.at(0);
                sp_size[2] = wei_len;
                sp_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                w_size[1] = 1;
                w_size[2] = wei_len;
                w_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), w_size.data(), w_stride.data(), 3);

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
                         hid_shift + in_n.at(0) * hy_stride,
                         wei_shift_bias_temp,
                         hid_shift + in_n.at(0) * hy_stride);
                // Update time
                profileRNNkernels(handle, 1, ctime);

                if(dirMode != 0u)
                {
                    if(in_n.at(0) == in_n.at(seqLen - 1))
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
                                 hid_shift + wei_len,
                                 wei_shift_bias_temp + wei_len,
                                 hid_shift + wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else
                    {
                        int cur_batch = 0;
                        for(int ti = 0; ti < seqLen; ti++)
                        {
                            if(ti != (seqLen - 1))
                            {
                                offset = hid_shift + cur_batch * hy_stride;

                                sp_size[1] = in_n.at(ti + 1);
                                sp_size[2] = wei_len;
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                         offset + wei_len,
                                         wei_shift_bias_temp + wei_len,
                                         offset + wei_len);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                            cur_batch += in_n.at(ti);
                        }
                    }
                }
            }
        }

        // from hidden state
        int bacc   = 0;
        int baccbi = batch_n;
        for(int ti = 0; ti < seqLen; ti++)
        {
            baccbi -= in_n.at(seqLen - 1 - ti);
            wei_shift         = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift = 0;
            int use_time      = 0;

            for(int ri = 0; ri < bi; ri++)
            {
                int cur_time  = ri == 0 ? ti : seqLen - 1 - ti;
                int cur_batch = ri == 0 ? bacc : baccbi;
                offset        = hid_shift + cur_batch * hy_stride;
                if(ti > 0)
                {
                    pretime_shift =
                        ri == 0 ? hid_shift + (bacc - in_n.at(ti - 1)) * hy_stride
                                : hid_shift + (baccbi + in_n.at(seqLen - 1 - ti)) * hy_stride;
                    use_time = ri == 0 ? ti : seqLen - ti;
                }

                if(in_n.at(cur_time) > 0)
                {
                    if(ti == 0)
                    {
                        if(hx != nullptr)
                        {
                            miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                                              false,
                                                                              true,
                                                                              in_n.at(cur_time),
                                                                              wei_len,
                                                                              hy_h,
                                                                              uni_stride,
                                                                              uni_stride,
                                                                              hy_stride,
                                                                              1, // batch count
                                                                              0, // Stride A
                                                                              0, // Stride B
                                                                              0, // Stride C
                                                                              1, // alpha
                                                                              1, // beta
                                                                              xDesc[0].GetType()};

                            miopenStatus_t gemm_status =
                                CallGemm(handle,
                                         gemm_desc,
                                         hx,
                                         hx_shift + ri * hy_n * hy_h,
                                         w,
                                         wei_shift + ri * wei_len * uni_stride,
                                         workSpace,
                                         static_cast<int>(offset) + ri * wei_len,
                                         nullptr,
                                         GemmBackend_t::miopengemm);

                            if(gemm_status != miopenStatusSuccess)
                            {
                                if(gemm_status == miopenStatusNotImplemented)
                                {
                                    MIOPEN_LOG_E("GEMM not implemented");
                                }
                                else
                                {
                                    MIOPEN_LOG_E("GEMM failed");
                                }
                            }
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }
                    else
                    {
                        if(ri == 1 && hx != nullptr && in_n.at(cur_time) > in_n.at(use_time))
                        {
                            miopen::GemmDescriptor gemm_desc =
                                GemmDescriptor{false,
                                               false,
                                               true,
                                               (in_n.at(cur_time) - in_n.at(use_time)),
                                               wei_len,
                                               hy_h,
                                               uni_stride,
                                               uni_stride,
                                               hy_stride,
                                               1, // batch count
                                               0, // Stride A
                                               0, // Stride B
                                               0, // Stride C
                                               1, // alpha
                                               1, // beta
                                               xDesc[0].GetType()};
                            miopenStatus_t gemm_status =
                                CallGemm(handle,
                                         gemm_desc,
                                         hx,
                                         hx_shift + ri * hy_n * hy_h + in_n.at(use_time) * hy_h,
                                         w,
                                         wei_shift + ri * wei_len * uni_stride,
                                         workSpace,
                                         static_cast<int>(offset) + ri * wei_len +
                                             in_n.at(use_time) * hy_stride,
                                         nullptr,
                                         GemmBackend_t::miopengemm);

                            if(gemm_status != miopenStatusSuccess)
                            {
                                if(gemm_status == miopenStatusNotImplemented)
                                {
                                    MIOPEN_LOG_E("GEMM not implemented");
                                }
                                else
                                {
                                    MIOPEN_LOG_E("GEMM failed");
                                }
                            }
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }

                        if(in_n.at(use_time) > 0)
                        {
                            miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                                              false,
                                                                              true,
                                                                              in_n.at(use_time),
                                                                              wei_len,
                                                                              hy_h,
                                                                              hy_stride,
                                                                              uni_stride,
                                                                              hy_stride,
                                                                              1, // batch count
                                                                              0, // Stride A
                                                                              0, // Stride B
                                                                              0, // Stride C
                                                                              1, // alpha
                                                                              1, // beta
                                                                              xDesc[0].GetType()};

                            miopenStatus_t gemm_status =
                                CallGemm(handle,
                                         gemm_desc,
                                         workSpace,
                                         pretime_shift + hid_off + ri * hy_h,
                                         w,
                                         wei_shift + ri * wei_len * uni_stride,
                                         workSpace,
                                         static_cast<int>(offset) + ri * wei_len,
                                         nullptr,
                                         GemmBackend_t::miopengemm);

                            if(gemm_status != miopenStatusSuccess)
                            {
                                if(gemm_status == miopenStatusNotImplemented)
                                {
                                    MIOPEN_LOG_E("GEMM not implemented");
                                }
                                else
                                {
                                    MIOPEN_LOG_E("GEMM failed");
                                }
                            }
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }

                    // update hidden status
                    sp_size[1] = in_n.at(cur_time);
                    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                    {
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        activDesc.Forward(handle,
                                          &alpha,
                                          sp_desc,
                                          workSpace,
                                          &beta,
                                          sp_desc,
                                          workSpace,
                                          offset + ri * wei_len,
                                          offset + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenLSTM)
                    {
                        if(algoMode == miopenRNNdefault)
                        {
                            LSTMForwardHiddenStateUpdate(handle,
                                                         wDesc.GetType(),
                                                         true,
                                                         ti == 0,
                                                         ri,
                                                         in_n.at(0),
                                                         in_n.at(cur_time),
                                                         in_n.at(use_time),
                                                         hy_h,
                                                         hy_stride,
                                                         wei_len,
                                                         wei_stride,
                                                         cx,
                                                         hx_shift + ri * hy_n * hy_h,
                                                         workSpace,
                                                         offset + ri * wei_len,
                                                         offset + hy_h + ri * wei_len,
                                                         offset + 2 * hy_h + ri * wei_len,
                                                         offset + 3 * hy_h + ri * wei_len,
                                                         offset + bi * wei_len + ri * hy_h,
                                                         pretime_shift + bi * wei_len + ri * hy_h,
                                                         0,
                                                         offset + hid_off + ri * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                            continue;
                        }

                        // active gate i, f, o
                        sp_size[2] = hy_h * 3;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        sigDesc.Forward(handle,
                                        &alpha,
                                        sp_desc,
                                        workSpace,
                                        &beta,
                                        sp_desc,
                                        workSpace,
                                        offset + ri * wei_len,
                                        offset + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // active gate c
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         workSpace,
                                         &beta,
                                         sp_desc,
                                         workSpace,
                                         offset + 3 * hy_h + ri * wei_len,
                                         offset + 3 * hy_h + ri * wei_len);
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
                                 offset + ri * wei_len,
                                 offset + 3 * hy_h + ri * wei_len,
                                 offset + bi * wei_len + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(ti == 0)
                        {
                            if(cx != nullptr)
                            {
                                hx_size[1] = in_n.at(cur_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

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
                                         offset + hy_h + ri * wei_len,
                                         hx_shift + ri * hy_n * hy_h,
                                         offset + bi * wei_len + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }
                        else
                        {
                            if(ri == 1 && cx != nullptr && in_n.at(cur_time) > in_n.at(use_time))
                            {
                                hx_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

                                sp_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                         offset + hy_h + ri * wei_len +
                                             in_n.at(use_time) * hy_stride,
                                         hx_shift + ri * hy_n * hy_h + in_n.at(use_time) * hy_h,
                                         offset + bi * wei_len + ri * hy_h +
                                             in_n.at(use_time) * hy_stride);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                sp_size[1] = in_n.at(cur_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                            }

                            if(in_n.at(use_time) > 0)
                            {
                                if(in_n.at(use_time) != in_n.at(cur_time))
                                {
                                    sp_size[1] = in_n.at(use_time);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }

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
                                         offset + hy_h + ri * wei_len,
                                         pretime_shift + bi * wei_len + ri * hy_h,
                                         offset + bi * wei_len + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                if(in_n.at(use_time) != in_n.at(cur_time))
                                {
                                    sp_size[1] = in_n.at(cur_time);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }
                            }
                        }

                        // active cell state
                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         workSpace,
                                         &beta,
                                         sp_desc,
                                         workSpace,
                                         offset + bi * wei_len + ri * hy_h,
                                         offset + hid_off + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update hidden state
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
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + hid_off + ri * hy_h,
                                 offset + hid_off + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenGRU)
                    {
                        // active z, r gate
                        sp_size[2] = 2 * hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        sigDesc.Forward(handle,
                                        &alpha,
                                        sp_desc,
                                        workSpace,
                                        &beta,
                                        sp_desc,
                                        workSpace,
                                        offset + ri * wei_len,
                                        offset + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // calculate c gate
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                 offset + hy_h + ri * wei_len,
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + 2 * hy_h + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

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
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + hid_off + ri * hy_h,
                                 offset + 2 * hy_h + ri * wei_len);
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
                                         offset + 2 * hy_h + ri * wei_len,
                                         offset + 2 * hy_h + ri * wei_len);
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
                                 offset + ri * wei_len,
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + hid_off + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

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
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + hid_off + ri * hy_h,
                                 offset + hid_off + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;
                        if(ti == 0)
                        {
                            if(hx != nullptr)
                            {
                                hx_size[1] = in_n.at(cur_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

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
                                         offset + ri * wei_len,
                                         hx_shift + ri * hy_n * hy_h,
                                         offset + hid_off + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }
                        else
                        {
                            if(ri == 1 && hx != nullptr && in_n.at(cur_time) > in_n.at(use_time))
                            {
                                hx_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

                                sp_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                         offset + ri * wei_len + in_n.at(use_time) * hy_stride,
                                         hx_shift + ri * hy_n * hy_h + in_n.at(use_time) * hy_h,
                                         offset + hid_off + ri * hy_h +
                                             in_n.at(use_time) * hy_stride);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                sp_size[1] = in_n.at(cur_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                            }

                            if(in_n.at(use_time) > 0)
                            {
                                if(in_n.at(use_time) != in_n.at(cur_time))
                                {
                                    sp_size[1] = in_n.at(use_time);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }

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
                                         offset + ri * wei_len,
                                         pretime_shift + hid_off + ri * hy_h,
                                         offset + hid_off + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }
                    }
                }
            }

            bacc += in_n.at(ti);
        }

        // update hy, cy
        if(hy != nullptr || (rnnMode == miopenLSTM && cy != nullptr))
        {
            hx_size[2] = hy_h;
            sp_size[2] = hy_h;

            bacc   = batch_n;
            baccbi = 0;
            for(int ti = seqLen - 1; ti >= 0; ti--)
            {
                bacc -= in_n.at(ti);
                for(int ri = 0; ri < bi; ri++)
                {
                    int cur_time  = ri == 0 ? ti : seqLen - 1 - ti;
                    int cur_batch = ri == 0 ? bacc : baccbi;
                    int use_batch = 0;

                    if(ti < seqLen - 1)
                    {
                        int use_time = ri == 0 ? ti + 1 : seqLen - 2 - ti;
                        use_batch    = in_n.at(use_time);
                    }

                    if(in_n.at(cur_time) > use_batch)
                    {
                        offset = hid_shift + cur_batch * hy_stride;

                        sp_size[1] = in_n.at(cur_time) - use_batch;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        hx_size[1] = sp_size[1];
                        hx_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

                        if(hy != nullptr)
                        {
                            CopyTensor(handle,
                                       sp_desc,
                                       workSpace,
                                       hx_desc,
                                       hy,
                                       static_cast<int>(offset) + hid_off + ri * hy_h +
                                           use_batch * hy_stride,
                                       hx_shift + ri * hy_n * hy_h + use_batch * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }

                        if(rnnMode == miopenLSTM && cy != nullptr)
                        {
                            CopyTensor(handle,
                                       sp_desc,
                                       workSpace,
                                       hx_desc,
                                       cy,
                                       static_cast<int>(offset) + bi * wei_len + ri * hy_h +
                                           use_batch * hy_stride,
                                       hx_shift + ri * hy_n * hy_h + use_batch * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }
                }
                baccbi += in_n.at(seqLen - 1 - ti);
            }
        }
    }

    // output
    prelayer_shift = (static_cast<int>(nLayers) - 1) * batch_n * hy_stride + hid_off;

    sp_size[1] = batch_n;
    sp_size[2] = hy_h * bi;
    y_size[1]  = batch_n;
    y_size[2]  = out_h;
    y_desc     = miopen::TensorDescriptor(wDesc.GetType(), y_size.data(), y_stride.data(), 3);
    sp_desc    = miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

    CopyTensor(handle, sp_desc, workSpace, y_desc, y, prelayer_shift, 0);
    // Update time
    profileRNNkernels(handle, 2, ctime);

#else
    (void)hx;
    (void)cx;
    (void)offset;
    (void)alpha0;
    (void)alpha1;
    (void)beta_t;
    (void)alpha;
    (void)bi_stride;
    (void)wei_shift_bias;
    MIOPEN_THROW("GEMM is not supported");
#endif
}

void RNNDescriptor::RNNForwardTraining(Handle& handle,
                                       const int seqLen,
                                       c_array_view<const miopenTensorDescriptor_t> xDesc,
                                       ConstData_t x,
                                       const TensorDescriptor& hxDesc,
                                       ConstData_t hx,
                                       const TensorDescriptor& cxDesc,
                                       ConstData_t cx,
                                       const TensorDescriptor& wDesc,
                                       ConstData_t w,
                                       c_array_view<const miopenTensorDescriptor_t> yDesc,
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
    (void)workSpace;

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

    if(in_h <= 0 || hy_h <= 0 || hy_n <= 0 || hy_d <= 0 || out_h <= 0 || seqLen <= 0)
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
            if(batchval <= 0)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Input batch is ZERO!");
            }
        }
        else
        {
            if(batchval > in_n.back() || batchval < 0)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Incorrect input batch size at time " + std::to_string(i) +
                                 "! Batch size must not ascend!");
            }
        }
        in_n.push_back(batchval);
        batch_n += batchval;
    }

    int bi = dirMode != 0u ? 2 : 1;
    if(out_h != (bi * hy_h))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Output size doesn't match hidden state size!");
    }

    float ctime    = 0.;
    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * static_cast<int>(workspaceScale);
    int out_stride = out_h;
    int wei_stride = hy_h * bi * static_cast<int>(nHiddenTensorsPerLayer);
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

    sp_size[2]   = reserveSpaceSize / GetTypeSize(wDesc.GetType());
    sp_stride[0] = sp_size[2];
    sp_stride[1] = sp_size[2];
    sp_desc      = miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
    SetTensor(handle, sp_desc, reserveSpace, &beta);
    // Update time
    profileRNNkernels(handle, 0, ctime);
    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = hy_stride;
    sp_size[2]   = 1;
    w_stride[0]  = wei_stride;
    w_stride[1]  = wei_stride;
    x_stride[0]  = batch_n * in_stride;
    x_stride[1]  = in_stride;
    y_stride[0]  = batch_n * out_stride;
    y_stride[1]  = out_stride;
    if(hy != nullptr || (rnnMode == miopenLSTM && cy != nullptr))
    {
        hx_size[2]   = hy_d * hy_n * hy_h;
        hx_stride[0] = hx_size[2];
        hx_stride[1] = hx_size[2];
        hx_desc = miopen::TensorDescriptor(wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);
        if(hy != nullptr)
        {
            SetTensor(handle, hx_desc, hy, &beta);
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }
        if(rnnMode == miopenLSTM && cy != nullptr)
        {
            SetTensor(handle, hx_desc, cy, &beta);
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }
    }
    hx_stride[0] = in_n.at(0) * uni_stride;
    hx_stride[1] = uni_stride;

#if MIOPEN_USE_GEMM

    int wei_shift, prelayer_shift;
    int wei_len = 0;
    int hid_off = 0;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        // printf("run rnn gpu fwd \n");
        wei_len = hy_h;
        hid_off = static_cast<int>(nLayers) * batch_n * hy_stride;
        break;
    case miopenLSTM:
        // printf("run lstm gpu fwd \n");
        wei_len = hy_h * 4;
        hid_off = bi * hy_h * 5;
        break;
    case miopenGRU:
        // printf("run gru gpu fwd \n");
        wei_len = hy_h * 3;
        hid_off = bi * hy_h * 3;
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
        int wei_shift_bias_temp = static_cast<int>(wei_shift_bias) + li * 2 * wei_stride;

        // from input
        if(li == 0)
        {
            if(inputMode == miopenRNNskip)
            {
                x_size[1]  = batch_n;
                x_size[2]  = hy_h;
                sp_size[1] = batch_n;
                sp_size[2] = hy_h;
                x_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), x_size.data(), x_stride.data(), 3);
                sp_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
                {
                    CopyTensor(handle, x_desc, x, sp_desc, reserveSpace, 0, gi * hy_h);
                    // Update time
                    profileRNNkernels(handle, 1, ctime);
                }
            }
            else
            {
                miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                                  false,
                                                                  true,
                                                                  batch_n,
                                                                  wei_len * bi,
                                                                  in_h,
                                                                  in_stride,
                                                                  in_stride,
                                                                  hy_stride,
                                                                  1, // batch count
                                                                  0, // Stride A
                                                                  0, // Stride B
                                                                  0, // Stride C
                                                                  1, // alpha
                                                                  1, // beta
                                                                  xDesc[0].GetType()};

                miopenStatus_t gemm_status = CallGemm(handle,
                                                      gemm_desc,
                                                      x,
                                                      0,
                                                      w,
                                                      0,
                                                      reserveSpace,
                                                      hid_shift,
                                                      nullptr,
                                                      GemmBackend_t::miopengemm);

                if(gemm_status != miopenStatusSuccess)
                {
                    if(gemm_status == miopenStatusNotImplemented)
                    {
                        MIOPEN_LOG_E("GEMM not implemented");
                    }
                    else
                    {
                        MIOPEN_LOG_E("GEMM failed");
                    }
                }
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
        }
        else
        {
            wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            prelayer_shift = (li - 1) * batch_n * hy_stride + hid_off;

            bool use_dropout = !float_equal(miopen::deref(dropoutDesc).dropout, 0);
            if(use_dropout)
            {
                std::vector<int> drop_size(2), drop_in_str(2, 1), drop_out_str(2, 1);
                drop_size[0]    = batch_n;
                drop_size[1]    = hy_h * bi;
                drop_in_str[0]  = hy_stride;
                drop_out_str[0] = hy_h * bi;

                auto drop_in_desc = miopen::TensorDescriptor(
                    wDesc.GetType(), drop_size.data(), drop_in_str.data(), 2);
                auto drop_out_desc = miopen::TensorDescriptor(
                    wDesc.GetType(), drop_size.data(), drop_out_str.data(), 2);

                size_t drop_rsv_size = drop_out_desc.GetElementSize();
                size_t drop_rsv_start =
                    algoMode == miopenRNNdefault && rnnMode == miopenLSTM
                        ? nLayers * batch_n * hy_stride + nLayers * batch_n * hy_h * bi
                        : 2 * nLayers * batch_n * hy_stride;

                size_t drop_in_offset  = prelayer_shift;
                size_t drop_out_offset = drop_rsv_start + (li - 1) * batch_n * hy_h * bi;
                size_t drop_rsv_offset = (drop_rsv_start + (nLayers - 1) * batch_n * hy_h * bi) *
                                             (wDesc.GetType() == miopenFloat ? 4 : 2) +
                                         (li - 1) * drop_rsv_size;

                miopen::deref(dropoutDesc)
                    .DropoutForward(handle,
                                    drop_in_desc,
                                    drop_in_desc,
                                    reserveSpace,
                                    drop_out_desc,
                                    reserveSpace,
                                    reserveSpace,
                                    drop_rsv_size,
                                    drop_in_offset,
                                    drop_out_offset,
                                    drop_rsv_offset);
                // Update time
                profileRNNkernels(handle, 1, ctime);

                prelayer_shift = drop_out_offset;
            }

            miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                              false,
                                                              true,
                                                              batch_n,
                                                              wei_len * bi,
                                                              hy_h * bi,
                                                              use_dropout ? hy_h * bi : hy_stride,
                                                              bi_stride,
                                                              hy_stride,
                                                              1, // batch count
                                                              0, // Stride A
                                                              0, // Stride B
                                                              0, // Stride C
                                                              1, // alpha
                                                              1, // beta
                                                              xDesc[0].GetType()};

            miopenStatus_t gemm_status = CallGemm(handle,
                                                  gemm_desc,
                                                  reserveSpace,
                                                  prelayer_shift,
                                                  w,
                                                  wei_shift,
                                                  reserveSpace,
                                                  hid_shift,
                                                  nullptr,
                                                  GemmBackend_t::miopengemm);

            if(gemm_status != miopenStatusSuccess)
            {
                if(gemm_status == miopenStatusNotImplemented)
                {
                    MIOPEN_LOG_E("GEMM not implemented");
                }
                else
                {
                    MIOPEN_LOG_E("GEMM failed");
                }
            }
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }

        if(biasMode != 0u)
        {
            alpha0 = 1;
            alpha1 = 1;
            beta_t = 0;

            w_size[1]  = 1;
            w_size[2]  = wei_stride;
            sp_size[1] = batch_n;
            sp_size[2] = wei_stride;
            w_desc = miopen::TensorDescriptor(wDesc.GetType(), w_size.data(), w_stride.data(), 3);
            sp_desc =
                miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                     wei_shift_bias_temp,
                     hid_shift);
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }

        if(rnnMode == miopenGRU)
        {
            sp_size[1] = batch_n;
            sp_size[2] = hy_h;
            sp_desc =
                miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

            alpha0 = 0;
            alpha1 = 0;
            beta_t = 0;
            for(int bs = 0; bs < bi; bs++)
            {
                CopyTensor(handle,
                           sp_desc,
                           reserveSpace,
                           sp_desc,
                           reserveSpace,
                           hid_shift + bs * wei_len + 2 * hy_h,
                           hid_shift + hid_off + bs * hy_h);
                // Update time
                profileRNNkernels(handle, 1, ctime);

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
                         hid_shift + bs * wei_len + 2 * hy_h,
                         hid_shift + bs * wei_len + 2 * hy_h,
                         hid_shift + bs * wei_len + 2 * hy_h);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
        }

        if(biasMode != 0u)
        {
            wei_shift_bias_temp += wei_stride;

            alpha0 = 1;
            alpha1 = 1;
            beta_t = 0;

            if(hx != nullptr)
            {
                sp_size[1] = batch_n;
                sp_size[2] = wei_stride;
                sp_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                         wei_shift_bias_temp,
                         hid_shift);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
            else
            {
                sp_size[1] = batch_n - in_n.at(0);
                sp_size[2] = wei_len;
                sp_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                w_size[1] = 1;
                w_size[2] = wei_len;
                w_desc =
                    miopen::TensorDescriptor(wDesc.GetType(), w_size.data(), w_stride.data(), 3);

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
                         hid_shift + in_n.at(0) * hy_stride,
                         wei_shift_bias_temp,
                         hid_shift + in_n.at(0) * hy_stride);
                // Update time
                profileRNNkernels(handle, 1, ctime);

                if(dirMode != 0u)
                {
                    if(in_n.at(0) == in_n.at(seqLen - 1))
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
                                 hid_shift + wei_len,
                                 wei_shift_bias_temp + wei_len,
                                 hid_shift + wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else
                    {
                        int cur_batch = 0;
                        for(int ti = 0; ti < seqLen; ti++)
                        {
                            if(ti != (seqLen - 1))
                            {
                                offset = hid_shift + cur_batch * hy_stride;

                                sp_size[1] = in_n.at(ti + 1);
                                sp_size[2] = wei_len;
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                         static_cast<int>(offset) + wei_len,
                                         wei_shift_bias_temp + wei_len,
                                         static_cast<int>(offset) + wei_len);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                            cur_batch += in_n.at(ti);
                        }
                    }
                }
            }
        }

        // from hidden state
        int bacc   = 0;
        int baccbi = batch_n;
        for(int ti = 0; ti < seqLen; ti++)
        {
            baccbi -= in_n.at(seqLen - 1 - ti);
            wei_shift         = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift = 0;
            int use_time      = 0;

            for(int ri = 0; ri < bi; ri++)
            {
                int cur_time  = ri == 0 ? ti : seqLen - 1 - ti;
                int cur_batch = ri == 0 ? bacc : baccbi;
                offset        = hid_shift + cur_batch * hy_stride;
                if(ti > 0)
                {
                    pretime_shift =
                        ri == 0 ? hid_shift + (bacc - in_n.at(ti - 1)) * hy_stride
                                : hid_shift + (baccbi + in_n.at(seqLen - 1 - ti)) * hy_stride;
                    use_time = ri == 0 ? ti : seqLen - ti;
                }

                if(in_n.at(cur_time) > 0)
                {
                    if(ti == 0)
                    {
                        if(hx != nullptr)
                        {
                            miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                                              false,
                                                                              true,
                                                                              in_n.at(cur_time),
                                                                              wei_len,
                                                                              hy_h,
                                                                              uni_stride,
                                                                              uni_stride,
                                                                              hy_stride,
                                                                              1, // batch count
                                                                              0, // Stride A
                                                                              0, // Stride B
                                                                              0, // Stride C
                                                                              1, // alpha
                                                                              1, // beta
                                                                              xDesc[0].GetType()};

                            miopenStatus_t gemm_status =
                                CallGemm(handle,
                                         gemm_desc,
                                         hx,
                                         hx_shift + ri * hy_n * hy_h,
                                         w,
                                         wei_shift + ri * wei_len * uni_stride,
                                         reserveSpace,
                                         static_cast<int>(offset) + ri * wei_len,
                                         nullptr,
                                         GemmBackend_t::miopengemm);

                            if(gemm_status != miopenStatusSuccess)
                            {
                                if(gemm_status == miopenStatusNotImplemented)
                                {
                                    MIOPEN_LOG_E("GEMM not implemented");
                                }
                                else
                                {
                                    MIOPEN_LOG_E("GEMM failed");
                                }
                            }
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }
                    else
                    {
                        if(ri == 1 && hx != nullptr && in_n.at(cur_time) > in_n.at(use_time))
                        {
                            miopen::GemmDescriptor gemm_desc =
                                GemmDescriptor{false,
                                               false,
                                               true,
                                               (in_n.at(cur_time) - in_n.at(use_time)),
                                               wei_len,
                                               hy_h,
                                               uni_stride,
                                               uni_stride,
                                               hy_stride,
                                               1, // batch count
                                               0, // Stride A
                                               0, // Stride B
                                               0, // Stride C
                                               1, // alpha
                                               1, // beta
                                               xDesc[0].GetType()};

                            miopenStatus_t gemm_status =
                                CallGemm(handle,
                                         gemm_desc,
                                         hx,
                                         hx_shift + ri * hy_n * hy_h + in_n.at(use_time) * hy_h,
                                         w,
                                         wei_shift + ri * wei_len * uni_stride,
                                         reserveSpace,
                                         static_cast<int>(offset) + ri * wei_len +
                                             in_n.at(use_time) * hy_stride,
                                         nullptr,
                                         GemmBackend_t::miopengemm);

                            if(gemm_status != miopenStatusSuccess)
                            {
                                if(gemm_status == miopenStatusNotImplemented)
                                {
                                    MIOPEN_LOG_E("GEMM not implemented");
                                }
                                else
                                {
                                    MIOPEN_LOG_E("GEMM failed");
                                }
                            }
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }

                        if(in_n.at(use_time) > 0)
                        {
                            miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                                              false,
                                                                              true,
                                                                              in_n.at(use_time),
                                                                              wei_len,
                                                                              hy_h,
                                                                              hy_stride,
                                                                              uni_stride,
                                                                              hy_stride,
                                                                              1, // batch count
                                                                              0, // Stride A
                                                                              0, // Stride B
                                                                              0, // Stride C
                                                                              1, // alpha
                                                                              1, // beta
                                                                              xDesc[0].GetType()};

                            miopenStatus_t gemm_status =
                                CallGemm(handle,
                                         gemm_desc,
                                         reserveSpace,
                                         pretime_shift + hid_off + ri * hy_h,
                                         w,
                                         wei_shift + ri * wei_len * uni_stride,
                                         reserveSpace,
                                         static_cast<int>(offset) + ri * wei_len,
                                         nullptr,
                                         GemmBackend_t::miopengemm);

                            if(gemm_status != miopenStatusSuccess)
                            {
                                if(gemm_status == miopenStatusNotImplemented)
                                {
                                    MIOPEN_LOG_E("GEMM not implemented");
                                }
                                else
                                {
                                    MIOPEN_LOG_E("GEMM failed");
                                }
                            }
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }

                    // update hidden status
                    sp_size[1] = in_n.at(cur_time);
                    if(rnnMode == miopenRNNRELU || rnnMode == miopenRNNTANH)
                    {
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        activDesc.Forward(handle,
                                          &alpha,
                                          sp_desc,
                                          reserveSpace,
                                          &beta,
                                          sp_desc,
                                          reserveSpace,
                                          offset + ri * wei_len,
                                          offset + ri * wei_len + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenLSTM)
                    {
                        if(algoMode == miopenRNNdefault)
                        {
                            LSTMForwardHiddenStateUpdate(handle,
                                                         wDesc.GetType(),
                                                         false,
                                                         ti == 0,
                                                         ri,
                                                         in_n.at(0),
                                                         in_n.at(cur_time),
                                                         in_n.at(use_time),
                                                         hy_h,
                                                         hy_stride,
                                                         wei_len,
                                                         wei_stride,
                                                         cx,
                                                         hx_shift + ri * hy_n * hy_h,
                                                         reserveSpace,
                                                         offset + ri * wei_len,
                                                         offset + hy_h + ri * wei_len,
                                                         offset + 2 * hy_h + ri * wei_len,
                                                         offset + 3 * hy_h + ri * wei_len,
                                                         offset + bi * wei_len + ri * hy_h,
                                                         pretime_shift + bi * wei_len + ri * hy_h,
                                                         (li * batch_n + cur_batch) * bi * hy_h +
                                                             ri * hy_h +
                                                             nLayers * batch_n * hy_stride,
                                                         offset + hid_off + ri * hy_h);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                            continue;
                        }

                        // active gate i, f, o
                        sp_size[2] = hy_h * 3;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        sigDesc.Forward(handle,
                                        &alpha,
                                        sp_desc,
                                        reserveSpace,
                                        &beta,
                                        sp_desc,
                                        reserveSpace,
                                        offset + ri * wei_len,
                                        offset + ri * wei_len + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // active gate c
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         reserveSpace,
                                         &beta,
                                         sp_desc,
                                         reserveSpace,
                                         offset + 3 * hy_h + ri * wei_len,
                                         offset + 3 * hy_h + ri * wei_len +
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
                                 offset + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + 3 * hy_h + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + bi * wei_len + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(ti == 0)
                        {
                            if(cx != nullptr)
                            {
                                hx_size[1] = in_n.at(cur_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

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
                                         offset + hy_h + ri * wei_len +
                                             nLayers * batch_n * hy_stride,
                                         hx_shift + ri * hy_n * hy_h,
                                         offset + bi * wei_len + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }
                        else
                        {
                            if(ri == 1 && cx != nullptr && in_n.at(cur_time) > in_n.at(use_time))
                            {
                                hx_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

                                sp_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                         offset + hy_h + ri * wei_len +
                                             in_n.at(use_time) * hy_stride +
                                             nLayers * batch_n * hy_stride,
                                         hx_shift + ri * hy_n * hy_h + in_n.at(use_time) * hy_h,
                                         offset + bi * wei_len + ri * hy_h +
                                             in_n.at(use_time) * hy_stride);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                sp_size[1] = in_n.at(cur_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                            }

                            if(in_n.at(use_time) > 0)
                            {
                                if(in_n.at(use_time) != in_n.at(cur_time))
                                {
                                    sp_size[1] = in_n.at(use_time);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }

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
                                         offset + hy_h + ri * wei_len +
                                             nLayers * batch_n * hy_stride,
                                         pretime_shift + bi * wei_len + ri * hy_h,
                                         offset + bi * wei_len + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                if(in_n.at(use_time) != in_n.at(cur_time))
                                {
                                    sp_size[1] = in_n.at(cur_time);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }
                            }
                        }

                        // active cell state
                        tanhDesc.Forward(handle,
                                         &alpha,
                                         sp_desc,
                                         reserveSpace,
                                         &beta,
                                         sp_desc,
                                         reserveSpace,
                                         offset + bi * wei_len + ri * hy_h,
                                         offset + bi * wei_len + ri * hy_h +
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
                                 offset + 2 * hy_h + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + bi * wei_len + ri * hy_h + nLayers * batch_n * hy_stride,
                                 offset + hid_off + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenGRU)
                    {
                        // active z, r gate
                        sp_size[2] = 2 * hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        sigDesc.Forward(handle,
                                        &alpha,
                                        sp_desc,
                                        reserveSpace,
                                        &beta,
                                        sp_desc,
                                        reserveSpace,
                                        offset + ri * wei_len,
                                        offset + ri * wei_len + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // calculate c gate
                        sp_size[2] = hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        CopyTensor(handle,
                                   sp_desc,
                                   reserveSpace,
                                   sp_desc,
                                   reserveSpace,
                                   static_cast<int>(offset) + 2 * hy_h + ri * wei_len,
                                   static_cast<int>(offset) + hid_off + ri * hy_h +
                                       static_cast<int>(nLayers) * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
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
                                 offset + hy_h + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + 2 * hy_h + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

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
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + hid_off + ri * hy_h,
                                 offset + 2 * hy_h + ri * wei_len);
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
                                         offset + 2 * hy_h + ri * wei_len,
                                         offset + 2 * hy_h + ri * wei_len +
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
                                 offset + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + 2 * hy_h + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + hid_off + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

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
                                 offset + 2 * hy_h + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + hid_off + ri * hy_h,
                                 offset + hid_off + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 1;

                        if(ti == 0)
                        {
                            if(hx != nullptr)
                            {
                                hx_size[1] = in_n.at(cur_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

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
                                         offset + ri * wei_len + nLayers * batch_n * hy_stride,
                                         hx_shift + ri * hy_n * hy_h,
                                         offset + hid_off + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }
                        else
                        {
                            if(ri == 1 && hx != nullptr && in_n.at(cur_time) > in_n.at(use_time))
                            {
                                hx_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

                                sp_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                         offset + ri * wei_len + in_n.at(use_time) * hy_stride +
                                             nLayers * batch_n * hy_stride,
                                         hx_shift + ri * hy_n * hy_h + in_n.at(use_time) * hy_h,
                                         offset + hid_off + ri * hy_h +
                                             in_n.at(use_time) * hy_stride);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                sp_size[1] = in_n.at(cur_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                            }

                            if(in_n.at(use_time) > 0)
                            {
                                if(in_n.at(use_time) != in_n.at(cur_time))
                                {
                                    sp_size[1] = in_n.at(use_time);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }

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
                                         offset + ri * wei_len + nLayers * batch_n * hy_stride,
                                         pretime_shift + hid_off + ri * hy_h,
                                         offset + hid_off + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }
                    }
                }
            }

            bacc += in_n.at(ti);
        }

        // update hy, cy
        if(hy != nullptr || (rnnMode == miopenLSTM && cy != nullptr))
        {
            hx_size[2] = hy_h;
            sp_size[2] = hy_h;

            bacc   = batch_n;
            baccbi = 0;
            for(int ti = seqLen - 1; ti >= 0; ti--)
            {
                bacc -= in_n.at(ti);
                for(int ri = 0; ri < bi; ri++)
                {
                    int cur_time  = ri == 0 ? ti : seqLen - 1 - ti;
                    int cur_batch = ri == 0 ? bacc : baccbi;
                    int use_batch = 0;

                    if(ti < seqLen - 1)
                    {
                        int use_time = ri == 0 ? ti + 1 : seqLen - 2 - ti;
                        use_batch    = in_n.at(use_time);
                    }

                    if(in_n.at(cur_time) > use_batch)
                    {
                        offset = hid_shift + cur_batch * hy_stride;

                        sp_size[1] = in_n.at(cur_time) - use_batch;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                        hx_size[1] = sp_size[1];
                        hx_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

                        if(hy != nullptr)
                        {
                            CopyTensor(handle,
                                       sp_desc,
                                       reserveSpace,
                                       hx_desc,
                                       hy,
                                       static_cast<int>(offset) + hid_off + ri * hy_h +
                                           use_batch * hy_stride,
                                       hx_shift + ri * hy_n * hy_h + use_batch * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }

                        if(rnnMode == miopenLSTM && cy != nullptr)
                        {
                            CopyTensor(handle,
                                       sp_desc,
                                       reserveSpace,
                                       hx_desc,
                                       cy,
                                       static_cast<int>(offset) + bi * wei_len + ri * hy_h +
                                           use_batch * hy_stride,
                                       hx_shift + ri * hy_n * hy_h + use_batch * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }
                }
                baccbi += in_n.at(seqLen - 1 - ti);
            }
        }
    }

    // output
    prelayer_shift = (static_cast<int>(nLayers) - 1) * batch_n * hy_stride + hid_off;

    sp_size[1] = batch_n;
    sp_size[2] = hy_h * bi;
    y_size[1]  = batch_n;
    y_size[2]  = out_h;
    y_desc     = miopen::TensorDescriptor(wDesc.GetType(), y_size.data(), y_stride.data(), 3);
    sp_desc    = miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

    CopyTensor(handle, sp_desc, reserveSpace, y_desc, y, prelayer_shift, 0);
    // Update time
    profileRNNkernels(handle, 2, ctime);

#else
    (void)bi_stride;
    (void)alpha;
    (void)offset;
    (void)alpha0;
    (void)alpha1;
    (void)beta_t;
    (void)hx;
    (void)cx;
    (void)wei_shift_bias;
    MIOPEN_THROW("GEMM is not supported");
#endif
};

void RNNDescriptor::RNNBackwardData(Handle& handle,
                                    const int seqLen,
                                    c_array_view<const miopenTensorDescriptor_t> yDesc,
                                    ConstData_t y,
                                    c_array_view<const miopenTensorDescriptor_t> dyDesc,
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
                                    c_array_view<const miopenTensorDescriptor_t> dxDesc,
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

    // Suppress warning
    (void)y;
    (void)yDesc;
    (void)hxDesc;
    (void)cxDesc;
    (void)dcxDesc;
    (void)dcyDesc;
    (void)dhyDesc;
    (void)wDesc;

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

    std::vector<int> in_n;
    int in_h  = dxDesc[0].GetLengths()[1];
    int hy_d  = dhxDesc.GetLengths()[0];
    int hy_n  = dhxDesc.GetLengths()[1];
    int hy_h  = dhxDesc.GetLengths()[2];
    int out_h = dyDesc[0].GetLengths()[1];

    if(in_h <= 0 || hy_h <= 0 || hy_n <= 0 || hy_d <= 0 || out_h <= 0 || seqLen <= 0)
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
            if(batchval <= 0)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Input batch is ZERO!");
            }
        }
        else
        {
            if(batchval > in_n.back() || batchval < 0)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Incorrect input batch size at time " + std::to_string(i) +
                                 "! Batch size must not ascend!");
            }
        }
        in_n.push_back(batchval);
        batch_n += dxDesc[i].GetLengths()[0];
    }

    int bi = dirMode != 0u ? 2 : 1;
    if(out_h != (bi * hy_h))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Output size doesn't match hidden state size!");
    }

    float ctime    = 0.;
    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * static_cast<int>(workspaceScale);
    int out_stride = out_h;
    int wei_stride = hy_h * bi * static_cast<int>(nHiddenTensorsPerLayer);
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

    sp_size[2]   = workSpaceSize / GetTypeSize(wDesc.GetType());
    sp_stride[0] = sp_size[2];
    sp_stride[1] = sp_size[2];
    sp_desc      = miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
    SetTensor(handle, sp_desc, workSpace, &beta);
    // Update time
    profileRNNkernels(handle, 0, ctime);
    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = hy_stride;
    sp_size[2]   = 1;
    x_stride[0]  = batch_n * in_stride;
    x_stride[1]  = in_stride;
    y_stride[0]  = batch_n * out_stride;
    y_stride[1]  = out_stride;
    if(dhx != nullptr || (rnnMode == miopenLSTM && dcx != nullptr))
    {
        hx_size[2]   = hy_d * hy_n * hy_h;
        hx_stride[0] = hx_size[2];
        hx_stride[1] = hx_size[2];
        hx_desc = miopen::TensorDescriptor(wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);
        if(dhx != nullptr)
        {
            SetTensor(handle, hx_desc, dhx, &beta);
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }
        if(rnnMode == miopenLSTM && dcx != nullptr)
        {
            SetTensor(handle, hx_desc, dcx, &beta);
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }
    }
    hx_stride[0] = in_n.at(0) * uni_stride;
    hx_stride[1] = uni_stride;

#if MIOPEN_USE_GEMM

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

    for(int li = static_cast<int>(nLayers) - 1; li >= 0; li--)
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
            y_desc = miopen::TensorDescriptor(wDesc.GetType(), y_size.data(), y_stride.data(), 3);
            sp_desc =
                miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

            CopyTensor(handle, y_desc, dy, sp_desc, workSpace, 0, hid_shift + dhd_off);
            // Update time
            profileRNNkernels(handle, 1, ctime); // start timing
        }
        else
        {
            prelayer_shift                   = (li + 1) * batch_n * hy_stride;
            miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                              false,
                                                              false,
                                                              batch_n,
                                                              hy_h * bi,
                                                              wei_len * bi,
                                                              hy_stride,
                                                              bi_stride,
                                                              hy_stride,
                                                              1, // batch count
                                                              0, // Stride A
                                                              0, // Stride B
                                                              0, // Stride C
                                                              1, // alpha
                                                              1, // beta
                                                              yDesc[0].GetType()};

            miopenStatus_t gemm_status = CallGemm(handle,
                                                  gemm_desc,
                                                  workSpace,
                                                  prelayer_shift,
                                                  w,
                                                  wei_shift,
                                                  workSpace,
                                                  hid_shift + dhd_off,
                                                  nullptr,
                                                  GemmBackend_t::miopengemm);

            if(gemm_status != miopenStatusSuccess)
            {
                if(gemm_status == miopenStatusNotImplemented)
                {
                    MIOPEN_LOG_E("GEMM not implemented");
                }
                else
                {
                    MIOPEN_LOG_E("GEMM failed");
                }
            }
            // Update time
            profileRNNkernels(handle, 1, ctime);

            if(!float_equal(miopen::deref(dropoutDesc).dropout, 0))
            {
                std::vector<int> drop_size(2), drop_in_str(2, 1);
                drop_size[0]   = batch_n;
                drop_size[1]   = hy_h * bi;
                drop_in_str[0] = hy_stride;

                auto drop_in_desc = miopen::TensorDescriptor(
                    wDesc.GetType(), drop_size.data(), drop_in_str.data(), 2);

                size_t drop_rsv_size = drop_in_desc.GetElementSize();
                size_t drop_rsv_start =
                    algoMode == miopenRNNdefault && rnnMode == miopenLSTM
                        ? nLayers * batch_n * hy_stride + nLayers * batch_n * hy_h * bi
                        : 2 * nLayers * batch_n * hy_stride;

                size_t drop_rsv_offset = (drop_rsv_start + (nLayers - 1) * batch_n * hy_h * bi) *
                                             (wDesc.GetType() == miopenFloat ? 4 : 2) +
                                         li * drop_rsv_size;

                miopen::deref(dropoutDesc)
                    .DropoutBackward(handle,
                                     drop_in_desc,
                                     drop_in_desc,
                                     workSpace,
                                     drop_in_desc,
                                     workSpace,
                                     reserveSpace,
                                     drop_rsv_size,
                                     hid_shift + dhd_off,
                                     hid_shift + dhd_off,
                                     drop_rsv_offset);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
        }

        // from hidden state
        int bacc   = batch_n;
        int baccbi = 0;
        for(int ti = seqLen - 1; ti >= 0; ti--)
        {
            bacc -= in_n.at(ti);

            // from post state
            for(int ri = 0; ri < bi; ri++)
            {
                cur_time  = ri == 0 ? ti : seqLen - 1 - ti;
                cur_batch = ri == 0 ? bacc : baccbi;
                offset    = hid_shift + cur_batch * hy_stride;
                if(ti < seqLen - 1)
                {
                    use_time  = ri == 0 ? ti + 1 : seqLen - 1 - ti;
                    pre_batch = ri == 0 ? bacc + in_n.at(ti) : baccbi - in_n.at(seqLen - 2 - ti);
                }
                if(ti > 0)
                {
                    use_time2 = ri == 0 ? ti : seqLen - ti;
                    pre_batch2 =
                        ri == 0 ? bacc - in_n.at(ti - 1) : baccbi + in_n.at(seqLen - 1 - ti);
                }

                if(in_n.at(cur_time) > 0)
                {
                    if(ti == seqLen - 1)
                    {
                        if(dhy != nullptr)
                        {
                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            hx_size[1] = in_n.at(cur_time);
                            hx_size[2] = hy_h;
                            sp_size[1] = in_n.at(cur_time);
                            sp_size[2] = hy_h;
                            hx_desc    = miopen::TensorDescriptor(
                                wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);
                            sp_desc = miopen::TensorDescriptor(
                                wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     hx_desc,
                                     dhy,
                                     &alpha1,
                                     sp_desc,
                                     workSpace,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     hx_shift + ri * hy_n * hy_h,
                                     offset + dhd_off + ri * hy_h,
                                     offset + dhd_off + ri * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }
                    else
                    {
                        if(ri == 0 && dhy != nullptr && in_n.at(cur_time) > in_n.at(use_time))
                        {
                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 0;

                            hx_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                            hx_size[2] = hy_h;
                            sp_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                            sp_size[2] = hy_h;
                            hx_desc    = miopen::TensorDescriptor(
                                wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);
                            sp_desc = miopen::TensorDescriptor(
                                wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     hx_desc,
                                     dhy,
                                     &alpha1,
                                     sp_desc,
                                     workSpace,
                                     &beta_t,
                                     sp_desc,
                                     workSpace,
                                     hx_shift + ri * hy_n * hy_h + in_n.at(use_time) * hy_h,
                                     offset + dhd_off + ri * hy_h + in_n.at(use_time) * hy_stride,
                                     offset + dhd_off + ri * hy_h + in_n.at(use_time) * hy_stride);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }

                        pretime_shift =
                            li * batch_n * hy_stride + pre_batch * hy_stride + ri * wei_len;

                        if(in_n.at(use_time) > 0)
                        {
                            if(rnnMode == miopenGRU)
                            {
                                sp_size[1] = in_n.at(use_time);
                                sp_size[2] = hy_h;
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                         pretime_shift - ri * 2 * hy_h + dhd_off,
                                         pretime_shift + nLayers * batch_n * hy_stride,
                                         offset + dhd_off + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                CopyTensor(handle,
                                           sp_desc,
                                           workSpace,
                                           sp_desc,
                                           workSpace,
                                           pretime_shift + 2 * hy_h,
                                           static_cast<int>(offset) + ri * wei_len + 2 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                CopyTensor(handle,
                                           sp_desc,
                                           reserveSpace,
                                           sp_desc,
                                           workSpace,
                                           pretime_shift - ri * 2 * hy_h + dhd_off +
                                               static_cast<int>(nLayers) * batch_n * hy_stride,
                                           pretime_shift + 2 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                            miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                                              false,
                                                                              false,
                                                                              in_n.at(use_time),
                                                                              hy_h,
                                                                              wei_len,
                                                                              hy_stride,
                                                                              uni_stride,
                                                                              hy_stride,
                                                                              1, // batch count
                                                                              0, // Stride A
                                                                              0, // Stride B
                                                                              0, // Stride C
                                                                              1, // alpha
                                                                              1, // beta
                                                                              yDesc[0].GetType()};

                            miopenStatus_t gemm_status =
                                CallGemm(handle,
                                         gemm_desc,
                                         workSpace,
                                         pretime_shift,
                                         w,
                                         weitime_shift + ri * wei_len * uni_stride,
                                         workSpace,
                                         static_cast<int>(offset) + dhd_off + ri * hy_h,
                                         nullptr,
                                         GemmBackend_t::miopengemm);

                            if(gemm_status != miopenStatusSuccess)
                            {
                                if(gemm_status == miopenStatusNotImplemented)
                                {
                                    MIOPEN_LOG_E("GEMM not implemented");
                                }
                                else
                                {
                                    MIOPEN_LOG_E("GEMM failed");
                                }
                            }
                            // Update time
                            profileRNNkernels(handle, 1, ctime);

                            if(rnnMode == miopenGRU)
                            {
                                CopyTensor(handle,
                                           sp_desc,
                                           workSpace,
                                           sp_desc,
                                           workSpace,
                                           static_cast<int>(offset) + ri * wei_len + 2 * hy_h,
                                           pretime_shift + 2 * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }
                    }

                    // update hidden status
                    sp_size[1] = in_n.at(cur_time);
                    sp_size[2] = hy_h;
                    sp_desc    = miopen::TensorDescriptor(
                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                           offset + ri * wei_len + nLayers * batch_n * hy_stride,
                                           offset + ri * wei_len,
                                           offset + ri * wei_len,
                                           offset + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                    else if(rnnMode == miopenLSTM)
                    {
                        if(algoMode == miopenRNNdefault)
                        {
                            LSTMBackwardHiddenStateUpdate(
                                handle,
                                wDesc.GetType(),
                                ti == 0,
                                ti == seqLen - 1,
                                ri,
                                in_n.at(0),
                                in_n.at(cur_time),
                                in_n.at(use_time),
                                in_n.at(use_time2),
                                hy_h,
                                hy_stride,
                                wei_len,
                                wei_stride,
                                cx,
                                hx_shift + ri * hy_n * hy_h,
                                reserveSpace,
                                offset + ri * wei_len,
                                offset + hy_h + ri * wei_len,
                                offset + 2 * hy_h + ri * wei_len,
                                offset + 3 * hy_h + ri * wei_len,
                                (li * batch_n + cur_batch) * bi * hy_h + ri * hy_h +
                                    nLayers * batch_n * hy_stride,
                                li * batch_n * hy_stride + pre_batch2 * hy_stride + bi * wei_len +
                                    ri * hy_h,
                                dcy,
                                hx_shift + ri * hy_n * hy_h,
                                workSpace,
                                offset + ri * wei_len,
                                offset + hy_h + ri * wei_len,
                                offset + 2 * hy_h + ri * wei_len,
                                offset + 3 * hy_h + ri * wei_len,
                                offset + bi * wei_len + ri * hy_h,
                                li * batch_n * hy_stride + pre_batch * hy_stride + bi * wei_len +
                                    ri * hy_h,
                                offset + dhd_off + ri * hy_h,
                                li * batch_n * hy_stride + pre_batch * hy_stride + hy_h +
                                    ri * wei_len);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                            continue;
                        }

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        // update cell state
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
                                 offset + dhd_off + ri * hy_h,
                                 offset + 2 * hy_h + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + bi * wei_len + ri * hy_h);
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
                                          offset + bi * wei_len + ri * hy_h +
                                              nLayers * batch_n * hy_stride,
                                          offset + bi * wei_len + ri * hy_h,
                                          offset + bi * wei_len + ri * hy_h,
                                          offset + bi * wei_len + ri * hy_h);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        if(ti == seqLen - 1)
                        {
                            if(dcy != nullptr)
                            {
                                hx_size[1] = in_n.at(cur_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         hx_desc,
                                         dcy,
                                         &alpha1,
                                         sp_desc,
                                         workSpace,
                                         &beta_t,
                                         sp_desc,
                                         workSpace,
                                         hx_shift + ri * hy_n * hy_h,
                                         offset + bi * wei_len + ri * hy_h,
                                         offset + bi * wei_len + ri * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }
                        else
                        {
                            if(ri == 0 && dcy != nullptr && in_n.at(cur_time) > in_n.at(use_time))
                            {
                                hx_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                hx_size[2] = hy_h;
                                sp_size[1] = in_n.at(cur_time) - in_n.at(use_time);
                                sp_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);
                                sp_desc = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                                OpTensor(handle,
                                         miopenTensorOpAdd,
                                         &alpha0,
                                         hx_desc,
                                         dcy,
                                         &alpha1,
                                         sp_desc,
                                         workSpace,
                                         &beta_t,
                                         sp_desc,
                                         workSpace,
                                         hx_shift + ri * hy_n * hy_h + in_n.at(use_time) * hy_h,
                                         offset + bi * wei_len + ri * hy_h +
                                             in_n.at(use_time) * hy_stride,
                                         offset + bi * wei_len + ri * hy_h +
                                             in_n.at(use_time) * hy_stride);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                sp_size[1] = in_n.at(cur_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                            }

                            pretime_shift = li * batch_n * hy_stride + pre_batch * hy_stride;
                            alpha0        = 1;
                            alpha1        = 1;
                            beta_t        = 1;

                            if(in_n.at(cur_time) != in_n.at(use_time))
                            {
                                sp_size[1] = in_n.at(use_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                            }

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
                                     pretime_shift + bi * wei_len + ri * hy_h,
                                     pretime_shift + hy_h + ri * wei_len +
                                         nLayers * batch_n * hy_stride,
                                     offset + bi * wei_len + ri * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);

                            if(in_n.at(cur_time) != in_n.at(use_time))
                            {
                                sp_size[1] = in_n.at(cur_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                            }
                        }

                        // update forget gate
                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

                        if(ti == 0)
                        {
                            if(cx != nullptr)
                            {
                                hx_size[1] = in_n.at(cur_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

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
                                         offset + bi * wei_len + ri * hy_h,
                                         hx_shift + ri * hy_n * hy_h,
                                         offset + hy_h + ri * wei_len);
                            }
                        }
                        else
                        {
                            if(ri == 1 && cx != nullptr && in_n.at(cur_time) > in_n.at(use_time2))
                            {
                                hx_size[1] = in_n.at(cur_time) - in_n.at(use_time2);
                                hx_size[2] = hy_h;
                                sp_size[1] = in_n.at(cur_time) - in_n.at(use_time2);
                                sp_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);
                                sp_desc = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                         offset + bi * wei_len + ri * hy_h +
                                             in_n.at(use_time2) * hy_stride,
                                         hx_shift + ri * hy_n * hy_h + in_n.at(use_time2) * hy_h,
                                         offset + hy_h + ri * wei_len +
                                             in_n.at(use_time2) * hy_stride);

                                sp_size[1] = in_n.at(cur_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                            }

                            if(in_n.at(use_time2) > 0)
                            {
                                pretime_shift = li * batch_n * hy_stride + pre_batch2 * hy_stride;

                                if(in_n.at(cur_time) != in_n.at(use_time2))
                                {
                                    sp_size[1] = in_n.at(use_time2);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }

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
                                         offset + bi * wei_len + ri * hy_h,
                                         pretime_shift + bi * wei_len + ri * hy_h,
                                         offset + hy_h + ri * wei_len);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                if(in_n.at(cur_time) != in_n.at(use_time2))
                                {
                                    sp_size[1] = in_n.at(cur_time);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }
                            }
                        }

                        // update input gate
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
                                 offset + bi * wei_len + ri * hy_h,
                                 offset + 3 * hy_h + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update output gate
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
                                 offset + dhd_off + ri * hy_h,
                                 offset + bi * wei_len + ri * hy_h + nLayers * batch_n * hy_stride,
                                 offset + 2 * hy_h + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // update c gate
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
                                 offset + bi * wei_len + ri * hy_h,
                                 offset + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + 3 * hy_h + ri * wei_len);
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
                                          offset + 3 * hy_h + ri * wei_len +
                                              nLayers * batch_n * hy_stride,
                                          offset + 3 * hy_h + ri * wei_len,
                                          offset + 3 * hy_h + ri * wei_len,
                                          offset + 3 * hy_h + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        sp_size[2] = 3 * hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

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
                                         offset + ri * wei_len + nLayers * batch_n * hy_stride,
                                         offset + ri * wei_len,
                                         offset + ri * wei_len,
                                         offset + ri * wei_len);
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
                                 offset + dhd_off + ri * hy_h,
                                 offset + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + 2 * hy_h + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        alpha0 = 1;
                        alpha1 = 1;
                        beta_t = 0;

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
                                 offset + dhd_off + ri * hy_h,
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + 2 * hy_h + ri * wei_len);
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
                                          offset + 2 * hy_h + ri * wei_len +
                                              nLayers * batch_n * hy_stride,
                                          offset + 2 * hy_h + ri * wei_len,
                                          offset + 2 * hy_h + ri * wei_len,
                                          offset + 2 * hy_h + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // r gate
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
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + dhd_off + ri * hy_h + nLayers * batch_n * hy_stride,
                                 offset + hy_h + ri * wei_len);
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
                                 reserveSpace,
                                 offset + 2 * hy_h + ri * wei_len,
                                 offset + hy_h + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + dhd_off + ri * hy_h + nLayers * batch_n * hy_stride);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        // z gate
                        if(ti == 0)
                        {
                            if(hx != nullptr)
                            {
                                hx_size[1] = in_n.at(cur_time);
                                hx_size[2] = hy_h;
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         hx_desc,
                                         hx,
                                         &alpha1,
                                         sp_desc,
                                         workSpace,
                                         &beta_t,
                                         sp_desc,
                                         workSpace,
                                         hx_shift + ri * hy_n * hy_h,
                                         offset + dhd_off + ri * hy_h,
                                         offset + ri * wei_len);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }
                        }
                        else
                        {
                            if(ri == 1 && hx != nullptr && in_n.at(cur_time) > in_n.at(use_time2))
                            {
                                hx_size[1] = in_n.at(cur_time) - in_n.at(use_time2);
                                hx_size[2] = hy_h;
                                sp_size[1] = in_n.at(cur_time) - in_n.at(use_time2);
                                hx_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);
                                sp_desc = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                                OpTensor(handle,
                                         miopenTensorOpMul,
                                         &alpha0,
                                         hx_desc,
                                         hx,
                                         &alpha1,
                                         sp_desc,
                                         workSpace,
                                         &beta_t,
                                         sp_desc,
                                         workSpace,
                                         hx_shift + ri * hy_n * hy_h + in_n.at(use_time2) * hy_h,
                                         offset + dhd_off + ri * hy_h +
                                             in_n.at(use_time2) * hy_stride,
                                         offset + ri * wei_len + in_n.at(use_time2) * hy_stride);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                sp_size[1] = in_n.at(cur_time);
                                sp_desc    = miopen::TensorDescriptor(
                                    wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                            }

                            if(in_n.at(use_time2) > 0)
                            {
                                if(in_n.at(use_time2) != in_n.at(cur_time))
                                {
                                    sp_size[1] = in_n.at(use_time2);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }

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
                                         hid_shift + pre_batch2 * hy_stride + dhd_off + ri * hy_h,
                                         offset + dhd_off + ri * hy_h,
                                         offset + ri * wei_len);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

                                if(in_n.at(use_time2) != in_n.at(cur_time))
                                {
                                    sp_size[1] = in_n.at(cur_time);
                                    sp_desc    = miopen::TensorDescriptor(
                                        wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                                }
                            }
                        }

                        alpha0 = -1;
                        alpha1 = 1;
                        beta_t = 1;

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
                                 offset + 2 * hy_h + ri * wei_len + nLayers * batch_n * hy_stride,
                                 offset + dhd_off + ri * hy_h,
                                 offset + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);

                        sp_size[2] = 2 * hy_h;
                        sp_desc    = miopen::TensorDescriptor(
                            wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
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
                                         offset + ri * wei_len + nLayers * batch_n * hy_stride,
                                         offset + ri * wei_len,
                                         offset + ri * wei_len,
                                         offset + ri * wei_len);
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                }
            }

            baccbi += in_n.at(seqLen - 1 - ti);
        }

        // dcx, dhx
        if(dhx != nullptr || (rnnMode == miopenLSTM && dcx != nullptr))
        {
            hx_size[2] = hy_h;
            sp_size[2] = hy_h;

            bacc   = 0;
            baccbi = batch_n;
            for(int ti = 0; ti < seqLen; ti++)
            {
                baccbi -= in_n.at(seqLen - 1 - ti);
                for(int ri = 0; ri < bi; ri++)
                {
                    cur_time      = ri == 0 ? ti : seqLen - 1 - ti;
                    cur_batch     = ri == 0 ? bacc : baccbi;
                    use_time      = 0;
                    int use_batch = 0;

                    if(ti > 0)
                    {
                        use_time  = ri == 0 ? ti - 1 : seqLen - ti;
                        use_batch = in_n.at(use_time);
                    }

                    if(in_n.at(cur_time) > use_batch)
                    {
                        pretime_shift = li * batch_n * hy_stride + cur_batch * hy_stride;

                        if(rnnMode == miopenLSTM || rnnMode == miopenGRU)
                        {
                            sp_size[1] = in_n.at(cur_time) - use_batch;
                            hx_size[1] = in_n.at(cur_time) - use_batch;
                            hx_desc    = miopen::TensorDescriptor(
                                wDesc.GetType(), hx_size.data(), hx_stride.data(), 3);
                            sp_desc = miopen::TensorDescriptor(
                                wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);
                        }

                        if(dhx != nullptr)
                        {
                            if(rnnMode == miopenGRU)
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
                                         pretime_shift + 2 * hy_h + ri * wei_len +
                                             use_batch * hy_stride,
                                         pretime_shift + hy_h + ri * wei_len +
                                             use_batch * hy_stride + nLayers * batch_n * hy_stride,
                                         pretime_shift + dhd_off + ri * hy_h +
                                             use_batch * hy_stride + nLayers * batch_n * hy_stride);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                                miopen::GemmDescriptor gemm_desc =
                                    GemmDescriptor{false,
                                                   false,
                                                   false,
                                                   (in_n.at(cur_time) - use_batch),
                                                   hy_h,
                                                   hy_h,
                                                   hy_stride,
                                                   uni_stride,
                                                   uni_stride,
                                                   1, // batch count
                                                   0, // Stride A
                                                   0, // Stride B
                                                   0, // Stride C
                                                   1, // alpha
                                                   0, // beta
                                                   yDesc[0].GetType()};

                                miopenStatus_t gemm_status = CallGemm(
                                    handle,
                                    gemm_desc,
                                    reserveSpace,
                                    pretime_shift + dhd_off + ri * hy_h + use_batch * hy_stride +
                                        static_cast<int>(nLayers) * batch_n * hy_stride,
                                    w,
                                    weitime_shift + 2 * hy_h * uni_stride +
                                        ri * wei_len * uni_stride,
                                    dhx,
                                    hx_shift + ri * hy_n * hy_h + use_batch * hy_h,
                                    nullptr,
                                    GemmBackend_t::miopengemm);

                                if(gemm_status != miopenStatusSuccess)
                                {
                                    if(gemm_status == miopenStatusNotImplemented)
                                    {
                                        MIOPEN_LOG_E("GEMM not implemented");
                                    }
                                    else
                                    {
                                        MIOPEN_LOG_E("GEMM failed");
                                    }
                                }
                                // Update time
                                profileRNNkernels(handle, 1, ctime);

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
                                         pretime_shift + dhd_off + ri * hy_h +
                                             use_batch * hy_stride,
                                         pretime_shift + ri * wei_len + use_batch * hy_stride +
                                             nLayers * batch_n * hy_stride,
                                         hx_shift + ri * hy_n * hy_h + use_batch * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }

                            miopen::GemmDescriptor gemm_desc =
                                GemmDescriptor{false,
                                               false,
                                               false,
                                               (in_n.at(cur_time) - use_batch),
                                               hy_h,
                                               wei_len_t,
                                               hy_stride,
                                               uni_stride,
                                               uni_stride,
                                               1, // batch count
                                               0, // Stride A
                                               0, // Stride B
                                               0, // Stride C
                                               1, // alpha
                                               1, // beta
                                               yDesc[0].GetType()};

                            miopenStatus_t gemm_status =
                                CallGemm(handle,
                                         gemm_desc,
                                         workSpace,
                                         pretime_shift + ri * wei_len + use_batch * hy_stride,
                                         w,
                                         weitime_shift + ri * wei_len * uni_stride,
                                         dhx,
                                         hx_shift + ri * hy_n * hy_h + use_batch * hy_h,
                                         nullptr,
                                         GemmBackend_t::miopengemm);

                            if(gemm_status != miopenStatusSuccess)
                            {
                                if(gemm_status == miopenStatusNotImplemented)
                                {
                                    MIOPEN_LOG_E("GEMM not implemented");
                                }
                                else
                                {
                                    MIOPEN_LOG_E("GEMM failed");
                                }
                            }
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }

                        if(rnnMode == miopenLSTM && dcx != nullptr)
                        {
                            alpha0 = 1;
                            alpha1 = 1;
                            beta_t = 1;
                            if(algoMode == miopenRNNdefault)
                            {
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
                                         pretime_shift + bi * wei_len + ri * hy_h +
                                             use_batch * hy_stride,
                                         pretime_shift + hy_h + ri * wei_len +
                                             use_batch * hy_stride,
                                         hx_shift + ri * hy_n * hy_h + use_batch * hy_h);
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                                continue;
                            }
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
                                     pretime_shift + bi * wei_len + ri * hy_h +
                                         use_batch * hy_stride,
                                     pretime_shift + hy_h + ri * wei_len + use_batch * hy_stride +
                                         nLayers * batch_n * hy_stride,
                                     hx_shift + ri * hy_n * hy_h + use_batch * hy_h);
                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                    }
                }
                bacc += in_n.at(ti);
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
        x_desc     = miopen::TensorDescriptor(wDesc.GetType(), x_size.data(), x_stride.data(), 3);
        sp_desc    = miopen::TensorDescriptor(wDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

        alpha0 = 1;
        alpha1 = 1;
        beta_t = 0;

        for(int gi = 0; gi < nHiddenTensorsPerLayer * bi; gi++)
        {
            OpTensor(handle,
                     miopenTensorOpAdd,
                     &alpha0,
                     sp_desc,
                     workSpace,
                     &alpha1,
                     x_desc,
                     dx,
                     &beta_t,
                     x_desc,
                     dx,
                     gi * hy_h,
                     0,
                     0);
            // Update time
            profileRNNkernels(handle, (gi == nHiddenTensorsPerLayer * bi - 1) ? 2 : 1, ctime);
        }
    }
    else
    {
        miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                          false,
                                                          false,
                                                          batch_n,
                                                          in_h,
                                                          wei_len * bi,
                                                          hy_stride,
                                                          in_stride,
                                                          in_stride,
                                                          1, // batch count
                                                          0, // Stride A
                                                          0, // Stride B
                                                          0, // Stride C
                                                          1, // alpha
                                                          0, // beta
                                                          yDesc[0].GetType()};
        miopenStatus_t gemm_status       = CallGemm(
            handle, gemm_desc, workSpace, 0, w, 0, dx, 0, nullptr, GemmBackend_t::miopengemm);
        if(gemm_status != miopenStatusSuccess)
        {
            if(gemm_status == miopenStatusNotImplemented)
            {
                MIOPEN_LOG_E("GEMM not implemented");
            }
            else
            {
                MIOPEN_LOG_E("GEMM failed");
            }
        }
        // Update time
        profileRNNkernels(handle, 2, ctime);
    }
#else
    (void)wei_stride;
    (void)bi_stride;
    (void)alpha;
    (void)offset;
    (void)alpha0;
    (void)alpha1;
    (void)beta_t;
    (void)hx;
    (void)cx;
    (void)dhy;
    (void)dcy;
    (void)reserveSpace;
    (void)in_h;
    MIOPEN_THROW("GEMM is not supported");
#endif
};

void RNNDescriptor::RNNBackwardWeights(Handle& handle,
                                       const int seqLen,
                                       c_array_view<const miopenTensorDescriptor_t> xDesc,
                                       ConstData_t x,
                                       const TensorDescriptor& hxDesc,
                                       ConstData_t hx,
                                       c_array_view<const miopenTensorDescriptor_t> dyDesc,
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

    if(in_h <= 0 || hy_h <= 0 || hy_n <= 0 || hy_d <= 0 || out_h <= 0 || seqLen <= 0)
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
            if(batchval <= 0)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Input batch is ZERO!");
            }
        }
        else
        {
            if(batchval > in_n.back() || batchval < 0)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Incorrect input batch size at time " + std::to_string(i) +
                                 "! Batch size must not ascend!");
            }
        }
        in_n.push_back(batchval);
        batch_n += xDesc[i].GetLengths()[0];
    }

    int bi = dirMode != 0u ? 2 : 1;
    if(out_h != (bi * hy_h))
    {
        MIOPEN_THROW(miopenStatusBadParm, "Output size doesn't match hidden state size!");
    }

    float ctime    = 0.;
    int in_stride  = in_h;
    int hy_stride  = hy_h * bi * static_cast<int>(workspaceScale);
    int wei_stride = hy_h * bi * static_cast<int>(nHiddenTensorsPerLayer);
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

    float alpha0, alpha1, beta_t = 0;

    std::vector<int> sp_size(3, 1), sp_stride(3, 1), w_size(3, 1), w_stride(3, 1);
    miopen::TensorDescriptor sp_desc, w_desc;

    sp_stride[0] = batch_n * hy_stride;
    sp_stride[1] = hy_stride;
    w_size[2]    = dwDesc.GetElementSize();
    w_stride[0]  = w_size[2];
    w_stride[1]  = w_size[2];
    w_desc       = miopen::TensorDescriptor(dwDesc.GetType(), w_size.data(), w_stride.data(), 3);
    SetTensor(handle, w_desc, dw, &beta_t);
    // Update time
    profileRNNkernels(handle, 0, ctime);
    w_stride[0] = wei_stride;
    w_stride[1] = wei_stride;
    w_size[2]   = 1;

#if MIOPEN_USE_GEMM

    int wei_len   = 0;
    int hid_off   = 0;
    int use_time  = 0;
    int pre_batch = 0;

    switch(rnnMode)
    {
    case miopenRNNRELU:
    case miopenRNNTANH:
        // printf("run rnn gpu bwd weights \n");
        wei_len = hy_h;
        hid_off = static_cast<int>(nLayers) * batch_n * hy_stride;
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
                miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                                  true,
                                                                  false,
                                                                  wei_len * bi,
                                                                  in_h,
                                                                  batch_n,
                                                                  hy_stride,
                                                                  in_stride,
                                                                  in_stride,
                                                                  1, // batch count
                                                                  0, // Stride A
                                                                  0, // Stride B
                                                                  0, // Stride C
                                                                  1, // alpha
                                                                  1, // beta
                                                                  xDesc[0].GetType()};

                miopenStatus_t gemm_status = CallGemm(handle,
                                                      gemm_desc,
                                                      workSpace,
                                                      0,
                                                      x,
                                                      0,
                                                      dw,
                                                      0,
                                                      nullptr,
                                                      GemmBackend_t::miopengemm);

                if(gemm_status != miopenStatusSuccess)
                {
                    if(gemm_status == miopenStatusNotImplemented)
                    {
                        MIOPEN_LOG_E("GEMM not implemented");
                    }
                    else
                    {
                        MIOPEN_LOG_E("GEMM failed");
                    }
                }
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
        }
        else
        {
            bool use_dropout    = !float_equal(miopen::deref(dropoutDesc).dropout, 0);
            auto prelayer_shift = static_cast<int>(
                use_dropout ? (algoMode == miopenRNNdefault && rnnMode == miopenLSTM
                                   ? nLayers * batch_n * hy_stride + nLayers * batch_n * hy_h * bi
                                   : 2 * nLayers * batch_n * hy_stride) +
                                  (li - 1) * batch_n * hy_h * bi
                            : (li - 1) * batch_n * hy_stride + hid_off);

            miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                              true,
                                                              false,
                                                              wei_len * bi,
                                                              hy_h * bi,
                                                              batch_n,
                                                              hy_stride,
                                                              use_dropout ? hy_h * bi : hy_stride,
                                                              bi_stride,
                                                              1, // batch count
                                                              0, // Stride A
                                                              0, // Stride B
                                                              0, // Stride C
                                                              1, // alpha
                                                              1, // beta
                                                              xDesc[0].GetType()};

            miopenStatus_t gemm_status = CallGemm(handle,
                                                  gemm_desc,
                                                  workSpace,
                                                  hid_shift,
                                                  reserveSpace,
                                                  prelayer_shift,
                                                  dw,
                                                  wei_shift,
                                                  nullptr,
                                                  GemmBackend_t::miopengemm);

            if(gemm_status != miopenStatusSuccess)
            {
                if(gemm_status == miopenStatusNotImplemented)
                {
                    MIOPEN_LOG_E("GEMM not implemented");
                }
                else
                {
                    MIOPEN_LOG_E("GEMM failed");
                }
            }
            // Update time
            profileRNNkernels(handle, 1, ctime);
        }

        if(biasMode != 0u)
        {
            wei_shift = static_cast<int>(wei_shift_bias) + li * 2 * wei_stride;

            sp_size[1] = batch_n;
            sp_size[2] = wei_stride;
            w_size[1]  = 1;
            w_size[2]  = wei_stride;
            w_desc = miopen::TensorDescriptor(dwDesc.GetType(), w_size.data(), w_stride.data(), 3);
            sp_desc =
                miopen::TensorDescriptor(dwDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

            alpha0 = 0;
            alpha1 = 1;
            beta_t = 1;

            OpTensor(handle,
                     miopenTensorOpAdd,
                     &alpha0,
                     w_desc,
                     dw,
                     &alpha1,
                     sp_desc,
                     workSpace,
                     &beta_t,
                     w_desc,
                     dw,
                     wei_shift,
                     hid_shift,
                     wei_shift);

            // Update time
            profileRNNkernels(handle, 1, ctime);
        }

        // between time
        // Calculate feedback for c gate in GRU
        if(rnnMode == miopenGRU)
        {
            sp_size[1] = batch_n;
            sp_size[2] = hy_h;
            sp_desc =
                miopen::TensorDescriptor(dwDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

            for(int ri = 0; ri < bi; ri++)
            {
                CopyTensor(handle,
                           sp_desc,
                           reserveSpace,
                           sp_desc,
                           workSpace,
                           hid_shift + hid_off + ri * hy_h +
                               static_cast<int>(nLayers) * batch_n * hy_stride,
                           hid_shift + 2 * hy_h + ri * wei_len);
                // Update time
                profileRNNkernels(handle, 1, ctime);
            }
        }

        if(biasMode != 0u)
        {
            wei_shift = static_cast<int>(wei_shift_bias) + li * 2 * wei_stride + wei_stride;

            alpha0 = 1;
            alpha1 = 1;
            beta_t = 0;

            if(hx != nullptr)
            {
                if(rnnMode == miopenGRU)
                {
                    sp_size[1] = batch_n;
                    sp_size[2] = wei_stride;
                    w_size[1]  = 1;
                    w_size[2]  = wei_stride;
                    w_desc     = miopen::TensorDescriptor(
                        dwDesc.GetType(), w_size.data(), w_stride.data(), 3);
                    sp_desc = miopen::TensorDescriptor(
                        dwDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                    OpTensor(handle,
                             miopenTensorOpAdd,
                             &alpha0,
                             w_desc,
                             dw,
                             &alpha1,
                             sp_desc,
                             workSpace,
                             &beta_t,
                             w_desc,
                             dw,
                             wei_shift,
                             hid_shift,
                             wei_shift);

                    // Update time
                    profileRNNkernels(handle, 1, ctime);
                }
                else
                {
                    CopyTensor(handle, w_desc, dw, w_desc, dw, wei_shift - wei_stride, wei_shift);
                    // Update time
                    profileRNNkernels(handle, 1, ctime);
                }
            }
            else
            {
                sp_size[1] = 1;
                sp_size[2] = wei_len;
                w_size[1]  = 1;
                w_size[2]  = wei_len;
                w_desc =
                    miopen::TensorDescriptor(dwDesc.GetType(), w_size.data(), w_stride.data(), 3);
                sp_desc =
                    miopen::TensorDescriptor(dwDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                for(int bs = 0; bs < batch_n; bs++)
                {
                    if(!(hx == nullptr && bs < in_n.at(0)))
                    {
                        OpTensor(handle,
                                 miopenTensorOpAdd,
                                 &alpha0,
                                 sp_desc,
                                 workSpace,
                                 &alpha1,
                                 w_desc,
                                 dw,
                                 &beta_t,
                                 w_desc,
                                 dw,
                                 hid_shift + bs * hy_stride,
                                 wei_shift,
                                 wei_shift);

                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }
                }

                if(dirMode != 0u)
                {
                    sp_size[1] = 1;
                    sp_size[2] = wei_len;
                    w_size[1]  = 1;
                    w_size[2]  = wei_len;
                    w_desc     = miopen::TensorDescriptor(
                        dwDesc.GetType(), w_size.data(), w_stride.data(), 3);
                    sp_desc = miopen::TensorDescriptor(
                        dwDesc.GetType(), sp_size.data(), sp_stride.data(), 3);

                    int cur_batch = 0;
                    for(int ti = 0; ti < seqLen - 1; ti++)
                    {
                        for(int bs = 0; bs < in_n.at(ti + 1); bs++)
                        {
                            OpTensor(handle,
                                     miopenTensorOpAdd,
                                     &alpha0,
                                     sp_desc,
                                     workSpace,
                                     &alpha1,
                                     w_desc,
                                     dw,
                                     &beta_t,
                                     w_desc,
                                     dw,
                                     hid_shift + (cur_batch + bs) * hy_stride + wei_len,
                                     wei_shift + wei_len,
                                     wei_shift + wei_len);

                            // Update time
                            profileRNNkernels(handle, 1, ctime);
                        }
                        cur_batch += in_n.at(ti);
                    }
                }
            }
        }

        int pretime_shift, hx_shift, cur_time;
        bool comb_check = true;
        if(seqLen > 2)
        {
            if(in_n.at(0) != in_n.at(seqLen - 2))
            {
                comb_check = false;
            }
        }

        if(comb_check)
        {
            hx_shift  = li * hy_n * bi_stride;
            wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            for(int ri = 0; ri < bi; ri++)
            {
                hid_shift =
                    ri == 0 ? li * batch_n * hy_stride
                            : (li * batch_n * hy_stride + in_n.at(0) * (seqLen - 1) * hy_stride);
                cur_time = ri == 0 ? 0 : seqLen - 1;

                if(in_n.at(cur_time) > 0 && hx != nullptr)
                {
                    miopen::GemmDescriptor gemm_desc = GemmDescriptor{false,
                                                                      true,
                                                                      false,
                                                                      wei_len,
                                                                      hy_h,
                                                                      in_n.at(cur_time),
                                                                      hy_stride,
                                                                      uni_stride,
                                                                      uni_stride,
                                                                      1, // batch count
                                                                      0, // Stride A
                                                                      0, // Stride B
                                                                      0, // Stride C
                                                                      1, // alpha
                                                                      1, // beta
                                                                      xDesc[0].GetType()};

                    miopenStatus_t gemm_status = CallGemm(handle,
                                                          gemm_desc,
                                                          workSpace,
                                                          hid_shift + ri * wei_len,
                                                          hx,
                                                          hx_shift + ri * hy_n * hy_h,
                                                          dw,
                                                          wei_shift + ri * wei_len * uni_stride,
                                                          nullptr,
                                                          GemmBackend_t::miopengemm);

                    if(gemm_status != miopenStatusSuccess)
                    {
                        if(gemm_status == miopenStatusNotImplemented)
                        {
                            MIOPEN_LOG_E("GEMM not implemented");
                        }
                        else
                        {
                            MIOPEN_LOG_E("GEMM failed");
                        }
                    }

                    // Update time
                    if(li == nLayers - 1 && ri == bi - 1 && seqLen == 1)
                        profileRNNkernels(handle, 2, ctime);
                    else
                        profileRNNkernels(handle, 1, ctime);
                }

                if(seqLen > 1)
                {
                    if(ri == 1 && hx != nullptr && in_n.at(0) > in_n.at(seqLen - 1))
                    {
                        miopen::GemmDescriptor gemm_desc =
                            GemmDescriptor{false,
                                           true,
                                           false,
                                           wei_len,
                                           hy_h,
                                           (in_n.at(0) - in_n.at(seqLen - 1)),
                                           hy_stride,
                                           uni_stride,
                                           uni_stride,
                                           1, // batch count
                                           0, // Stride A
                                           0, // Stride B
                                           0, // Stride C
                                           1, // alpha
                                           1, // beta
                                           xDesc[0].GetType()};

                        miopenStatus_t gemm_status =
                            CallGemm(handle,
                                     gemm_desc,
                                     workSpace,
                                     hid_shift + ri * wei_len -
                                         (in_n.at(0) - in_n.at(seqLen - 1)) * hy_stride,
                                     hx,
                                     hx_shift + ri * hy_n * hy_h + in_n.at(seqLen - 1) * hy_h,
                                     dw,
                                     wei_shift + ri * wei_len * uni_stride,
                                     nullptr,
                                     GemmBackend_t::miopengemm);

                        if(gemm_status != miopenStatusSuccess)
                        {
                            if(gemm_status == miopenStatusNotImplemented)
                            {
                                MIOPEN_LOG_E("GEMM not implemented");
                            }
                            else
                            {
                                MIOPEN_LOG_E("GEMM failed");
                            }
                        }
                        // Update time
                        profileRNNkernels(handle, 1, ctime);
                    }

                    hid_shift = ri == 0 ? (li * batch_n * hy_stride + in_n.at(0) * hy_stride)
                                        : (li * batch_n * hy_stride);
                    pretime_shift =
                        ri == 0 ? li * batch_n * hy_stride + hid_off
                                : li * batch_n * hy_stride + in_n.at(0) * hy_stride + hid_off;

                    miopen::GemmDescriptor gemm_desc =
                        GemmDescriptor{false,
                                       true,
                                       false,
                                       wei_len,
                                       hy_h,
                                       in_n.at(0) * (seqLen - 2) + in_n.at(seqLen - 1),
                                       hy_stride,
                                       hy_stride,
                                       uni_stride,
                                       1, // batch count
                                       0, // Stride A
                                       0, // Stride B
                                       0, // Stride C
                                       1, // alpha
                                       1, // beta
                                       xDesc[0].GetType()};

                    miopenStatus_t gemm_status = CallGemm(handle,
                                                          gemm_desc,
                                                          workSpace,
                                                          hid_shift + ri * wei_len,
                                                          reserveSpace,
                                                          pretime_shift + ri * hy_h,
                                                          dw,
                                                          wei_shift + ri * wei_len * uni_stride,
                                                          nullptr,
                                                          GemmBackend_t::miopengemm);

                    if(gemm_status != miopenStatusSuccess)
                    {
                        if(gemm_status == miopenStatusNotImplemented)
                        {
                            MIOPEN_LOG_E("GEMM not implemented");
                        }
                        else
                        {
                            MIOPEN_LOG_E("GEMM failed");
                        }
                    }
                    // Update time
                    if(li == nLayers - 1 && ri == bi - 1)
                        profileRNNkernels(handle, 2, ctime);
                    else
                        profileRNNkernels(handle, 1, ctime);
                }
            }
        }
        else
        {
            int bacc   = 0;
            int baccbi = batch_n;
            for(int ti = 0; ti < seqLen; ti++)
            {
                baccbi -= in_n.at(seqLen - 1 - ti);

                hx_shift  = li * hy_n * bi_stride;
                wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

                for(int ri = 0; ri < bi; ri++)
                {
                    hid_shift = ri == 0 ? (li * batch_n * hy_stride + bacc * hy_stride)
                                        : (li * batch_n * hy_stride + baccbi * hy_stride);
                    cur_time = ri == 0 ? ti : seqLen - 1 - ti;
                    if(ti > 0)
                    {
                        pre_batch =
                            ri == 0 ? bacc - in_n.at(ti - 1) : baccbi + in_n.at(seqLen - 1 - ti);
                        use_time = ri == 0 ? ti : seqLen - ti;
                    }

                    if(in_n.at(cur_time) > 0)
                    {
                        if(ti == 0)
                        {
                            if(hx != nullptr)
                            {
                                miopen::GemmDescriptor gemm_desc =
                                    GemmDescriptor{false,
                                                   true,
                                                   false,
                                                   wei_len,
                                                   hy_h,
                                                   in_n.at(cur_time),
                                                   hy_stride,
                                                   uni_stride,
                                                   uni_stride,
                                                   1, // batch count
                                                   0, // Stride A
                                                   0, // Stride B
                                                   0, // Stride C
                                                   1, // alpha
                                                   1, // beta
                                                   xDesc[0].GetType()};

                                miopenStatus_t gemm_status =
                                    CallGemm(handle,
                                             gemm_desc,
                                             workSpace,
                                             hid_shift + ri * wei_len,
                                             hx,
                                             hx_shift + ri * hy_n * hy_h,
                                             dw,
                                             wei_shift + ri * wei_len * uni_stride,
                                             nullptr,
                                             GemmBackend_t::miopengemm);

                                if(gemm_status != miopenStatusSuccess)
                                {
                                    if(gemm_status == miopenStatusNotImplemented)
                                    {
                                        MIOPEN_LOG_E("GEMM not implemented");
                                    }
                                    else
                                    {
                                        MIOPEN_LOG_E("GEMM failed");
                                    }
                                }
                                // Update time
                                if(li == nLayers - 1 && ti == seqLen - 1 && ri == bi - 1)
                                    profileRNNkernels(handle, 2, ctime);
                                else
                                    profileRNNkernels(handle, 1, ctime);
                            }
                        }
                        else
                        {
                            if(ri == 1 && hx != nullptr && in_n.at(cur_time) > in_n.at(use_time))
                            {
                                miopen::GemmDescriptor gemm_desc =
                                    GemmDescriptor{false,
                                                   true,
                                                   false,
                                                   wei_len,
                                                   hy_h,
                                                   (in_n.at(cur_time) - in_n.at(use_time)),
                                                   hy_stride,
                                                   uni_stride,
                                                   uni_stride,
                                                   1, // batch count
                                                   0, // Stride A
                                                   0, // Stride B
                                                   0, // Stride C
                                                   1, // alpha
                                                   1, // beta
                                                   xDesc[0].GetType()};

                                miopenStatus_t gemm_status = CallGemm(
                                    handle,
                                    gemm_desc,
                                    workSpace,
                                    hid_shift + ri * wei_len + in_n.at(use_time) * hy_stride,
                                    hx,
                                    hx_shift + ri * hy_n * hy_h + in_n.at(use_time) * hy_h,
                                    dw,
                                    wei_shift + ri * wei_len * uni_stride,
                                    nullptr,
                                    GemmBackend_t::miopengemm);

                                if(gemm_status != miopenStatusSuccess)
                                {
                                    if(gemm_status == miopenStatusNotImplemented)
                                    {
                                        MIOPEN_LOG_E("GEMM not implemented");
                                    }
                                    else
                                    {
                                        MIOPEN_LOG_E("GEMM failed");
                                    }
                                }
                                // Update time
                                profileRNNkernels(handle, 1, ctime);
                            }

                            pretime_shift =
                                li * batch_n * hy_stride + pre_batch * hy_stride + hid_off;

                            if(in_n.at(use_time) > 0)
                            {
                                miopen::GemmDescriptor gemm_desc =
                                    GemmDescriptor{false,
                                                   true,
                                                   false,
                                                   wei_len,
                                                   hy_h,
                                                   in_n.at(use_time),
                                                   hy_stride,
                                                   hy_stride,
                                                   uni_stride,
                                                   1, // batch count
                                                   0, // Stride A
                                                   0, // Stride B
                                                   0, // Stride C
                                                   1, // alpha
                                                   1, // beta
                                                   xDesc[0].GetType()};

                                miopenStatus_t gemm_status =
                                    CallGemm(handle,
                                             gemm_desc,
                                             workSpace,
                                             hid_shift + ri * wei_len,
                                             reserveSpace,
                                             pretime_shift + ri * hy_h,
                                             dw,
                                             wei_shift + ri * wei_len * uni_stride,
                                             nullptr,
                                             GemmBackend_t::miopengemm);

                                if(gemm_status != miopenStatusSuccess)
                                {
                                    if(gemm_status == miopenStatusNotImplemented)
                                    {
                                        MIOPEN_LOG_E("GEMM not implemented");
                                    }
                                    else
                                    {
                                        MIOPEN_LOG_E("GEMM failed");
                                    }
                                }
                                // Update time
                                if(li == nLayers - 1 && ti == seqLen - 1 && ri == bi - 1)
                                    profileRNNkernels(handle, 2, ctime);
                                else
                                    profileRNNkernels(handle, 1, ctime);
                            }
                        }
                    }
                }

                bacc += in_n.at(ti);
            }
        }
    }
#else
    (void)in_stride;
    (void)alpha0;
    (void)wei_shift_bias;
    (void)alpha1;
    (void)bi_stride;
    (void)uni_stride;
    (void)hx;
    (void)workSpace;
    (void)reserveSpace;
    MIOPEN_THROW("GEMM is not supported");
#endif
};

} // namespace miopen
