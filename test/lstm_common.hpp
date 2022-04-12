/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_TEST_LSTM_COMMON_HPP
#define GUARD_MIOPEN_TEST_LSTM_COMMON_HPP

#include "driver.hpp"
#include "dropout_util.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "rnn_util.hpp"
#include "random.hpp"
#include <array>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/rnn.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>
#include <cfloat>

#define MIO_LSTM_TEST_DEBUG 0
#define MIO_RNN_TIME_EVERYTHING 0

#define WORKAROUND_ISSUE_692 1

//****************************************************
// FORWARD BASE
//****************************************************
template <class T>
struct verify_forward_lstm
{
    std::vector<T> input{};
    std::vector<T> initHidden{};
    std::vector<T> initCell{};
    std::vector<T> weights{};
    std::vector<int> batch_seq{};
    int hiddenSize{};
    int seqLength{};
    int nLayers{};
    int biasMode{};
    int dirMode{};
    int inputMode{};
    int batch_n{};
    int inputVecLen{};
    miopenRNNDescriptor_t rnnDesc{};
    size_t realHiddenSize{};
    bool nohx{};
    bool nocx{};
    bool nohy{};
    bool nocy{};

    void LSTMFwdCPUVerify(miopen::Handle& handle,
                          bool use_dropout,
                          const miopen::DropoutDescriptor& dropoutDesc,
                          const std::vector<T>& in,
                          const std::vector<T>& wei,
                          std::vector<T>& hy_host,
                          const std::vector<T>& hx,
                          std::vector<T>& cy_host,
                          const std::vector<T>& cx,
                          std::vector<T>& out_host,
                          const std::vector<int>& in_n,
                          int in_h,
                          int seqLength_cpu,
                          int bidirection,
                          int biased,
                          int hy_d,
                          int hy_n,
                          int hy_h,
                          int out_h,
                          int inputMode_cpu,
                          std::vector<T>& rsvspace,
                          bool hx_is_null = false,
                          bool cx_is_null = false) const;
};

//****************************************************
// BACKWARDS DATA
//****************************************************
template <class T>
struct verify_backward_data_lstm
{
    std::vector<T> yin;        // Y
    std::vector<T> dy;         // dY
    std::vector<T> dhy;        // dHY
    std::vector<T> dcy;        // dHY
    std::vector<T> initHidden; // HX
    std::vector<T> initCell;   // CX
    std::vector<T> weights;
    std::vector<int> batch_seq;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool nocx;
    bool nodhy;
    bool nodcy;
    bool nodhx;
    bool nodcx;
    bool use_dropout;
    typename std::vector<T>::iterator RSVgpu;
    typename std::vector<T>::iterator RSVcpu;

    verify_backward_data_lstm(miopenRNNDescriptor_t pRD,
                              const std::vector<T>& py,
                              const std::vector<T>& pdy,
                              const std::vector<T>& pdhy,
                              const std::vector<T>& phx,
                              const std::vector<T>& pdcy,
                              const std::vector<T>& pcx,
                              const std::vector<T>& pW,
                              std::vector<T>& pRSVgpu,
                              std::vector<T>& pRSVcpu,
                              const std::vector<int>& pBS,
                              const int pHS,
                              const int pBN,
                              const int pS,
                              const int pNL,
                              const int pBM,
                              const int pDM,
                              const int pIM,
                              const int pVL,
                              const size_t pHXZ,
                              const bool pnohx        = false,
                              const bool pnocx        = false,
                              const bool pnodhy       = false,
                              const bool pnodcy       = false,
                              const bool pnodhx       = false,
                              const bool pnodcx       = false,
                              const bool puse_dropout = false)
        : yin(py),
          dy(pdy),
          dhy(pdhy),
          dcy(pdcy),
          initHidden(phx),
          initCell(pcx),
          weights(pW),
          batch_seq(pBS),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          nocx(pnocx),
          nodhy(pnodhy),
          nodcy(pnodcy),
          nodhx(pnodhx),
          nodcx(pnodcx),
          use_dropout(puse_dropout),
          RSVgpu(pRSVgpu.begin()),
          RSVcpu(pRSVcpu.begin())
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);

        if(!nocx)
            initCell = pcx; // this may be intentionally a nullptr
        else
            initCell.resize(realHiddenSize);

        if(!nodhy)
            dhy = pdhy; // this may be intentionally a nullptr
        else
            dhy.resize(realHiddenSize);

        if(!nodcy)
            dcy = pdcy; // this may be intentionally a nullptr
        else
            dcy.resize(realHiddenSize);
    }

    void LSTMBwdDataCPUVerify(bool use_dropout_cpu,
                              const miopen::DropoutDescriptor& dropoutDesc,
                              std::vector<T>& din_host,
                              const std::vector<T>& wei,
                              const std::vector<T>& dhy,
                              std::vector<T>& dhx_host,
                              const std::vector<T>& hx,
                              const std::vector<T>& dcy,
                              std::vector<T>& dcx_host,
                              const std::vector<T>& cx,
                              const std::vector<T>& out,
                              const std::vector<T>& dout,
                              const std::vector<int>& in_n,
                              int in_h,
                              int seqLength_cpu,
                              int bidirection,
                              int,
                              int hy_d,
                              int hy_n,
                              int hy_h,
                              int out_h,
                              int inputMode_cpu,
                              std::vector<T>& rsvspace,
                              std::vector<T>& wkspace,
                              bool cx_is_null  = false,
                              bool dhy_is_null = false,
                              bool dcy_is_null = false) const;

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> cpu() const;
    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> gpu() const;

    void fail(int badtensor) const
    {

        std::cout << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                std::cout << batch_seq.at(i) << ",";
            }
            else
            {
                std::cout << batch_seq.at(i);
            }
        }
        std::cout << " -m lstm "
                  << " -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen << " -l "
                  << nLayers << " -F 0 "
                  << " -r " << dirMode << " -b " << biasMode << " -p " << inputMode << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        std::cout << "Backward Data LSTM: " << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Output dx failed verification." << std::endl; break;
        case(1): std::cout << "Hidden state dhx tensor failed verification." << std::endl; break;
        case(2): std::cout << "Hidden cell dcx tensor failed verification." << std::endl; break;
        case(3): std::cout << "Workspace space tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};
//~~~~~~~~~~~~ END BACKWARD DATA ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS WEIGHTS
//****************************************************
template <class T>
struct verify_backward_weights_lstm
{
    std::vector<T> input;      // Y
    std::vector<T> dy;         // dY
    std::vector<T> initHidden; // HX
    std::vector<T> workSpace;
    std::vector<int> batch_seq;
    int weightSize;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode;
    int inputMode;
    int batch_n;
    int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    size_t realHiddenSize;
    bool nohx;
    bool use_dropout;
    typename std::vector<T> reserveSpace_gpu;
    typename std::vector<T> reserveSpace_cpu;

    verify_backward_weights_lstm(miopenRNNDescriptor_t pRD,
                                 const std::vector<T>& px,
                                 const std::vector<T>& pdy,
                                 const std::vector<T>& phx,
                                 const std::vector<T>& pRSVgpu,
                                 const std::vector<T>& pRSVcpu,
                                 const std::vector<T>& pWS,
                                 const std::vector<int>& pBS,
                                 const int pHS,
                                 const int pW,
                                 const int pBN,
                                 const int pS,
                                 const int pNL,
                                 const int pBM,
                                 const int pDM,
                                 const int pIM,
                                 const int pVL,
                                 const size_t pHXZ,
                                 const bool pnohx        = false,
                                 const bool puse_dropout = false)
        : input(px),
          dy(pdy),
          initHidden(phx),
          workSpace(pWS),
          batch_seq(pBS),
          weightSize(pW),
          hiddenSize(pHS),
          seqLength(pS),
          nLayers(pNL),
          biasMode(pBM),
          dirMode(pDM),
          inputMode(pIM),
          batch_n(pBN),
          inputVecLen(pVL),
          rnnDesc(pRD),
          realHiddenSize(pHXZ),
          nohx(pnohx),
          use_dropout(puse_dropout),
          reserveSpace_gpu(pRSVgpu),
          reserveSpace_cpu(pRSVcpu)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);
    }

    void LSTMBwdWeightCPUVerify(bool use_dropout_cpu,
                                const std::vector<T>& in,
                                std::vector<T>& dwei_host,
                                const std::vector<T>& hx,
                                const std::vector<T>& dout,
                                const std::vector<int>& in_n,
                                int in_h,
                                int seqLength_cpu,
                                int bidirection,
                                int biased,
                                int hy_d,
                                int hy_n,
                                int hy_h,
                                int out_h,
                                int inputMode_cpu,
                                const std::vector<T>& rsvspace,
                                const std::vector<T>& wkspace,
                                bool hx_is_null = false) const;

    std::vector<T> cpu() const;
    std::vector<T> gpu() const;

    void fail(int) const
    {
        std::cout << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                std::cout << batch_seq.at(i) << ",";
            }
            else
            {
                std::cout << batch_seq.at(i);
            }
        }
        std::cout << " -m lstm "
                  << " -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen << " -l "
                  << nLayers << " -F 0 "
                  << " -r " << dirMode << " -b " << biasMode << " -p " << inputMode << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        std::cout << "Backward Weights LSTM: " << std::endl;
    }
};
//~~~~~~~~~~~~ END BACKWARD WEIGHTS ~~~~~~~~~~~~~~~~~~~~~~~~

/**********************************************
 * CPU verification functions
 *
 **********************************************/

template <class T>
void verify_forward_lstm<T>::LSTMFwdCPUVerify(
    miopen::Handle& handle,
    bool use_dropout,
    const miopen::DropoutDescriptor& dropoutDesc,
    const std::vector<T>& in,
    const std::vector<T>& wei, // [ input_state_weight_trans
                               // hidden_state_weight0_trans input1_trans
                               // hidden1_trans ... output_weight;
                               // bidirectional reversed weights ]
    std::vector<T>& hy_host,   // current/final hidden state
    const std::vector<T>& hx,  // initial hidden state
    std::vector<T>& cy_host,   // current/final cell state
    const std::vector<T>& cx,  // initial cell state
    std::vector<T>& out_host,
    const std::vector<int>& in_n, // input batch size
    int in_h,                     // input data length
    int seqLength_cpu,            // Number of iterations to unroll over
    int bidirection,              // whether using bidirectional net
    int biased,                   // whether using bias
    int hy_d,                     // 1 by numlayer (number of stacks of hidden layers) for
                                  // unidirection, 2 by numlayer for bidirection
    int hy_n,                     // equal to input batch size in_n[0]
    int hy_h,                     // hidden state number
    int out_h,                    // 1 by hy_h related function for unidirection, 2 by hy_h
                                  // related function for bidirection
    int inputMode_cpu,
    std::vector<T>& rsvspace,
    bool hx_is_null,
    bool cx_is_null) const
{
    int batch_n_cpu = sumvc(in_n);

    int numlayer = bidirection == 1 ? hy_d / 2 : hy_d;
    int bi       = bidirection == 1 ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    if(inputMode_cpu == 1)
    {
        if(in_h != hy_h)
        {
            std::cout
                << "Verification cannot be completed: The input tensor size must equal to the "
                << "hidden state size of the network in SKIP_INPUT mode!" << std::endl;
            return;
        }
        in_h = 0;
    }

    int wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;

    // initial dropoput
    std::vector<prngStates> dropout_states_host;
    std::vector<unsigned char> dropout_reservespace_host;
    std::vector<T> dropout_hid_state;
    miopenTensorDescriptor_t dropout_inputTensor{}, dropout_outputTensor{};
    if(use_dropout)
    {
        size_t states_size  = dropoutDesc.stateSizeInBytes / sizeof(prngStates);
        dropout_states_host = std::vector<prngStates>(states_size);
        InitKernelStateEmulator(dropout_states_host, dropoutDesc);

        std::array<int, 2> drop_in_len  = {{batch_n_cpu, hy_h * bi}};
        std::array<int, 2> drop_in_str  = {{hy_stride, 1}};
        std::array<int, 2> drop_out_str = {{hy_h * bi, 1}};
        miopenCreateTensorDescriptor(&dropout_inputTensor);
        miopenCreateTensorDescriptor(&dropout_outputTensor);
        miopenSetTensorDescriptor(
            dropout_inputTensor, miopenFloat, 2, drop_in_len.data(), drop_in_str.data());
        miopenSetTensorDescriptor(
            dropout_outputTensor, miopenFloat, 2, drop_in_len.data(), drop_out_str.data());

        size_t reserveSpaceSizeInBytes = 0;
        miopenDropoutGetReserveSpaceSize(dropout_inputTensor, &reserveSpaceSizeInBytes);
        size_t reserve_size       = reserveSpaceSizeInBytes / sizeof(unsigned char);
        dropout_reservespace_host = std::vector<unsigned char>(reserve_size * (numlayer - 1),
                                                               static_cast<unsigned char>(1));

        dropout_hid_state =
            std::vector<T>((numlayer - 1) * batch_n_cpu * hy_h * bi, static_cast<T>(0));
    }

    // forward emulator
    for(int li = 0; li < numlayer; li++)
    {
        int hid_shift = li * batch_n_cpu * hy_stride;
        int hx_shift  = li * in_n.at(0) * h_stride;

        // from input
        if(li == 0)
        {
            if(inputMode_cpu == 1)
            {
                for(int bs = 0; bs < batch_n_cpu; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        for(int gi = 0; gi < 4; gi++)
                        {
                            rsvspace.at(hid_shift + bs * hy_stride + gi * hy_h + h) +=
                                in.at(bs * in_stride + h);
                            if(bidirection == 1)
                            {
                                rsvspace.at(hid_shift + bs * hy_stride + (gi + 4) * hy_h + h) +=
                                    in.at(bs * in_stride + h);
                            }
                        }
                    }
                }

                // from bias
                if(biased == 1)
                {
                    for(int bs = 0; bs < batch_n_cpu; bs++)
                    {
                        for(int h = 0; h < wei_stride; h++)
                        {
                            rsvspace.at(hid_shift + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias + h);
                        }
                    }
                }
            }
            else
            {
                RNN_mm_cpu<T>(in.data(),
                              in_h,
                              batch_n_cpu,
                              in_stride,
                              0,
                              wei.data(),
                              in_h,
                              hy_h * bi * 4,
                              in_stride,
                              RNN_MM_TRANSPOSE,
                              &rsvspace[hid_shift],
                              hy_h * bi * 4,
                              batch_n_cpu,
                              hy_stride,
                              0,
                              1,
                              1);

                // from bias
                if(biased == 1)
                {
                    for(int bs = 0; bs < batch_n_cpu; bs++)
                    {
                        for(int h = 0; h < wei_stride; h++)
                        {
                            rsvspace.at(hid_shift + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias + h);
                        }
                    }
                }
            }
        }
        else
        {
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;
            int prelayer_shift = (li - 1) * batch_n_cpu * hy_stride + bi * 5 * hy_h;
            if(use_dropout)
            {
                auto dropout_states_tmp = dropout_states_host;
                size_t drop_out_offset  = (li - 1) * batch_n_cpu * hy_h * bi;

                DropoutForwardVerify<T>(handle,
                                        dropoutDesc,
                                        miopen::deref(dropout_inputTensor),
                                        rsvspace,
                                        miopen::deref(dropout_outputTensor),
                                        dropout_hid_state,
                                        dropout_reservespace_host,
                                        dropout_states_tmp,
                                        prelayer_shift,
                                        drop_out_offset,
                                        drop_out_offset);

                prelayer_shift = drop_out_offset;
            }

            RNN_mm_cpu<T>(use_dropout ? &dropout_hid_state[prelayer_shift]
                                      : &rsvspace[prelayer_shift],
                          hy_h * bi,
                          batch_n_cpu,
                          use_dropout ? hy_h * bi : hy_stride,
                          0,
                          &wei[wei_shift],
                          hy_h * bi,
                          hy_h * bi * 4,
                          bi_stride,
                          RNN_MM_TRANSPOSE,
                          &rsvspace[hid_shift],
                          hy_h * bi * 4,
                          batch_n_cpu,
                          hy_stride,
                          0,
                          1,
                          1);

            // from bias
            if(biased == 1)
            {
                int wei_shift_bias_temp = wei_shift_bias + li * 2 * wei_stride;

                for(int bs = 0; bs < batch_n_cpu; bs++)
                {
                    for(int h = 0; h < wei_stride; h++)
                    {
                        rsvspace.at(hid_shift + bs * hy_stride + h) +=
                            wei.at(wei_shift_bias_temp + h);
                    }
                }
            }
        }

        // from hidden state
        int bacc   = 0;
        int baccbi = batch_n_cpu;
        for(int ti = 0; ti < seqLength_cpu; ti++)
        {
            baccbi -= in_n.at(seqLength_cpu - 1 - ti);
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    RNN_mm_cpu<T>(&hx[hx_shift],
                                  hy_h,
                                  in_n.at(ti),
                                  uni_stride,
                                  0,
                                  &wei[wei_shift],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[hid_shift + bacc * hy_stride],
                                  hy_h * 4,
                                  in_n.at(ti),
                                  hy_stride,
                                  0,
                                  1,
                                  1);

                    // from bias
                    if(biased == 1)
                    {
                        int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                        for(int bs = 0; bs < in_n.at(ti); bs++)
                        {
                            for(int h = 0; h < 4 * hy_h; h++)
                            {
                                rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                    wei.at(wei_shift_bias_temp + h);
                            }
                        }
                    }

                    if(bidirection == 1)
                    {
                        RNN_mm_cpu<T>(&hx[hx_shift + hy_n * hy_h],
                                      hy_h,
                                      in_n.at(seqLength_cpu - 1 - ti),
                                      uni_stride,
                                      0,
                                      &wei[wei_shift + 4 * hy_h * uni_stride],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      RNN_MM_TRANSPOSE,
                                      &rsvspace[hid_shift + baccbi * hy_stride + 4 * hy_h],
                                      hy_h * 4,
                                      in_n.at(seqLength_cpu - 1 - ti),
                                      hy_stride,
                                      0,
                                      1,
                                      1);

                        // from bias
                        if(biased == 1)
                        {
                            int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                            for(int bs = 0; bs < in_n.at(seqLength_cpu - 1 - ti); bs++)
                            {
                                for(int h = 0; h < 4 * hy_h; h++)
                                {
                                    rsvspace.at(hid_shift + baccbi * hy_stride + 4 * hy_h +
                                                bs * hy_stride + h) +=
                                        wei.at(wei_shift_bias_temp + 4 * hy_h + h);
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                RNN_mm_cpu<T>(&hy_host[hx_shift],
                              hy_h,
                              in_n.at(ti),
                              uni_stride,
                              0,
                              &wei[wei_shift],
                              hy_h,
                              hy_h * 4,
                              uni_stride,
                              RNN_MM_TRANSPOSE,
                              &rsvspace[hid_shift + bacc * hy_stride],
                              hy_h * 4,
                              in_n.at(ti),
                              hy_stride,
                              0,
                              1,
                              1);

                // from bias
                if(biased == 1)
                {
                    int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                    for(int bs = 0; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < 4 * hy_h; h++)
                        {
                            rsvspace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) +=
                                wei.at(wei_shift_bias_temp + h);
                        }
                    }
                }

                if(bidirection == 1)
                {

                    if(!hx_is_null && in_n.at(seqLength_cpu - 1 - ti) > in_n.at(seqLength_cpu - ti))
                    {
                        RNN_mm_cpu<T>(
                            &hx[hx_shift + hy_n * hy_h + in_n.at(seqLength_cpu - ti) * hy_h],
                            hy_h,
                            (in_n.at(seqLength_cpu - 1 - ti) - in_n.at(seqLength_cpu - ti)),
                            uni_stride,
                            0,
                            &wei[wei_shift + 4 * hy_h * uni_stride],
                            hy_h,
                            hy_h * 4,
                            uni_stride,
                            RNN_MM_TRANSPOSE,
                            &rsvspace[hid_shift +
                                      (baccbi + in_n.at(seqLength_cpu - ti)) * hy_stride +
                                      4 * hy_h],
                            hy_h * 4,
                            (in_n.at(seqLength_cpu - 1 - ti) - in_n.at(seqLength_cpu - ti)),
                            hy_stride,
                            0,
                            1,
                            1);

                        // from bias
                        if(biased == 1)
                        {
                            int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                            for(int bs = in_n.at(seqLength_cpu - ti);
                                bs < in_n.at(seqLength_cpu - 1 - ti);
                                bs++)
                            {
                                for(int h = 0; h < 4 * hy_h; h++)
                                {
                                    rsvspace.at(hid_shift + baccbi * hy_stride + 4 * hy_h +
                                                bs * hy_stride + h) +=
                                        wei.at(wei_shift_bias_temp + 4 * hy_h + h);
                                }
                            }
                        }
                    }

                    RNN_mm_cpu<T>(&hy_host[hx_shift + hy_n * hy_h],
                                  hy_h,
                                  in_n.at(seqLength_cpu - ti),
                                  uni_stride,
                                  0,
                                  &wei[wei_shift + 4 * hy_h * uni_stride],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[hid_shift + baccbi * hy_stride + 4 * hy_h],
                                  hy_h * 4,
                                  in_n.at(seqLength_cpu - ti),
                                  hy_stride,
                                  0,
                                  1,
                                  1);

                    // from bias
                    if(biased == 1)
                    {
                        int wei_shift_bias_temp = wei_shift_bias + (li * 2 + 1) * wei_stride;

                        for(int bs = 0; bs < in_n.at(seqLength_cpu - ti); bs++)
                        {
                            for(int h = 0; h < 4 * hy_h; h++)
                            {
                                rsvspace.at(hid_shift + baccbi * hy_stride + 4 * hy_h +
                                            bs * hy_stride + h) +=
                                    wei.at(wei_shift_bias_temp + 4 * hy_h + h);
                            }
                        }
                    }
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h), 2) *
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h),
                                  1);
                    if(ti == 0)
                    {
                        if(!cx_is_null)
                        {
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                activfunc(
                                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h),
                                    2) *
                                cx.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                    else
                    {
                        int prec_shift = li * batch_n_cpu * hy_stride +
                                         (bacc - in_n.at(ti - 1)) * hy_stride + bi * 4 * hy_h;

                        rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                            activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h),
                                      2) *
                            rsvspace.at(prec_shift + bs * hy_stride + h);
                    }

                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) +=
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h),
                                  2) *
                        activfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1);

                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h), 2);
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h), 2);
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h),
                                  2);
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h),
                                  1);
                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h +
                                numlayer * batch_n_cpu * hy_stride) =
                        activfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1);

                    cy_host.at(hx_shift + bs * uni_stride + h) =
                        rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h);
                    hy_host.at(hx_shift + bs * uni_stride + h) =
                        rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h);
                }
            }

            if(bidirection == 1)
            {
                for(int bs = 0; bs < in_n.at(seqLength_cpu - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                    h) +=
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2) *
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1);
                        if(ti == 0)
                        {
                            if(!cx_is_null)
                            {
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                            hy_h + h) +=
                                    activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                          5 * hy_h + h),
                                              2) *
                                    cx.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                            }
                        }
                        else
                        {

                            if(!cx_is_null &&
                               in_n.at(seqLength_cpu - 1 - ti) > in_n.at(seqLength_cpu - ti))
                            {
                                if(bs >= in_n.at(seqLength_cpu - ti))
                                {
                                    rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                bi * 4 * hy_h + hy_h + h) +=
                                        activfunc(rsvspace.at(hid_shift +
                                                              (baccbi + bs) * hy_stride + 5 * hy_h +
                                                              h),
                                                  2) *
                                        cx.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                                }
                            }

                            if(bs < in_n.at(seqLength_cpu - ti))
                            {
                                int prec_shift =
                                    li * batch_n_cpu * hy_stride +
                                    (baccbi + in_n.at(seqLength_cpu - 1 - ti)) * hy_stride +
                                    bi * 4 * hy_h + hy_h;

                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                            hy_h + h) +=
                                    activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                          5 * hy_h + h),
                                              2) *
                                    rsvspace.at(prec_shift + bs * hy_stride + h);
                            }
                        }

                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h +
                                    h) +=
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h),
                                2) *
                            activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                  bi * 4 * hy_h + hy_h + h),
                                      1);

                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h +
                                    numlayer * batch_n_cpu * hy_stride) =
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2);
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h +
                                    numlayer * batch_n_cpu * hy_stride) =
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h),
                                2);
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h +
                                    numlayer * batch_n_cpu * hy_stride) =
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h),
                                2);
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h +
                                    numlayer * batch_n_cpu * hy_stride) =
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1);
                        rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                    h + numlayer * batch_n_cpu * hy_stride) =
                            activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                  bi * 4 * hy_h + hy_h + h),
                                      1);

                        cy_host.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) = rsvspace.at(
                            hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h + h);
                        hy_host.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) = rsvspace.at(
                            hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h + hy_h + h);
                    }
                }
            }

            bacc += in_n.at(ti);
        }
    }

    // output
    int prelayer_shift = (numlayer - 1) * batch_n_cpu * hy_stride + bi * 5 * hy_h;

    for(int bs = 0; bs < batch_n_cpu; bs++)
    {
        for(int h = 0; h < out_h; h++)
        {
            out_host.at(bs * out_stride + h) = rsvspace.at(prelayer_shift + bs * hy_stride + h);
        }
    }

    if(use_dropout)
    {
        for(int i = 0; i < (numlayer - 1) * batch_n_cpu * hy_h * bi; i++)
        {
            rsvspace.at(numlayer * batch_n_cpu * hy_stride * 2 + i) = dropout_hid_state.at(i);
        }
        auto p_drop_rsv = reinterpret_cast<unsigned char*>(&rsvspace.at(
            numlayer * batch_n_cpu * hy_stride * 2 + (numlayer - 1) * batch_n_cpu * hy_h * bi));
        for(int i = 0; i < (numlayer - 1) * batch_n_cpu * hy_h * bi; i++)
        {
            *(p_drop_rsv + i) = dropout_reservespace_host.at(i);
        }
    }
}

template <class T>
void verify_backward_data_lstm<T>::LSTMBwdDataCPUVerify(
    bool use_dropout_cpu,
    const miopen::DropoutDescriptor& dropoutDesc,
    std::vector<T>& din_host,
    const std::vector<T>& wei,     // [ input_state_weight_trans
                                   // hidden_state_weight0_trans input1_trans
                                   // hidden1_trans ... output_weight;
                                   // bidirectional reversed weights ]
    const std::vector<T>& dhy_cpu, // current/final hidden state
    std::vector<T>& dhx_host,
    const std::vector<T>& hx,      // initial hidden state
    const std::vector<T>& dcy_cpu, // current/final cell state
    std::vector<T>& dcx_host,
    const std::vector<T>& cx,
    const std::vector<T>& out,
    const std::vector<T>& dout,
    const std::vector<int>& in_n, // input batch size
    int in_h,                     // input data length
    int seqLength_cpu,            // Number of iterations to unroll over
    int bidirection,              // whether using bidirectional net
    int,                          // whether using bias
    int hy_d,                     // 1 by numlayer (number of stacks of hidden layers)
                                  // for unidirection, 2 by numlayer for bidirection
    int hy_n,                     // equal to input batch size in_n[0]
    int hy_h,                     // hidden state number
    int out_h,                    // 1 by hy_h related function for unidirection, 2 by
                                  // hy_h related function for bidirection
    int inputMode_cpu,
    std::vector<T>& rsvspace,
    std::vector<T>& wkspace,
    bool cx_is_null,
    bool dhy_is_null,
    bool dcy_is_null) const
{
    int batch_n_cpu = sumvc(in_n);
    (void)out;
    (void)hx;

    int numlayer = bidirection == 1 ? hy_d / 2 : hy_d;
    int bi       = bidirection == 1 ? 2 : 1;

    int in_stride  = in_h;
    int out_stride = out_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;

    if(inputMode_cpu == 1)
    {
        if(in_h != hy_h)
        {
            std::cout
                << "Verification cannot be completed: The input tensor size must equal to the "
                << "hidden state size of the network in SKIP_INPUT mode!" << std::endl;
            return;
        }
        in_h = 0;
    }

    // initial dropoput
    miopenTensorDescriptor_t dropout_inputTensor{};
    std::vector<unsigned char> dropout_reservespace_host;
    if(use_dropout_cpu)
    {
        std::array<int, 2> drop_in_len = {{batch_n_cpu, hy_h * bi}};
        std::array<int, 2> drop_in_str = {{hy_stride, 1}};
        miopenCreateTensorDescriptor(&dropout_inputTensor);
        miopenSetTensorDescriptor(
            dropout_inputTensor, miopenFloat, 2, drop_in_len.data(), drop_in_str.data());

        size_t reserveSpaceSizeInBytes = 0;
        miopenDropoutGetReserveSpaceSize(dropout_inputTensor, &reserveSpaceSizeInBytes);
        size_t reserve_size       = reserveSpaceSizeInBytes / sizeof(unsigned char);
        dropout_reservespace_host = std::vector<unsigned char>(reserve_size * (numlayer - 1),
                                                               static_cast<unsigned char>(0));

        auto p_drop_rsv = reinterpret_cast<unsigned char*>(&rsvspace.at(
            numlayer * batch_n_cpu * hy_stride * 2 + (numlayer - 1) * batch_n_cpu * hy_h * bi));
        for(int i = 0; i < (numlayer - 1) * batch_n_cpu * hy_h * bi; i++)
        {
            dropout_reservespace_host.at(i) = *(p_drop_rsv + i);
        }
    }

    // bwd data emulator
    for(int li = numlayer - 1; li >= 0; li--)
    {
        int wei_shift = (in_h + hy_h) * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
        int hid_shift = li * batch_n_cpu * hy_stride;
        int hx_shift  = li * in_n.at(0) * h_stride;

        if(li == numlayer - 1)
        {
            for(int bs = 0; bs < batch_n_cpu; bs++)
            {
                for(int h = 0; h < out_h; h++)
                {
                    wkspace.at(hid_shift + bi * 5 * hy_h + bs * hy_stride + h) +=
                        dout.at(bs * out_stride + h);
                }
            }
        }
        else
        {
            int prelayer_shift = (li + 1) * batch_n_cpu * hy_stride;

            RNN_mm_cpu<T>(&wkspace[prelayer_shift],
                          hy_h * bi * 4,
                          batch_n_cpu,
                          hy_stride,
                          0,
                          &wei[wei_shift],
                          hy_h * bi,
                          hy_h * bi * 4,
                          bi_stride,
                          0,
                          &wkspace[hid_shift + bi * 5 * hy_h],
                          hy_h * bi,
                          batch_n_cpu,
                          hy_stride,
                          0,
                          1,
                          1);

            if(use_dropout_cpu)
            {
                DropoutBackwardVerify<T>(dropoutDesc,
                                         miopen::deref(dropout_inputTensor),
                                         wkspace,
                                         miopen::deref(dropout_inputTensor),
                                         wkspace,
                                         dropout_reservespace_host,
                                         hid_shift + bi * 5 * hy_h,
                                         hid_shift + bi * 5 * hy_h,
                                         li * batch_n_cpu * hy_h * bi);
            }
        }

        // from hidden state
        int bacc   = batch_n_cpu;
        int baccbi = 0;
        for(int ti = seqLength_cpu - 1; ti >= 0; ti--)
        {
            bacc -= in_n.at(ti);

            if(ti == seqLength_cpu - 1)
            {
                for(int bs = 0; bs < in_n.at(ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        if(!dhy_is_null)
                        {
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) +=
                                dhy_cpu.at(hx_shift + bs * uni_stride + h);
                        }
                        if(!dcy_is_null)
                        {
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                dcy_cpu.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                }

                if(bidirection == 1)
                {
                    for(int bs = 0; bs < in_n.at(seqLength_cpu - 1 - ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            if(!dhy_is_null)
                            {
                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h +
                                           hy_h + h) +=
                                    dhy_cpu.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                            }
                            if(!dcy_is_null)
                            {
                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                           hy_h + h) +=
                                    dcy_cpu.at(hx_shift + bs * uni_stride + hy_n * hy_h + h);
                            }
                        }
                    }
                }
            }
            else
            {
                if(!dhy_is_null && in_n.at(ti) > in_n.at(ti + 1))
                {
                    for(int bs = in_n.at(ti + 1); bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) +=
                                dhy_cpu.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                }

                if(!dcy_is_null && in_n.at(ti) > in_n.at(ti + 1))
                {
                    for(int bs = in_n.at(ti + 1); bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                dcy_cpu.at(hx_shift + bs * uni_stride + h);
                        }
                    }
                }

                int pretime_shift = li * batch_n_cpu * hy_stride + (bacc + in_n.at(ti)) * hy_stride;
                int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

                RNN_mm_cpu<T>(&wkspace[pretime_shift],
                              hy_h * 4,
                              in_n.at(ti + 1),
                              hy_stride,
                              0,
                              &wei[weitime_shift],
                              hy_h,
                              hy_h * 4,
                              uni_stride,
                              0,
                              &wkspace[hid_shift + bacc * hy_stride + bi * 5 * hy_h],
                              hy_h,
                              in_n.at(ti + 1),
                              hy_stride,
                              0,
                              1,
                              1);

                if(bidirection == 1)
                {
                    pretime_shift = li * batch_n_cpu * hy_stride +
                                    (baccbi - in_n.at(seqLength_cpu - 2 - ti)) * hy_stride +
                                    hy_h * 4;
                    weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride +
                                    hy_h * 4 * uni_stride;

                    RNN_mm_cpu<T>(&wkspace[pretime_shift],
                                  hy_h * 4,
                                  in_n.at(seqLength_cpu - 1 - ti),
                                  hy_stride,
                                  0,
                                  &wei[weitime_shift],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  &wkspace[hid_shift + baccbi * hy_stride + bi * 5 * hy_h + hy_h],
                                  hy_h,
                                  in_n.at(seqLength_cpu - 1 - ti),
                                  hy_stride,
                                  0,
                                  1,
                                  1);
                }
            }

            for(int bs = 0; bs < in_n.at(ti); bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    if(ti < seqLength_cpu - 1)
                    {
                        if(bs < in_n.at(ti + 1))
                        {
                            int pretime_shift =
                                li * batch_n_cpu * hy_stride + (bacc + in_n.at(ti)) * hy_stride;

                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                                wkspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h + h) *
                                activfunc(rsvspace.at(pretime_shift + bs * hy_stride + hy_h + h),
                                          2);
                        }
                    }
                    wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) +=
                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) *
                        dervactivfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1) *
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h),
                                  2);

                    if(ti == 0)
                    {
                        if(!cx_is_null)
                        {
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h) +=
                                wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h +
                                           h) *
                                cx.at(hx_shift + bs * uni_stride + h) *
                                dervactivfunc(
                                    rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h), 2);
                        }
                    }
                    else
                    {
                        int pretime_shift =
                            li * batch_n_cpu * hy_stride + (bacc - in_n.at(ti - 1)) * hy_stride;

                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h) +=
                            wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) *
                            rsvspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h + h) *
                            dervactivfunc(
                                rsvspace.at(hid_shift + (bacc + bs) * hy_stride + hy_h + h), 2);
                    }
                    wkspace.at(hid_shift + (bacc + bs) * hy_stride + h) +=
                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) *
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h),
                                  1) *
                        dervactivfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h), 2);
                    wkspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h) +=
                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 5 * hy_h + h) *
                        activfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h),
                            1) *
                        dervactivfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 2 * hy_h + h), 2);
                    wkspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h) +=
                        wkspace.at(hid_shift + (bacc + bs) * hy_stride + bi * 4 * hy_h + h) *
                        activfunc(rsvspace.at(hid_shift + (bacc + bs) * hy_stride + h), 2) *
                        dervactivfunc(
                            rsvspace.at(hid_shift + (bacc + bs) * hy_stride + 3 * hy_h + h), 1);
                }
            }

            if(bidirection == 1)
            {
                for(int bs = 0; bs < in_n.at(seqLength_cpu - 1 - ti); bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        if(ti < seqLength_cpu - 1)
                        {
                            int pretime_shift =
                                li * batch_n_cpu * hy_stride +
                                (baccbi - in_n.at(seqLength_cpu - 2 - ti)) * hy_stride;

                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                       hy_h + h) +=
                                wkspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h + hy_h +
                                           h) *
                                activfunc(
                                    rsvspace.at(pretime_shift + bs * hy_stride + 5 * hy_h + h), 2);
                        }
                        wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h + hy_h +
                                   h) +=
                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h +
                                       hy_h + h) *
                            dervactivfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                      bi * 4 * hy_h + hy_h + h),
                                          1) *
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h),
                                2);

                        if(ti == 0)
                        {
                            if(!cx_is_null)
                            {
                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h) +=
                                    wkspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                               bi * 4 * hy_h + hy_h + h) *
                                    cx.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) *
                                    dervactivfunc(rsvspace.at(hid_shift +
                                                              (baccbi + bs) * hy_stride + 5 * hy_h +
                                                              h),
                                                  2);
                            }
                        }
                        else
                        {
                            if(!cx_is_null &&
                               in_n.at(seqLength_cpu - 1 - ti) > in_n.at(seqLength_cpu - ti) &&
                               bs >= in_n.at(seqLength_cpu - ti))
                            {
                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h) +=
                                    wkspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                               bi * 4 * hy_h + hy_h + h) *
                                    cx.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) *
                                    dervactivfunc(rsvspace.at(hid_shift +
                                                              (baccbi + bs) * hy_stride + 5 * hy_h +
                                                              h),
                                                  2);
                            }

                            if(bs < in_n.at(seqLength_cpu - ti))
                            {
                                int pretime_shift =
                                    li * batch_n_cpu * hy_stride +
                                    (baccbi + in_n.at(seqLength_cpu - 1 - ti)) * hy_stride;

                                wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 5 * hy_h + h) +=
                                    wkspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                               bi * 4 * hy_h + hy_h + h) *
                                    rsvspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h +
                                                hy_h + h) *
                                    dervactivfunc(rsvspace.at(hid_shift +
                                                              (baccbi + bs) * hy_stride + 5 * hy_h +
                                                              h),
                                                  2);
                            }
                        }
                        wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h) +=
                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                       hy_h + h) *
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1) *
                            dervactivfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2);
                        wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h) +=
                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 5 * hy_h +
                                       hy_h + h) *
                            activfunc(rsvspace.at(hid_shift + (baccbi + bs) * hy_stride +
                                                  bi * 4 * hy_h + hy_h + h),
                                      1) *
                            dervactivfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 6 * hy_h + h),
                                2);
                        wkspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h) +=
                            wkspace.at(hid_shift + (baccbi + bs) * hy_stride + bi * 4 * hy_h +
                                       hy_h + h) *
                            activfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 4 * hy_h + h),
                                2) *
                            dervactivfunc(
                                rsvspace.at(hid_shift + (baccbi + bs) * hy_stride + 7 * hy_h + h),
                                1);
                    }
                }
            }

            baccbi += in_n.at(seqLength_cpu - 1 - ti);
        }

        // dcx, dhx
        int pretime_shift = li * batch_n_cpu * hy_stride;
        int weitime_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;

        RNN_mm_cpu<T>(&wkspace[pretime_shift],
                      hy_h * 4,
                      in_n.at(0),
                      hy_stride,
                      0,
                      &wei[weitime_shift],
                      hy_h,
                      hy_h * 4,
                      uni_stride,
                      0,
                      &dhx_host[hx_shift],
                      hy_h,
                      in_n.at(0),
                      uni_stride,
                      0,
                      1,
                      1);

        for(int bs = 0; bs < in_n.at(0); bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                dcx_host.at(hx_shift + bs * uni_stride + h) +=
                    wkspace.at(pretime_shift + bs * hy_stride + bi * 4 * hy_h + h) *
                    activfunc(rsvspace.at(pretime_shift + bs * hy_stride + hy_h + h), 2);
            }
        }

        if(bidirection == 1)
        {
            int ti = seqLength_cpu - 1, cur_bat = 0, pre_bat = batch_n_cpu;

            while(ti >= 0)
            {
                pre_bat -= in_n.at(ti);
                if(in_n.at(ti) > cur_bat)
                {
                    pretime_shift = li * batch_n_cpu * hy_stride + (pre_bat + cur_bat) * hy_stride;

                    RNN_mm_cpu<T>(&wkspace[pretime_shift + 4 * hy_h],
                                  hy_h * 4,
                                  (in_n.at(ti) - cur_bat),
                                  hy_stride,
                                  0,
                                  &wei[weitime_shift + 4 * hy_h * uni_stride],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  &dhx_host[hx_shift + hy_n * hy_h + cur_bat * hy_h],
                                  hy_h,
                                  (in_n.at(ti) - cur_bat),
                                  uni_stride,
                                  0,
                                  1,
                                  1);

                    for(int bs = cur_bat; bs < in_n.at(ti); bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            dcx_host.at(hx_shift + bs * uni_stride + hy_n * hy_h + h) +=
                                wkspace.at(pretime_shift + (bs - cur_bat) * hy_stride +
                                           bi * 4 * hy_h + hy_h + h) *
                                activfunc(rsvspace.at(pretime_shift + (bs - cur_bat) * hy_stride +
                                                      5 * hy_h + h),
                                          2);
                        }
                    }
                }
                cur_bat = in_n.at(ti--);
            }
        }
    }

    // dinput
    if(inputMode_cpu == 1)
    {
        for(int bs = 0; bs < batch_n_cpu; bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                for(int gi = 0; gi < 4; gi++)
                {
                    din_host.at(bs * in_stride + h) += wkspace.at(bs * hy_stride + gi * hy_h + h);
                    if(bidirection == 1)
                    {
                        din_host.at(bs * in_stride + h) +=
                            wkspace.at(bs * hy_stride + (gi + 4) * hy_h + h);
                    }
                }
            }
        }
    }
    else
    {
        RNN_mm_cpu<T>(wkspace.data(),
                      hy_h * bi * 4,
                      batch_n_cpu,
                      hy_stride,
                      0,
                      wei.data(),
                      in_h,
                      hy_h * bi * 4,
                      in_stride,
                      0,
                      din_host.data(),
                      in_h,
                      batch_n_cpu,
                      in_stride,
                      0,
                      1,
                      1);
    }
}

template <class T>
void verify_backward_weights_lstm<T>::LSTMBwdWeightCPUVerify(
    bool use_dropout_cpu,
    const std::vector<T>& in,
    std::vector<T>& dwei_host, // [ input_state_weight_trans
                               // hidden_state_weight0_trans
                               // input1_trans hidden1_trans ...
                               // output_weight; bidirectional
                               // reversed weights ]
    const std::vector<T>& hx,  // initial hidden state
    const std::vector<T>& dout,
    const std::vector<int>& in_n, // input batch size
    int in_h,                     // input data length
    int seqLength_cpu,            // Number of iterations to unroll over
    int bidirection,              // whether using bidirectional net
    int biased,                   // whether using bias
    int hy_d,                     // 1 by numlayer (number of stacks of hidden
                                  // layers) for unidirection, 2 by numlayer for
                                  // bidirection
    int hy_n,                     // equal to input batch size in_n[0]
    int hy_h,                     // hidden state number
    int out_h,                    // 1 by hy_h related function for unidirection, 2
                                  // by hy_h related function for bidirection
    int inputMode_cpu,
    const std::vector<T>& rsvspace,
    const std::vector<T>& wkspace,
    bool hx_is_null) const
{
    int batch_n_cpu = sumvc(in_n);
    int numlayer    = bidirection == 1 ? hy_d / 2 : hy_d;
    int bi          = bidirection == 1 ? 2 : 1;

    int in_stride  = in_h;
    int wei_stride = bi * 4 * hy_h;
    int hy_stride  = bi * 6 * hy_h;
    int h_stride   = bi * hy_h;
    int uni_stride = hy_h;
    int bi_stride  = hy_h * bi;
    (void)dout;
    (void)out_h;

    if(inputMode_cpu == 1)
    {
        if(in_h != hy_h)
        {
            std::cout
                << "Verification cannot be completed: The input tensor size must equal to the "
                << "hidden state size of the network in SKIP_INPUT mode!" << std::endl;
            return;
        }
        in_h = 0;
    }

    int wei_shift_bias = (in_h + hy_h + (bi * hy_h + hy_h) * (numlayer - 1)) * wei_stride;

    // bwd weights emulator
    for(int li = 0; li < numlayer; li++)
    {
        // between layers
        if(li == 0)
        {
            if(inputMode_cpu != 1)
            {
                RNN_mm_cpu<T>(wkspace.data(),
                              hy_h * bi * 4,
                              batch_n_cpu,
                              hy_stride,
                              RNN_MM_TRANSPOSE,
                              in.data(),
                              in_h,
                              batch_n_cpu,
                              in_stride,
                              0,
                              dwei_host.data(),
                              in_h,
                              hy_h * bi * 4,
                              in_stride,
                              0,
                              1,
                              1);
            }

            if(biased == 1)
            {
                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n_cpu; w++)
                    {
                        dwei_host.at(wei_shift_bias + h) += wkspace.at(w * hy_stride + h);
                    }
                }
            }
        }
        else
        {
            int prelayer_shift =
                use_dropout_cpu
                    ? 2 * numlayer * batch_n_cpu * hy_stride + (li - 1) * batch_n_cpu * hy_h * bi
                    : (li - 1) * batch_n_cpu * hy_stride + bi * hy_h * 5;
            int hid_shift = li * batch_n_cpu * hy_stride;
            int wei_shift = (in_h + hy_h) * wei_stride + (li - 1) * (bi * hy_h + hy_h) * wei_stride;

            RNN_mm_cpu<T>(&wkspace[hid_shift],
                          hy_h * bi * 4,
                          batch_n_cpu,
                          hy_stride,
                          RNN_MM_TRANSPOSE,
                          &rsvspace[prelayer_shift],
                          hy_h * bi,
                          batch_n_cpu,
                          use_dropout_cpu ? hy_h * bi : hy_stride,
                          0,
                          &dwei_host[wei_shift],
                          hy_h * bi,
                          hy_h * bi * 4,
                          bi_stride,
                          0,
                          1,
                          1);

            if(biased == 1)
            {
                wei_shift = wei_shift_bias + li * 2 * wei_stride;

                for(int h = 0; h < wei_stride; h++)
                {
                    for(int w = 0; w < batch_n_cpu; w++)
                    {
                        dwei_host.at(wei_shift + h) += wkspace.at(hid_shift + w * hy_stride + h);
                    }
                }
            }
        }

        // between time
        int bacc = 0;
        for(int ti = 0; ti < seqLength_cpu; ti++)
        {
            int hid_shift = li * batch_n_cpu * hy_stride + bacc * hy_stride;
            int hx_shift  = li * in_n.at(0) * h_stride;
            int wei_shift = in_h * wei_stride + li * (bi * hy_h + hy_h) * wei_stride;
            int pretime_shift;

            // between time
            if(ti == 0)
            {
                if(!hx_is_null)
                {
                    RNN_mm_cpu<T>(&wkspace[hid_shift],
                                  hy_h * 4,
                                  in_n.at(ti),
                                  hy_stride,
                                  RNN_MM_TRANSPOSE,
                                  &hx[hx_shift],
                                  hy_h,
                                  in_n.at(ti),
                                  uni_stride,
                                  0,
                                  &dwei_host[wei_shift],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  1,
                                  1);

                    if(biased == 1)
                    {
                        int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 4; h++)
                        {
                            for(int w = 0; w < in_n.at(ti); w++)
                            {
                                dwei_host.at(bias_shift + h) +=
                                    wkspace.at(hid_shift + w * hy_stride + h);
                            }
                        }
                    }
                }
            }
            else
            {
                pretime_shift = li * batch_n_cpu * hy_stride +
                                (bacc - in_n.at(ti - 1)) * hy_stride + bi * 5 * hy_h;

                RNN_mm_cpu<T>(&wkspace[hid_shift],
                              hy_h * 4,
                              in_n.at(ti),
                              hy_stride,
                              RNN_MM_TRANSPOSE,
                              &rsvspace[pretime_shift],
                              hy_h,
                              in_n.at(ti),
                              hy_stride,
                              0,
                              &dwei_host[wei_shift],
                              hy_h,
                              hy_h * 4,
                              uni_stride,
                              0,
                              1,
                              1);

                if(biased == 1)
                {
                    int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                    for(int h = 0; h < hy_h * 4; h++)
                    {
                        for(int w = 0; w < in_n.at(ti); w++)
                        {
                            dwei_host.at(bias_shift + h) +=
                                wkspace.at(hid_shift + w * hy_stride + h);
                        }
                    }
                }
            }

            if(bidirection == 1)
            {
                if(ti == seqLength_cpu - 1)
                {
                    if(!hx_is_null)
                    {
                        RNN_mm_cpu<T>(&wkspace[hid_shift + 4 * hy_h],
                                      hy_h * 4,
                                      in_n.at(ti),
                                      hy_stride,
                                      RNN_MM_TRANSPOSE,
                                      &hx[hx_shift + hy_n * hy_h],
                                      hy_h,
                                      in_n.at(ti),
                                      uni_stride,
                                      0,
                                      &dwei_host[wei_shift + 4 * hy_h * uni_stride],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      0,
                                      1,
                                      1);

                        if(biased == 1)
                        {
                            int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 4; h++)
                            {
                                for(int w = 0; w < in_n.at(ti); w++)
                                {
                                    dwei_host.at(bias_shift + hy_h * 4 + h) +=
                                        wkspace.at(hid_shift + hy_h * 4 + w * hy_stride + h);
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(!hx_is_null && in_n.at(ti) > in_n.at(ti + 1))
                    {
                        RNN_mm_cpu<T>(&wkspace[hid_shift + 4 * hy_h + in_n.at(ti + 1) * hy_stride],
                                      hy_h * 4,
                                      (in_n.at(ti) - in_n.at(ti + 1)),
                                      hy_stride,
                                      RNN_MM_TRANSPOSE,
                                      &hx[hx_shift + hy_n * hy_h + in_n.at(ti + 1) * hy_h],
                                      hy_h,
                                      (in_n.at(ti) - in_n.at(ti + 1)),
                                      uni_stride,
                                      0,
                                      &dwei_host[wei_shift + 4 * hy_h * uni_stride],
                                      hy_h,
                                      hy_h * 4,
                                      uni_stride,
                                      0,
                                      1,
                                      1);

                        if(biased == 1)
                        {
                            int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                            for(int h = 0; h < hy_h * 4; h++)
                            {
                                for(int w = in_n.at(ti + 1); w < in_n.at(ti); w++)
                                {
                                    dwei_host.at(bias_shift + hy_h * 4 + h) +=
                                        wkspace.at(hid_shift + hy_h * 4 + w * hy_stride + h);
                                }
                            }
                        }
                    }

                    pretime_shift = li * batch_n_cpu * hy_stride +
                                    (bacc + in_n.at(ti)) * hy_stride + bi * 5 * hy_h;

                    RNN_mm_cpu<T>(&wkspace[hid_shift + 4 * hy_h],
                                  hy_h * 4,
                                  in_n.at(ti + 1),
                                  hy_stride,
                                  RNN_MM_TRANSPOSE,
                                  &rsvspace[pretime_shift + hy_h],
                                  hy_h,
                                  in_n.at(ti + 1),
                                  hy_stride,
                                  0,
                                  &dwei_host[wei_shift + 4 * hy_h * uni_stride],
                                  hy_h,
                                  hy_h * 4,
                                  uni_stride,
                                  0,
                                  1,
                                  1);

                    if(biased == 1)
                    {
                        int bias_shift = wei_shift_bias + li * 2 * wei_stride + wei_stride;

                        for(int h = 0; h < hy_h * 4; h++)
                        {
                            for(int w = 0; w < in_n.at(ti + 1); w++)
                            {
                                dwei_host.at(bias_shift + hy_h * 4 + h) +=
                                    wkspace.at(hid_shift + hy_h * 4 + w * hy_stride + h);
                            }
                        }
                    }
                }
            }

            bacc += in_n.at(ti);
        }
    }
}
//////=========END CPU VERIFICATION FUNCTIONS=============

//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T>
struct verify_forward_infer_lstm : verify_forward_lstm<T>
{
    using verify_forward_lstm<T>::input;
    using verify_forward_lstm<T>::initHidden;
    using verify_forward_lstm<T>::initCell;
    using verify_forward_lstm<T>::weights;
    using verify_forward_lstm<T>::batch_seq;
    using verify_forward_lstm<T>::hiddenSize;
    using verify_forward_lstm<T>::seqLength;
    using verify_forward_lstm<T>::nLayers;
    using verify_forward_lstm<T>::biasMode;
    using verify_forward_lstm<T>::dirMode;
    using verify_forward_lstm<T>::inputMode;
    using verify_forward_lstm<T>::batch_n;
    using verify_forward_lstm<T>::inputVecLen;
    using verify_forward_lstm<T>::rnnDesc;
    using verify_forward_lstm<T>::realHiddenSize;
    using verify_forward_lstm<T>::nohx;
    using verify_forward_lstm<T>::nocx;
    using verify_forward_lstm<T>::nohy;
    using verify_forward_lstm<T>::nocy;

    verify_forward_infer_lstm(miopenRNNDescriptor_t pRD,
                              const std::vector<T>& px,
                              const std::vector<T>& phx,
                              const std::vector<T>& pcx,
                              const std::vector<T>& pW,
                              const std::vector<int>& pBS,
                              const int pHS,
                              const int pBN,
                              const int pS,
                              const int pNL,
                              const int pBM,
                              const int pDM,
                              const int pIM,
                              const int pVL,
                              const size_t pHXZ,
                              const bool pnohx = false,
                              const bool pnocx = false,
                              const bool pnohy = false,
                              const bool pnocy = false)
    {
        input          = px;
        initHidden     = phx;
        initCell       = pcx;
        weights        = pW;
        batch_seq      = pBS;
        hiddenSize     = pHS;
        seqLength      = pS;
        nLayers        = pNL;
        biasMode       = pBM;
        dirMode        = pDM;
        inputMode      = pIM;
        batch_n        = pBN;
        inputVecLen    = pVL;
        rnnDesc        = pRD;
        realHiddenSize = pHXZ;
        nohx           = pnohx;
        nocx           = pnocx;
        nohy           = pnohy;
        nocy           = pnocy;

        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);

        if(!nocx)
            initCell = pcx; // this may be intentionally a nullptr
        else
            initCell.resize(realHiddenSize);
    }

    std::vector<T> cpu() const
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = dirMode != 0 ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t out_sz = 0;

        size_t reserveSpaceSize;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        if(miopen::deref(rnnDesc).algoMode == miopenRNNdefault)
        {
            reserveSpaceSize /= sizeof(T);
            reserveSpaceSize -=
                nLayers * std::accumulate(batch_seq.begin(), batch_seq.begin() + seqLength, 0) *
                hiddenSize * bi;
            reserveSpaceSize *= 2;
            reserveSpaceSize *= sizeof(T);
        }
        std::vector<T> reserveSpace(reserveSpaceSize / sizeof(T));
        std::vector<T> output(out_sz / sizeof(T));
        std::vector<T> hiddenState(initHidden.size());
        std::vector<T> cellState(initCell.size());

        this->LSTMFwdCPUVerify(handle,
                               false,
                               miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                               input,
                               weights,     // [ input_state_weight_trans
                                            // hidden_state_weight0_trans input1_trans
                                            // hidden1_trans ... output_weight;
                                            // bidirectional reversed weights ]
                               hiddenState, // current/final hidden state
                               initHidden,  // initial hidden state
                               cellState,   // current/final cell state
                               initCell,    // initial cell state
                               output,
                               batch_seq,       // input batch size
                               inputVecLen,     // input data length
                               seqLength,       // Number of iterations to unroll over
                               dirMode,         // whether using bidirectional net
                               biasMode,        // whether using bias
                               bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers)
                                                // for unidirection, 2 by numlayer for bidirection
                               batch_seq.at(0), // equal to input batch size in_n[0]
                               hiddenSize,      // hidden state number
                               bi_stride, // 1 by hy_h related function for unidirection, 2 by hy_h
                                          // related function for bidirection
                               inputMode,
                               reserveSpace,
                               nohx,
                               nocx);

#if(MIO_LSTM_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            printf("CPU outdata[%d]: %f\n", i, output[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward inference LSTM pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_LSTM_TEST_DEBUG > 0)
        std::cout << "Done with LSTM forward inference CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return output;
    }

    std::vector<T> gpu() const
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        size_t out_sz        = 0;
        size_t workSpaceSize = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);

        std::vector<T> workSpace(workSpaceSize / sizeof(T));

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T));
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);
        auto hy          = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto cy = initCell;
        std::fill(cy.begin(), cy.end(), 0.);

        auto workSpace_dev = handle.Write(workSpace);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens.data(), 3);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen.data(), 1);

        miopenRNNForwardInference(&handle,
                                  rnnDesc,
                                  seqLength,
                                  inputDescs.data(),
                                  input_dev.get(),
                                  &hiddenDesc,
                                  ((nohx) ? nullptr : handle.Write(initHidden).get()),
                                  &hiddenDesc,
                                  ((nocx) ? nullptr : handle.Write(initCell).get()),
                                  &weightDesc,
                                  weights_dev.get(),
                                  outputDescs.data(),
                                  output_dev.get(),
                                  &hiddenDesc,
                                  ((nohy) ? nullptr : handle.Write(hy).get()),
                                  &hiddenDesc,
                                  ((nocy) ? nullptr : handle.Write(cy).get()),
                                  workSpace_dev.get(),
                                  workSpaceSize);

#if(MIO_LSTM_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n", i, outdata[i]);
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer LSTM pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_LSTM_TEST_DEBUG > 0)
        std::cout << "Done with LSTM forward inference GPU" << std::endl;
#endif
        return (handle.Read<T>(output_dev, output.size()));
    }

    void fail(int) const
    {
        std::cout << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                std::cout << batch_seq.at(i) << ",";
            }
            else
            {
                std::cout << batch_seq.at(i);
            }
        }
        std::cout << " -m lstm -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen
                  << " -l " << nLayers << " -F 0 -r " << dirMode << " -b " << biasMode << " -p "
                  << inputMode << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers << std::endl;
        std::cout << "Forward Inference LSTM: " << std::endl;
        std::cout << "Output tensor output failed verification." << std::endl;
    }
};
//~~~~~~~~~~~~ END FWD INFERENCE ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_lstm : verify_forward_lstm<T>
{
    using verify_forward_lstm<T>::input;
    using verify_forward_lstm<T>::initHidden;
    using verify_forward_lstm<T>::initCell;
    using verify_forward_lstm<T>::weights;
    using verify_forward_lstm<T>::batch_seq;
    using verify_forward_lstm<T>::hiddenSize;
    using verify_forward_lstm<T>::seqLength;
    using verify_forward_lstm<T>::nLayers;
    using verify_forward_lstm<T>::biasMode;
    using verify_forward_lstm<T>::dirMode;
    using verify_forward_lstm<T>::inputMode;
    using verify_forward_lstm<T>::batch_n;
    using verify_forward_lstm<T>::inputVecLen;
    using verify_forward_lstm<T>::rnnDesc;
    using verify_forward_lstm<T>::realHiddenSize;
    using verify_forward_lstm<T>::nohx;
    using verify_forward_lstm<T>::nocx;
    using verify_forward_lstm<T>::nohy;
    using verify_forward_lstm<T>::nocy;

    bool use_dropout;
    typename std::vector<T>::iterator RSVgpu;
    typename std::vector<T>::iterator RSVcpu;

    verify_forward_train_lstm(miopenRNNDescriptor_t pRD,
                              const std::vector<T>& px,
                              const std::vector<T>& phx,
                              const std::vector<T>& pcx,
                              const std::vector<T>& pW,
                              const std::vector<int>& pBS,
                              std::vector<T>& pRSVgpu,
                              std::vector<T>& pRSVcpu,
                              const int pHS,
                              const int pBN,
                              const int pS,
                              const int pNL,
                              const int pBM,
                              const int pDM,
                              const int pIM,
                              const int pVL,
                              const size_t pHXZ,
                              const bool pnohx        = false,
                              const bool pnocx        = false,
                              const bool pnohy        = false,
                              const bool pnocy        = false,
                              const bool puse_dropout = false)
        : RSVgpu(pRSVgpu.begin()), RSVcpu(pRSVcpu.begin())
    {
        input          = px;
        initHidden     = phx;
        initCell       = pcx;
        weights        = pW;
        batch_seq      = pBS;
        hiddenSize     = pHS;
        seqLength      = pS;
        nLayers        = pNL;
        biasMode       = pBM;
        dirMode        = pDM;
        inputMode      = pIM;
        batch_n        = pBN;
        inputVecLen    = pVL;
        rnnDesc        = pRD;
        realHiddenSize = pHXZ;
        nohx           = pnohx;
        nocx           = pnocx;
        nohy           = pnohy;
        nocy           = pnocy;
        use_dropout    = puse_dropout;

        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);

        if(!nocx)
            initCell = pcx; // this may
        else
            initCell.resize(realHiddenSize);
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> cpu() const
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        int bi        = dirMode != 0 ? 2 : 1;
        int hy_h      = hiddenSize;
        int bi_stride = bi * hy_h;
        size_t out_sz = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);

        size_t inputBatchLenSum =
            std::accumulate(batch_seq.begin(), batch_seq.begin() + seqLength, 0);
        size_t reserveSpaceSize;
        reserveSpaceSize = 2 * 6 * miopen::deref(rnnDesc).nLayers * inputBatchLenSum * hiddenSize *
                           ((dirMode != 0) ? 2 : 1);
        if(use_dropout)
        {
            reserveSpaceSize += (miopen::deref(rnnDesc).nLayers - 1) * inputBatchLenSum *
                                hiddenSize * ((dirMode != 0) ? 2 : 1);
            reserveSpaceSize *= sizeof(T);
            reserveSpaceSize += (miopen::deref(rnnDesc).nLayers - 1) * inputBatchLenSum *
                                hiddenSize * ((dirMode != 0) ? 2 : 1);
            reserveSpaceSize = (reserveSpaceSize + sizeof(T) - 1) / sizeof(T);
        }

        std::vector<T> reserveSpace(reserveSpaceSize);
        std::vector<T> output(out_sz / sizeof(T));
        std::vector<T> hiddenState(initHidden.size());
        std::vector<T> cellState(initCell.size());

        this->LSTMFwdCPUVerify(handle,
                               use_dropout,
                               miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                               input,
                               weights,     // [ input_state_weight_trans
                                            // hidden_state_weight0_trans input1_trans
                                            // hidden1_trans ... output_weight;
                                            // bidirectional reversed weights ]
                               hiddenState, // current/final hidden state
                               initHidden,  // initial hidden state
                               cellState,   // current/final cell state
                               initCell,    // initial cell state
                               output,
                               batch_seq,       // input batch size
                               inputVecLen,     // input data length
                               seqLength,       // Number of iterations to unroll over
                               dirMode,         // whether using bidirectional net
                               biasMode,        // whether using bias
                               bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers)
                                                // for unidirection, 2 by numlayer for bidirection
                               batch_seq.at(0), // equal to input batch size in_n[0]
                               hiddenSize,      // hidden state number
                               bi_stride, // 1 by hy_h related function for unidirection, 2 by hy_h
                                          // related function for bidirection
                               inputMode,
                               reserveSpace,
                               nohx,
                               nocx);

#if(MIO_LSTM_TEST_DEBUG == 2)
        for(int i = 0; i < output.size(); i++)
        {
            std::cout << "CPU outdata[" << i << "]: " << output[i] << std::endl;
        }
#endif

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward train LSTM pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        std::copy(reserveSpace.begin(), reserveSpace.end(), RSVcpu);
        auto retSet = std::make_tuple(
            output, (nohy ? initHidden : hiddenState), (nocy ? initCell : cellState));

#if(MIO_LSTM_TEST_DEBUG > 0)
        std::cout << "Done with LSTM forward train CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
#endif
        return retSet;
    }

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> gpu() const
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        size_t out_sz           = 0;
        size_t workSpaceSize    = 0;
        size_t reserveSpaceSize = 0;

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);

        std::vector<T> workSpace(workSpaceSize / sizeof(T));
        std::vector<T> reserveSpace((reserveSpaceSize + sizeof(T) - 1) / sizeof(T));

        auto input_dev = handle.Write(input);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz / sizeof(T));
        auto output_dev = handle.Write(output);

        auto weights_dev = handle.Write(weights);
        auto hy          = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);
        auto cy     = initCell;
        std::fill(cy.begin(), cy.end(), 0.);
        auto cy_dev = handle.Write(cy);

        auto workSpace_dev    = handle.Write(workSpace);
        auto reserveSpace_dev = handle.Write(reserveSpace);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens.data(), 3);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen.data(), 1);

        miopenRNNForwardTraining(&handle,
                                 rnnDesc,
                                 seqLength,
                                 inputDescs.data(),
                                 input_dev.get(),
                                 &hiddenDesc,
                                 ((nohx) ? nullptr : handle.Write(initHidden).get()),
                                 &hiddenDesc,
                                 ((nocx) ? nullptr : handle.Write(initCell).get()),
                                 &weightDesc,
                                 weights_dev.get(),
                                 outputDescs.data(),
                                 output_dev.get(),
                                 &hiddenDesc,
                                 ((nohy) ? nullptr : hy_dev.get()),
                                 &hiddenDesc,
                                 ((nocy) ? nullptr : cy_dev.get()),
                                 workSpace_dev.get(),
                                 workSpaceSize,
                                 reserveSpace_dev.get(),
                                 reserveSpaceSize);

#if(MIO_LSTM_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n", i, outdata[i]);
        }
#endif
        reserveSpace =
            handle.Read<T>(reserveSpace_dev, (reserveSpaceSize + sizeof(T) - 1) / sizeof(T));
        std::copy(reserveSpace.begin(), reserveSpace.end(), RSVgpu);
        auto retSet = std::make_tuple(handle.Read<T>(output_dev, output.size()),
                                      (nohy ? initHidden : handle.Read<T>(hy_dev, hy.size())),
                                      (nocy ? initCell : handle.Read<T>(cy_dev, cy.size())));

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train LSTM pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
#if(MIO_LSTM_TEST_DEBUG > 0)
        std::cout << "Done with RNN forward train GPU" << std::endl;
#endif
        return retSet;
    }

    void fail(int badtensor) const
    {

        std::cout << "./bin/MIOpenDriver rnn -n ";
        for(int i = 0; i < seqLength; i++)
        {
            if(i < seqLength - 1)
            {
                std::cout << batch_seq.at(i) << ",";
            }
            else
            {
                std::cout << batch_seq.at(i);
            }
        }
        std::cout << " -m lstm "
                  << " -k " << seqLength << " -H " << hiddenSize << " -W " << inputVecLen << " -l "
                  << nLayers << " -F 0 "
                  << " -r " << dirMode << " -b " << biasMode << " -p " << inputMode << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        std::cout << "Forward Train LSTM: " << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Hidden state tensor failed verification." << std::endl; break;
        case(2): std::cout << "Cell state tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};
//~~~~~~~~~~~~ END FWD TRAIN ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS DATA CPU & GPU
//****************************************************
template <class T>
std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>>
verify_backward_data_lstm<T>::cpu() const
{

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    auto&& handle = get_handle();

    int bi        = dirMode != 0 ? 2 : 1;
    int hy_h      = hiddenSize;
    int bi_stride = bi * hy_h;
    size_t workSpaceSize;

    std::vector<miopen::TensorDescriptor> inputCPPDescs;
    std::vector<miopenTensorDescriptor_t> inputDescs;
    createTensorDescArray(
        inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

    // Outputs ----------
    size_t in_sz = 0;
    miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
    miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
    std::vector<T> workSpace(workSpaceSize / sizeof(T));
    std::vector<T> dx(in_sz / sizeof(T));
    std::vector<T> dhx(initHidden.size());
    std::vector<T> dcx(initHidden.size());

    size_t inputBatchLenSum = std::accumulate(batch_seq.begin(), batch_seq.begin() + seqLength, 0);
    size_t reserveSpaceSize;
    reserveSpaceSize = 2 * 6 * miopen::deref(rnnDesc).nLayers * inputBatchLenSum * hiddenSize *
                       ((dirMode != 0) ? 2 : 1);
    if(use_dropout)
    {
        reserveSpaceSize += (miopen::deref(rnnDesc).nLayers - 1) * inputBatchLenSum * hiddenSize *
                            ((dirMode != 0) ? 2 : 1);
        reserveSpaceSize *= sizeof(T);
        reserveSpaceSize += (miopen::deref(rnnDesc).nLayers - 1) * inputBatchLenSum * hiddenSize *
                            ((dirMode != 0) ? 2 : 1);
        reserveSpaceSize = (reserveSpaceSize + sizeof(T) - 1) / sizeof(T);
    }

    std::vector<T> reserveSpace(reserveSpaceSize);
    std::copy(RSVcpu, RSVcpu + reserveSpaceSize, reserveSpace.begin());

    this->LSTMBwdDataCPUVerify(use_dropout,
                               miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                               dx,              // DX (output)
                               weights,         // [ input_state_weight_trans
                                                //   hidden_state_weight0_trans input1_trans
                                                //   hidden1_trans ... output_weight;
                                                //   bidirectional reversed weights ]
                               dhy,             // current/final hidden state
                               dhx,             // DHX (output)
                               initHidden,      // HX initial hidden state
                               dcy,             // DCY current/final cell state
                               dcx,             // DCX (output)
                               initCell,        // CX
                               yin,             // Y
                               dy,              // DY
                               batch_seq,       // input batch size
                               inputVecLen,     // input data length
                               seqLength,       // Number of iterations to unroll over
                               dirMode,         // whether using bidirectional net
                               biasMode,        // whether using bias
                               bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers)
                                                // for unidirection, 2 by numlayer for bidirection
                               batch_seq.at(0), // equal to input batch size in_n[0]
                               hiddenSize,      // hidden state number
                               bi_stride,       // 1 by hy_h related function for unidirection, 2 by
                               // hy_h related function for bidirection
                               inputMode,
                               reserveSpace,
                               workSpace,
                               nocx,
                               nodhy,
                               nodcy);

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: CPU backward data LSTM pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif
    std::copy(reserveSpace.begin(), reserveSpace.end(), RSVcpu);
    auto retSet =
        std::make_tuple(dx, (nodhx ? initHidden : dhx), (nodcx ? initCell : dcx), workSpace);

#if(MIO_LSTM_TEST_DEBUG > 0)
    std::cout << "Done with LSTM backward data CPU" << std::endl;
    std::cout << "---------------------------------\n" << std::endl;
#endif
    return retSet;
}

template <class T>
std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>>
verify_backward_data_lstm<T>::gpu() const
{

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    auto&& handle = get_handle();

    size_t workSpaceSize = 0;

    std::vector<miopen::TensorDescriptor> inputCPPDescs;
    std::vector<miopenTensorDescriptor_t> inputDescs;
    createTensorDescArray(
        inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

    std::vector<miopen::TensorDescriptor> outputCPPDescs;
    std::vector<miopenTensorDescriptor_t> outputDescs;
    createTensorDescArray(outputCPPDescs,
                          outputDescs,
                          batch_seq,
                          hiddenSize * ((dirMode != 0) ? 2 : 1),
                          miopen::deref(rnnDesc).dataType);

    miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
    std::vector<T> workSpace(workSpaceSize / sizeof(T));
    auto workSpace_dev = handle.Write(workSpace);

    size_t reserveSpaceSize;
    miopenGetRNNTrainingReserveSize(
        &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
    std::vector<T> reserveSpace((reserveSpaceSize + sizeof(T) - 1) / sizeof(T));
    std::copy(RSVgpu, RSVgpu + reserveSpace.size(), reserveSpace.begin());

    auto yin_dev          = handle.Write(yin);
    auto dyin_dev         = handle.Write(dy);
    auto reserveSpace_dev = handle.Write(reserveSpace);
    auto weights_dev      = handle.Write(weights);

    std::vector<int> hlens(3, 0);
    hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
    hlens[1] = batch_seq[0];
    hlens[2] = hiddenSize;
    miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens.data(), 3);

    std::vector<int> wlen(1, 0);
    wlen[0] = weights.size();
    miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen.data(), 1);

    size_t in_sz = 0;
    miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
    std::vector<T> dx(in_sz / sizeof(T));
    auto dx_dev = handle.Write(dx);

    std::vector<T> dhx(initHidden.size());
    auto dhx_dev = handle.Write(dhx);

    std::vector<T> dcx(initHidden.size());
    auto dcx_dev = handle.Write(dcx);

    miopenRNNBackwardData(&handle,
                          rnnDesc,
                          seqLength,
                          outputDescs.data(),
                          yin_dev.get(),
                          outputDescs.data(),
                          dyin_dev.get(),
                          &hiddenDesc,
                          ((nodhy) ? nullptr : handle.Write(dhy).get()),
                          &hiddenDesc,
                          ((nodcy) ? nullptr : handle.Write(dcy).get()),
                          &weightDesc,
                          weights_dev.get(),
                          &hiddenDesc,
                          ((nohx) ? nullptr : handle.Write(initHidden).get()),
                          &hiddenDesc,
                          ((nocx) ? nullptr : handle.Write(initCell).get()),
                          inputDescs.data(),
                          dx_dev.get(),
                          &hiddenDesc,
                          ((nodhx) ? nullptr : dhx_dev.get()),
                          &hiddenDesc,
                          ((nodcx) ? nullptr : dcx_dev.get()),
                          workSpace_dev.get(),
                          workSpaceSize,
                          reserveSpace_dev.get(),
                          reserveSpace.size() * sizeof(T));

    reserveSpace = handle.Read<T>(reserveSpace_dev, reserveSpace.size());
    std::copy(reserveSpace.begin(), reserveSpace.end(), RSVgpu);
    auto retSet = std::make_tuple(handle.Read<T>(dx_dev, dx.size()),
                                  (nodhx ? initHidden : handle.Read<T>(dhx_dev, dhx.size())),
                                  (nodcx ? initCell : handle.Read<T>(dcx_dev, dcx.size())),
                                  handle.Read<T>(workSpace_dev, workSpace.size()));

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: GPU backward data LSTM pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif
#if(MIO_LSTM_TEST_DEBUG > 0)
    std::cout << "Done with LSTM backward data GPU" << std::endl;
#endif
    return retSet;
}
//~~~~~~~~~~~~ END BACKWARD DATA CPU & GPU ~~~~~~~~~~~~~~~~~~~~~~~~

//****************************************************
// BACKWARDS WEIGHTS CPU & GPU
//****************************************************
template <class T>
std::vector<T> verify_backward_weights_lstm<T>::cpu() const
{

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    int bi        = dirMode != 0 ? 2 : 1;
    int hy_h      = hiddenSize;
    int bi_stride = bi * hy_h;
    std::vector<T> dweights(weightSize);

    this->LSTMBwdWeightCPUVerify(use_dropout,
                                 input,
                                 dweights,   // (output) [ input_state_weight_trans
                                             // hidden_state_weight0_trans
                                             // input1_trans hidden1_trans ...
                                             // output_weight; bidirectional
                                             // reversed weights ]
                                 initHidden, // initial hidden state
                                 dy,
                                 batch_seq,       // input batch size
                                 inputVecLen,     // input data length
                                 seqLength,       // Number of iterations to unroll over
                                 dirMode,         // whether using bidirectional net
                                 biasMode,        // whether using bias
                                 bi * nLayers,    // 1 by numlayer (number of stacks of hidden
                                                  // layers) for unidirection, 2 by numlayer for
                                                  // bidirection
                                 batch_seq.at(0), // equal to input batch size in_n[0]
                                 hiddenSize,      // hidden state number
                                 bi_stride,       // 1 by hy_h related function for unidirection, 2
                                                  // by hy_h related function for bidirection
                                 inputMode,
                                 reserveSpace_cpu,
                                 workSpace,
                                 nohx);

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: CPU backward_weights LSTM pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif
#if(MIO_LSTM_TEST_DEBUG > 0)
    std::cout << "Done with LSTM backward weights CPU" << std::endl;
    std::cout << "---------------------------------\n" << std::endl;
#endif
    return dweights;
}

template <class T>
std::vector<T> verify_backward_weights_lstm<T>::gpu() const
{

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    auto&& handle = get_handle();

    std::vector<miopen::TensorDescriptor> inputCPPDescs;
    std::vector<miopenTensorDescriptor_t> inputDescs;
    createTensorDescArray(
        inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

    std::vector<miopen::TensorDescriptor> outputCPPDescs;
    std::vector<miopenTensorDescriptor_t> outputDescs;
    createTensorDescArray(outputCPPDescs,
                          outputDescs,
                          batch_seq,
                          hiddenSize * ((dirMode != 0) ? 2 : 1),
                          miopen::deref(rnnDesc).dataType);

    auto workSpace_dev    = handle.Write(workSpace);
    auto reserveSpace_dev = handle.Write(reserveSpace_gpu);
    std::vector<T> dweights(weightSize);
    auto dweights_dev = handle.Write(dweights);
    miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, &weightSize, 1);

    std::vector<int> hlens(3, 0);
    hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
    hlens[1] = batch_seq[0];
    hlens[2] = hiddenSize;
    miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens.data(), 3);
    auto dy_dev    = handle.Write(dy);
    auto input_dev = handle.Write(input);

    miopenRNNBackwardWeights(&handle,
                             rnnDesc,
                             seqLength,
                             inputDescs.data(),
                             input_dev.get(),
                             &hiddenDesc,
                             ((nohx) ? nullptr : handle.Write(initHidden).get()),
                             outputDescs.data(),
                             dy_dev.get(),
                             &weightDesc,
                             dweights_dev.get(),
                             workSpace_dev.get(),
                             workSpace.size() * sizeof(T),
                             reserveSpace_dev.get(),
                             reserveSpace_gpu.size() * sizeof(T));

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: GPU backwards_weights LSTM pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif
#if(MIO_LSTM_TEST_DEBUG > 0)
    std::cout << "Done with LSTM backward weights GPU" << std::endl;
#endif
    auto retvec = handle.Read<T>(dweights_dev, dweights.size());
    return retvec;
}
//~~~~~~~~~~~~ END BACKWARD WEIGHTS CPU & GPU ~~~~~~~~~~~~~~~~~~~~~~~~

//====================== DRIVER ============================
template <class T>
struct lstm_basic_driver : test_driver
{
    std::vector<int> batchSeq;
    int seqLength{};
    int inVecLen{};
    int hiddenSize{};
    int numLayers{};
    int inputMode{};
    int biasMode{};
    int dirMode{};
    int algoMode{};
    int batchSize{};
    int useDropout{};

    // Null pointer input
    bool nohx          = false;
    bool nodhy         = false;
    bool nocx          = false;
    bool nodcy         = false;
    bool nohy          = false;
    bool nodhx         = false;
    bool nocy          = false;
    bool nodcx         = false;
    bool flatBatchFill = false;

    lstm_basic_driver() {}

    void run()
    {

#if(MIOPEN_BACKEND_OPENCL == 1)
#if WORKAROUND_ISSUE_692 == 1
        std::cout << "Skip test for Issue #692: " << std::endl;
        exit(EXIT_SUCCESS); // NOLINT (concurrency-mt-unsafe)
#endif
        if(type == miopenHalf)
            exit(EXIT_SUCCESS); // NOLINT (concurrency-mt-unsafe)
#endif

        if(batchSeq.empty() || 0 == batchSeq[0])
        {
            std::cout << "Empty batch sequence. Filling uniformly with batch size: " << batchSize
                      << std::endl;
            if(flatBatchFill)
            {
                batchSeq.clear();
                batchSeq.resize(seqLength, batchSize);
            }
            else
            {
                batchSeq = generate_batchSeq(batchSize, seqLength)[0];
            }
        }

        if(batchSeq.size() != seqLength)
        {
            std::cerr << "FAILED: Batch sequence vector length, does not match sequence length."
                      << std::endl;
            std::abort();
        }

#if(MIO_LSTM_TEST_DEBUG == 2)
        for(int i = 0; i < seqLength; i++)
        {
            std::cout << "batch seq[" << i << "]: " << batchSeq.at(i) << std::endl;
        }
#endif

        auto&& handle = get_handle();

        int batch_n = std::accumulate(batchSeq.begin(), batchSeq.end(), 0);

        miopenRNNDescriptor_t rnnDesc;
        miopenCreateRNNDescriptor(&rnnDesc);
        miopenDropoutDescriptor_t DropoutDesc;
        miopenCreateDropoutDescriptor(&DropoutDesc);
        size_t statesSizeInBytes = 0;

        if(useDropout != 0)
        {
// Workaround for issue #2335.
// OpenCL error creating buffer: 0 Invalid Buffer Size
#if MIOPEN_BACKEND_OPENCL
            std::cout << "Skip test for Issue #2335: " << std::endl;
            return;
#endif
            miopenHandle_t mio_handle;
            miopenCreateWithStream(&mio_handle, handle.GetStream());

            float dropout_rate              = 0.5;
            unsigned long long dropout_seed = 0ULL;
            miopenDropoutGetStatesSize(mio_handle, &statesSizeInBytes);

#if MIOPEN_BACKEND_OPENCL
            cl_context ctx;
            clGetCommandQueueInfo(
                handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
            cl_mem dropout_state_buf =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, statesSizeInBytes, nullptr, nullptr);
#elif MIOPEN_BACKEND_HIP
            void* dropout_state_buf;
            hipMalloc(static_cast<void**>(&dropout_state_buf), statesSizeInBytes);
#endif

            miopenSetDropoutDescriptor(DropoutDesc,
                                       mio_handle,
                                       dropout_rate,
                                       dropout_state_buf,
                                       statesSizeInBytes,
                                       dropout_seed,
                                       false,
                                       false,
                                       MIOPEN_RNG_PSEUDO_XORWOW);

            miopenSetRNNDescriptor_V2(rnnDesc,
                                      hiddenSize,
                                      numLayers,
                                      DropoutDesc,
                                      miopenRNNInputMode_t(inputMode),
                                      miopenRNNDirectionMode_t(dirMode),
                                      miopenLSTM,
                                      miopenRNNBiasMode_t(biasMode),
                                      miopenRNNAlgo_t(algoMode),
                                      type);
        }
        else
        {
            miopenSetRNNDescriptor(rnnDesc,
                                   hiddenSize,
                                   numLayers,
                                   miopenRNNInputMode_t(inputMode),
                                   miopenRNNDirectionMode_t(dirMode),
                                   miopenLSTM,
                                   miopenRNNBiasMode_t(biasMode),
                                   miopenRNNAlgo_t(algoMode),
                                   type); // defined in superclass testdriver
        }

        // Create input tensor
        // If we are in skip mode, take the real input size to be the vector length.
        auto inVecReal    = (inputMode != 0) ? hiddenSize : inVecLen;
        std::size_t in_sz = inVecReal * batch_n;
        std::vector<T> input(in_sz);
        srand(0);
        for(std::size_t i = 0; i < in_sz; i++)
        {
            input[i] = /*(((GET_RAND()%2)==1)?-1:1)**/ 0.001 * float(GET_RAND() % 100);
        }

        std::size_t hx_sz = ((dirMode != 0) ? 2 : 1) * hiddenSize * batchSize * numLayers;
        std::vector<T> hx(hx_sz);
        std::vector<T> cx(hx_sz);
        std::vector<T> dhyin(hx_sz);
        std::vector<T> dcyin(hx_sz);

        size_t wei_bytes = 0;
        std::vector<int> inlens(2, 0);
        inlens.at(0) = batchSeq.at(0);
        inlens.at(1) = inVecReal;
        auto firstInputDesc =
            miopen::TensorDescriptor(miopen::deref(rnnDesc).dataType, inlens.data(), 2);
        miopenGetRNNParamsSize(
            &handle, rnnDesc, &firstInputDesc, &wei_bytes, miopen::deref(rnnDesc).dataType);
        auto wei_sz = int(wei_bytes / sizeof(T));
        std::vector<T> weights(wei_sz);
        for(std::size_t i = 0; i < wei_sz; i++)
        {
            weights[i] = (((GET_RAND() % 2) == 1) ? -1 : 1) * 0.001 * float(GET_RAND() % 100);
        }

#if(MIO_LSTM_TEST_DEBUG > 0)
        printf("inputMode: %d, biasMode: %d, dirMode: %d\n", inputMode, biasMode, dirMode);
        printf("hz: %d, batch_n: %d, seqLength: %d, inputLen: %d, numLayers: %d\n",
               hiddenSize,
               batch_n,
               seqLength,
               inVecLen,
               numLayers);
        std::cout << "nohx: " << nohx;
        std::cout << ", nocx: " << nocx;
        std::cout << ", nodhy: " << nodhy;
        std::cout << ", nodcy: " << nodcy << std::endl;
        std::cout << "nohy: " << nohy;
        std::cout << ", nocy: " << nocy;
        std::cout << ", nodhx: " << nodhx;
        std::cout << ", nodcx: " << nodcx << std::endl;
#endif

        if(!nohx)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                hx[i] = 0.001 * float(GET_RAND() % 100);
            }
        }

        if(!nodhy)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                dhyin[i] = 0.001 * float(GET_RAND() % 100);
            }
        }

        if(!nocx)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                cx[i] = 0.001 * float(GET_RAND() % 100);
            }
        }

        if(!nodcy)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                dcyin[i] = 0.001 * float(GET_RAND() % 100);
            }
        }

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batchSeq, inVecLen, miopen::deref(rnnDesc).dataType);
        size_t reserveSpaceSize;
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batchSeq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);
        size_t out_sz;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        size_t workSpaceSize;
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);

        size_t total_mem = statesSizeInBytes + reserveSpaceSize + workSpaceSize + 2 * out_sz +
                           (in_sz + wei_sz + (nohx ? 0 : hx_sz) + (nohy ? 0 : hx_sz) +
                            (nodhx ? 0 : hx_sz) + (nodhy ? 0 : hx_sz) + (nocx ? 0 : hx_sz) +
                            (nocy ? 0 : hx_sz) + (nodcx ? 0 : hx_sz) + (nodcy ? 0 : hx_sz)) *
                               sizeof(T);
        size_t device_mem = handle.GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
        }

        reserveSpaceSize = (reserveSpaceSize + sizeof(T) - 1) / sizeof(T);
        std::vector<T> rsvgpu(reserveSpaceSize, T(0));

        size_t inputBatchLenSum =
            std::accumulate(batchSeq.begin(), batchSeq.begin() + seqLength, 0);
        reserveSpaceSize =
            2 * 6 * numLayers * inputBatchLenSum * hiddenSize * ((dirMode != 0) ? 2 : 1);
        if(useDropout != 0)
        {
            reserveSpaceSize +=
                (numLayers - 1) * inputBatchLenSum * hiddenSize * ((dirMode != 0) ? 2 : 1);
            reserveSpaceSize *= sizeof(T);
            reserveSpaceSize +=
                (numLayers - 1) * inputBatchLenSum * hiddenSize * ((dirMode != 0) ? 2 : 1);
            reserveSpaceSize = (reserveSpaceSize + sizeof(T) - 1) / sizeof(T);
        }

        std::vector<T> rsvcpu(reserveSpaceSize, T(0));

        auto fwdTrainOutputPair = verify(verify_forward_train_lstm<T>{
            rnnDesc,         input,      hx,      cx,        weights,   batchSeq, rsvgpu,
            rsvcpu,          hiddenSize, batch_n, seqLength, numLayers, biasMode, dirMode,
            inputMode,       inVecReal,  hx_sz,   nohx,      nocx,      nohy,     nocy,
            bool(useDropout)});

        /// RETURNS std::make_tuple(output, hiddenState, cellState, reserveSpace);
        auto yin = std::get<0>(fwdTrainOutputPair.second);
        // auto curHiddenState = std::get<1>(fwdTrainOutputPair.second);
        // auto curCellState   = std::get<2>(fwdTrainOutputPair.second);

        std::vector<T> dyin(yin.size());
        for(std::size_t i = 0; i < yin.size(); i++)
        {
            dyin[i] = /*(((GET_RAND()%2)==1)?-1:1)**/ 0.001 * float(GET_RAND() % 100);
        }

#if(MIO_LSTM_TEST_DEBUG > 0)
        printf("Running backward data LSTM.\n");
#endif
        auto bwdDataOutputPair = verify(verify_backward_data_lstm<T>{
            rnnDesc,   yin,      dyin,    dhyin,     hx,         dcyin,           cx,
            weights,   rsvgpu,   rsvcpu,  batchSeq,  hiddenSize, batch_n,         seqLength,
            numLayers, biasMode, dirMode, inputMode, inVecReal,  hx_sz,           nohx,
            nocx,      nodhy,    nodcy,   nodhx,     nodcx,      bool(useDropout)});

        // RETURNS:  std::make_tuple(dx, dhx, dcx, reserveSpace, workSpace);
        auto workSpaceBwdData = std::get<3>(bwdDataOutputPair.second);

#if(MIO_LSTM_TEST_DEBUG > 0)
        printf("Running backward weights LSTM.\n");
        printf("reserve sz: %d, workSpace sz: %d, weight sz: %d\n",
               reserveSpaceBwdData.size(),
               workSpaceBwdData.size(),
               wei_sz);
        fflush(nullptr);
#endif
        // auto dweights_pair =
        verify(verify_backward_weights_lstm<T>{
            rnnDesc,  input,      dyin,      hx,      rsvgpu,    rsvcpu,          workSpaceBwdData,
            batchSeq, hiddenSize, wei_sz,    batch_n, seqLength, numLayers,       biasMode,
            dirMode,  inputMode,  inVecReal, hx_sz,   nohx,      bool(useDropout)});

/// \todo Resolve the issue and remove workaround.
/// ROCm3.3, Radeon VII: Many test cases always fail with:
/// "Forward Inference LSTM:"
/// "Output tensor output failed verification."
#if 0
        if(useDropout == 0)
        {
            verify(verify_forward_infer_lstm<T>{rnnDesc,
                                                input,
                                                hx,
                                                cx,
                                                weights,
                                                batchSeq,
                                                hiddenSize,
                                                batch_n,
                                                seqLength,
                                                numLayers,
                                                biasMode,
                                                dirMode,
                                                inputMode,
                                                inVecReal,
                                                hx_sz,
                                                nohx,
                                                nocx,
                                                nohy,
                                                nocy});
        }
#endif
        /* normal hx/cx/dhy/dcy input test end */

        // DLOWELL: Subtracting delta weights may produce NAN and infinities. Further investigation
        // is needed.
        //        auto dweights = std::get<1>(dweights_pair);
        //        std::transform(weightData.begin( ), weightData.end( ), dweights.begin( ),
        //        weightData.begin( ),std::minus<T>( ));
        //        verify(verify_forward_infer_lstm<T>{rnnDesc, inputData,
        //                                        curHiddenState, curCellState, weightData,
        //                                        batchSeq,
        //                                        hiddenSize, batch_n,
        //                                        seqLength, numLayers,
        //                                        biasMode, dirMode,
        //                                        inputMode, inVecReal});
    }
};

#endif
