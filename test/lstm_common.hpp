/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include "cpu_rnn.hpp"
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
    bool use_seqPadding{};
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
    bool use_seqPadding;
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
                              const bool pnohx           = false,
                              const bool pnocx           = false,
                              const bool pnodhy          = false,
                              const bool pnodcy          = false,
                              const bool pnodhx          = false,
                              const bool pnodcx          = false,
                              const bool puse_dropout    = false,
                              const bool puse_seqPadding = false)
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
          use_seqPadding(puse_seqPadding),
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
                  << " -r " << dirMode << " -b " << biasMode << " -p " << inputMode << " -q "
                  << use_seqPadding << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        std::cout << "Backward Data LSTM: " << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Output dx tensor report." << std::endl; break;
        case(1): std::cout << "Hidden state dhx tensor report." << std::endl; break;
        case(2): std::cout << "Hidden cell dcx tensor report." << std::endl; break;
        case(3): std::cout << "Workspace space tensor report." << std::endl; break;
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
    bool use_seqPadding;
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
                                 const bool pnohx           = false,
                                 const bool puse_dropout    = false,
                                 const bool puse_seqPadding = false)
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
          use_seqPadding(puse_seqPadding),
          reserveSpace_gpu(pRSVgpu),
          reserveSpace_cpu(pRSVcpu)
    {
        if(!nohx)
            initHidden = phx; // this may be intentionally a nullptr
        else
            initHidden.resize(realHiddenSize);
    }

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
                  << " -r " << dirMode << " -b " << biasMode << " -p " << inputMode << " -q "
                  << use_seqPadding << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        std::cout << "Backward Weights LSTM: " << std::endl;
    }
};
//~~~~~~~~~~~~ END BACKWARD WEIGHTS ~~~~~~~~~~~~~~~~~~~~~~~~

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
    using verify_forward_lstm<T>::use_seqPadding;

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
                              const bool pnohx           = false,
                              const bool pnocx           = false,
                              const bool pnohy           = false,
                              const bool pnocy           = false,
                              const bool puse_seqPadding = false)
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
        use_seqPadding = puse_seqPadding;

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

        LSTMFwdCPUVerify(handle,
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
                         bi_stride,       // 1 by hy_h related function for unidirection, 2 by hy_h
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
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

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
                  << inputMode << " -q " << use_seqPadding << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers << std::endl;
        std::cout << "Forward Inference LSTM: " << std::endl;
        std::cout << "Output tensor report." << std::endl;
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
    using verify_forward_lstm<T>::use_seqPadding;

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
                              const bool pnohx           = false,
                              const bool pnocx           = false,
                              const bool pnohy           = false,
                              const bool pnocy           = false,
                              const bool puse_dropout    = false,
                              const bool puse_seqPadding = false)
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
        use_seqPadding = puse_seqPadding;

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
        int out_h     = hiddenSize * ((dirMode != 0) ? 2 : 1);
        size_t out_sz = getSuperTensorSize(batch_seq,
                                           seqLength,
                                           inputVecLen,
                                           hiddenSize,
                                           batch_seq[0],
                                           dirMode != 0,
                                           false,
                                           use_seqPadding);

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(
            outputCPPDescs, outputDescs, batch_seq, out_h, miopen::deref(rnnDesc).dataType);

        size_t inputBatchLenSum =
            std::accumulate(batch_seq.begin(), batch_seq.begin() + seqLength, 0);

        size_t reserveSpaceSize;
        reserveSpaceSize = 2 * 6 * miopen::deref(rnnDesc).nLayers * inputBatchLenSum * out_h;

        if(use_dropout)
        {
            reserveSpaceSize += (miopen::deref(rnnDesc).nLayers - 1) * inputBatchLenSum * out_h;
            reserveSpaceSize *= sizeof(T);
            reserveSpaceSize += (miopen::deref(rnnDesc).nLayers - 1) * inputBatchLenSum * out_h;
            reserveSpaceSize = (reserveSpaceSize + sizeof(T) - 1) / sizeof(T);
        }

        std::vector<T> reserveSpace(reserveSpaceSize);
        std::vector<T> output(out_sz);
        std::vector<T> hiddenState(initHidden.size());
        std::vector<T> cellState(initCell.size());

        std::vector<T> converted_input;
        std::vector<T> converted_output;

        const std::vector<T>* packed_input;
        std::vector<T>* packed_output;

        if(use_seqPadding)
        {
            size_t packedXInSize, packedYOutSize;
            std::tie(packedXInSize, packedYOutSize) =
                GetTempPackedBuffersSize(batch_seq, inputVecLen, out_h);

            converted_input.resize(packedXInSize);
            converted_output.resize(packedYOutSize);

            ChangeDataPadding(input, converted_input, batch_seq, batch_seq[0], inputVecLen, false);

            packed_input  = &converted_input;
            packed_output = &converted_output;
        }
        else
        {
            packed_input  = &input;
            packed_output = &output;
        }

        LSTMFwdCPUVerify(handle,
                         use_dropout,
                         miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                         *packed_input,
                         weights,     // [ input_state_weight_trans
                                      // hidden_state_weight0_trans input1_trans
                                      // hidden1_trans ... output_weight;
                                      // bidirectional reversed weights ]
                         hiddenState, // current/final hidden state
                         initHidden,  // initial hidden state
                         cellState,   // current/final cell state
                         initCell,    // initial cell state
                         *packed_output,
                         batch_seq,       // input batch size
                         inputVecLen,     // input data length
                         seqLength,       // Number of iterations to unroll over
                         dirMode,         // whether using bidirectional net
                         biasMode,        // whether using bias
                         bi * nLayers,    // 1 by numlayer (number of stacks of hidden layers)
                                          // for unidirection, 2 by numlayer for bidirection
                         batch_seq.at(0), // equal to input batch size in_n[0]
                         hiddenSize,      // hidden state number
                         bi_stride,       // 1 by hy_h related function for unidirection, 2 by hy_h
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

        if(use_seqPadding)
        {
            ChangeDataPadding(*packed_output, output, batch_seq, batch_seq[0], out_h, true);
        }

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

        std::vector<miopen::TensorDescriptor> inputCPPDescs;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        createTensorDescArray(
            inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

        auto input_dev = handle.Write(input);

        std::vector<miopen::TensorDescriptor> outputCPPDescs;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        createTensorDescArray(outputCPPDescs,
                              outputDescs,
                              batch_seq,
                              hiddenSize * ((dirMode != 0) ? 2 : 1),
                              miopen::deref(rnnDesc).dataType);

        std::vector<T> output(getSuperTensorSize(batch_seq,
                                                 seqLength,
                                                 inputVecLen,
                                                 hiddenSize,
                                                 batch_seq[0],
                                                 dirMode != 0,
                                                 false,
                                                 use_seqPadding));
        std::fill(output.begin(), output.end(), static_cast<T>(0));
        auto output_dev = handle.Write(output);

        size_t workSpaceSize    = 0;
        size_t reserveSpaceSize = 0;
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        miopenGetRNNTrainingReserveSize(
            &handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);

        std::vector<T> workSpace(workSpaceSize / sizeof(T));
        std::vector<T> reserveSpace((reserveSpaceSize + sizeof(T) - 1) / sizeof(T));

        auto workSpace_dev    = handle.Write(workSpace);
        auto reserveSpace_dev = handle.Write(reserveSpace);

        auto weights_dev = handle.Write(weights);

        auto hy = initHidden;
        std::fill(hy.begin(), hy.end(), 0.);
        auto hy_dev = handle.Write(hy);

        auto cy = initCell;
        std::fill(cy.begin(), cy.end(), 0.);
        auto cy_dev = handle.Write(cy);

        std::vector<int> hlens(3, 0);
        hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
        hlens[1] = batch_seq[0];
        hlens[2] = hiddenSize;
        miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

        std::vector<int> wlen(1, 0);
        wlen[0] = weights.size();
        miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

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

        std::vector<T> output_gpu = handle.Read<T>(output_dev, output.size());

        auto retSet = std::make_tuple(output_gpu,
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
                  << " -r " << dirMode << " -b " << biasMode << " -p " << inputMode << " -q "
                  << use_seqPadding << std::endl;

        std::cout << "inputMode: " << inputMode << " biasMode: " << biasMode
                  << " dirMode: " << dirMode << std::endl;
        std::cout << "hz: " << hiddenSize << " batch_n: " << batch_n << " seqLength: " << seqLength
                  << " inputLen: " << inputVecLen << " numLayers: " << nLayers
                  << " useDropout: " << int(use_dropout) << std::endl;
        std::cout << "Forward Train LSTM: " << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Output tensor report." << std::endl; break;
        case(1): std::cout << "Hidden state tensor report." << std::endl; break;
        case(2): std::cout << "Cell state tensor report." << std::endl; break;
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
    int out_h     = hiddenSize * ((dirMode != 0) ? 2 : 1);
    size_t workSpaceSize;

    std::vector<miopen::TensorDescriptor> inputCPPDescs;
    std::vector<miopenTensorDescriptor_t> inputDescs;
    createTensorDescArray(
        inputCPPDescs, inputDescs, batch_seq, inputVecLen, miopen::deref(rnnDesc).dataType);

    // Outputs ----------
    size_t in_sz = getSuperTensorSize(batch_seq,
                                      seqLength,
                                      inputVecLen,
                                      hiddenSize,
                                      batch_seq[0],
                                      dirMode != 0,
                                      true,
                                      use_seqPadding);

    miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
    std::vector<T> workSpace(workSpaceSize / sizeof(T));
    std::vector<T> dx(in_sz);
    std::vector<T> dhx(initHidden.size());
    std::vector<T> dcx(initHidden.size());

    size_t inputBatchLenSum = std::accumulate(batch_seq.begin(), batch_seq.begin() + seqLength, 0);
    size_t reserveSpaceSize;
    reserveSpaceSize = 2ULL * 6 * miopen::deref(rnnDesc).nLayers * inputBatchLenSum * hiddenSize *
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

    std::vector<T> converted_dinput;
    std::vector<T> converted_output;
    std::vector<T> converted_doutput;

    std::vector<T>* packed_dinput;
    const std::vector<T>* packed_output;
    const std::vector<T>* packed_doutput;

    // WA: bug in workSpace using
    std::vector<T> wa_workSpace;
    std::vector<T>* wa_shifted_workSpace;

    if(use_seqPadding)
    {
        size_t packedXInSize, packedYOutSize;
        std::tie(packedXInSize, packedYOutSize) =
            GetTempPackedBuffersSize(batch_seq, inputVecLen, out_h);

        converted_dinput.resize(packedXInSize);
        converted_output.resize(packedYOutSize);
        converted_doutput.resize(packedYOutSize);

        ChangeDataPadding(yin, converted_output, batch_seq, batch_seq[0], out_h, false);
        ChangeDataPadding(dy, converted_doutput, batch_seq, batch_seq[0], out_h, false);

        packed_dinput  = &converted_dinput;
        packed_output  = &converted_output;
        packed_doutput = &converted_doutput;

        // WA
        wa_workSpace.resize(workSpaceSize / sizeof(T) - (packedXInSize + packedYOutSize));
        wa_shifted_workSpace = &wa_workSpace;
    }
    else
    {
        packed_dinput        = &dx;
        packed_output        = &yin;
        packed_doutput       = &dy;
        wa_shifted_workSpace = &workSpace;
    }

    LSTMBwdDataCPUVerify(use_dropout,
                         miopen::deref(miopen::deref(rnnDesc).dropoutDesc),
                         *packed_dinput,  // DX (output)
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
                         *packed_output,  // Y
                         *packed_doutput, // DY
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
                         *wa_shifted_workSpace,
                         nocx,
                         nodhy,
                         nodcy);

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: CPU backward data LSTM pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif

    if(use_seqPadding)
    {
        ChangeDataPadding(converted_dinput, dx, batch_seq, batch_seq[0], inputVecLen, true);

        // WA
        std::copy(converted_doutput.begin(), converted_doutput.end(), workSpace.begin());

        std::copy(converted_dinput.begin(),
                  converted_dinput.end(),
                  workSpace.begin() + converted_doutput.size());

        std::copy(wa_workSpace.begin(),
                  wa_workSpace.end(),
                  workSpace.begin() + converted_doutput.size() + converted_dinput.size());
    }

    std::copy(reserveSpace.begin(), reserveSpace.end(), RSVcpu);

    // TODO: remove workSpace
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
    miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);

    std::vector<int> wlen(1, 0);
    wlen[0] = weights.size();
    miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, wlen);

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
    // TODO: remove workSpace
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
    int out_h     = hiddenSize * ((dirMode != 0) ? 2 : 1);
    std::vector<T> dweights(weightSize);

    std::vector<T> converted_input;
    std::vector<T> converted_doutput;

    const std::vector<T>* packed_input;
    const std::vector<T>* packed_doutput;

    // WA: bug in workSpace using
    std::vector<T> wa_workSpace;
    const std::vector<T>* wa_shifted_workSpace;

    if(use_seqPadding)
    {
        size_t packedXInSize, packedYOutSize;
        std::tie(packedXInSize, packedYOutSize) =
            GetTempPackedBuffersSize(batch_seq, inputVecLen, out_h);

        converted_input.resize(packedXInSize);
        converted_doutput.resize(packedYOutSize);

        ChangeDataPadding(input, converted_input, batch_seq, batch_seq[0], inputVecLen, false);
        ChangeDataPadding(dy, converted_doutput, batch_seq, batch_seq[0], out_h, false);

        packed_input   = &converted_input;
        packed_doutput = &converted_doutput;

        // WA
        wa_workSpace.resize(workSpace.size() - (packedXInSize + packedYOutSize));
        std::copy(workSpace.begin() + packedXInSize + packedYOutSize,
                  workSpace.end(),
                  wa_workSpace.begin());
        wa_shifted_workSpace = &wa_workSpace;
    }
    else
    {
        packed_input         = &input;
        packed_doutput       = &dy;
        wa_shifted_workSpace = &workSpace;
    }

    LSTMBwdWeightCPUVerify(use_dropout,
                           *packed_input,
                           dweights,   // (output) [ input_state_weight_trans
                                       // hidden_state_weight0_trans
                                       // input1_trans hidden1_trans ...
                                       // output_weight; bidirectional
                                       // reversed weights ]
                           initHidden, // initial hidden state
                           *packed_doutput,
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
                           *wa_shifted_workSpace,
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
    miopen::TensorDescriptor weightDesc(miopen::deref(rnnDesc).dataType, {weightSize});

    std::vector<int> hlens(3, 0);
    hlens[0] = nLayers * (dirMode != 0 ? 2 : 1);
    hlens[1] = batch_seq[0];
    hlens[2] = hiddenSize;
    miopen::TensorDescriptor hiddenDesc(miopen::deref(rnnDesc).dataType, hlens);
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
    bool usePadding    = false;

    lstm_basic_driver() {}

    void run()
    {
        const double Data_scale = 0.001;
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

        if(usePadding)
        {
            miopenSetRNNPaddingMode(rnnDesc, miopenRNNPaddingMode_t::miopenRNNIOWithPadding);
        }

        // Create input tensor
        // If we are in skip mode, take the real input size to be the vector length.
        auto inVecReal = (inputMode != 0) ? hiddenSize : inVecLen;

        int batch_padding = usePadding ? batchSeq[0] : 0;

        std::size_t in_sz = getSuperTensorSize(batchSeq,
                                               seqLength,
                                               inVecReal,
                                               hiddenSize,
                                               batch_padding,
                                               dirMode != 0,
                                               true,
                                               usePadding);
        std::vector<T> input(in_sz);
        for(std::size_t i = 0; i < in_sz; i++)
        {
            input[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
        }

        std::size_t hx_sz = ((dirMode != 0) ? 2ULL : 1ULL) * hiddenSize * batchSize * numLayers;
        std::vector<T> hx(hx_sz);
        std::vector<T> cx(hx_sz);
        std::vector<T> dhyin(hx_sz);
        std::vector<T> dcyin(hx_sz);

        size_t wei_bytes = 0;
        std::vector<int> inlens(2, 0);
        inlens.at(0)        = batchSeq.at(0);
        inlens.at(1)        = inVecReal;
        auto firstInputDesc = miopen::TensorDescriptor(miopen::deref(rnnDesc).dataType, inlens);
        miopenGetRNNParamsSize(
            &handle, rnnDesc, &firstInputDesc, &wei_bytes, miopen::deref(rnnDesc).dataType);
        auto wei_sz = int(wei_bytes / sizeof(T));
        std::vector<T> weights(wei_sz);
        for(std::size_t i = 0; i < wei_sz; i++)
        {
            weights[i] = prng::gen_descreet_uniform_sign<T>(Data_scale, 100);
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
                hx[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
            }
        }

        if(!nodhy)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                dhyin[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
            }
        }

        if(!nocx)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                cx[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
            }
        }

        if(!nodcy)
        {
            for(std::size_t i = 0; i < hx_sz; i++)
            {
                dcyin[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
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

        size_t out_sz = getSuperTensorSize(batchSeq,
                                           seqLength,
                                           inVecLen,
                                           hiddenSize,
                                           batchSeq[0],
                                           dirMode != 0,
                                           false,
                                           usePadding);

        size_t workSpaceSize;
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);

        size_t total_mem = statesSizeInBytes + reserveSpaceSize + workSpaceSize +
                           (2 * out_sz + in_sz + wei_sz + (nohx ? 0 : hx_sz) + (nohy ? 0 : hx_sz) +
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
            2ULL * 6 * numLayers * inputBatchLenSum * hiddenSize * ((dirMode != 0) ? 2 : 1);
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
            rnnDesc,          input,      hx,      cx,        weights,   batchSeq, rsvgpu,
            rsvcpu,           hiddenSize, batch_n, seqLength, numLayers, biasMode, dirMode,
            inputMode,        inVecReal,  hx_sz,   nohx,      nocx,      nohy,     nocy,
            bool(useDropout), usePadding});

        /// RETURNS std::make_tuple(output, hiddenState, cellState, reserveSpace);
        auto yin = std::get<0>(fwdTrainOutputPair.second);
        // auto curHiddenState = std::get<1>(fwdTrainOutputPair.second);
        // auto curCellState   = std::get<2>(fwdTrainOutputPair.second);

        if(yin.size() != out_sz)
        {
            std::cout << "FWD FAILED: yin.size() != out_sz." << std::endl
                      << "yin.size()=" << yin.size() << std::endl
                      << "out_sz=" << out_sz << std::endl;

            exit(-1); // NOLINT (concurrency-mt-unsafe)
        }

        std::vector<T> dyin(out_sz);
        for(std::size_t i = 0; i < out_sz; i++)
        {
            dyin[i] = prng::gen_descreet_unsigned<T>(Data_scale, 100);
        }

#if(MIO_LSTM_TEST_DEBUG > 0)
        printf("Running backward data LSTM.\n");
#endif
        auto bwdDataOutputPair =
            verify(verify_backward_data_lstm<T>{rnnDesc,   yin,       dyin,
                                                dhyin,     hx,        dcyin,
                                                cx,        weights,   rsvgpu,
                                                rsvcpu,    batchSeq,  hiddenSize,
                                                batch_n,   seqLength, numLayers,
                                                biasMode,  dirMode,   inputMode,
                                                inVecReal, hx_sz,     nohx,
                                                nocx,      nodhy,     nodcy,
                                                nodhx,     nodcx,     bool(useDropout),
                                                usePadding});

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
            rnnDesc,  input,      dyin,      hx,      rsvgpu,    rsvcpu,           workSpaceBwdData,
            batchSeq, hiddenSize, wei_sz,    batch_n, seqLength, numLayers,        biasMode,
            dirMode,  inputMode,  inVecReal, hx_sz,   nohx,      bool(useDropout), usePadding});

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
