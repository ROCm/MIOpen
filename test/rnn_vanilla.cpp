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

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"
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




//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_rnn
{
    const tensor<T> input;
    const tensor<T> initHidden;
    std::vector<int> batch_seq;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode; 
    int inputMode;
    int rnnMode;
    int batch_n;
    miopenRNNDescriptor_t rnnDesc;
    
    verify_forward_train_rnn(miopenRNNDescriptor_t pRD,
                         const std::vector<miopen::TensorDescriptor>& pxd,
                         tensor<T>& px,
                         const tensor<T>& phx,
                         tensor<T>& pWS,
                         tensor<T>& pRS,
                         const std::vector<int> pBS,
                         const int pHS,
                         const int pBN,
                         const int pS,
                         const int pNL,
                         const int pBM,
                         const int pDM,
                         const int pIM,
                         const int pRM,
                         const int pVL)
    {
        rnnDesc      = pRD;
        inputDescs   = pxd;
        input        = px;
        initHidden   = phx;
        batch_seq    = pBS;
        seqLength    = pS;
        nLayers      = pNL;
        biasMode     = pBM;
        dirMode      = pDM;
        inputMode    = pIM;
        rnnMode      = pRM;
        batch_n      = pBN;
        hiddenSize   = pHS;
        inputVecLen  = pVL;
    }
        
    std::tuple<std::vector<T>, std::vector<T>,std::vector<T>,std::vector<T>,std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        
        auto&& handle = get_handle();
        
        tensor<T> hiddenState = initHidden;
        tensor<T> output = input;
        std::fill(output.begin(), output.end(), 0.);
        
        int bacc, baccbi; // accumulation of batch
        int bi = dirMode ? 2 : 1;
        
        int in_stride  = inputVecLen;
        int hy_stride  = hiddenSize * bi;
        int out_stride = hy_stride;

        int in_h = (inputMode)?0:inputVecLen;
        int hy_h = hiddenSize;
        int uni_stride = hy_h;

        size_t in_sz  = 0;
        size_t out_sz = 0;
        size_t wei_sz = 0;
        size_t hy_sz  = 0;
        size_t workSpaceSize;
        size_t reserveSpaceSize;

        miopenStatus_t errcode = miopenStatusSuccess;
        errcode |= miopenGetRNNInputTensorSize(handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        errcode |= miopenGetRNNInputTensorSize(handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        errcode |= miopenGetRNNHiddenTensorSize(handle, rnnDesc, seqLength, inputDescs.data(), &hy_sz);
        errcode |= miopenGetRNNWorkspaceSize(handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        errcode |= miopenGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        errcode |= miopenGetRNNParamsSize(handle, rnnDesc, inputs[0], &wei_sz, miopenFloat);
        assert(errcode != miopenStatusSuccess);
        
        std::vector<T> workSpace(workSpaceSize, 0.);
        std::vector<T> reserveSpace(reserveSpaceSize, 0.);
        std::vector<T> output(out_sz, 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);
        std::vector<T> weights(initHidden.size(), 0.);
        
        std::vector<miopen::TensorDescriptor> inputDescs;
        std::vector<int> lens(2,0);
        // -----------------------
        for(int i = 0; i < batch_seq.size(); i++)
        {
            lens[0] = batch_seq[i];
            lens[1] = inputVecLen;
            inputDescs.push_back(miopen::TensorDescriptor(miopenFloat, lens.data(), 2));
        }
        
        
        // initial weights
        int wei_len = (bi * (in_h + hy_h) + (nLayers - 1) * bi * (bi + 1) * hy_h) * hy_h;
        if(biasMode)
        {
            int in_bias = (inputMode == 1) ? 1 : 2;
            wei_len += (bi * in_bias + (nLayers - 1) * bi * 2) * hy_h;
        }

        int wei_shift_bias = ((in_h + hy_h) * bi + (bi * hy_h + hy_h) * bi * (nLayers - 1)) * hy_h;

        // forward emulator
        for(int li = 0; li < nLayers; li++)
        {
            int hid_shift = li * batch_n * hy_h * bi;
            int hx_shift  = li * bi * batch_seq[0] * hy_h;

            // from input
            if(li == 0)
            {
                if(inputMode == 1)
                {
                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            hiddenState[hid_shift + bs * hy_stride + h] += input[bs * in_stride + h];
                            if(dirMode)
                            {
                                hiddenState[hid_shift + bs * hy_stride + hy_h + h] += input[bs * in_stride + h];
                            }
                        }
                    }

                    // from bias
                    if(biasMode)
                    {
                        for(int bs = 0; bs < batch_n; bs++)
                        {
                            for(int h = 0; h < hy_stride; h++)
                            {
                                hiddenState[hid_shift + bs * hy_stride + h] += weights[wei_shift_bias + h];
                            }
                        }
                    }
                }
                else
                {
                    RNN_mm_cpu<T>(const_cast<T*>(input), 
                                   in_h,
                                   batch_n,
                                   in_stride,
                                   0,
                                   const_cast<T*>(weights), 
                                   hy_h * bi,
                                   in_h,
                                   hy_stride,
                                   0,
                                   &hiddenState[hid_shift],
                                   hy_h * bi,
                                   batch_n,
                                   hy_stride,
                                   0,
                                   1,
                                   1);

                    // from bias
                    if(biasMode)
                    {
                        for(int bs = 0; bs < batch_n; bs++)
                        {
                            for(int h = 0; h < hy_stride; h++)
                            {
                                hiddenState[hid_shift + bs * hy_stride + h] +=
                                    (weights[wei_shift_bias + h] + weights[wei_shift_bias + hy_stride + h]);
                            }
                        }
                    }
                }
            }
            else
            {
                int wei_shift = bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h;
                int prelayer_shift = (li - 1) * batch_n * hy_h * bi;

                RNN_mm_cpu<T>(const_cast<T*>(&workSpace[prelayer_shift]),
                               hy_h * bi,
                               batch_n,
                               hy_stride,
                               0,
                               const_cast<T*>(&weights[wei_shift]),
                               hy_h * bi,
                               hy_h * bi,
                               hy_stride,
                               0,
                               &hiddenState[hid_shift],
                               hy_h * bi,
                               batch_n,
                               hy_stride,
                               0,
                               1,
                               1);

                // from bias
                if(biasMode)
                {
                    int wei_shift_bias_temp =
                        (inputMode == 1) ? (wei_shift_bias + bi * hy_h + bi * (li - 1) * 2 * hy_h)
                                         : (wei_shift_bias + bi * li * 2 * hy_h);

                    for(int bs = 0; bs < batch_n; bs++)
                    {
                        for(int h = 0; h < hy_stride; h++)
                        {
                            hiddenState[hid_shift + bs * hy_stride + h] +=
                                (weights[wei_shift_bias_temp + h] +
                                 weights[wei_shift_bias_temp + hy_stride + h]);
                        }
                    }
                }
            }

            // from hidden state
            bacc   = 0;
            baccbi = batch_n;
            for(int ti = 0; ti < seqLength; ti++)
            {
                baccbi -= batch_seq[seqLength - 1 - ti];

                int wei_shift =
                    li == 0 ? (in_h * hy_h * bi)
                            : (bi * (in_h + hy_h) * hy_h + (li - 1) * bi * (bi * hy_h + hy_h) * hy_h +
                               bi * hy_h * hy_stride);

                if(ti == 0)
                {
                    RNN_mm_cpu<T>(const_cast<T*>(&initHidden[hx_shift]),
                                   hy_h,
                                   batch_seq[ti],
                                   uni_stride,
                                   0,
                                   const_cast<T*>(&weights[wei_shift]),
                                   hy_h,
                                   hy_h,
                                   hy_stride,
                                   0,
                                   &hiddenState[hid_shift + bacc * hy_stride],
                                   hy_h,
                                   batch_seq[ti],
                                   hy_stride,
                                   0,
                                   1,
                                   1);

                    if(dirMode)
                    {
                        RNN_mm_cpu<T>(const_cast<T*>(&initHidden[hx_shift + hy_h]),
                                       hy_h,
                                       batch_seq[seqLength - 1 - ti],
                                       uni_stride,
                                       0,
                                       const_cast<T*>(&weights[wei_shift + hy_h]),
                                       hy_h,
                                       hy_h,
                                       hy_stride,
                                       0,
                                       &hiddenState[hid_shift + baccbi * hy_stride + hy_h],
                                       hy_h,
                                       batch_seq[seqLength - 1 - ti],
                                       hy_stride,
                                       0,
                                       1,
                                       1);
                    }
                }
                else
                {
                    RNN_mm_cpu<T>(const_cast<T*>(&hiddenState[hx_shift]),
                                   hy_h,
                                   batch_seq[ti],
                                   uni_stride,
                                   0,
                                   const_cast<T*>(&weights[wei_shift]),
                                   hy_h,
                                   hy_h,
                                   hy_stride,
                                   0,
                                   &hiddenState[hid_shift + bacc * hy_stride],
                                   hy_h,
                                   batch_seq[ti],
                                   hy_stride,
                                   0,
                                   1,
                                   1);

                    if(dirMode)
                    {
                        RNN_mm_cpu<T>(const_cast<T*>(&hiddenState[hx_shift + hy_h]),
                                       hy_h,
                                       batch_seq[seqLength - 1 - ti],
                                       uni_stride,
                                       0,
                                       const_cast<T*>(&weights[wei_shift + hy_h]),
                                       hy_h,
                                       hy_h,
                                       hy_stride,
                                       0,
                                       &hiddenState[hid_shift + baccbi * hy_stride + hy_h],
                                       hy_h,
                                       batch_seq[seqLength - 1 - ti],
                                       hy_stride,
                                       0,
                                       1,
                                       1);
                    }
                }

                for(int bs = 0; bs < in_n[ti]; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        workSpace[hid_shift + bacc * hy_stride + bs * hy_stride + h] =
                            activfunc(hiddenState[hid_shift + bacc * hy_stride + bs * hy_stride + h],
                                      rnnMode); // squash_func
                        hiddenState[hx_shift + bs * uni_stride + h] =
                            workSpace[hid_shift + bacc * hy_stride + bs * hy_stride + h];

                        reserveSpace[hid_shift + bacc * hy_stride + bs * hy_stride + h] =
                            hiddenState[hid_shift + bacc * hy_stride + bs * hy_stride + h];

                        reserveSpace[hid_shift + bacc * hy_stride + bs * hy_stride + h +
                                 nLayers * batch_n * hy_h * bi] =
                            activfunc(hiddenState[hid_shift + bacc * hy_stride + bs * hy_stride + h],
                                      rnnMode);
                    }
                }

                if(dirMode)
                {
                    for(int bs = 0; bs < batch_seq[seqLength - 1 - ti]; bs++)
                    {
                        for(int h = 0; h < hy_h; h++)
                        {
                            workSpace[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h] =
                                activfunc(hiddenState[hid_shift + baccbi * hy_stride + hy_h +
                                                    bs * hy_stride + h],
                                          rnnMode); // squash_func
                            hiddenState[hx_shift + hy_h + bs * uni_stride + h] =
                                workSpace[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h];

                            reserveSpace[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h] =
                                hiddenState[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h];

                            reserveSpace[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h +
                                     nLayers * batch_n * hy_h * bi] =
                                activfunc(hiddenState[hid_shift + baccbi * hy_stride + hy_h +
                                                    bs * hy_stride + h], rnnMode);
                        }
                    }
                }

                bacc += batch_seq[ti];
            }

            // hy clean
            for(int bs = batch_seq[seqLength - 1]; bs < batch_seq[0]; bs++)
            {
                for(int h = 0; h < hy_h; h++)
                {
                    hiddenState[hx_shift + bs * uni_stride + h] = 0;
                }
            }
        }

        // output
        int prelayer_shift = (nLayers - 1) * batch_n * hy_h * bi;

        for(int bs = 0; bs < batch_n; bs++)
        {
            for(int h = 0; h < out_h; h++)
            {
                output[bs * out_stride + h] = workSpace[prelayer_shift + bs * hy_stride + h];
            }
        }
 

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_train_bn_spatial pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(output, hiddenState, weights, workSpace, reserveSpace);
    }

    std::tuple<std::vector<T>, std::vector<T>,std::vector<T>,std::vector<T>,std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        tensor<T> hiddenState = initHidden;
        tensor<T> output = input;
        std::fill(output.begin(), output.end(), 0.);
        
        int bacc, baccbi; // accumulation of batch
        int bi = dirMode ? 2 : 1;
        
        int in_stride  = inputVecLen;
        int hy_stride  = hiddenSize * bi;
        int out_stride = hy_stride;

        int in_h = (inputMode)?0:inputVecLen;
        int hy_h = hiddenSize;
        
        /*    
            const tensor<T> input;
            const tensor<T> initHidden;
            tensor<T> weights;
            tensor<T> workSpace;
            tensor<T> reserveSpace;
            std::vector<int> batch_seq;
            int hiddenSize;
            int seqLength;
            int nLayers;
            int biasMode;
            int dirMode; 
            int inputMode;
            int rnnMode;
            int batch_n;
            tensor<T> hiddenState;
            tensor<T> output;
            miopenRNNDescriptor_t rnnDesc;
        */ 
        
        size_t in_sz  = 0;
        size_t out_sz = 0;
        size_t wei_sz = 0;
        size_t hy_sz  = 0;
        size_t workSpaceSize;
        size_t reserveSpaceSize;


        // TODO: Implement this here!
        std::vector<miopen::TensorDescriptor> inputDescs;
        std::vector<int> lens(2,0);
        lens[1] = inputVecLen;
        // -----------------------
        for(int i = 0; i < batch_seq.size(); i++)
        {
            lens[0] = batch_seq[i];
            inputDescs.push_back(miopen::TensorDescriptor(miopenFloat, lens.data(), 2));
        }
        auto outputDescs = inputDescs;

        miopenStatus_t errcode = miopenStatusSuccess;
        errcode |= miopenGetRNNInputTensorSize(handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        errcode |= miopenGetRNNInputTensorSize(handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        errcode |= miopenGetRNNHiddenTensorSize(handle, rnnDesc, seqLength, inputDescs.data(), &hy_sz);
        errcode |= miopenGetRNNWorkspaceSize(handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        errcode |= miopenGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        errcode |= miopenGetRNNParamsSize(handle, rnnDesc, inputDescs[0], &wei_sz, miopenFloat);
        assert(errcode != miopenStatusSuccess);
        
        std::vector<T> workSpace(workSpaceSize, 0.);
        std::vector<T> reserveSpace(reserveSpaceSize, 0.);
        std::vector<T> output(out_sz, 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);
        std::vector<T> weights(initHidden.size(), 0.); 
        
        auto input_dev    = handle.Write(input.data);
        auto output = input;
        std::fill(output.begin(), output.end(), 0.);
        auto output_dev  = handle.Write(output.data);
        
        auto weights_dev  = handle.Write(weights.data);
        auto hx_dev  = handle.Write(initHidden.data);
		auto hy = initHidden;
		std::fill(hy.begin(), hy.end(), 0.);
		auto hy_dev  = handle.Write(hy);
		
        
        auto runMean = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(rand_gen{});
        auto runVar  = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(rand_gen{});
        auto saveMean   = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto saveInvVar = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width};

        
        miopenRNNForwardTraining(handle,
                         rnnDesc,
                         seqLength,
                         inputTensors.data(),
                         in_dev->GetMem(),
                         hiddenTensor,
                         hx_dev->GetMem(),
                         hiddenTensor,
                         cx_dev->GetMem(),
                         weightTensor,
                         wei_dev->GetMem(),
                         outputTensors.data(),
                         out_dev->GetMem(),
                         hiddenTensor,
                         hy_dev->GetMem(),
                         hiddenTensor,
                         cy_dev->GetMem(),
                         workspace_dev->GetMem(),
                         workspace_dev->GetSize(),
                         reservespace_dev->GetMem(),
                         reservespace_dev->GetSize());
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        auto runMean = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(rand_gen{});
        auto runVar  = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width}.generate(rand_gen{});
        auto saveMean   = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width};
        auto saveInvVar = tensor<T>{rs_n_batch, rs_channels, rs_height, rs_width};

        // in buffers
        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);

        // out buffers
        auto runMean_dev    = handle.Write(runMean.data);
        auto runVar_dev     = handle.Write(runVar.data);
        auto saveMean_dev   = handle.Create<T>(channels);
        auto saveInvVar_dev = handle.Create<T>(channels);
        auto out_dev        = handle.Create<T>(n_batch * channels * height * width);

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        T alpha = 1, beta = 0;
        miopen::BatchNormForwardTraining(handle,
                                         miopenBNSpatial,
                                         &alpha,
                                         &beta,
                                         input.desc,
                                         in_dev.get(),
                                         out.desc,
                                         out_dev.get(),
                                         scale.desc,
                                         scale_dev.get(),
                                         shift_dev.get(),
                                         expAvgFactor,
                                         runMean_dev.get(),
                                         runVar_dev.get(),
                                         epsilon,
                                         saveMean_dev.get(),
                                         saveInvVar_dev.get());

        saveMean.data   = handle.Read<T>(saveMean_dev, saveMean.data.size());
        saveInvVar.data = handle.Read<T>(saveInvVar_dev, saveInvVar.data.size());
        runMean.data    = handle.Read<T>(runMean_dev, runMean.data.size());
        runVar.data     = handle.Read<T>(runVar_dev, runVar.data.size());
        out.data        = handle.Read<T>(out_dev, out.data.size());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train_bn_spatial pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    void fail(int badtensor)
    {

        std::cout << "Forward Train Spatial Batch Normalization: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Running Mean output tensor failed verification." << std::endl; break;
        case(2):
            std::cout << "Running Variance output tensor failed verification." << std::endl;
            break;
        case(3): std::cout << "Saved Mean tensor failed verification." << std::endl; break;
        case(4): std::cout << "Saved Variance tensor failed verification." << std::endl; break;
        }
    }
};

//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T>
struct verify_backward_data_rnn
{

    const tensor<T> input;
    const tensor<T> scale;
    const tensor<T> shift;

    tensor<T> cpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {

            double elemStd        = 0.;
            double variance_accum = 0.;
            double mean_accum     = 0.;
            double inhat          = 0.;
            double invVar         = 0.;

            mean_accum = 0.;
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    // #1 calculate the mean
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // iterating through the stack of images in the mini_batch
                        mean_accum += input(bidx, cidx, row, column);
                    } // end for (n)
                }     // end for (column)
            }         // end for (row)
            mean_accum /= nhw;

            elemStd        = 0.;
            variance_accum = 0.;
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // using out buffer as scratchpad
                        out(bidx, cidx, row, column) = elemStd =
                            (input(bidx, cidx, row, column) - mean_accum); // (x_i - mean)
                        variance_accum += (elemStd * elemStd);             // sum{ (x_i - mean)^2 }
                    }                                                      // end for(n)
                }                                                          // end for (column)
            }                                                              // end for (row)
            variance_accum /= nhw; // (1/N)*sum{ (x_i - mean)^2 }

            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = 1.0 / sqrt(variance_accum + epsilon);

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        elemStd =
                            out(bidx, cidx, row, column); // using saved values from output tensor
                        inhat = elemStd * invVar;
                        // #5 Gamma and Beta adjust // y_i = gamma*x_hat + beta
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, 0, 0) * inhat + shift(0, cidx, 0, 0);
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    tensor<T> gpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);
        auto out_dev   = handle.Write(out.data);

        T alpha = 1, beta = 0;

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormForwardInference(handle,
                                          miopenBNSpatial,
                                          &alpha,
                                          &beta,
                                          input.desc,
                                          in_dev.get(),
                                          out.desc,
                                          out_dev.get(),
                                          scale.desc,
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          nullptr,
                                          nullptr,
                                          epsilon);
        out.data = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int)
    {
        std::cout << "Forward Inference Spatial Batch Normalization Recalc: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_backward_weights_rnn
{

    const tensor<T> input;
    const tensor<T> scale;
    const tensor<T> shift;
    const tensor<T> estMean;
    const tensor<T> estVar;
    tensor<T> cpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        par_for(channels, 1, [&](int cidx) {
            double elemStd  = 0.;
            double variance = 0.;
            double mean     = 0.;
            double inhat    = 0.;
            double invVar   = 0.;

            mean     = estMean(0, cidx, 0, 0);
            variance = estVar(0, cidx, 0, 0);
            invVar   = 1.0 / sqrt(variance + epsilon);
            // process the batch per channel
            for(int bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns

                        elemStd = input(bidx, cidx, row, column) - mean;
                        inhat   = elemStd * invVar;
                        out(bidx, cidx, row, column) =
                            scale(0, cidx, 0, 0) * inhat + shift(0, cidx, 0, 0);
                    }
                }
            }
        });
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_bn_spatial_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return out;
    }

    tensor<T> gpu()
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev      = handle.Write(input.data);
        auto estMean_dev = handle.Write(estMean.data);
        auto estVar_dev  = handle.Write(estVar.data);
        auto scale_dev   = handle.Write(scale.data);
        auto shift_dev   = handle.Write(shift.data);
        auto out_dev     = handle.Write(out.data);

        T alpha = 1, beta = 0;

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormForwardInference(handle,
                                          miopenBNSpatial,
                                          &alpha,
                                          &beta,
                                          input.desc,
                                          in_dev.get(),
                                          out.desc,
                                          out_dev.get(),
                                          scale.desc,
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          estMean_dev.get(),
                                          estVar_dev.get(),
                                          epsilon);
        out.data = handle.Read<T>(out_dev, out.data.size());
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_bn_spatial_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int)
    {
        std::cout << "Forward Inference Spatial Batch Normalization Use Estimated: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

//****************************************************
// BACKWARDS PROPAGATION
//****************************************************
template <class T>
struct verify_forward_inference_rnn
{

    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<T> scale;

    std::tuple<tensor<T>, tensor<T>, tensor<T>> cpu()
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_cstride = height * width;
        const auto nhw                = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {

            double elemStd = 0.;
            unsigned int xhat_index;
            double mean     = 0.;
            double invVar   = 0.;
            double dyelem   = 0.;
            double variance = 0.;

            std::vector<double> xhat(n_batch * in_cstride, 0.0);

#if(MIO_HEIRARCH_SEL == 1)
            std::vector<double> variance_accum_arr(height, 0.0);
            std::vector<double> mean_accum_arr(height, 0.0);
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);
#endif

            // process the batch per channel
            mean = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // #1 calculate the mean
                        mean += x_input(bidx, cidx, row, column);
                    }
                } // for (column)
            }     // for (row)
#else
            for (int row = 0; row < height; row++){ //via rows
                for(int column = 0; column < width; column++){// via columns
                    for (int bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        mean_accum_arr[row] += x_input(bidx,cidx,row,column);
                    }	
                }// for (column)
            }// for (row)  
            for(int i = 0; i<height; i++) mean += mean_accum_arr[i];
#endif
            mean /= nhw;

            elemStd  = 0.;
            variance = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        elemStd = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        variance += elemStd * elemStd;                     // sum{ (x_i - mean)^2 }
                    }                                                      // end for(n)
                }                                                          // for (column)
            }                                                              // for (row)
#else
            for (int row = 0; row < height; row++){ //via rows
                for(int column = 0; column < width; column++){// via columns
                    for (int bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        elemStd = x_input(bidx,cidx,row,column) - mean;
                        variance_accum_arr[row] += elemStd*elemStd;
                    }	
                }// for (column)
            }// for (row)  
            for(int i = 0; i<height; i++) variance += variance_accum_arr[i];
#endif
            variance /= nhw; // (1/(N*H*W))*sum{ (x_i - mean)^2 }
            invVar = 1. / double(sqrt(variance + epsilon));

            dscale(0, cidx, 0, 0) = 0.;

#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_input(bidx, cidx, row, column) - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * invVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift(0, cidx, 0, 0) += dyelem;
                        dscale(0, cidx, 0, 0) += xhat[xhat_index] * dyelem;
                    } // end for(n_batch)
                }     // for (column)
            }         // for (row)
#else   
            
            for (int row = 0; row < height; row++){ //via rows
                for(int column = 0; column < width; column++){// via columns
                    for (int bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                        xhat_index = in_cstride*bidx + (width*row + column);
                        //per (x-dims) channel load a block of data into LDS
                        elemStd             = x_input(bidx,cidx,row,column) - mean;// (x_i - mean)
                        xhat[xhat_index]    = elemStd*invVar;
                        dyelem              = dy_input(bidx,cidx,row,column);
                        dshift_accum_arr[row] += dyelem;
                        dscale_accum_arr[row] += xhat[xhat_index]*dyelem;
                        //dscale_accum_arr[row] += x_input(bidx,cidx,row,column);;//dscale_accum_arr[row] += xhat[xhat_index];
                        //dscale_accum_arr[row] += 1.0;//DEBUG
                    }	
                }// for (column)
            }// for (row)  
            for(int i = 0; i<height; i++) {
                dshift(0,cidx,0,0) += dshift_accum_arr[i];    
                dscale(0,cidx,0,0) += dscale_accum_arr[i];    
            }
#endif

            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    for(int bidx = 0; bidx < n_batch; bidx++)
                    { // via mini_batch
                        xhat_index = in_cstride * bidx + (width * row + column);

                        double tmp1 =
                            nhw * dy_input(bidx, cidx, row, column) - dshift(0, cidx, 0, 0);
                        double tmp2 = -xhat[xhat_index] * dscale(0, cidx, 0, 0);
                        double tmp3 = (scale(0, cidx, 0, 0) * invVar) / nhw;
                        dx_out(bidx, cidx, row, column) = tmp3 * (tmp2 + tmp1);
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        });           // for (channel)

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<T>, tensor<T>> gpu()
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_height, ss_width) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dscale = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<T>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        T alpha = 1, beta = 0;

        auto xin_dev    = handle.Write(x_input.data);
        auto dyin_dev   = handle.Write(dy_input.data);
        auto scale_dev  = handle.Write(scale.data);
        auto dscale_dev = handle.Write(dscale.data);
        auto dshift_dev = handle.Write(dshift.data);
        auto dx_out_dev = handle.Write(dx_out.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormBackward(handle,
                                  miopenBNSpatial,
                                  &alpha,
                                  &beta,
                                  &alpha,
                                  &beta,
                                  x_input.desc,
                                  xin_dev.get(),
                                  dy_input.desc,
                                  dyin_dev.get(),
                                  dx_out.desc,
                                  dx_out_dev.get(),
                                  scale.desc,
                                  scale_dev.get(),
                                  dscale_dev.get(),
                                  dshift_dev.get(),
                                  epsilon,
                                  nullptr,
                                  nullptr);

        dx_out.data = handle.Read<T>(dx_out_dev, dx_out.data.size());
        dscale.data = handle.Read<T>(dscale_dev, dscale.data.size());
        dshift.data = handle.Read<T>(dshift_dev, dshift.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor)
    {
        std::cout << "Backward Batch Spatial Normalization Recalc Mean and Variance: " << std::endl;
        std::cout << "X Input tensor: " << x_input.desc.ToString() << std::endl;
        std::cout << "Delta Y Input tensor: " << dy_input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0):
            std::cout << "Delta X output tensor output failed verification." << std::endl;
            break;
        case(1): std::cout << "Delta scale output tensor failed verification." << std::endl; break;
        case(2): std::cout << "Delta shift output tensor failed verification." << std::endl; break;
        }
    }
};





//====================== DRIVER ============================
template <class T>
struct rnn_vanilla_driver : test_driver
{
    tensor<T> x;
    tensor<T> hx;
    tensor<T> w;
    tensor<T> workSpace;
    tensor<T> reserveSpace;
    
    rnn_vanilla_driver()
    {
        this->batch_factor = 4;

        // this->verbose=true;
        add(batchSeq, "batch-seq", get_rnn_batchseq());
        add(seqLength, "seq-len", get_rnn_seq_len());
        add(inVecLen, "vector-len", get_rnn_vector_len());
        add(hiddenSize, "hidden-size", get_rnn_hidden_size());
        add(numLayers, "num-layers", get_rnn_num_layers());
        add(inputMode, "in-mode", {0, 1});
        add(biasMode, "bias-mode", {0, 1});
        add(dirMode, "dir-mode", {0,1});
        add(rnnMode, "rnn-mode", {0,1});
    }

    void run()
    {
        
     /*
        std::vector<T>& wei,     // [ input_state_weight_trans
                                 // hidden_state_weight0_trans input1_trans
                                 // hidden1_trans ... output_weight;
                                 // bidirectional reversed weights ]
        std::vector<T>& hy_host, // current/final hidden state
        std::vector<T>& hx,      // initial hidden state
        std::vector<T>& out_host,
        std::vector<int>& in_n, // input batch size
        int in_h,               // input data length
        int seqLength,          // Number of iterations to unroll over
        bool bidirection,       // whether using bidirectional net
        bool biased,            // whether using bias
        int hy_d,  // 1 by numlayer (number of stacks of hidden layers) for
                   // unidirection, 2 by numlayer for bidirection
        int hy_n,  // equal to input batch size in_n[0]
        int hy_h,  // hidden state number
        int out_h, // 1 by hy_h related function for unidirection, 2 by hy_h
                   // related function for bidirection
        int squash,
      */   
        int batch_n = 0;
        for(auto& n : batch_seq) batch_n += n;
        
        
        
        T* hid_state = new T[hy_d * batch_n * hy_h];
        memset(hid_state, 0, hy_d * batch_n * hy_h * sizeof(T));

        T* wk_state = new T[hy_d * batch_n * hy_h];
        memset(wk_state, 0, hy_d * batch_n * hy_h * sizeof(T));

        T* out_state = new T[batch_n * out_h];
        memset(out_state, 0, batch_n * out_h * sizeof(T));

        int numlayer = bidirection ? hy_d / 2 : hy_d;
        int bacc, baccbi; // accumulation of batch
        int bi = bidirection ? 2 : 1;

        int in_stride  = in_h;
        int hy_stride  = hy_h * bi;
        int out_stride = out_h;

        // initial input
        T* in_state = new T[batch_n * in_h];
        for(int h = 0; h < batch_n; h++)
        {
            for(int w = 0; w < in_h; w++)
            {
                in_state[h * in_h + w] = in[h * in_h + w];
            }
        }

        
        if(biased)
        {
            int in_bias = (inputMode == 1) ? 1 : 2;
            wei_len += (bi * in_bias + (numlayer - 1) * bi * 2) * hy_h;
        }

        T* wei_state = new T[wei_len];
        for(int h = 0; h < wei_len; h++)
        {
            wei_state[h] = wei[h];
        }
      
    }
};

int main(int argc, const char* argv[])
{
#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    test_drive<rnn_vanilla_driver<float>>(argc, argv);

#if(MIO_RNN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: RNN test pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif
    exit(0);
}
