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
#include "rnn_util.hpp"
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



#define MIO_RNN_TIME_EVERYTHING 0
#define RNN_MM_TRANSPOSE 1

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_rnn
{
    std::vector<T> input;
    std::vector<T> initHidden;
    std::vector<int> batch_seq;
    int hiddenSize;
    int seqLength;
    int nLayers;
    int biasMode;
    int dirMode; 
    int inputMode;
    int rnnMode;
    int batch_n;
	int inputVecLen;
    miopenRNNDescriptor_t rnnDesc;
    
    verify_forward_train_rnn(miopenRNNDescriptor_t pRD,
                         const std::vector<T>& px,
                         const std::vector<T>& phx,
                         const std::vector<int>& pBS,
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
        int bacc, baccbi; // accumulation of batch
        int bi = dirMode ? 2 : 1;
        
        int in_stride  = inputVecLen;
        int hy_stride  = hiddenSize * bi;
        int out_stride = hy_stride;

        int in_h = (inputMode)?0:inputVecLen;
        int hy_h = hiddenSize;
        int uni_stride = hy_h;
        int bi_stride = hy_h * bi;

        auto in_sz  = input.size();
        size_t out_sz = 0;
        size_t wei_sz = 0;
        size_t hy_sz  = 0;
        size_t workSpaceSize;
        size_t reserveSpaceSize;
		
        miopenTensorDescriptor_t inDesc;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        std::vector<int> inlens(2,0);
        inlens[1] = inputVecLen;
        
        miopenTensorDescriptor_t outDesc;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        std::vector<int> outlens(2,0);
        outlens[1] = hiddenSize*((dirMode)?2:1);
        // -----------------------
        for(int i = 0; i < batch_seq.size(); i++)
        {
            inlens[0] = batch_seq[i];
            miopenCreateTensorDescriptor(&inDesc);
            miopenSetTensorDescriptor(inDesc,
                                    miopenFloat,
                                    2,
                                    inlens.data(), 
                                    nullptr);
            inputDescs.push_back(inDesc);
            
            outlens[0] = batch_seq[i];
            miopenCreateTensorDescriptor(&outDesc);
            miopenSetTensorDescriptor(outDesc,
                                    miopenFloat,
                                    2,
                                    outlens.data(), 
                                    nullptr);
            outputDescs.push_back(outDesc);
        }
        

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        miopenGetRNNHiddenTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &hy_sz);
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        miopenGetRNNTrainingReserveSize(&handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        miopenGetRNNParamsSize(&handle, rnnDesc, inputDescs[0], &wei_sz, miopenFloat);
        
        wei_sz /= sizeof(T);
        hy_sz  /= sizeof(T);
        out_sz /= sizeof(T);
        std::vector<T> workSpace(workSpaceSize/sizeof(T), 0.);
        std::vector<T> reserveSpace(reserveSpaceSize/sizeof(T), 0.);
        std::vector<T> output(out_sz, 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);
        std::vector<T> weights(wei_sz, 0.);
          
        
        
        // initial weights
        int wei_len = (bi * (in_h + hy_h) + (nLayers - 1) * bi * (bi + 1) * hy_h) * hy_h;
        if(biasMode)
        {
            int in_bias = ((inputMode == 1) ? 1 : 2);
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
                            hiddenState.at(hid_shift + bs * hy_stride + h) += input[bs * in_stride + h];
                            if(dirMode)
                            {
                                hiddenState.at(hid_shift + bs * hy_stride + hy_h + h) += input[bs * in_stride + h];
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
                                hiddenState.at(hid_shift + bs * hy_stride + h) += weights[wei_shift_bias + h];
                            }
                        }
                    }
                }
                else
                {
                    RNN_mm_cpu<T>(input.data(), 
                                   in_h,
                                   batch_n,
                                   in_stride,
                                   0,
                                   weights.data(), 
                                   hy_h * bi,
                                   in_h,
                                   in_stride,
                                   RNN_MM_TRANSPOSE,
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
                                hiddenState.at(hid_shift + bs * hy_stride + h) +=
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

                RNN_mm_cpu<T>(&workSpace[prelayer_shift],
                               hy_h * bi,
                               batch_n,
                               hy_stride,
                               0,
                               &weights[wei_shift],
                               hy_h * bi,
                               hy_h * bi,
                               bi_stride,
                               RNN_MM_TRANSPOSE,
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
                            hiddenState.at(hid_shift + bs * hy_stride + h) +=
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
                    RNN_mm_cpu<T>(&initHidden[hx_shift],
                                   hy_h,
                                   batch_seq[ti],
                                   uni_stride,
                                   0,
                                   &weights[wei_shift],
                                   hy_h,
                                   hy_h,
                                   uni_stride,
                                   RNN_MM_TRANSPOSE,
                                   &hiddenState[hid_shift + bacc * hy_stride],
                                   hy_h,
                                   batch_seq[ti],
                                   hy_stride,
                                   0,
                                   1,
                                   1);

                    if(dirMode)
                    {
                        RNN_mm_cpu<T>(&initHidden[hx_shift + hy_h],
                                       hy_h,
                                       batch_seq[seqLength - 1 - ti],
                                       uni_stride,
                                       0,
                                       &weights[wei_shift + hy_h],
                                       hy_h,
                                       hy_h,
                                       uni_stride,
                                       RNN_MM_TRANSPOSE,
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
                    RNN_mm_cpu<T>(&hiddenState[hx_shift],
                                   hy_h,
                                   batch_seq[ti],
                                   uni_stride,
                                   0,
                                   &weights[wei_shift],
                                   hy_h,
                                   hy_h,
                                   uni_stride,
                                   RNN_MM_TRANSPOSE,
                                   &hiddenState[hid_shift + bacc * hy_stride],
                                   hy_h,
                                   batch_seq[ti],
                                   hy_stride,
                                   0,
                                   1,
                                   1);

                    if(dirMode)
                    {
                        RNN_mm_cpu<T>(&hiddenState[hx_shift + hy_h],
                                       hy_h,
                                       batch_seq[seqLength - 1 - ti],
                                       uni_stride,
                                       0,
                                       &weights[wei_shift + hy_h],
                                       hy_h,
                                       hy_h,
                                       uni_stride,
                                       RNN_MM_TRANSPOSE,
                                       &hiddenState[hid_shift + baccbi * hy_stride + hy_h],
                                       hy_h,
                                       batch_seq[seqLength - 1 - ti],
                                       hy_stride,
                                       0,
                                       1,
                                       1);
                    }
                }

                for(int bs = 0; bs < batch_seq[ti]; bs++)
                {
                    for(int h = 0; h < hy_h; h++)
                    {
                        workSpace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) =
                            activfunc(hiddenState[hid_shift + bacc * hy_stride + bs * hy_stride + h],
                                      rnnMode); // squash_func
                        hiddenState.at(hx_shift + bs * uni_stride + h) =
                            workSpace[hid_shift + bacc * hy_stride + bs * hy_stride + h];

                        reserveSpace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h) =
                            hiddenState[hid_shift + bacc * hy_stride + bs * hy_stride + h];

                        reserveSpace.at(hid_shift + bacc * hy_stride + bs * hy_stride + h +
                                 nLayers * batch_n * hy_h * bi) =
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
                            workSpace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h) =
                                activfunc(hiddenState[hid_shift + baccbi * hy_stride + hy_h +
                                                    bs * hy_stride + h],
                                          rnnMode); // squash_func
                            hiddenState.at(hx_shift + hy_h + bs * uni_stride + h) =
                                workSpace[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h];

                            reserveSpace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h) =
                                hiddenState[hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h];

                            reserveSpace.at(hid_shift + baccbi * hy_stride + hy_h + bs * hy_stride + h +
                                     nLayers * batch_n * hy_h * bi) =
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
                    hiddenState.at(hx_shift + bs * uni_stride + h) = 0;
                }
            }
        }

        // output
        int prelayer_shift = (nLayers - 1) * batch_n * hy_h * bi;

        for(int bs = 0; bs < batch_n; bs++)
        {
            for(int h = 0; h < hy_h; h++)
            {
                //printf("out index: %d\n", bs*out_stride+h); fflush(nullptr);
                output.at(bs * out_stride + h) = workSpace[prelayer_shift + bs * hy_stride + h];
            }
        }
 

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_train_bn_spatial pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        auto retSet = std::make_tuple(output, hiddenState, weights, workSpace, reserveSpace);
        return retSet;
    }

    std::tuple<std::vector<T>, std::vector<T>,std::vector<T>,std::vector<T>,std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif


        auto&& handle = get_handle();
        
        int bacc, baccbi; // accumulation of batch
        int bi = dirMode ? 2 : 1;
        
        int in_stride  = inputVecLen;
        int hy_stride  = hiddenSize * bi;
        int out_stride = hy_stride;

        int in_h = (inputMode)?0:inputVecLen;
        int hy_h = hiddenSize;
        
        size_t in_sz  = 0;
        size_t out_sz = 0;
        size_t wei_sz = 0;
        size_t hy_sz  = 0;
        size_t workSpaceSize = 0;
        size_t reserveSpaceSize = 0;

        miopenTensorDescriptor_t inDesc;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        std::vector<int> inlens(2,0);
        inlens[1] = inputVecLen;
        
        miopenTensorDescriptor_t outDesc;
        std::vector<miopenTensorDescriptor_t> outputDescs;
        std::vector<int> outlens(2,0);
        outlens[1] = hiddenSize*((dirMode)?2:1);
        // -----------------------
        for(int i = 0; i < batch_seq.size(); i++)
        {
            inlens[0] = batch_seq[i];
            miopenCreateTensorDescriptor(&inDesc);
            miopenSetTensorDescriptor(inDesc,
                                    miopenFloat,
                                    2,
                                    inlens.data(), 
                                    nullptr);
            inputDescs.push_back(inDesc);
            
            outlens[0] = batch_seq[i];
            miopenCreateTensorDescriptor(&outDesc);
            miopenSetTensorDescriptor(outDesc,
                                    miopenFloat,
                                    2,
                                    outlens.data(), 
                                    nullptr);
            outputDescs.push_back(outDesc);
        }
        

        
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        miopenGetRNNHiddenTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &hy_sz);
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        miopenGetRNNTrainingReserveSize(&handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        miopenGetRNNParamsSize(&handle, rnnDesc, inputDescs[0], &wei_sz, miopenFloat);
        auto wei_elems = int(wei_sz/sizeof(T));
        
        std::vector<T> workSpace(workSpaceSize/sizeof(T), 0.);
        std::vector<T> reserveSpace(reserveSpaceSize/sizeof(T), 0.);
        //std::vector<T> output(out_sz/4, 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);
        std::vector<T> weights(wei_elems, 0.); 
        
        auto input_dev    = handle.Write(input);
        auto output = input;
        std::fill(output.begin(), output.end(), 0.);
        auto output_dev  = handle.Write(output);
        
        auto weights_dev  = handle.Write(weights);
        auto hx_dev  = handle.Write(initHidden);
		auto hy = initHidden;
		std::fill(hy.begin(), hy.end(), 0.);
		auto hy_dev  = handle.Write(hy);
		
		auto workSpace_dev = handle.Write(workSpace);
		auto reserveSpace_dev = handle.Write(reserveSpace);

		std::vector<int> hlens(3,0);
		hlens[0] = nLayers*(dirMode)?2:1;
		hlens[1] = batch_seq[0];
		hlens[2] = hiddenSize;
		miopen::TensorDescriptor hiddenDesc(miopenFloat, hlens.data(), 3);
        
		miopen::TensorDescriptor weightDesc(miopenFloat, &wei_elems, 1);
		
        miopenRNNForwardTraining(&handle,
                         rnnDesc,
                         seqLength,
                         inputDescs.data(),
                         input_dev.get(),
                         &hiddenDesc,
                         hx_dev.get(),
                         &hiddenDesc,
                         nullptr,
                         &weightDesc,
                         weights_dev.get(),
                         outputDescs.data(),
                         output_dev.get(),
                         &hiddenDesc,
                         hy_dev.get(),
                         &hiddenDesc,
                         nullptr,
                         workSpace_dev.get(),
                         workSpaceSize,
                         reserveSpace_dev.get(),
                         reserveSpaceSize);
						 
        auto retSet = std::make_tuple(handle.Read<T>(output_dev, output.size()),
                                    handle.Read<T>(hy_dev, hy.size()), 
                                    handle.Read<T>(weights_dev, weights.size()), 
                                    handle.Read<T>(workSpace_dev, workSpaceSize/sizeof(T)),
                                    handle.Read<T>(reserveSpace_dev, reserveSpaceSize/sizeof(T)));
		
#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return retSet;
    }

    void fail(int badtensor)
    {

        std::cout << "Forward Train RNN vanilla: " << std::endl;
        //std::cout << "Input tensor: " << input.desc.ToString() << std::endl;

        /*switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Running Mean output tensor failed verification." << std::endl; break;
        case(2):
            std::cout << "Running Variance output tensor failed verification." << std::endl;
            break;
        case(3): std::cout << "Saved Mean tensor failed verification." << std::endl; break;
        case(4): std::cout << "Saved Variance tensor failed verification." << std::endl; break;
        }*/
    }
};






//====================== DRIVER ============================
template <class T>
struct rnn_vanilla_driver : test_driver
{
	std::vector<int> batchSeq;
    int seqLength;
	int inVecLen;
	int hiddenSize;
	int numLayers;
	int inputMode;
	int biasMode;
	int dirMode;
	int rnnMode;
	int batchSize;
    
    rnn_vanilla_driver()
    {
        this->batch_factor = 4;
		std::vector<int> modes(2,0);
		modes[1] = 1;
		
        // this->verbose=true;
        add(batchSize, "batch-size", generate_data(get_rnn_batchSize(),{5}));
        add(seqLength, "seq-len", generate_data(get_rnn_seq_len()));
        add(inVecLen, "vector-len", generate_data(get_rnn_vector_len()));
        add(hiddenSize, "hidden-size", generate_data(get_rnn_hidden_size()));
        add(numLayers, "num-layers", generate_data(get_rnn_num_layers()));
        add(inputMode, "in-mode", generate_data(modes));
        add(biasMode, "bias-mode", generate_data(modes));
        add(dirMode, "dir-mode", generate_data(modes));
        add(rnnMode, "rnn-mode", generate_data(modes));
		//add(batchSeq, "batch-seq", lazy_generate_data([=]{ return generate_batchSeq(batchSize, seqLength); }, {10}));
        
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
        
        /*
             verify_forward_train_rnn(miopenRNNDescriptor_t pRD,
                         tensor<T>& px,
                         tensor<T>& phx,
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

    */ 
        int modval = 5;
        int currentval = batchSize;
        batchSeq.clear();
        for(int i = 0; i < seqLength; i++)
        {
            printf("adding a value to batch sequence.\n");
            if(i>0){
                int nvalue = currentval - rand()%modval;
                currentval = (nvalue<1) ? 1 : nvalue;
                printf("current value: %d\n", currentval);
            }
                batchSeq.push_back(currentval);
        }

        
        int batch_n = 0;
        for(auto& n : batchSeq) batch_n += n;
        
        // Need to multiply the number of layers by 2 for bidirectional.
        //int numlayer = bidirection ? hy_d / 2 : hy_d;
        int bacc, baccbi; // accumulation of batch
        int bi = dirMode ? 2 : 1;
	
        miopenRNNDescriptor_t rnnDesc;
        miopenCreateRNNDescriptor(&rnnDesc);
        
        miopenRNNAlgo_t algoMode = miopenRNNdefault;
        miopenSetRNNDescriptor(rnnDesc, 
                                hiddenSize, 
                                numLayers, 
                                miopenRNNInputMode_t(inputMode), 
                                miopenRNNDirectionMode_t(dirMode), 
                                miopenRNNMode_t(rnnMode), 
                                miopenRNNBiasMode_t(biasMode), 
                                miopenRNNAlgo_t(algoMode),
                                miopenFloat);
        
        //Create input tensor
        std::size_t in_sz = inVecLen*seqLength*batch_n;
        auto inputTensor = tensor<T>{in_sz}.generate(rand_gen{});
        auto inputData = inputTensor.data;
        
        std::size_t hx_sz = ((dirMode)?2:1)*hiddenSize*batchSize;
        
        auto hxTensor = tensor<T>{hx_sz}.generate(rand_gen{});
        auto hxData = hxTensor.data;
        printf("hz: %d, batch_n: %d, seqLength: %d\n" ,hiddenSize, batch_n, seqLength);
        verify(verify_forward_train_rnn<T>{rnnDesc, inputData, 
                                        hxData, batchSeq, 
                                        hiddenSize, batch_n, 
                                        seqLength, numLayers, 
                                        biasMode, dirMode, 
                                        inputMode, rnnMode, inVecLen});
        
        
        
/*
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
      */
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
