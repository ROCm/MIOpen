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


//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T>
struct verify_forward_infer_rnn
{
    std::vector<T> input;
    std::vector<T> initHidden;
    std::vector<T> weights;
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
    
    verify_forward_infer_rnn(miopenRNNDescriptor_t pRD,
                         const std::vector<T>& px,
                         const std::vector<T>& phx,
                         const std::vector<T>& pW,
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
        weights      = pW,
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
        
    std::vector<T> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        
        auto&& handle = get_handle();

        int bi = dirMode ? 2 : 1;        
        int hy_h = hiddenSize;
        int uni_stride = hy_h;
        int bi_stride = bi*hy_h;
        size_t out_sz = 0;

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
        miopenGetRNNTrainingReserveSize(&handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        std::vector<T> reserveSpace(reserveSpaceSize/sizeof(T), 0.);
        std::vector<T> output(out_sz/sizeof(T), 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);

          
        RNNFwdTrainCPUVerify(input,
                             weights,     // [ input_state_weight_trans
                                          // hidden_state_weight0_trans input1_trans
                                          // hidden1_trans ... output_weight;
                                          // bidirectional reversed weights ]
                             hiddenState, // current/final hidden state
                             initHidden,  // initial hidden state
                             output,
                             batch_seq,   // input batch size
                             inputVecLen, // input data length
                             seqLength,   // Number of iterations to unroll over
                             dirMode,     // whether using bidirectional net
                             biasMode,    // whether using bias
                             bi*nLayers,   // 1 by numlayer (number of stacks of hidden layers) for
                                          // unidirection, 2 by numlayer for bidirection
                             batch_seq.at(0),  // equal to input batch size in_n[0]
                             hiddenSize,  // hidden state number
                             bi_stride,   // 1 by hy_h related function for unidirection, 2 by hy_h
                                          // related function for bidirection
                             rnnMode,
                             inputMode,
                             reserveSpace);  
        
#if (MIO_RNN_SP_TEST_DEBUG == 2)     
        for(int i = 0; i < output.size(); i++)
        {
            printf("CPU outdata[%d]: %f\n" ,i ,output[i]);
            
        }
#endif
 

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_inference pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        auto retSet = std::make_tuple(output, hiddenState, weights, reserveSpace);
        std::cout << "Done with RNN forward inference CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
        return output;
    }

    std::vector<T> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();
        
        size_t out_sz = 0;
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
        
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);

        std::vector<T> workSpace(workSpaceSize/sizeof(T), 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);
        
        auto input_dev    = handle.Write(input);
        
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz/sizeof(T),0.);
        auto output_dev  = handle.Write(output);
        
        auto weights_dev  = handle.Write(weights);
        auto hx_dev  = handle.Write(initHidden);
		auto hy = initHidden;
		std::fill(hy.begin(), hy.end(), 0.);
		auto hy_dev  = handle.Write(hy);
		
		auto workSpace_dev = handle.Write(workSpace);

		std::vector<int> hlens(3,0);
		hlens[0] = nLayers*(dirMode)?2:1;
		hlens[1] = batch_seq[0];
		hlens[2] = hiddenSize;
		miopen::TensorDescriptor hiddenDesc(miopenFloat, hlens.data(), 3);
        
        std::vector<int> wlen(1,0);
        wlen[0] = weights.size();
		miopen::TensorDescriptor weightDesc(miopenFloat, wlen.data(), 1);
		
        miopenRNNForwardInference(&handle,
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
                         workSpaceSize);
        
#if (MIO_RNN_SP_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n" ,i ,outdata[i]);
        }	
#endif
		
#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        std::cout << "Done with RNN forward inference GPU" << std::endl;
        return (handle.Read<T>(output_dev, output.size()));
    }

    void fail(int badtensor)
    {
        std::cout << "Forward Inference RNN vanilla: " << std::endl;
        std::cout << "Output tensor output failed verification." << std::endl;
    }
};
//~~~~~~~~~~~~ END FWD INFERENCE ~~~~~~~~~~~~~~~~~~~~~~~~


























//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T>
struct verify_forward_train_rnn
{
    std::vector<T> input;
    std::vector<T> initHidden;
    std::vector<T> weights;
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
                         const std::vector<T>& pW,
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
        weights      = pW,
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
        
    std::tuple<std::vector<T>, std::vector<T>,std::vector<T>,std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        
        auto&& handle = get_handle();

        int bi = dirMode ? 2 : 1;        
        int hy_h = hiddenSize;
        int uni_stride = hy_h;
        int bi_stride = bi*hy_h;
        size_t out_sz = 0;

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
        miopenGetRNNTrainingReserveSize(&handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        std::vector<T> reserveSpace(reserveSpaceSize/sizeof(T), 0.);
        std::vector<T> output(out_sz/sizeof(T), 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);

          
        RNNFwdTrainCPUVerify(input,
                             weights,     // [ input_state_weight_trans
                                          // hidden_state_weight0_trans input1_trans
                                          // hidden1_trans ... output_weight;
                                          // bidirectional reversed weights ]
                             hiddenState, // current/final hidden state
                             initHidden,  // initial hidden state
                             output,
                             batch_seq,   // input batch size
                             inputVecLen, // input data length
                             seqLength,   // Number of iterations to unroll over
                             dirMode,     // whether using bidirectional net
                             biasMode,    // whether using bias
                             bi*nLayers,   // 1 by numlayer (number of stacks of hidden layers) for
                                          // unidirection, 2 by numlayer for bidirection
                             batch_seq.at(0),  // equal to input batch size in_n[0]
                             hiddenSize,  // hidden state number
                             bi_stride,   // 1 by hy_h related function for unidirection, 2 by hy_h
                                          // related function for bidirection
                             rnnMode,
                             inputMode,
                             reserveSpace);  
        
#if (MIO_RNN_SP_TEST_DEBUG == 2)     
        for(int i = 0; i < output.size(); i++)
        {
            printf("CPU outdata[%d]: %f\n" ,i ,output[i]);
            
        }
#endif
 

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_train pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        auto retSet = std::make_tuple(output, hiddenState, weights, reserveSpace);
        std::cout << "Done with RNN forward train CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
        return retSet;
    }

    std::tuple<std::vector<T>, std::vector<T>,std::vector<T>,std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();
        
        size_t out_sz = 0;
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
        
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        miopenGetRNNTrainingReserveSize(&handle, rnnDesc, seqLength, inputDescs.data(), &reserveSpaceSize);
        
        std::vector<T> workSpace(workSpaceSize/sizeof(T), 0.);
        std::vector<T> reserveSpace(reserveSpaceSize/sizeof(T), 0.);
        std::vector<T> hiddenState(initHidden.size(), 0.);
        
        auto input_dev    = handle.Write(input);
        
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        std::vector<T> output(out_sz/sizeof(T),0.);
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
        
        std::vector<int> wlen(1,0);
        wlen[0] = weights.size();
		miopen::TensorDescriptor weightDesc(miopenFloat, wlen.data(), 1);
		
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
        
#if (MIO_RNN_SP_TEST_DEBUG == 2)
        auto outdata = handle.Read<T>(output_dev, output.size());
        for(int i = 0; i < outdata.size(); i++)
        {
            printf("GPU outdata[%d]: %f\n" ,i ,outdata[i]);
            
        }	
#endif
        
        auto retSet = std::make_tuple(handle.Read<T>(output_dev, output.size()),
                                    handle.Read<T>(hy_dev, hy.size()), 
                                    handle.Read<T>(weights_dev, weights.size()), 
                                    handle.Read<T>(reserveSpace_dev, reserveSpaceSize/sizeof(T)));
		
#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        std::cout << "Done with RNN forward train GPU" << std::endl;
        return retSet;
    }

    void fail(int badtensor)
    {

        std::cout << "Forward Train RNN vanilla: " << std::endl;

        switch(badtensor)
        {
        case(0): 
            std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Hidden state tensor failed verification." << std::endl; break;
        case(2): std::cout << "Weight tensor failed verification." << std::endl; break;
        case(3): std::cout << "Reserved space tensor failed verification." << std::endl; break;
        }
    }
};
//~~~~~~~~~~~~ END FWD TRAIN ~~~~~~~~~~~~~~~~~~~~~~~~

























//****************************************************
// BACKWARDS DATA
//****************************************************
template <class T>
struct verify_backward_data_rnn
{
    std::vector<T> yin; // Y
    std::vector<T> dy; // dY
    std::vector<T> dhy; // dHY
    std::vector<T> initHidden; // HX
    std::vector<T> weights;
    std::vector<T> reserveSpace;
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
    
    verify_backward_data_rnn(miopenRNNDescriptor_t pRD,
                         const std::vector<T>& py,
                         const std::vector<T>& pdy,
                         const std::vector<T>& pdhy,
                         const std::vector<T>& phx,
                         const std::vector<T>& pW,
                         const std::vector<T>& pRS,
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
        yin          = py;
        dy           = pdy;
        dhy          = pdhy;
        initHidden   = phx;
        weights      = pW,
        reserveSpace = pRS;
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
        
    std::tuple<std::vector<T>, std::vector<T>,std::vector<T>,std::vector<T>> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        
        auto&& handle = get_handle();

        int bi = dirMode ? 2 : 1;        
        int hy_h = hiddenSize;
        int uni_stride = hy_h;
        int bi_stride = bi*hy_h;
        size_t out_sz = 0;

        size_t reserveSpaceSize;
        size_t workSpaceSize;
		
        miopenTensorDescriptor_t inDesc;
        std::vector<miopenTensorDescriptor_t> inputDescs;
        std::vector<int> inlens(2,0);
        inlens[1] = inputVecLen;
        
        miopenTensorDescriptor_t oneYdesc;
        std::vector<miopenTensorDescriptor_t> yDescs;
        
        
        std::vector<int> ylens(2,0);
        ylens[1] = hiddenSize*((dirMode)?2:1);
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
            
            ylens[0] = batch_seq[i];
            miopenCreateTensorDescriptor(&oneYdesc);
            miopenSetTensorDescriptor(oneYdesc,
                                    miopenFloat,
                                    2,
                                    ylens.data(), 
                                    nullptr);
            yDescs.push_back(oneYdesc);
        }

        
        size_t in_sz = 0;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        std::vector<T> dx(in_sz/sizeof(T), 0.);
        
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        std::vector<T> workSpace(workSpaceSize/sizeof(T), 0.);
                
        std::vector<T> dhx(initHidden.size(), 0.);
          
        RNNBwdDataCPUVerify(dx, //OUTPUT
                            weights, // [ input_state_weight_trans
                                                 // hidden_state_weight0_trans input1_trans
                                                 // hidden1_trans ... output_weight;
                                                 // bidirectional reversed weights ]
                            dhy,        // dhy -- input: current/final hidden state
                            dhx,        // dhx OUTPUT
                            initHidden, // HX initial hidden state
                            yin,     // Y input
                            dy,    // dY -- input
                            batch_seq, // input batch size
                            inputVecLen,               // input data length
                            seqLength,          // Number of iterations to unroll over
                            dirMode,       // whether using bidirectional net
                            biasMode,            // whether using bias
                            bi*nLayers,  // 1 by numlayer (number of stacks of hidden layers)
                                       // for unidirection, 2 by numlayer for bidirection
                            batch_seq.at(0),  // equal to input batch size in_n[0]
                            hiddenSize,  // hidden state number
                            bi_stride, // 1 by hy_h related function for unidirection, 2 by
                                       // hy_h related function for bidirection
                            rnnMode,
                            inputMode,
                            reserveSpace,
                            workSpace);


#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_data_rnn_vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        auto retSet = std::make_tuple(dx, dhx, reserveSpace, workSpace);
        std::cout << "Done with RNN backward data CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
        return retSet;
    }

    std::tuple<std::vector<T>, std::vector<T>,std::vector<T>,std::vector<T>> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();
        
        size_t out_sz = 0;
        size_t workSpaceSize = 0;

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
        
        miopenGetRNNWorkspaceSize(&handle, rnnDesc, seqLength, inputDescs.data(), &workSpaceSize);
        std::vector<T> workSpace(workSpaceSize/sizeof(T), 0.);
        auto workSpace_dev = handle.Write(workSpace);
                
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, outputDescs.data(), &out_sz);
        auto yin_dev  = handle.Write(yin);
        auto dyin_dev = handle.Write(dy);
        auto dhyin_dev = handle.Write(dhy);
        auto reserveSpace_dev = handle.Write(reserveSpace);
        auto weights_dev  = handle.Write(weights);
        auto hx_dev  = handle.Write(initHidden);
	


		std::vector<int> hlens(3,0);
		hlens[0] = nLayers*(dirMode)?2:1;
		hlens[1] = batch_seq[0];
		hlens[2] = hiddenSize;
		miopen::TensorDescriptor hiddenDesc(miopenFloat, hlens.data(), 3);
        
        std::vector<int> wlen(1,0);
        wlen[0] = weights.size();
		miopen::TensorDescriptor weightDesc(miopenFloat, wlen.data(), 1);
		
        size_t in_sz = 0;
        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLength, inputDescs.data(), &in_sz);
        std::vector<T> dx(in_sz/sizeof(T), 0.);
        auto dx_dev = handle.Write(dx);
                
        std::vector<T> dhx(initHidden.size(), 0.);
        auto dhx_dev = handle.Write(dhx);
          
        /*
            Input:
                std::vector<T> yin; // Y
                std::vector<T> dy;  // dY
                std::vector<T> dhy; // dHY
                std::vector<T> initHidden; // HX
                std::vector<T> weights;
                std::vector<T> reserveSpace;
         * 
            Output:
                std::make_tuple(dx, dhx, reserveSpace, workSpace);
         */ 
                
        miopenRNNBackwardData(&handle,
                                rnnDesc,
                                seqLength,
                                outputDescs.data(),
                                yin_dev.get(),// TODO up
                                outputDescs.data(),
                                dyin_dev.get(),// TODO up
                                &hiddenDesc,
                                dhyin_dev.get(),// TODO up
                                &hiddenDesc,
                                nullptr,
                                &weightDesc,
                                weights_dev.get(),
                                &hiddenDesc,
                                hx_dev.get(), // TODO up
                                &hiddenDesc,
                                nullptr,
                                inputDescs.data(),
                                dx_dev.get(), // TODO up
                                &hiddenDesc,
                                dhx_dev.get(),
                                &hiddenDesc,
                                nullptr,
                                workSpace_dev.get(), // TODO up
                                workSpaceSize,
                                reserveSpace_dev.get(),// TODO up remove extra
                                reserveSpace.size());
        

        auto retSet = std::make_tuple(handle.Read<T>(dx_dev, dx.size()),
                                    handle.Read<T>(dhx_dev, dhx.size()),  
                                    handle.Read<T>(reserveSpace_dev, reserveSpace.size()),
                                      handle.Read<T>(workSpace_dev, workSpace.size()));
		
#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward data RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        std::cout << "Done with RNN backward data GPU" << std::endl;
        return retSet;
    }

    void fail(int badtensor)
    {

        std::cout << "Backward Data RNN vanilla: " << std::endl;

        switch(badtensor)
        {
        case(0): 
            std::cout << "Output dx failed verification." << std::endl; break;
        case(1): std::cout << "Hidden state dhx tensor failed verification." << std::endl; break;
        case(2): std::cout << "Weight tensor failed verification." << std::endl; break;
        case(3): std::cout << "Reserved space tensor failed verification." << std::endl; break;
        }
    }
};
//~~~~~~~~~~~~ END BACKWARD DATA ~~~~~~~~~~~~~~~~~~~~~~~~










//****************************************************
// BACKWARDS WEIGHTS
//****************************************************
template <class T>
struct verify_backward_weights_rnn
{
    std::vector<T> input; // Y
    std::vector<T> dy; // dY
    std::vector<T> initHidden; // HX
    std::vector<T> reserveSpace;
    std::vector<T> workSpace;
    std::vector<int> batch_seq;
    int weightSize;
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
    
    verify_backward_weights_rnn(miopenRNNDescriptor_t pRD,
                         const std::vector<T>& px,
                         const std::vector<T>& pdy,
                         const std::vector<T>& phx,
                         const std::vector<T>& pRS,
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
                         const int pRM,
                         const int pVL)
    {
        rnnDesc      = pRD;
        input        = px;
        dy           = pdy;
        initHidden   = phx;
        reserveSpace = pRS;
        workSpace    = pWS;
        batch_seq    = pBS;
        seqLength    = pS;
        nLayers      = pNL;
        biasMode     = pBM;
        dirMode      = pDM;
        inputMode    = pIM;
        rnnMode      = pRM;
        batch_n      = pBN;
        hiddenSize   = pHS;
        weightSize   = pW;
        inputVecLen  = pVL;       
     }
        
    std::vector<T> cpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        
        auto&& handle = get_handle();

        int bi = dirMode ? 2 : 1;        
        int hy_h = hiddenSize;
        int bi_stride = bi*hy_h;
        std::vector<T> dweights(weightSize, 0.);
        
        RNNBwdWeightCPUVerify(input,
                              dweights, // [ input_state_weight_trans
                                                                  // hidden_state_weight0_trans
                                                                  // input1_trans hidden1_trans ...
                                                                  // output_weight; bidirectional
                                                                  // reversed weights ]
                              initHidden,        // initial hidden state
                              dy,
                              batch_seq, // input batch size
                              inputVecLen,               // input data length
                              seqLength,    // Number of iterations to unroll over
                              dirMode, // whether using bidirectional net
                              biasMode,      // whether using bias
                              bi*nLayers,  // 1 by numlayer (number of stacks of hidden
                                                  // layers) for unidirection, 2 by numlayer for
                                                  // bidirection
                              batch_seq.at(0),  // equal to input batch size in_n[0]
                              hiddenSize,  // hidden state number
                              bi_stride, // 1 by hy_h related function for unidirection, 2
                                                  // by hy_h related function for bidirection
                              rnnMode,
                              inputMode,
                              reserveSpace,
                              workSpace);


#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_weights_rnn_vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        std::cout << "Done with RNN backward weights CPU" << std::endl;
        std::cout << "---------------------------------\n" << std::endl;
        return dweights;
    }

    
    std::vector<T> gpu()
    {

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();
        
        size_t out_sz = 0;

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
        
        auto workSpace_dev = handle.Write(workSpace);
        auto reserveSpace_dev = handle.Write(reserveSpace);
        std::vector<T> dweights(weightSize, 0.);
        auto dweights_dev  = handle.Write(dweights);
        miopen::TensorDescriptor weightDesc(miopenFloat, &weightSize, 1);

		std::vector<int> hlens(3,0);
		hlens[0] = nLayers*(dirMode)?2:1;
		hlens[1] = batch_seq[0];
		hlens[2] = hiddenSize;
		miopen::TensorDescriptor hiddenDesc(miopenFloat, hlens.data(), 3);
        auto hx_dev = handle.Write(initHidden);
        auto dy_dev = handle.Write(dy);
        auto input_dev = handle.Write(input);
        
        std::vector<int> wlen(1,0);
        wlen[0] = weightSize;
        printf("weight size: %d\n", weightSize);

        miopenRNNBackwardWeights(&handle,
                                   rnnDesc,
                                   seqLength,
                                   inputDescs.data(),
                                   input_dev.get(),
                                   &hiddenDesc,
                                   hx_dev.get(),
                                   outputDescs.data(),
                                   dy_dev.get(),
                                   &weightDesc,
                                   dweights_dev.get(),
                                   workSpace_dev.get(),
                                   workSpace.size(),
                                   reserveSpace_dev.get(),
                                   reserveSpace.size());

#if(MIO_RNN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backwards_weights RNN vanilla pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        std::cout << "Done with RNN backward weights GPU" << std::endl;
        auto retvec = handle.Read<T>(dweights_dev, dweights.size());
        return retvec;
    }

    void fail(int badtensor = 0)
    {
        std::cout << "Backward Weights RNN vanilla: " << std::endl;
    }
};
//~~~~~~~~~~~~ END BACKWARD WEIGHTS ~~~~~~~~~~~~~~~~~~~~~~~~








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
        //this->batch_factor = 4;
		std::vector<int> modes(2,0);
		modes[1] = 1;
		std::vector<int> defaultBS(1,5);
       
        // this->verbose=true;
        add(batchSize, "batch-size", generate_data(get_rnn_batchSize(),{5}));
        add(seqLength, "seq-len", generate_data(get_rnn_seq_len()));
        add(inVecLen, "vector-len", generate_data(get_rnn_vector_len()));
        add(hiddenSize, "hidden-size", generate_data(get_rnn_hidden_size()));
        add(numLayers, "num-layers", generate_data(get_rnn_num_layers()));

        biasMode = 0;
        dirMode  = 1;
        rnnMode  = 0;
        inputMode= 0;
//        add(inputMode, "in-mode", generate_data(modes));
//        add(biasMode, "bias-mode", generate_data(modes));
//        add(dirMode, "dir-mode", generate_data(modes));
//        add(rnnMode, "rnn-mode", generate_data(modes));
		add(batchSeq, "batch-seq", lazy_generate_data([=]{ return generate_batchSeq(batchSize, seqLength); }, defaultBS));
        
    }

    void run()
    { 
        
#if (MIO_RNN_SP_TEST_DEBUG == 2)       
        for(int i = 0; i < seqLength; i++)
        {
            std::cout << "batch seq[" << i <<"]: " << batchSeq.at(i) << std::endl;
        }
#endif
        int batch_n = 0;
        for(auto& n : batchSeq) batch_n += n;
        
        // Need to multiply the number of layers by 2 for bidirectional.
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
        auto inVecReal = (inputMode)?hiddenSize:inVecLen;
        std::size_t in_sz = inVecReal*batch_n;
        auto inputTensor = tensor<T>{in_sz};
        srand(0);
        for(int i = 0; i < in_sz; i++){
            inputTensor[i] = /*(((rand()%2)==1)?-1:1)**/0.001*float(rand()%100);
        }
        auto inputData = inputTensor.data;
        
        std::size_t hx_sz = ((dirMode)?2:1)*hiddenSize*batchSize*numLayers;
        
        auto hxTensor = tensor<T>{hx_sz};
        for(int i = 0; i < hx_sz; i++){
            hxTensor[i] = /*(((rand()%2)==1)?-1:1)**/0.001*float(rand()%100);
        }
        
        auto hxData = hxTensor.data;
        
        auto iVL = inVecLen;
        if(inputMode == miopenRNNskip)
            iVL = 0;
        
        auto wei_sz = hiddenSize * bi * (iVL + hiddenSize + (numLayers - 1) * (bi + 1) * hiddenSize);
        if(biasMode)
        {
            auto in_bias = inputMode ? 1 : 2;
            wei_sz += (in_bias + (numLayers - 1) * 2) * hiddenSize * bi;
        }
        auto weightTensor = tensor<T>{std::size_t(wei_sz)};
        for(int i = 0; i < wei_sz; i++){
            weightTensor[i] = (((rand()%2)==1)?-1:1)*0.001*float(rand()%100);
        }

        auto weightData = weightTensor.data;   
        printf("inputMode: %d, biasMode: %d, rnnMode: %d, dirMode: %d\n", inputMode, biasMode, rnnMode, dirMode);
        printf("hz: %d, batch_n: %d, seqLength: %d, inputLen: %d, numLayers: %d\n" ,hiddenSize, batch_n, seqLength,inVecLen,numLayers);
        auto fwdTrainOutputPair = verify(verify_forward_train_rnn<T>{rnnDesc, inputData, 
                                        hxData, weightData, batchSeq, 
                                        hiddenSize, batch_n, 
                                        seqLength, numLayers, 
                                        biasMode, dirMode, 
                                        inputMode, rnnMode, inVecReal});
                      
        ///RETURNS std::make_tuple(output, hiddenState, weights, reserveSpace);
        auto reserveSpaceFwdTrain = std::get<3>(fwdTrainOutputPair.second);
        auto curHiddenState = std::get<1>(fwdTrainOutputPair.second);
        
        std::vector<T> dhyin(hx_sz, 0.);
        for(int i = 0; i < hx_sz; i++){
            dhyin[i] = /*(((rand()%2)==1)?-1:1)**/0.001*float(rand()%100);
        }
        
        
        auto yin  = std::get<0>(fwdTrainOutputPair.second);
        
        std::vector<T> dyin(yin.size(), 0.);
        for(int i = 0; i < yin.size(); i++){
            dyin[i] = /*(((rand()%2)==1)?-1:1)**/0.001*float(rand()%100);
        }
        
        
  
        printf("Running backward data RNN.\n");
        auto bwdDataOutputPair = verify(verify_backward_data_rnn<T>{rnnDesc, yin, dyin, dhyin,
                                        curHiddenState, weightData, reserveSpaceFwdTrain, batchSeq, 
                                        hiddenSize, batch_n, 
                                        seqLength, numLayers, 
                                        biasMode, dirMode, 
                                        inputMode, rnnMode, inVecReal});      
                                        
        //RETURNS:  std::make_tuple(dx, dhx, reserveSpace, workSpace);

                                        
        auto workSpaceBwdData = std::get<2>(bwdDataOutputPair.second);                              
        auto reserveSpaceBwdData = std::get<3>(bwdDataOutputPair.second);                              
        printf("Running backward weights RNN.\n"); fflush(nullptr);
        printf("reserve sz: %d, workSpace sz: %d, weight sz: %d\n" ,reserveSpaceBwdData.size(), workSpaceBwdData.size(), wei_sz);fflush(nullptr);
        
        auto dweights_pair = verify(verify_backward_weights_rnn<T>{rnnDesc, inputData, dyin, curHiddenState,
                                        reserveSpaceBwdData, workSpaceBwdData, batchSeq, 
                                        hiddenSize, wei_sz, batch_n, 
                                        seqLength, numLayers, 
                                        biasMode, dirMode, 
                                        inputMode, rnnMode, inVecReal});     
        
        
        auto dweights = std::get<1>(dweights_pair);
        std::transform(weightData.begin( ), weightData.end( ), dweights.begin( ), weightData.begin( ),std::plus<T>( ));
        verify(verify_forward_infer_rnn<T>{rnnDesc, inputData, 
                                        curHiddenState, weightData, batchSeq, 
                                        hiddenSize, batch_n, 
                                        seqLength, numLayers, 
                                        biasMode, dirMode, 
                                        inputMode, rnnMode, inVecReal});
                                        
        
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

