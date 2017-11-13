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

        std::cout << "Wall clock: CPU forward_train_bn_spatial pass time: "
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
        //add(inputMode, "in-mode", generate_data(modes));
        //add(biasMode, "bias-mode", generate_data(modes));
        biasMode = 0;
        dirMode  = 1;
        rnnMode  = 0;
        inputMode= 0;
        //add(dirMode, "dir-mode", generate_data(modes));
        //add(rnnMode, "rnn-mode", generate_data(modes));
		//add(batchSeq, "batch-seq", lazy_generate_data([=]{ return generate_batchSeq(batchSize, seqLength); }, defaultBS));
        
    }

    void run()
    { 
        int modval = 4;
        int scale = 0.9;
        int currentval = batchSize;
        batchSeq.clear();
        for(int i = 0; i < seqLength; i++)
        {            
            if(i>0){
                int nvalue = currentval - rand()%modval;
                currentval = (nvalue<1) ? 1 : nvalue;
                printf("current value: %d\n", currentval);
            }
            printf("adding a value to batch sequence: %d\n", currentval);
            batchSeq.push_back(currentval);
        }
        
        
        for(int i = 0; i < seqLength; i++)
        {
            std::cout << "batch seq[" << i <<"]: " << batchSeq[i] << std::endl;
        }
        
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
        std::size_t in_sz = inVecReal*batch_n;//inVecReal*seqLength*batch_n;
        auto inputTensor = tensor<T>{in_sz};
        srand(0);
        for(int i = 0; i < in_sz; i++){
            inputTensor[i] = 0.01*float(rand()%100);//scale*static_cast<T>((static_cast<double>(rand()) * (1.0 / RAND_MAX)));
        }
        auto inputData = inputTensor.data;
        
        std::size_t hx_sz = ((dirMode)?2:1)*hiddenSize*batchSize*numLayers;
        
        auto hxTensor = tensor<T>{hx_sz};//.generate(rand_gen_small{});
        for(int i = 0; i < hx_sz; i++){
            hxTensor[i] = 0.01*float(rand()%100);//= scale*static_cast<T>((static_cast<double>(rand()) * (1.0 / RAND_MAX)));
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
        auto weightTensor = tensor<T>{std::size_t(wei_sz)};//.generate(rand_gen{});
        for(int i = 0; i < wei_sz; i++){
            weightTensor[i] = 0.01*float(rand()%100);//= scale*static_cast<T>((static_cast<double>(rand()) * (1.0 / RAND_MAX)));
        }

        auto weightData = weightTensor.data;
                
        printf("inputMode: %d, biasMode: %d, rnnMode: %d, dirMode: %d\n", inputMode, biasMode, rnnMode, dirMode);
        printf("hz: %d, batch_n: %d, seqLength: %d, inputLen: %d, numLayers: %d\n" ,hiddenSize, batch_n, seqLength,inVecLen,numLayers);
        verify(verify_forward_train_rnn<T>{rnnDesc, inputData, 
                                        hxData, weightData, batchSeq, 
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

