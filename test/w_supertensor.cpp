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
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <utility>

#include "driver.hpp"
#include "test.hpp"

struct superTensorTest
{
    miopenRNNDescriptor_t rnnDesc;

    int num_layer;
    int wei_hh;
    miopenRNNMode_t mode;
    miopenRNNBiasMode_t biasMode;
    miopenRNNDirectionMode_t directionMode;
    miopenRNNInputMode_t inMode;
    miopenRNNAlgo_t algo;
    miopenDataType_t dataType;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t paramTensor;
    miopenTensorDescriptor_t biasTensor;

    superTensorTest()
    {
        miopenCreateRNNDescriptor(&rnnDesc);

        wei_hh = 4;
        inMode = miopenRNNlinear;
        // inMode        = miopenRNNskip;
        directionMode = miopenRNNunidirection;
        // directionMode = miopenRNNbidirection;
        num_layer = 3 * ((directionMode == miopenRNNbidirection) ? 2 : 1);
        // mode          = miopenRNNRELU;
        mode = miopenGRU;
        // mode = miopenLSTM;
        // biasMode = miopenRNNNoBias;
        biasMode = miopenRNNwithBias;
        algo     = miopenRNNdefault;
        dataType = miopenFloat;

        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&paramTensor);
        miopenCreateTensorDescriptor(&biasTensor);
    }

    void run()
    {
        miopenSetRNNDescriptor(
            rnnDesc, wei_hh, num_layer, inMode, directionMode, mode, biasMode, algo, dataType);

        int seqLen    = 1;
        int in_size   = 2;
        size_t in_sz  = 0;
        size_t wei_sz = 0;

        size_t batch_size = 4;

        auto&& handle = get_handle();

        std::array<int, 2> in_lens = {batch_size, in_size};
        miopenSetTensorDescriptor(inputTensor, miopenFloat, 2, in_lens.data(), nullptr);
        miopenSetTensorDescriptor(weightTensor, miopenFloat, 2, in_lens.data(), nullptr);

        miopenGetRNNInputTensorSize(&handle, rnnDesc, seqLen, &inputTensor, &in_sz);

        miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);

        wei_sz = wei_sz / sizeof(miopenFloat);

        printf("inputTensor size: %lu weightTensor size: %lu\n", in_sz, wei_sz);

        std::vector<float> wei_h(wei_sz, 0);
        std::vector<float> bias_h(wei_hh, 1);

        auto wei_dev  = handle.Write(wei_h);
        auto bias_dev = handle.Write(bias_h);

        int num_HiddenLayer = (mode == miopenRNNRELU) ? 1 : (mode == miopenGRU ? 3 : 4);

        for(int layer = 0; layer < num_layer; layer++)
        {

            for(int layerID = 0; layerID < num_HiddenLayer * 2; layerID++)
            {

#if 1
                size_t paramSize = 0;
                miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerID, &paramSize);

                miopenGetRNNLayerParam(&handle,
                                       rnnDesc,
                                       layer,
                                       inputTensor,
                                       weightTensor,
                                       wei_dev.get(),
                                       layerID,
                                       paramTensor,
                                       nullptr);

                std::vector<float> param_h_in(paramSize, layer * 10 + layerID);
                auto param_dev_in  = handle.Write(param_h_in);
                auto param_dev_out = handle.Create(paramSize);

                miopenSetRNNLayerParam(&handle,
                                       rnnDesc,
                                       layer,
                                       inputTensor,
                                       weightTensor,
                                       wei_dev.get(),
                                       layerID,
                                       paramTensor,
                                       param_dev_in.get());
                miopenGetRNNLayerParam(&handle,
                                       rnnDesc,
                                       layer,
                                       inputTensor,
                                       weightTensor,
                                       wei_dev.get(),
                                       layerID,
                                       paramTensor,
                                       param_dev_out.get());

                auto param_h_out = handle.Read<float>(param_dev_out, paramSize / sizeof(float));

                for(int i = 0; i < param_h_out.size(); i++)
                {
                    if(param_h_out[i] != param_h_in[i])
                    {
                        fprintf(stderr,
                                "mismatch at %d in %f != out %f\n",
                                i,
                                param_h_in[i],
                                param_h_out[i]);
                        exit(1);
                    }
                }
#endif

                size_t biasSize = 0;

                miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerID, &biasSize);

                miopenGetRNNLayerBias(&handle,
                                      rnnDesc,
                                      layer,
                                      inputTensor,
                                      weightTensor,
                                      wei_dev.get(),
                                      layerID,
                                      biasTensor,
                                      nullptr);

// fprintf(stderr, "biasSize: %d\n", biasSize);

#if 1
                std::vector<float> bias_h_in(biasSize, layer * 10 + layerID);
                auto bias_dev_in  = handle.Write(bias_h_in);
                auto bias_dev_out = handle.Create(biasSize);

                miopenSetRNNLayerBias(&handle,
                                      rnnDesc,
                                      layer,
                                      inputTensor,
                                      weightTensor,
                                      wei_dev.get(),
                                      layerID,
                                      biasTensor,
                                      bias_dev_in.get());

                miopenGetRNNLayerBias(&handle,
                                      rnnDesc,
                                      layer,
                                      inputTensor,
                                      weightTensor,
                                      wei_dev.get(),
                                      layerID,
                                      biasTensor,
                                      bias_dev_out.get());
#endif

                // auto bias_h_out = handle.Read<float>(bias_dev_out, biasSize / sizeof(float));

                // for(int i = 0; i < bias_h_out.size(); i++)
                //{
                // if(bias_h_out[i] != bias_h_in[i])
                //{
                // fprintf(stderr, "mismatch at %d in %f != out %f\n", i, bias_h_in[i],
                // bias_h_out[i]);
                // exit(1);
                //}
                //}
            }
        }

#if 0
        auto wei_h_out = handle.Read<float>(wei_dev, wei_sz);

        for(int i = 0; i < wei_h_out.size(); i++)
        {
            fprintf(stderr, "%d %f\n", i, wei_h_out[i]);
        }
#endif
    }
};

int main() { run_test<superTensorTest>(); }
