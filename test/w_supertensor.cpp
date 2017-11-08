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

    int layer;
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


    superTensorTest() { 
        miopenCreateRNNDescriptor(&rnnDesc);

        wei_hh        = 16;
        layer         = 1;
        inMode        = miopenRNNlinear;
        directionMode = miopenRNNunidirection;
        mode          = miopenRNNRELU;
        biasMode      = miopenRNNNoBias;
        algo          = miopenRNNdefault;
        dataType      = miopenFloat;

        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&paramTensor);
        miopenCreateTensorDescriptor(&biasTensor);
    }

    void run()
    {
        miopenSetRNNDescriptor(
                rnnDesc, wei_hh, layer, inMode, directionMode, mode, biasMode, algo, dataType);

        int seqLen = 1;
        int in_size = 16;
        size_t in_sz  = 0;
        size_t wei_sz = 0;


        auto&& handle = get_handle();

        std::array<int, 2> in_lens = {in_size, 1};
        miopenSetTensorDescriptor(inputTensor, miopenFloat, 2, in_lens.data(), nullptr);
        miopenSetTensorDescriptor(weightTensor, miopenFloat, 2, in_lens.data(), nullptr);

        std::array<int, 2> wei_lens = {wei_hh, 1};
        miopenSetTensorDescriptor(paramTensor, miopenFloat, 2, wei_lens.data(), nullptr);

        miopenSetTensorDescriptor(biasTensor, miopenFloat, 2, wei_lens.data(), nullptr);
        
        miopenGetRNNInputTensorSize(
                &handle, rnnDesc, seqLen, &inputTensor, &in_sz);

        miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);

        printf("inputTensor size: %lu weightTensor size: %lu\n", in_sz, wei_sz);

        auto in_dev = handle.Create(in_sz);
        std::vector<float> wei_h(wei_sz, 0); 
        std::vector<float> param_h(wei_hh, 1);
        std::vector<float> bias_h(wei_hh, 1);

        auto param_dev = handle.Write(param_h);
        auto wei_dev = handle.Write(wei_h);
        auto bias_dev = handle.Write(bias_h);

        for(int layerID = 0; layerID < 8; layerID++)
        {
            miopenSetRNNLayerParam(&handle, rnnDesc, layer, inputTensor, weightTensor, wei_dev.get(), layerID, paramTensor, param_dev.get());
            miopenSetRNNLayerBias(&handle, rnnDesc, layer, inputTensor, weightTensor, wei_dev.get(), layerID, biasTensor, bias_dev.get());
        }

        wei_h = handle.Read<float>(wei_dev, wei_sz);

        for(int i = 0; i < wei_h.size(); i++)
        {
            printf("%d %f\n", i, wei_h[i]);
        }

    }
};

int main() { run_test<superTensorTest>(); }
