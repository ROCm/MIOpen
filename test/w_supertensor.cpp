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
//#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/allocator.hpp>
#include <utility>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>

#include "verify.hpp"
#include "driver.hpp"
#include "test.hpp"

std::vector<float> generate_w_tensor(miopenRNNDescriptor_t rnnDesc,
                                     miopenRNNMode_t mode,
                                     miopenRNNInputMode_t inMode,
                                     miopenRNNDirectionMode_t directionMode,
                                     miopenRNNBiasMode_t biasMode,
                                     miopenTensorDescriptor_t inputTensor,
                                     int num_layer)
{
    size_t wei_sz = 0;
    auto&& handle = get_handle();
    miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);

    wei_sz = wei_sz / sizeof(float);
    std::vector<float> wei_h(wei_sz, 0);

    int offset = 0;

    int num_HiddenLayer = (mode == miopenRNNRELU) ? 1 : (mode == miopenGRU ? 3 : 4);

    if(directionMode == miopenRNNbidirection)
    {
        for(int k = 0; k < num_layer * 4; k++)
        {
            for(int j = 0; j < num_HiddenLayer; j++)
            {
                int layer   = k % 2 + (k / 4) * 2;
                int layerId = (k % 4 > 1) ? j + num_HiddenLayer : j;

                std::size_t paramSize = 0;
                miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerId, &paramSize);

                if((inMode == miopenRNNskip) && (layer < 2) && (layerId < num_HiddenLayer))
                {
                    EXPECT(paramSize == 0);
                    continue;
                }

                paramSize /= sizeof(float);

                for(std::size_t i = 0; i < paramSize; i++)
                {
                    wei_h[offset + i] = layer * 10 + layerId;
                }

                offset += paramSize;
            }
        }

        if(biasMode == miopenRNNwithBias)
        {
            for(int k = 0; k < num_layer * 4; k++)
            {
                for(int j = 0; j < num_HiddenLayer; j++)
                {

                    int layer   = k % 2 + (k / 4) * 2;
                    int layerID = (k % 4 > 1) ? j + num_HiddenLayer : j;

                    std::size_t biasSize = 0;
                    miopenGetRNNLayerBiasSize(&handle, rnnDesc, layer, layerID, &biasSize);

                    biasSize /= sizeof(float);

                    for(std::size_t i = 0; i < biasSize; i++)
                    {
                        wei_h[offset + i] = -(layer * 10 + layerID);
                    }
                    offset += biasSize;
                }
            }
        }
    }
    else
    {
        for(int layer = 0; layer < num_layer; layer++)
        {
            // int layerID = (inMode == miopenRNNskip && layer < 1) ? num_HiddenLayer : 0;

            for(int layerID = 0; layerID < num_HiddenLayer * 2; layerID++)
            {
                std::size_t paramSize = 0;
                miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerID, &paramSize);

                if((inMode == miopenRNNskip) && (layer < 1) && (layerID < num_HiddenLayer))
                {
                    EXPECT(paramSize == 0);
                    continue;
                }

                paramSize /= sizeof(float);

                for(std::size_t i = 0; i < paramSize; i++)
                {
                    wei_h[offset + i] = layer * 10 + layerID;
                }

                offset += paramSize;
            }
        }

        if(biasMode == miopenRNNwithBias)
        {
            for(int layer = 0; layer < num_layer; layer++)
            {
                // int layerID = (inMode == miopenRNNskip && layer < 1) ? num_HiddenLayer : 0;

                for(int layerID = 0; layerID < num_HiddenLayer * 2; layerID++)
                {
                    std::size_t biasSize = 0;
                    miopenGetRNNLayerBiasSize(&handle, rnnDesc, layer, layerID, &biasSize);

                    biasSize /= sizeof(float);

                    for(std::size_t i = 0; i < biasSize; i++)
                    {
                        wei_h[offset + i] = -(layer * 10 + layerID);
                    }
                    offset += biasSize;
                }
            }
        }
    }
    return wei_h;
}

struct verify_w_tensor_get
{
    miopenRNNDescriptor_t rnnDesc;

    miopenRNNMode_t mode;
    miopenRNNInputMode_t inMode;
    miopenRNNAlgo_t algo = miopenRNNdefault;
    miopenRNNDirectionMode_t directionMode;
    miopenRNNBiasMode_t biasMode;

    int num_layer;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t paramTensor;
    miopenTensorDescriptor_t biasTensor;

    miopen::Allocator::ManageDataPtr wei_dev;

    verify_w_tensor_get(miopenRNNDescriptor_t p_rnnDesc,
                        miopenRNNMode_t p_mode,
                        miopenRNNInputMode_t p_inMode,
                        miopenRNNDirectionMode_t p_directionMode,
                        miopenRNNBiasMode_t p_biasMode,
                        miopenTensorDescriptor_t p_inputTensor,
                        miopenTensorDescriptor_t p_weightTensor,
                        miopenTensorDescriptor_t p_paramTensor,
                        miopenTensorDescriptor_t p_biasTensor,
                        int p_num_layer)
    {
        rnnDesc       = p_rnnDesc;
        mode          = p_mode;
        inMode        = p_inMode;
        directionMode = p_directionMode;
        biasMode      = p_biasMode;
        inputTensor   = p_inputTensor;
        weightTensor  = p_weightTensor;
        paramTensor   = p_paramTensor;
        biasTensor    = p_biasTensor;
        num_layer     = p_num_layer;

        auto&& handle = get_handle();
        wei_dev       = handle.Write(generate_w_tensor(
            rnnDesc, mode, inMode, directionMode, biasMode, inputTensor, num_layer));
    }

    std::vector<float> gpu() const
    {
        auto&& handle       = get_handle();
        int num_HiddenLayer = (mode == miopenRNNRELU) ? 1 : (mode == miopenGRU ? 3 : 4);
        int bi              = (directionMode == miopenRNNbidirection) ? 2 : 1;

        size_t wei_sz = 0;
        miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);

        wei_sz = wei_sz / sizeof(float);
        std::vector<float> wei_h(wei_sz, 0);

        for(int layer = 0; layer < num_layer * bi; layer++)
        {
            int layerID = (inMode == miopenRNNskip && layer < bi) ? num_HiddenLayer : 0;

            for(; layerID < num_HiddenLayer * 2; layerID++)
            {

                size_t paramSize = 0;
                miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerID, &paramSize);

                size_t poffset = 0;
                auto err       = miopenGetRNNLayerParamOffset(
                    rnnDesc, layer, inputTensor, layerID, paramTensor, &poffset);

                if(err != miopenStatusSuccess)
                {
                    std::cout << "Error in call to miopenGetRNNLayerParamOffset" << std::endl;
                }

                auto param_dev_out = handle.Create(paramSize);

                miopenGetRNNLayerParam(&handle,
                                       rnnDesc,
                                       layer,
                                       inputTensor,
                                       weightTensor,
                                       wei_dev.get(),
                                       layerID,
                                       paramTensor,
                                       param_dev_out.get());

                paramSize /= sizeof(float);

                auto param_h_out = handle.Read<float>(param_dev_out, paramSize);

                memcpy(&wei_h[poffset], &param_h_out[0], sizeof(float) * paramSize);
            }
        }

        if(biasMode == miopenRNNwithBias)
        {
            for(int layer = 0; layer < num_layer * bi; layer++)
            {
                int layerID = 0;

                for(; layerID < num_HiddenLayer * 2; layerID++)
                {
                    size_t boffset  = 0;
                    size_t biasSize = 0;

                    miopenGetRNNLayerBiasSize(&handle, rnnDesc, layer, layerID, &biasSize);
                    auto err = miopenGetRNNLayerBiasOffset(
                        rnnDesc, layer, inputTensor, layerID, biasTensor, &boffset);

                    if(err != miopenStatusSuccess)
                    {
                        std::cout << "Error in call to miopenGetRNNLayerBiasOffset" << std::endl;
                    }

                    auto bias_dev_out = handle.Create(biasSize);

                    miopenGetRNNLayerBias(&handle,
                                          rnnDesc,
                                          layer,
                                          inputTensor,
                                          weightTensor,
                                          wei_dev.get(),
                                          layerID,
                                          biasTensor,
                                          bias_dev_out.get());

                    biasSize /= sizeof(float);

                    auto bias_h_out = handle.Read<float>(bias_dev_out, biasSize);

                    memcpy(&wei_h[boffset], &bias_h_out[0], sizeof(float) * biasSize);
                }
            }
        }

        // for(int i = 0; i < wei_sz; i++)
        //{
        // printf("GPU [%d]: %f\n", i, wei_h[i]);
        //}

        return wei_h;
    }

    std::vector<float> cpu() const
    {
        return generate_w_tensor(
            rnnDesc, mode, inMode, directionMode, biasMode, inputTensor, num_layer);
    }

    void fail(float = 0) const {}
};

struct verify_w_tensor_set
{
    miopenRNNDescriptor_t rnnDesc;
    miopenRNNMode_t mode;
    miopenRNNInputMode_t inMode;
    miopenRNNDirectionMode_t directionMode;
    miopenRNNBiasMode_t biasMode;
    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t paramTensor;
    miopenTensorDescriptor_t biasTensor;

    miopen::Allocator::ManageDataPtr wei_dev;

    int num_layer;

    verify_w_tensor_set(miopenRNNDescriptor_t p_rnnDesc,
                        miopenRNNMode_t p_mode,
                        miopenRNNInputMode_t p_inMode,
                        miopenRNNDirectionMode_t p_directionMode,
                        miopenRNNBiasMode_t p_biasMode,
                        miopenTensorDescriptor_t p_inputTensor,
                        miopenTensorDescriptor_t p_weightTensor,
                        miopenTensorDescriptor_t p_paramTensor,
                        miopenTensorDescriptor_t p_biasTensor,
                        int p_num_layer)
    {
        rnnDesc       = p_rnnDesc;
        mode          = p_mode;
        inMode        = p_inMode;
        directionMode = p_directionMode;
        biasMode      = p_biasMode;
        inputTensor   = p_inputTensor;
        weightTensor  = p_weightTensor;
        paramTensor   = p_paramTensor;
        biasTensor    = p_biasTensor;
        num_layer     = p_num_layer;

        size_t wei_sz = 0;
        auto&& handle = get_handle();
        miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);
        wei_dev = handle.Create(wei_sz);
    }

    std::vector<float> cpu() const
    {
        return generate_w_tensor(
            rnnDesc, mode, inMode, directionMode, biasMode, inputTensor, num_layer);
    }

    std::vector<float> gpu() const
    {
        auto&& handle       = get_handle();
        int num_HiddenLayer = (mode == miopenRNNRELU) ? 1 : (mode == miopenGRU ? 3 : 4);
        int bi              = (directionMode == miopenRNNbidirection) ? 2 : 1;

        size_t wei_sz = 0;
        miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);

        for(int layer = 0; layer < num_layer * bi; layer++)
        {
            int layerID = 0;

            if(inMode == miopenRNNskip && layer < bi)
            {
                layerID = num_HiddenLayer;
            }

            for(; layerID < num_HiddenLayer * 2; layerID++)
            {

                size_t paramSize = 0;
                miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerID, &paramSize);

                auto param_dev_out = handle.Create(paramSize);

                miopenGetRNNLayerParam(&handle,
                                       rnnDesc,
                                       layer,
                                       inputTensor,
                                       weightTensor,
                                       wei_dev.get(),
                                       layerID,
                                       paramTensor,
                                       nullptr);

                paramSize /= sizeof(float);
                std::vector<float> param_h_in(paramSize, layer * 10 + layerID);
                auto param_dev_in = handle.Write(param_h_in);

                miopenSetRNNLayerParam(&handle,
                                       rnnDesc,
                                       layer,
                                       inputTensor,
                                       weightTensor,
                                       wei_dev.get(),
                                       layerID,
                                       paramTensor,
                                       param_dev_in.get());
            }

            layerID = 0;

            for(; layerID < num_HiddenLayer * 2; layerID++)
            {
                if(biasMode == miopenRNNwithBias)
                {
                    size_t biasSize = 0;

                    miopenGetRNNLayerBiasSize(&handle, rnnDesc, layer, layerID, &biasSize);

                    auto bias_dev_out = handle.Create(biasSize);

                    miopenGetRNNLayerBias(&handle,
                                          rnnDesc,
                                          layer,
                                          inputTensor,
                                          weightTensor,
                                          wei_dev.get(),
                                          layerID,
                                          biasTensor,
                                          nullptr);

                    biasSize /= sizeof(float);
                    std::vector<float> bias_h_in(biasSize, -(layer * 10 + layerID));
                    auto bias_dev_in = handle.Write(bias_h_in);

                    miopenSetRNNLayerBias(&handle,
                                          rnnDesc,
                                          layer,
                                          inputTensor,
                                          weightTensor,
                                          wei_dev.get(),
                                          layerID,
                                          biasTensor,
                                          bias_dev_in.get());
                }
            }
        }

        wei_sz = wei_sz / sizeof(float);

        // auto wei_h = handle.Read<float>(wei_dev, wei_sz);
        // for(int i = 0; i < wei_sz; i++)
        //{
        // printf("GPU [%d]: %f\n", i, wei_h[i]);
        //}

        return handle.Read<float>(wei_dev, wei_sz);
    }

    void fail(float = 0) const {}
};

struct superTensorTest : test_driver
{
    miopenRNNDescriptor_t rnnDesc{};

    int num_layer{};
    int wei_hh{};
    int batch_size{};

    std::string mode;
    std::string biasMode;
    std::string directionMode;
    std::string inMode;
    miopenRNNAlgo_t algo = miopenRNNdefault;
    miopenDataType_t dataType{};

    int seqLen{};
    int in_size{};

    miopenTensorDescriptor_t inputTensor{};
    miopenTensorDescriptor_t weightTensor{};
    miopenTensorDescriptor_t paramTensor{};
    miopenTensorDescriptor_t biasTensor{};

    std::unordered_map<std::string, miopenRNNMode_t> mode_lookup = {
        {"RNNRELU", miopenRNNRELU}, {"LSTM", miopenLSTM}, {"GRU", miopenGRU}};
    std::unordered_map<std::string, miopenRNNBiasMode_t> biasMode_lookup = {
        {"RNNwithBias", miopenRNNwithBias}, {"RNNNoBias", miopenRNNNoBias}};
    std::unordered_map<std::string, miopenRNNDirectionMode_t> directionMode_lookup = {
        {"RNNunidirection", miopenRNNunidirection}, {"RNNbidirection", miopenRNNbidirection}};
    std::unordered_map<std::string, miopenRNNInputMode_t> inMode_lookup = {
        {"RNNskip", miopenRNNskip}, {"RNNlinear", miopenRNNlinear}};

    superTensorTest()
    {
        disabled_cache = true;
        miopenCreateRNNDescriptor(&rnnDesc);

        dataType = miopenFloat;

#if 1
        std::vector<int> get_seqLen     = {1, 2, 4};
        std::vector<int> get_batch_size = {2, 4, 8};
        std::vector<int> get_num_layer  = {4, 8, 16};
        std::vector<int> get_in_size    = {2, 8, 16};
        std::vector<int> get_wei_hh     = {4, 8, 16};
#else
        std::vector<int> get_seqLen     = {2};
        std::vector<int> get_batch_size = {4};
        std::vector<int> get_num_layer  = {8};
        std::vector<int> get_in_size    = {2};
        std::vector<int> get_wei_hh     = {6};
#endif

        add(seqLen, "seqLen", generate_data(get_seqLen));
        add(in_size, "in_size", generate_data(get_in_size));
        add(batch_size, "batch_size", generate_data(get_batch_size));
        add(num_layer, "num_layer", generate_data(get_num_layer));
        add(wei_hh, "wei_hh", generate_data(get_wei_hh));
        add(mode, "mode", generate_data({"RNNRELU", "LSTM", "GRU"}));
        add(biasMode, "biasMode", generate_data({"RNNwithBias", "RNNNoBias"}));
        add(directionMode, "directionMode", generate_data({"RNNunidirection", "RNNbidirection"}));
        add(inMode, "inMode", generate_data({"RNNskip", "RNNlinear"}));
        // add(algo, "algo", generate_data(get_algo));

        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&paramTensor);
        miopenCreateTensorDescriptor(&biasTensor);
    }

    void run()
    {

        miopenSetRNNDescriptor(rnnDesc,
                               wei_hh,
                               num_layer,
                               inMode_lookup[inMode],
                               directionMode_lookup[directionMode],
                               mode_lookup[mode],
                               biasMode_lookup[biasMode],
                               algo,
                               dataType);

        std::array<int, 2> in_lens = {{batch_size, in_size}};
        miopenSetTensorDescriptor(inputTensor, dataType, 2, in_lens.data(), nullptr);
        miopenSetTensorDescriptor(weightTensor, dataType, 2, in_lens.data(), nullptr);

        verify_equals(verify_w_tensor_set(rnnDesc,
                                          mode_lookup[mode],
                                          inMode_lookup[inMode],
                                          directionMode_lookup[directionMode],
                                          biasMode_lookup[biasMode],
                                          inputTensor,
                                          weightTensor,
                                          paramTensor,
                                          biasTensor,
                                          num_layer));

        verify_equals(verify_w_tensor_get(rnnDesc,
                                          mode_lookup[mode],
                                          inMode_lookup[inMode],
                                          directionMode_lookup[directionMode],
                                          biasMode_lookup[biasMode],
                                          inputTensor,
                                          weightTensor,
                                          paramTensor,
                                          biasTensor,
                                          num_layer));
    }
};

int main(int argc, const char* argv[]) { test_drive<superTensorTest>(argc, argv); }
