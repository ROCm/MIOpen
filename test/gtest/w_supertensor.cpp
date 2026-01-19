/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <vector>

#include <miopen/float_equal.hpp>
#include <miopen/miopen.h>

#include "gtest_common.hpp"
#include "test_parameter_name_generator.hpp"
#include "verify.hpp"

namespace {

using TestCase = std::tuple<NamedParameter<int>,
                            NamedParameter<int>,
                            NamedParameter<int>,
                            NamedParameter<int>,
                            NamedParameter<int>,
                            NamedParameter<miopenRNNMode_t>,
                            NamedParameter<miopenRNNBiasMode_t>,
                            NamedParameter<miopenRNNDirectionMode_t>,
                            NamedParameter<miopenRNNInputMode_t>>;

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
    auto status   = miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);

    EXPECT_EQUAL(status, miopenStatusSuccess);

    wei_sz /= sizeof(float);
    std::vector<float> wei_h(wei_sz, 0);

    int offset = 0;

    const int num_HiddenLayer = (mode == miopenRNNRELU) ? 1 : (mode == miopenGRU ? 3 : 4);

    if(directionMode == miopenRNNbidirection)
    {
        for(int k = 0; k < num_layer * 4; ++k)
        {
            for(int j = 0; j < num_HiddenLayer; ++j)
            {
                const int layer   = k % 2 + (k / 4) * 2;
                const int layerId = (k % 4 > 1) ? j + num_HiddenLayer : j;

                size_t paramSize = 0;

                status = miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerId, &paramSize);

                EXPECT_EQUAL(status, miopenStatusSuccess);

                if((inMode == miopenRNNskip) && (layer < 2) && (layerId < num_HiddenLayer))
                {
                    EXPECT_EQUAL(paramSize, 0);
                    continue;
                }

                paramSize /= sizeof(float);

                for(size_t i = 0; i < paramSize; ++i)
                {
                    wei_h[offset + i] = layer * 10 + layerId;
                }

                offset += paramSize;
            }
        }

        if(biasMode == miopenRNNwithBias)
        {
            for(int k = 0; k < num_layer * 4; ++k)
            {
                for(int j = 0; j < num_HiddenLayer; ++j)
                {
                    const int layer   = k % 2 + (k / 4) * 2;
                    const int layerID = (k % 4 > 1) ? j + num_HiddenLayer : j;

                    size_t biasSize = 0;
                    status = miopenGetRNNLayerBiasSize(&handle, rnnDesc, layer, layerID, &biasSize);

                    EXPECT_EQUAL(status, miopenStatusSuccess);

                    biasSize /= sizeof(float);

                    for(size_t i = 0; i < biasSize; ++i)
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
        for(int layer = 0; layer < num_layer; ++layer)
        {
            for(int layerID = 0; layerID < num_HiddenLayer * 2; ++layerID)
            {
                size_t paramSize = 0;

                status = miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerID, &paramSize);

                EXPECT_EQUAL(status, miopenStatusSuccess);

                if((inMode == miopenRNNskip) && (layer < 1) && (layerID < num_HiddenLayer))
                {
                    EXPECT_EQUAL(paramSize, 0);
                    continue;
                }

                paramSize /= sizeof(float);

                for(size_t i = 0; i < paramSize; ++i)
                {
                    wei_h[offset + i] = layer * 10 + layerID;
                }

                offset += paramSize;
            }
        }

        if(biasMode == miopenRNNwithBias)
        {
            for(int layer = 0; layer < num_layer; ++layer)
            {
                for(int layerID = 0; layerID < num_HiddenLayer * 2; ++layerID)
                {
                    size_t biasSize = 0;
                    status = miopenGetRNNLayerBiasSize(&handle, rnnDesc, layer, layerID, &biasSize);

                    EXPECT_EQUAL(status, miopenStatusSuccess);

                    biasSize /= sizeof(float);

                    for(size_t i = 0; i < biasSize; ++i)
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

        wei_dev = handle.Write(generate_w_tensor(
            rnnDesc, mode, inMode, directionMode, biasMode, inputTensor, num_layer));
    }

    std::vector<float> gpu() const
    {
        auto&& handle             = get_handle();
        const int num_HiddenLayer = (mode == miopenRNNRELU) ? 1 : (mode == miopenGRU ? 3 : 4);
        const int bi              = (directionMode == miopenRNNbidirection) ? 2 : 1;

        size_t wei_sz = 0;
        auto status   = miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);

        EXPECT_EQUAL(status, miopenStatusSuccess);

        wei_sz /= sizeof(float);
        std::vector<float> wei_h(wei_sz, 0);

        for(int layer = 0; layer < num_layer * bi; ++layer)
        {
            int layerID = (inMode == miopenRNNskip && layer < bi) ? num_HiddenLayer : 0;

            for(; layerID < num_HiddenLayer * 2; ++layerID)
            {
                size_t paramSize = 0;

                status = miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerID, &paramSize);

                EXPECT_EQUAL(status, miopenStatusSuccess);

                size_t poffset = 0;

                status = miopenGetRNNLayerParamOffset(
                    rnnDesc, layer, inputTensor, layerID, paramTensor, &poffset);

                EXPECT_EQUAL(status, miopenStatusSuccess);

                auto param_dev_out = handle.Create(paramSize);

                status = miopenGetRNNLayerParam(&handle,
                                                rnnDesc,
                                                layer,
                                                inputTensor,
                                                weightTensor,
                                                wei_dev.get(),
                                                layerID,
                                                paramTensor,
                                                param_dev_out.get());

                EXPECT_EQUAL(status, miopenStatusSuccess);

                const auto param_h_out =
                    handle.Read<float>(param_dev_out, paramSize / sizeof(float));

                memcpy(&wei_h[poffset], &param_h_out[0], paramSize);
            }
        }

        if(biasMode == miopenRNNwithBias)
        {
            for(int layer = 0; layer < num_layer * bi; ++layer)
            {
                for(int layerID = 0; layerID < num_HiddenLayer * 2; ++layerID)
                {
                    size_t boffset  = 0;
                    size_t biasSize = 0;

                    status = miopenGetRNNLayerBiasSize(&handle, rnnDesc, layer, layerID, &biasSize);

                    EXPECT_EQUAL(status, miopenStatusSuccess);

                    status = miopenGetRNNLayerBiasOffset(
                        rnnDesc, layer, inputTensor, layerID, biasTensor, &boffset);

                    EXPECT_EQUAL(status, miopenStatusSuccess);

                    auto bias_dev_out = handle.Create(biasSize);

                    status = miopenGetRNNLayerBias(&handle,
                                                   rnnDesc,
                                                   layer,
                                                   inputTensor,
                                                   weightTensor,
                                                   wei_dev.get(),
                                                   layerID,
                                                   biasTensor,
                                                   bias_dev_out.get());

                    EXPECT_EQUAL(status, miopenStatusSuccess);

                    const auto bias_h_out =
                        handle.Read<float>(bias_dev_out, biasSize / sizeof(float));

                    memcpy(&wei_h[boffset], &bias_h_out[0], biasSize);
                }
            }
        }

        return wei_h;
    }

    std::vector<float> cpu() const
    {
        return generate_w_tensor(
            rnnDesc, mode, inMode, directionMode, biasMode, inputTensor, num_layer);
    }
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
        auto status   = miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);
        EXPECT_EQUAL(status, miopenStatusSuccess);
        wei_dev = handle.Create(wei_sz);
    }

    std::vector<float> cpu() const
    {
        return generate_w_tensor(
            rnnDesc, mode, inMode, directionMode, biasMode, inputTensor, num_layer);
    }

    std::vector<float> gpu() const
    {
        auto&& handle             = get_handle();
        const int num_HiddenLayer = (mode == miopenRNNRELU) ? 1 : (mode == miopenGRU ? 3 : 4);
        const int bi              = (directionMode == miopenRNNbidirection) ? 2 : 1;

        size_t wei_sz = 0;
        auto status   = miopenGetRNNParamsSize(&handle, rnnDesc, inputTensor, &wei_sz, miopenFloat);

        EXPECT_EQUAL(status, miopenStatusSuccess);

        for(int layer = 0; layer < num_layer * bi; ++layer)
        {
            int layerID = (inMode == miopenRNNskip && layer < bi) ? num_HiddenLayer : 0;

            for(; layerID < num_HiddenLayer * 2; ++layerID)
            {

                size_t paramSize = 0;

                status = miopenGetRNNLayerParamSize(
                    &handle, rnnDesc, layer, inputTensor, layerID, &paramSize);

                EXPECT_EQUAL(status, miopenStatusSuccess);

                const auto param_dev_out = handle.Create(paramSize);

                status = miopenGetRNNLayerParam(&handle,
                                                rnnDesc,
                                                layer,
                                                inputTensor,
                                                weightTensor,
                                                wei_dev.get(),
                                                layerID,
                                                paramTensor,
                                                nullptr);

                EXPECT_EQUAL(status, miopenStatusSuccess);

                paramSize /= sizeof(float);
                std::vector<float> param_h_in(paramSize, layer * 10 + layerID);
                auto param_dev_in = handle.Write(param_h_in);

                status = miopenSetRNNLayerParam(&handle,
                                                rnnDesc,
                                                layer,
                                                inputTensor,
                                                weightTensor,
                                                wei_dev.get(),
                                                layerID,
                                                paramTensor,
                                                param_dev_in.get());

                EXPECT_EQUAL(status, miopenStatusSuccess);
            }

            for(layerID = 0; layerID < num_HiddenLayer * 2; ++layerID)
            {
                if(biasMode == miopenRNNwithBias)
                {
                    size_t biasSize = 0;

                    status = miopenGetRNNLayerBiasSize(&handle, rnnDesc, layer, layerID, &biasSize);

                    EXPECT_EQUAL(status, miopenStatusSuccess);

                    const auto bias_dev_out = handle.Create(biasSize);

                    status = miopenGetRNNLayerBias(&handle,
                                                   rnnDesc,
                                                   layer,
                                                   inputTensor,
                                                   weightTensor,
                                                   wei_dev.get(),
                                                   layerID,
                                                   biasTensor,
                                                   nullptr);

                    EXPECT_EQUAL(status, miopenStatusSuccess);

                    biasSize /= sizeof(float);
                    std::vector<float> bias_h_in(biasSize, -(layer * 10 + layerID));
                    auto bias_dev_in = handle.Write(bias_h_in);

                    status = miopenSetRNNLayerBias(&handle,
                                                   rnnDesc,
                                                   layer,
                                                   inputTensor,
                                                   weightTensor,
                                                   wei_dev.get(),
                                                   layerID,
                                                   biasTensor,
                                                   bias_dev_in.get());

                    EXPECT_EQUAL(status, miopenStatusSuccess);
                }
            }
        }

        wei_sz /= sizeof(float);

        return handle.Read<float>(wei_dev, wei_sz);
    }
};

inline auto GenCases()
{
    return testing::Combine(
        MakeNamedParameterValues<int>("seqLen", 1, 2, 4),
        MakeNamedParameterValues<int>("batch_size", 2, 4, 8),
        MakeNamedParameterValues<int>("num_layer", 4, 8, 16),
        MakeNamedParameterValues<int>("in_size", 2, 8, 16),
        MakeNamedParameterValues<int>("wei_hh", 4, 8, 16),
        MakeNamedParameterValues<miopenRNNMode_t>("mode", miopenRNNRELU, miopenLSTM, miopenGRU),
        MakeNamedParameterValues<miopenRNNBiasMode_t>(
            "biasMode", miopenRNNwithBias, miopenRNNNoBias),
        MakeNamedParameterValues<miopenRNNDirectionMode_t>(
            "directionMode", miopenRNNunidirection, miopenRNNbidirection),
        MakeNamedParameterValues<miopenRNNInputMode_t>("inMode", miopenRNNskip, miopenRNNlinear));
}

inline auto GetCases()
{
    static const auto cases = GenCases();
    return cases;
}

}; // namespace

std::ostream& operator<<(std::ostream& os, miopenRNNMode_t param)
{
    switch(param)
    {
    case miopenRNNRELU: os << "miopenRNNRELU"; break;

    case miopenRNNTANH: os << "miopenRNNTANH"; break;

    case miopenLSTM: os << "miopenLSTM"; break;

    case miopenGRU: os << "miopenGRU"; break;

    default: break;
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, miopenRNNBiasMode_t param)
{
    switch(param)
    {
    case miopenRNNNoBias: os << "miopenRNNNoBias"; break;

    case miopenRNNwithBias: os << "miopenRNNwithBias"; break;

    default: break;
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, miopenRNNDirectionMode_t param)
{
    switch(param)
    {
    case miopenRNNunidirection: os << "miopenRNNunidirection"; break;

    case miopenRNNbidirection: os << "miopenRNNbidirection"; break;

    default: break;
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, miopenRNNInputMode_t param)
{
    switch(param)
    {
    case miopenRNNlinear: os << "miopenRNNlinear"; break;

    case miopenRNNskip: os << "miopenRNNskip"; break;

    default: break;
    }

    return os;
}

struct WSuperTensorTest : public testing::TestWithParam<TestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        std::tie(
            seqLen, batch_size, num_layer, in_size, wei_hh, mode, biasMode, directionMode, inMode) =
            GetParam();

        auto status = miopenCreateRNNDescriptor(&rnnDesc);
        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenCreateTensorDescriptor(&inputTensor);
        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenCreateTensorDescriptor(&weightTensor);
        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenCreateTensorDescriptor(&paramTensor);
        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenCreateTensorDescriptor(&biasTensor);
        EXPECT_EQ(status, miopenStatusSuccess);
    }

    void Run()
    {
        if(inMode == miopenRNNskip && in_size != wei_hh)
        {
            return;
        }

        const std::array<int, 2> in_lens{{batch_size, in_size}};

        auto status = miopenSetRNNDescriptor(
            rnnDesc, wei_hh, num_layer, inMode, directionMode, mode, biasMode, algo, dataType);
        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenSetTensorDescriptor(inputTensor, dataType, 2, in_lens.data(), nullptr);
        EXPECT_EQ(status, miopenStatusSuccess);

        status = miopenSetTensorDescriptor(weightTensor, dataType, 2, in_lens.data(), nullptr);
        EXPECT_EQ(status, miopenStatusSuccess);

        Verify<verify_w_tensor_set>();
        Verify<verify_w_tensor_get>();
    }

private:
    template <typename TOperation>
    void Verify()
    {
        const auto operation = TOperation(rnnDesc,
                                          mode,
                                          inMode,
                                          directionMode,
                                          biasMode,
                                          inputTensor,
                                          weightTensor,
                                          paramTensor,
                                          biasTensor,
                                          num_layer);

        CompareResults(operation);
    }

    void CompareResults(const auto& operation)
    {
        const auto cpu_data = operation.cpu();
        const auto gpu_data = operation.gpu();
        const auto idx      = miopen::mismatch_idx(cpu_data, gpu_data, miopen::float_equal);

        EXPECT_GE(idx, miopen::range_distance(cpu_data));
    }

private:
    int seqLen{};
    int batch_size{};
    int num_layer{};
    int in_size{};
    int wei_hh{};
    miopenRNNMode_t mode{};
    miopenRNNBiasMode_t biasMode{};
    miopenRNNDirectionMode_t directionMode{};
    miopenRNNInputMode_t inMode{};
    miopenRNNDescriptor_t rnnDesc{};
    miopenRNNAlgo_t algo{miopenRNNdefault};
    miopenDataType_t dataType{miopenFloat};
    miopenTensorDescriptor_t inputTensor{};
    miopenTensorDescriptor_t weightTensor{};
    miopenTensorDescriptor_t paramTensor{};
    miopenTensorDescriptor_t biasTensor{};
};

struct TestNameGenerator
{
    std::string operator()(const auto& info)
    {
        const auto& [seqLen,
                     batch_size,
                     num_layer,
                     in_size,
                     wei_hh,
                     mode,
                     biasMode,
                     directionMode,
                     inMode] = info.param;
        std::stringstream ss;
        std::string str;

        ss << "seqLen_" << seqLen() << "_batch_size_" << batch_size() << "_num_layer_"
           << num_layer() << "_in_size_" << in_size() << "_wei_hh_" << wei_hh() << "_mode_"
           << mode() << "_directionMode_" << directionMode() << "_inMode_" << inMode()
           << "_test_id_" << info.index;

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

using CPU_WSuperTensor_NONE = WSuperTensorTest;

TEST_P(CPU_WSuperTensor_NONE, WSuperTensorTest) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, CPU_WSuperTensor_NONE, GetCases(), TestNameGenerator{});
