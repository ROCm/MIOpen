/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include <miopen/miopen.h>

#if MIOPEN_ENABLE_HEUR
#include "../tensor_holder.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/convolution.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/heur/heur.hpp>
#include <miopen/conv/heur/metadata.hpp>

void TestIsApplicable(void)
{
    tensor<float> test_in = tensor<float>(miopenFloat, miopenTensorNCHW, 64, 3, 224, 224);
    tensor<float> test_in_bad_layout =
        tensor<float>(miopenFloat, miopenTensorCHWN, 3, 224, 224, 64);
    tensor<int8_t> test_in_bad_type = tensor<int8_t>(miopenInt8, miopenTensorNCHW, 64, 3, 224, 224);

    tensor<float> test_weights = tensor<float>(miopenFloat, miopenTensorNCHW, 64, 3, 7, 7);
    tensor<float> test_weights_mismatch_height =
        tensor<float>(miopenFloat, miopenTensorNCHW, 64, 3, 5, 7);
    tensor<float> test_weights_bad_layout =
        tensor<float>(miopenFloat, miopenTensorCHWN, 3, 7, 7, 64);

    tensor<float> test_out = tensor<float>(miopenFloat, miopenTensorNCHW, 64, 64, 112, 112);
    tensor<float> test_out_bad_layout =
        tensor<float>(miopenFloat, miopenTensorCHWN, 64, 112, 112, 64);

    miopen::ConvolutionDescriptor conv_desc{};
    miopen::ConvolutionDescriptor conv_desc_mismatch_pads(std::vector<int>{0, 1});
    miopen::ConvolutionDescriptor conv_desc_mismatch_strides(std::vector<int>{0, 0},
                                                             std::vector<int>{2, 1},
                                                             std::vector<int>{1, 1},
                                                             std::vector<int>{0, 0},
                                                             1,
                                                             float(1));

    miopen::ConvolutionDescriptor conv_desc_bad_dilation(std::vector<int>{0, 0},
                                                         std::vector<int>{1, 1},
                                                         std::vector<int>{2, 2},
                                                         std::vector<int>{0, 0},
                                                         1,
                                                         float(1));

    miopen::conv::ProblemDescription conv_prob_desc(test_in.desc,
                                                    test_weights.desc,
                                                    test_out.desc,
                                                    conv_desc,
                                                    miopen::conv::Direction::Forward);

    miopen::conv::ProblemDescription conv_prob_desc_bad_layout(test_in_bad_layout.desc,
                                                               test_weights_bad_layout.desc,
                                                               test_out_bad_layout.desc,
                                                               conv_desc,
                                                               miopen::conv::Direction::Forward);

    miopen::conv::ProblemDescription conv_prob_desc_bad_type(test_in_bad_type.desc,
                                                             test_weights.desc,
                                                             test_out.desc,
                                                             conv_desc,
                                                             miopen::conv::Direction::Forward);

    miopen::conv::ProblemDescription conv_prob_desc_mismatch_height(
        test_in.desc,
        test_weights_mismatch_height.desc,
        test_out.desc,
        conv_desc,
        miopen::conv::Direction::Forward);

    miopen::conv::ProblemDescription conv_prob_desc_mimatch_pads(test_in.desc,
                                                                 test_weights.desc,
                                                                 test_out.desc,
                                                                 conv_desc_mismatch_pads,
                                                                 miopen::conv::Direction::Forward);

    miopen::conv::ProblemDescription conv_prob_desc_mismatch_stride(
        test_in.desc,
        test_weights.desc,
        test_out.desc,
        conv_desc_mismatch_strides,
        miopen::conv::Direction::Forward);

    miopen::conv::ProblemDescription conv_prob_desc_bad_dilation(test_in.desc,
                                                                 test_weights.desc,
                                                                 test_out.desc,
                                                                 conv_desc_bad_dilation,
                                                                 miopen::conv::Direction::Forward);

    EXPECT_EQ(miopen::ConvHeur::IsApplicable("gfx908", conv_prob_desc_bad_type), false)
        << "IsApplicable not catching bad type" << std::endl;
    EXPECT_EQ(miopen::ConvHeur::IsApplicable("gfx908", conv_prob_desc_bad_layout), false)
        << "IsApplicable not catching bad layout" << std::endl;
    EXPECT_EQ(miopen::ConvHeur::IsApplicable("gfx908", conv_prob_desc_mismatch_height), false)
        << "IsApplicable not catching mismatch kernel heights" << std::endl;
    EXPECT_EQ(miopen::ConvHeur::IsApplicable("gfx908", conv_prob_desc_mimatch_pads), false)
        << "IsApplicable not mismatch padding" << std::endl;
    EXPECT_EQ(miopen::ConvHeur::IsApplicable("gfx908", conv_prob_desc_mismatch_stride), false)
        << "IsApplicable not catching mismatch strides" << std::endl;
    EXPECT_EQ(miopen::ConvHeur::IsApplicable("gfx908", conv_prob_desc_bad_dilation), false)
        << "IsApplicable not catching bad dilation" << std::endl;
    EXPECT_EQ(miopen::ConvHeur::IsApplicable("gfx906", conv_prob_desc), false)
        << "IsApplicable not catching un supported arch" << std::endl;
}

void TestEstimateCaching(void)
{
    tensor<float> test_in      = tensor<float>(miopenFloat, miopenTensorNCHW, 64, 3, 224, 224);
    tensor<float> test_weights = tensor<float>(miopenFloat, miopenTensorNCHW, 64, 3, 7, 7);
    tensor<float> test_out     = tensor<float>(miopenFloat, miopenTensorNCHW, 64, 64, 112, 112);
    miopen::ConvolutionDescriptor conv_desc{};
    miopen::conv::ProblemDescription conv_prob_desc(test_in.desc,
                                                    test_weights.desc,
                                                    test_out.desc,
                                                    conv_desc,
                                                    miopen::conv::Direction::Forward);
    ASSERT_EQ(miopen::ConvHeur::IsApplicable("gfx908", conv_prob_desc), true)
        << "Problem description or arch is not applicable" << std::endl;
    bool is_cached = false;
    auto solvers   = miopen::ConvHeur{}.Estimate("gfx908", conv_prob_desc, is_cached);
    ASSERT_EQ(is_cached, false);
    auto solvers_cached = miopen::ConvHeur{}.Estimate("gfx908", conv_prob_desc, is_cached);
    ASSERT_EQ(is_cached, true);
    for(int i = 0; i < solvers.size(); i++)
    {
        EXPECT_EQ(solvers[i], solvers_cached[i])
            << "solvers do not match at index " << i << std::endl;
    }
}

void TestSolverMap(void)
{
    const auto& solver_map = miopen::GetSolverMap("gfx908");

    const std::vector<std::string> solvers = {"ConvBinWinogradRxSf2x3g1",
                                              "ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC",
                                              "ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC",
                                              "ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC",
                                              "GemmFwd1x1_0_1",
                                              "GemmBwdRest",
                                              "GemmBwd1x1_stride1",
                                              "GemmFwd1x1_0_2",
                                              "GemmBwd1x1_stride2"};
    for(size_t i = 0; i < 9; i++)
    {
        EXPECT_EQ(solver_map.at(i), solvers[i])
            << "solver maps do not match at index" << i << std::endl;
    }
}

void TestTranformFeatures(void)
{
    const std::vector<float> init_features = {
        312.0f, 1.0f, 171.0f, 181.0f, 1.0f, 2.0f, 2.0f, 314.0f, 1.0f, 191.0f, 203.0f, 19.0f,
        1.0f,   0.5f, 0.5f,   1.0f,   1.3f, 1.3f, 1.0f, 1.0f,   0.0f, 0.1f,   1.0f,   1.4f};
    const std::vector<float> manual_mean = {
        312.35211181640625,    1.0006444454193115,  171.07810974121094, 181.9158477783203,
        1.0000308752059937,    2.1867706775665283,  2.1919846534729004, 315.09002685546875,
        1.0008162260055542,    191.78111267089844,  203.95440673828125, 19.30280303955078,
        0.9998764991760254,    0.5854535102844238,  0.5849190354347229, 1.0000327825546265,
        1.3545188903808594,    1.3545188903808594,  1.0090287923812866, 1.0090287923812866,
        0.0001389347162330523, 0.08208919316530228, 0.9971132278442383, 1.3931447267532349};
    const std::vector<float> manual_std = {
        391.0542907714844,    0.08177872747182846, 181.0471954345703,   191.44808959960938,
        0.007857984863221645, 1.8559739589691162,  1.8844841718673706,  388.1698303222656,
        0.10612594336271286,  218.2330780029297,   230.737060546875,    95.23401641845703,
        0.011112269014120102, 1.0408027172088623,  1.0403735637664795,  0.00572739215567708,
        0.4816601276397705,   0.4816601276397705,  0.48016685247421265, 0.48016685247421265,
        0.011786249466240406, 0.3345409035682678,  0.8201566934585571,  8.84158992767334};

    std::vector<float> function_transform(init_features);

    miopen::TransformFeatures(function_transform, "gfx908");
    for(size_t i = 0; i < init_features.size(); i++)
    {
        EXPECT_FLOAT_EQ((init_features[i] - manual_mean[i]) / manual_std[i], function_transform[i])
            << "Transformed features don't match at index: " << i << std::endl;
    }
}

void TestGetDirectionMap(void)
{
    EXPECT_EQ(0, miopen::GetDirectionMap(miopen::conv::Direction::BackwardData, "gfx908"));
    EXPECT_EQ(1, miopen::GetDirectionMap(miopen::conv::Direction::BackwardWeights, "gfx908"));
    EXPECT_EQ(2, miopen::GetDirectionMap(miopen::conv::Direction::Forward, "gfx908"));
}

void TestGetPrecisionMap(void)
{
    EXPECT_EQ(0, miopen::GetPrecisionMap(miopenFloat, "gfx908"));
    EXPECT_EQ(1, miopen::GetPrecisionMap(miopenHalf, "gfx908"));
    EXPECT_EQ(2, miopen::GetPrecisionMap(miopenBFloat16, "gfx908"));
}

void TestGetLayoutMap(void)
{
    EXPECT_EQ(0, miopen::GetLayoutMap("NCHW", "gfx908"));
    EXPECT_EQ(1, miopen::GetLayoutMap("NCDHW", "gfx908"));
}

size_t find_max_index(const std::vector<float>& output_vector)
{
    size_t max_index = 0;
    for(size_t i = 0; i < output_vector.size(); i++)
    {
        if(output_vector[i] > output_vector[max_index])
            max_index = i;
    }
    return max_index;
}

void TestModelAccuracy(void)
{
    std::vector<std::vector<float>> test_vectors = {
        {-0.1446344405412674,   -0.0085555799305439,  0.48501718044281006,  0.49151259660720825,
         -0.005675659514963627, -0.6382020711898804,  -0.6313512921333313,  -0.649233877658844,
         -0.008292003534734249, 0.30213940143585205,  0.3109438419342041,   -0.18217183649539948,
         0.010758177377283573,  -0.574522852897644,   -0.5742151141166687,  -0.006217394024133682,
         -0.7351782321929932,   -0.7351782321929932,  -0.01861736737191677, -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, -1.2106455564498901,  -0.04445681348443031},
        {1.8189201354980469,    -0.0085555799305439,  -0.5403382778167725,  -0.6471299529075623,
         -0.005675659514963627, -0.6382020711898804,  -0.6313512921333313,  -0.1556694209575653,
         -0.008292003534734249, -0.5392463803291321,  -0.6319539546966553,  -0.17174085974693298,
         0.010758177377283573,  -0.574522852897644,   -0.5742151141166687,  -0.006217394024133682,
         -0.7351782321929932,   -0.7351782321929932,  -0.01861736737191677, -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, 1.2254846096038818,   -0.04445681348443031},
        {-0.1446344405412674,   -0.0085555799305439,  0.9085335731506348,   0.4497275650501251,
         -0.005675659514963627, -0.6382020711898804,  -0.6313512921333313,  -0.649233877658844,
         -0.008292003534734249, 0.6496683359146118,   0.27634209394454956,  -0.18217183649539948,
         0.010758177377283573,  -0.574522852897644,   -0.5742151141166687,  -0.006217394024133682,
         -0.7351782321929932,   -0.7351782321929932,  -0.01861736737191677, -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, -1.2106455564498901,  -0.04445681348443031},
        {-0.7914825081825256,   -0.0085555799305439,  -0.8412578105926514,  -0.7986007928848267,
         -0.005675659514963627, -0.6382020711898804,  -0.6313512921333313,  -0.1556694209575653,
         -0.008292003534734249, -0.7861747741699219,  -0.7573853135108948,  -0.18217183649539948,
         0.010758177377283573,  -0.574522852897644,   -0.5742151141166687,  -0.006217394024133682,
         -0.7351782321929932,   -0.7351782321929932,  -0.01861736737191677, -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, 0.007419522851705551, -0.04445681348443031},
        {-0.1446344405412674,   -0.0085555799305439,  0.5128800868988037,   0.05276959761977196,
         -0.005675659514963627, -0.6382020711898804,  -0.6313512921333313,  -0.649233877658844,
         -0.008292003534734249, 0.32500314712524414,  -0.05237457901239395, -0.18217183649539948,
         0.010758177377283573,  -0.574522852897644,   -0.5742151141166687,  -0.006217394024133682,
         -0.7351782321929932,   -0.7351782321929932,  -0.01861736737191677, -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, 0.007419522851705551, -0.04445681348443031},
        {-0.1446344405412674,   -0.0085555799305439,  0.16180731356143951,  0.7683385610580444,
         -0.005675659514963627, -0.6382020711898804,  -0.6313512921333313,  -0.4847123920917511,
         -0.008292003534734249, 0.0369199775159359,   0.5401804447174072,   -0.1613098680973053,
         0.010758177377283573,  -0.574522852897644,   -0.5742151141166687,  -0.006217394024133682,
         -0.7351782321929932,   -0.7351782321929932,  -0.01861736737191677, -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, 1.2254846096038818,   -0.04445681348443031},
        {-0.1446344405412674,   -0.0085555799305439,  0.00020237349963281304, -0.06213928759098053,
         -0.005675659514963627, -0.6382020711898804,  -0.6313512921333313,    -0.649233877658844,
         -0.008292003534734249, -0.09568973630666733, -0.14752940833568573,   -0.18217183649539948,
         0.010758177377283573,  -0.574522852897644,   -0.5742151141166687,    -0.006217394024133682,
         -0.7351782321929932,   -0.7351782321929932,  -0.01861736737191677,   -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, -1.2106455564498901,    -0.04445681348443031},
        {-0.6355230808258057,   -0.0085555799305439,  -0.7966771125793457,  -0.8038238883018494,
         -0.005675659514963627, -0.6382020711898804,  -0.6313512921333313,  0.008852057158946991,
         -0.008292003534734249, -0.7495928406715393,  -0.761710524559021,   0.13075770437717438,
         0.010758177377283573,  -0.574522852897644,   -0.5742151141166687,  -0.006217394024133682,
         -0.7351782321929932,   -0.7351782321929932,  -0.01861736737191677, -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, 0.007419522851705551, -0.04445681348443031},
        {4.436992645263672,     -0.0085555799305439,  -0.7632416486740112,  -0.7620388865470886,
         -0.005675659514963627, -0.6382020711898804,  -0.6313512921333313,  1.8185882568359375,
         -0.008292003534734249, -0.571255624294281,   -0.5757260918617249,  -0.1508788913488388,
         0.010758177377283573,  -0.574522852897644,   -0.5742151141166687,  -0.006217394024133682,
         1.3398629426956177,    1.3398629426956177,   -0.01861736737191677, -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, -1.2106455564498901,  -0.04445681348443031},
        {-0.6355230808258057,   -0.0085555799305439,  1.4323564767837524,   1.72417151927948,
         -0.005675659514963627, 2.590287446975708,    2.5474672317504883,   -0.8060433864593506,
         -0.008292003534734249, 3.0366432666778564,   3.546207904815674,    -0.18217183649539948,
         0.010758177377283573,  2.3763999938964844,   2.3782155513763428,   -0.006217394024133682,
         1.3398629426956177,    1.3398629426956177,   -0.01861736737191677, -0.01861736737191677,
         -0.012163897044956684, -0.24550098180770874, 0.007419522851705551, -0.04445681348443031}

    };

    std::vector<size_t> outputs = {6, 4, 6, 1, 1, 4, 6, 1, 8, 1};

    for(int i = 0; i < test_vectors.size(); i++)
    {
        std::vector<float> output = miopen::CallModel(test_vectors[i], "gfx908");
        EXPECT_EQ(outputs[i], find_max_index(output))
            << "Model not accurate for vector " << i << std::endl;
    }
}

TEST(HEUR_TEST, TestEstimateCaching) { TestEstimateCaching(); }

TEST(HEUR_TEST, TestIsApplicable) { TestIsApplicable(); }

TEST(HEUR_TEST, TestMetadata)
{
    TestSolverMap();
    TestTranformFeatures();
    TestGetDirectionMap();
    TestGetPrecisionMap();
    TestGetLayoutMap();
}

TEST(HEUR_TEST, TestModelAccuracy) { TestModelAccuracy(); }
#endif
