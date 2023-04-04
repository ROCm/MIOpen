#include <miopen/miopen.h>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include "../tensor_holder.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/convolution.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/solver.hpp>
#include "get_handle.hpp"
#include <unordered_map>

static std::vector<std::vector<std::vector<int>>> gfx908_ConvAsm1x1U_tensor_shapes{
    {{256, 2048, 7, 7}, {512, 2048, 1, 1}, {256, 512, 7, 7}},
    {{512, 192, 56, 56}, {288, 192, 1, 1}, {512, 288, 56, 56}},
    {{1, 4, 2, 2}, {4, 4, 1, 1}, {1, 4, 2, 2}}};

static std::vector<miopenDataType_t> gfx908_ConvAsm1x1U_data_types{
    miopenHalf, miopenFloat, miopenFloat};

static std::vector<miopen::conv::Direction> gfx908_ConvAsm1x1U_directions{
    miopen::conv::Direction::Forward,
    miopen::conv::Direction::BackwardData,
    miopen::conv::Direction::Forward};

static std::vector<std::string> gfx908_ConvAsm1x1U_expected_configs{
    "2,8,4,16,1,4,1,4", "1,16,1,64,2,2,1,4", ""};

static std::vector<bool> gfx908_ConvAsm1x1U_expected_valid{true, true, false};

template <typename T, typename G>
void TestParameterPredictionModel(miopen::Handle& handle,
                                  const std::vector<std::vector<int>>& tensor_shapes,
                                  const miopenDataType_t& data_type,
                                  const miopenTensorLayout_t& layout,
                                  const miopen::conv::Direction& direction,
                                  const miopen::ConvolutionDescriptor& conv_desc,
                                  const std::string& expected,
                                  const bool expected_valid)
{
    miopen::ConvolutionContext ctx;
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    bool valid              = false;
    tensor<G> input_tensor  = tensor<G>(data_type, layout, tensor_shapes[0]);
    tensor<G> weight_tensor = tensor<G>(data_type, layout, tensor_shapes[1]);
    tensor<G> output_tensor = tensor<G>(data_type, layout, tensor_shapes[2]);
    miopen::ProblemDescription problem_description(
        input_tensor.desc, weight_tensor.desc, output_tensor.desc, conv_desc, direction);
    T perf_config;
    perf_config.RunParmeterPredictionModel(ctx, problem_description, valid);
    ASSERT_EQ(valid, expected_valid)
        << "Expected parameters to be "
        << (expected_valid ? std::string("valid") : std::string("invalid")) << " but were "
        << (valid ? std::string("valid") : std::string("invalid"));
    if(expected_valid)
    {
        EXPECT_EQ(perf_config.ToString(), expected)
            << "Expected parameters: " << expected
            << "\nPredicted parameters: " << perf_config.ToString();
    }
}

void TestConvAsm1x1UGfx908(void)
{
    auto&& handle = get_handle();
    if(handle.GetDeviceName() != "gfx908")
        GTEST_SKIP();
    miopen::ConvolutionDescriptor conv_desc;
    for(int i = 0; i < gfx908_ConvAsm1x1U_tensor_shapes.size(); i++)
    {
        if(gfx908_ConvAsm1x1U_data_types[i] == miopenFloat)
        {
            TestParameterPredictionModel<miopen::solver::PerformanceConfigConvAsm1x1U, float>(
                handle,
                gfx908_ConvAsm1x1U_tensor_shapes[i],
                gfx908_ConvAsm1x1U_data_types[i],
                miopenTensorNCHW,
                gfx908_ConvAsm1x1U_directions[i],
                conv_desc,
                gfx908_ConvAsm1x1U_expected_configs[i],
                gfx908_ConvAsm1x1U_expected_valid[i]);
        }
        else if(gfx908_ConvAsm1x1U_data_types[i] == miopenHalf)
        {
            TestParameterPredictionModel<miopen::solver::PerformanceConfigConvAsm1x1U,
                                         half_float::half>(handle,
                                                           gfx908_ConvAsm1x1U_tensor_shapes[i],
                                                           gfx908_ConvAsm1x1U_data_types[i],
                                                           miopenTensorNCHW,
                                                           gfx908_ConvAsm1x1U_directions[i],
                                                           conv_desc,
                                                           gfx908_ConvAsm1x1U_expected_configs[i],
                                                           gfx908_ConvAsm1x1U_expected_valid[i]);
        }
    }
}

TEST(KERNEL_TUNING_NET_TESTS, TestConvAsm1x1UGfx908) { TestConvAsm1x1UGfx908(); }
#endif
