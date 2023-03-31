#include <miopen/miopen.h>
#include "../tensor_holder.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/convolution.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/solver.hpp>
#include "get_handle.hpp"

void Test_908_ConvAsm1x1U(void)
{
#if MIOPEN_ENABLE_AI_HEUR
    auto&& handle = get_handle();
    if(handle.GetDeviceName() != "gfx908")
        return;
    tensor<float> test_inputs_f  = tensor<float>(miopenHalf, miopenTensorNCHW, 256, 2048, 7, 7);
    tensor<float> test_weights_f = tensor<float>(miopenHalf, miopenTensorNCHW, 512, 2048, 1, 1);
    tensor<float> test_outputs_f = tensor<float>(miopenHalf, miopenTensorNCHW, 256, 512, 7, 7);

    tensor<float> test_inputs_b  = tensor<float>(miopenFloat, miopenTensorNCHW, 512, 192, 56, 56);
    tensor<float> test_weights_b = tensor<float>(miopenFloat, miopenTensorNCHW, 288, 192, 1, 1);
    tensor<float> test_outputs_b = tensor<float>(miopenFloat, miopenTensorNCHW, 512, 288, 56, 56);

    tensor<float> test_inputs_fallback  = tensor<float>(miopenFloat, miopenTensorNCHW, 1, 4, 2, 2);
    tensor<float> test_weights_fallback = tensor<float>(miopenFloat, miopenTensorNCHW, 4, 4, 1, 1);
    tensor<float> test_outputs_fallback = tensor<float>(miopenFloat, miopenTensorNCHW, 1, 4, 2, 2);

    miopen::ConvolutionDescriptor conv_desc_fallback{2,
                                                     miopenConvolution,
                                                     miopenPaddingDefault,
                                                     std::vector<int>{0, 0},
                                                     std::vector<int>{1, 1},
                                                     std::vector<int>{1, 1},
                                                     std::vector<int>{0, 0},
                                                     1,
                                                     float(1)};

    miopen::ConvolutionDescriptor conv_desc{std::vector<int>{0, 0},
                                            std::vector<int>{1, 1},
                                            std::vector<int>{1, 1},
                                            std::vector<int>{0, 0},
                                            1,
                                            float(1)};

    miopen::ProblemDescription forward_pd(test_inputs_f.desc,
                                          test_weights_f.desc,
                                          test_outputs_f.desc,
                                          conv_desc,
                                          miopen::conv::Direction::Forward);

    miopen::ConvolutionContext forward(forward_pd);

    forward.SetStream(&handle);
    forward.DetectRocm();

    miopen::ProblemDescription backward_pd(test_inputs_b.desc,
                                           test_weights_b.desc,
                                           test_outputs_b.desc,
                                           conv_desc,
                                           miopen::conv::Direction::BackwardData);

    miopen::ConvolutionContext backward(backward_pd);

    backward.SetStream(&handle);
    backward.DetectRocm();

    miopen::ProblemDescription fallback_pd(test_inputs_fallback.desc,
                                           test_weights_fallback.desc,
                                           test_outputs_fallback.desc,
                                           conv_desc_fallback,
                                           miopen::conv::Direction::Forward);

    miopen::ConvolutionContext fallback(fallback_pd);

    fallback.SetStream(&handle);
    fallback.DetectRocm();

    miopen::solver::PerformanceConfigConvAsm1x1U config_forward;
    config_forward.HeuristicInit(forward, forward_pd);
    miopen::solver::PerformanceConfigConvAsm1x1U config_backward;
    config_backward.HeuristicInit(backward, backward_pd);
    miopen::solver::PerformanceConfigConvAsm1x1U config_fallback;
    config_fallback.HeuristicInit(fallback, fallback_pd);
    EXPECT_EQ(config_forward.ToString(), "2,8,4,16,1,4,1,4")
        << "Forward fp16 test case failed, model predicted: " << config_forward.ToString()
        << " but should have predicted: 2,8,4,16,1,4,1,4";
    EXPECT_EQ(config_backward.ToString(), "1,16,1,64,2,2,1,4")
        << "Backward fp32 test case failed, model predicted: " << config_backward.ToString()
        << " but should have predicted: 1,16,1,64,2,2,1,4";
    EXPECT_EQ(config_fallback.ToString(), "1,4,4,1,1,1,1,1")
        << "Model did not fallback when producing erroneous tokens";
#endif
}

TEST(KERNEL_TUNING_NET_TESTS, Test_908_ConvAsm1x1U) { Test_908_ConvAsm1x1U(); }
