/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#include <miopen/manage_ptr.hpp>
#include <miopen/fusion_plan.hpp>

#include "test.hpp"
#include "get_handle.hpp"

void BNAlgTest(std::vector<int> inputs,
               miopenBatchNormMode_t bnmode,
               std::string& pgm,
               std::string& krn,
               std::string& alg)
{
    MIOPEN_LOG_I("*********************************************************");
    auto&& handle = get_handle();
    miopen::TensorDescriptor inputTensor;
    miopen::TensorDescriptor scaleTensor;
    miopenFusionOpDescriptor_t bNormOp = nullptr;

    // input descriptor
    STATUS(miopenSet4dTensorDescriptor(
        &inputTensor, miopenFloat, inputs[0], inputs[1], inputs[2], inputs[3]));
    miopen::FusionPlanDescriptor fp(miopenVerticalFusion, inputTensor);

    miopenCreateOpBatchNormInference(&fp, &bNormOp, bnmode, &scaleTensor);

    pgm = fp.GetProgramName(handle);
    krn = fp.GetKernelName(handle);
    alg = fp.GetAlgorithmName(handle);
}

void ConvAlgTest(std::vector<int> inputs,
                 std::vector<int> conv_filter,
                 std::vector<int> conv_desc,
                 std::string& pgm,
                 std::string& krn,
                 std::string& alg)
{
    MIOPEN_LOG_I("*********************************************************");
    auto&& handle = get_handle();
    miopen::TensorDescriptor inputTensor;
    miopen::TensorDescriptor convFilter;
    miopenConvolutionDescriptor_t convDesc{};
    miopenFusionOpDescriptor_t convoOp;

    // input descriptor
    STATUS(miopenSet4dTensorDescriptor(
        &inputTensor, miopenFloat, inputs[0], inputs[1], inputs[2], inputs[3]));
    // convolution descriptor
    STATUS(miopenSet4dTensorDescriptor(&convFilter,
                                       miopenFloat,
                                       conv_filter[0], // outputs k
                                       conv_filter[1], // inputs c
                                       conv_filter[2], // kernel size
                                       conv_filter[3]));

    STATUS(miopenCreateConvolutionDescriptor(&convDesc));
    STATUS(miopenInitConvolutionDescriptor(convDesc,
                                           miopenConvolution,
                                           conv_desc[0], // pad h
                                           conv_desc[1], // pad w
                                           conv_desc[2], // stride h
                                           conv_desc[3], // stride w
                                           conv_desc[4], // dilations
                                           conv_desc[5]));

    miopen::FusionPlanDescriptor fp(miopenVerticalFusion, inputTensor);

    STATUS(miopenCreateOpConvForward(&fp, &convoOp, convDesc, &convFilter));
    pgm = fp.GetProgramName(handle);
    krn = fp.GetKernelName(handle);
    alg = fp.GetAlgorithmName(handle);

    // Cleanup
    miopenDestroyConvolutionDescriptor(convDesc);
}

void ConvAlgFailTest(std::vector<int> inputs,
                     std::vector<int> conv_filter,
                     std::vector<int> conv_desc)
{
    MIOPEN_LOG_I("*********************************************************");
    miopen::TensorDescriptor inputTensor;
    miopen::TensorDescriptor convFilter;
    miopenConvolutionDescriptor_t convDesc{};
    miopenFusionOpDescriptor_t convoOp;

    // input descriptor
    STATUS(miopenSet4dTensorDescriptor(
        &inputTensor, miopenFloat, inputs[0], inputs[1], inputs[2], inputs[3]));
    // convolution descriptor
    STATUS(miopenSet4dTensorDescriptor(&convFilter,
                                       miopenFloat,
                                       conv_filter[0], // outputs k
                                       conv_filter[1], // inputs c
                                       conv_filter[2], // kernel size
                                       conv_filter[3]));

    STATUS(miopenCreateConvolutionDescriptor(&convDesc));
    STATUS(miopenInitConvolutionDescriptor(convDesc,
                                           miopenConvolution,
                                           conv_desc[0], // pad h
                                           conv_desc[1], // pad w
                                           conv_desc[2], // stride h
                                           conv_desc[3], // stride w
                                           conv_desc[4], // dilations
                                           conv_desc[5]));

    miopen::FusionPlanDescriptor fp(miopenVerticalFusion, inputTensor);

    EXPECT(miopenCreateOpConvForward(&fp, &convoOp, convDesc, &convFilter) != 0);

    // Cleanup
    miopenDestroyConvolutionDescriptor(convDesc);
}

void ConvBiasAlgTest(std::vector<int> inputs,
                     std::vector<int> conv_filter,
                     std::vector<int> conv_desc,
                     std::string& pgm,
                     std::string& krn,
                     std::string& alg)
{
    MIOPEN_LOG_I("*********************************************************");
    auto&& handle = get_handle();
    miopen::TensorDescriptor inputTensor;
    miopen::TensorDescriptor convFilter;
    miopen::TensorDescriptor biasTensor;
    miopenConvolutionDescriptor_t convDesc{};
    miopenFusionOpDescriptor_t convOp;
    miopenFusionOpDescriptor_t biasOp;

    // input descriptor
    STATUS(miopenSet4dTensorDescriptor(
        &inputTensor, miopenFloat, inputs[0], inputs[1], inputs[2], inputs[3]));
    // convolution descriptor
    STATUS(miopenSet4dTensorDescriptor(&convFilter,
                                       miopenFloat,
                                       conv_filter[0], // outputs k
                                       conv_filter[1], // inputs c
                                       conv_filter[2], // kernel size
                                       conv_filter[3]));
    // bias descriptor
    STATUS(miopenSet4dTensorDescriptor(&biasTensor, miopenFloat, 1, conv_filter[0], 1, 1));

    STATUS(miopenCreateConvolutionDescriptor(&convDesc));
    STATUS(miopenInitConvolutionDescriptor(convDesc,
                                           miopenConvolution,
                                           conv_desc[0], // pad h
                                           conv_desc[1], // pad w
                                           conv_desc[2], // stride h
                                           conv_desc[3], // stride w
                                           conv_desc[4], // dilations
                                           conv_desc[5]));

    miopen::FusionPlanDescriptor fp(miopenVerticalFusion, inputTensor);

    STATUS(miopenCreateOpConvForward(&fp, &convOp, convDesc, &convFilter));
    STATUS(miopenCreateOpBiasForward(&fp, &biasOp, &biasTensor));
    pgm = fp.GetProgramName(handle);
    krn = fp.GetKernelName(handle);
    alg = fp.GetAlgorithmName(handle);

    // Cleanup
    miopenDestroyConvolutionDescriptor(convDesc);
}