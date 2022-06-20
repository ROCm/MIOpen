/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include "get_handle.hpp"
#include "test.hpp"

void chk_getop_bounds()
{
    miopen::TensorDescriptor inputTensor;
    miopen::TensorDescriptor convFilter;
    miopenConvolutionDescriptor_t convDesc{};
    miopenFusionOpDescriptor_t convoOp;
    std::vector<int> inputs      = {100, 32, 8, 8};
    std::vector<int> conv_filter = {64, 32, 5, 5};
    std::vector<int> conv_desc   = {0, 0, 1, 1, 1, 1};
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
                                           conv_desc[0],
                                           conv_desc[1],
                                           conv_desc[2],
                                           conv_desc[3],
                                           conv_desc[4],
                                           conv_desc[5]));

    miopen::FusionPlanDescriptor fp(miopenVerticalFusion, inputTensor);

    STATUS(miopenCreateOpConvForward(&fp, &convoOp, convDesc, &convFilter));
    miopenFusionOpDescriptor_t op1;
    miopenFusionOpDescriptor_t op2;
    miopenStatus_t miopenError;
    STATUS(miopenFusionPlanGetOp(&fp, 0, &op1));
    miopenError = miopenFusionPlanGetOp(&fp, 1, &op2);
    EXPECT(miopenError != miopenStatusSuccess);
}

int main()
{
    /*
     * This test ensures that the GetOp API call for the fusion plan exercises
     * bound checking on the incoming index
     */
    chk_getop_bounds();
}
