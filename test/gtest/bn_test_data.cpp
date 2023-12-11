/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include "bn_test_data.hpp"

std::vector<BNTestCase> Network1()
{
    // pyt_mlperf_resnet50v1.5
    return {
        {192, 1, 8, 8, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 1, 0},
        {16, 8, 128, 256, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 0},
        {16, 8, 128, 256, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 2048, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 2048, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 2048, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 256, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 256, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 256, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 256, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 256, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 256, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 256, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 256, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 256, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 512, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 512, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 512, 14, 14, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 512, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 512, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 512, 28, 28, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 512, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 512, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 512, 7, 7, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 64, 112, 112, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 64, 112, 112, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 64, 112, 112, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0},
        {64, 64, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::Backward, 0, 1},
        {64, 64, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::ForwardTraining, 1, 1},
        {64, 64, 56, 56, miopenBNSpatial, miopen::batchnorm::Direction::ForwardInference, 1, 0}};
}

