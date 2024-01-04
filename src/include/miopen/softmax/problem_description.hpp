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

#pragma once

#include <miopen/problem_description_base.hpp>


namespace miopen {

struct NetworkConfig;

namespace softmax {


struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription( const void* alpha_,
                        const void* beta_,
                        const TensorDescriptor& xDesc_,
                        const TensorDescriptor& yDesc_,
                        miopenSoftmaxAlgorithm_t algorithm_,
                        miopenSoftmaxMode_t mode_,
)
        : alpha(alpha_), beta(beta_), xDesc(xDesc_), yDesc(yDesc_), algorithm(algorithm_), mode(mode_)
    {
        CheckParamsAndThrowExceptionIfNeccessary();

        int n, c, h, w;
        std::tie(n, c, h, w) = tien<4>(yDesc.GetLengths());

        int vector_size = mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? c * h * w : c;

        calculatedNumBatches = vector_size < 256 ? nextPow2(256 / vector_size) : 1;
    }

    void CheckParamsAndThrowExceptionIfNeccessary()
    {
        if(x == nullptr || y == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
        }

        if(xDesc.GetType() != yDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(xDesc.GetLengths() != yDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }
    }

    NetworkConfig MakeNetworkConfig() const override;

    int GetNumBatch() const {return calculatedNumBatches;}

private:
    const void* alpha;
    const void* beta;
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    miopenSoftmaxAlgorithm_t algorithm;
    miopenSoftmaxMode_t mode;

    int calculatedNumBatches;
};

} // namespace softmax
} // namespace miopen
