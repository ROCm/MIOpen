/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/sparse_softmax_cross_entropy_with_logits.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<uint64_t>& v)
{
    os << '{';
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

inline void LogCmdSparseSoftmaxCrossEntropyWithLogits(const miopenTensorDescriptor_t iDesc,
                                                      const bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(iDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "sparse_softmax_cross_entropy_with_logitsfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "sparse_softmax_cross_entropy_with_logitsfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "sparse_softmax_cross_entropy_with_logitsbfp16";
        }
        ss << " -Is ";
        ss << miopen::deref(iDesc).GetLengths();
        ss << " -F " << ((is_fwd) ? "1" : "2");
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenSparseSoftmaxCrossEntropyWithLogitsForward(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t inputDesc,
                                                 const void* input,
                                                 const miopenTensorDescriptor_t targetDesc,
                                                 const void* target,
                                                 const miopenTensorDescriptor_t outputDesc,
                                                 void* output,
                                                 const miopenTensorDescriptor_t backpropDesc,
                                                 void* backprop)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputDesc, input, targetDesc, target, outputDesc, output, backpropDesc, backprop);

    LogCmdSparseSoftmaxCrossEntropyWithLogits(inputDesc, true);

    return miopen::try_([&] {
        miopen::sparse_softmax_cross_entropy_with_logits::
            SparseSoftmaxCrossEntropyWithLogitsForward(miopen::deref(handle),
                                                       miopen::deref(inputDesc),
                                                       DataCast(input),
                                                       miopen::deref(targetDesc),
                                                       DataCast(target),
                                                       miopen::deref(outputDesc),
                                                       DataCast(output),
                                                       miopen::deref(backpropDesc),
                                                       DataCast(backprop));
    });
}

extern "C" miopenStatus_t
miopenSparseSoftmaxCrossEntropyWithLogitsBackward(miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t outputGradDesc,
                                                  const void* output_grad,
                                                  const miopenTensorDescriptor_t backpropDesc,
                                                  const void* backprop,
                                                  const miopenTensorDescriptor_t inputGradDesc,
                                                  void* input_grad)
{
    MIOPEN_LOG_FUNCTION(
        handle, outputGradDesc, output_grad, backpropDesc, backprop, inputGradDesc, input_grad);

    LogCmdSparseSoftmaxCrossEntropyWithLogits(inputGradDesc, false);

    return miopen::try_([&] {
        miopen::sparse_softmax_cross_entropy_with_logits::
            SparseSoftmaxCrossEntropyWithLogitsBackward(miopen::deref(handle),
                                                        miopen::deref(outputGradDesc),
                                                        DataCast(output_grad),
                                                        miopen::deref(backpropDesc),
                                                        DataCast(backprop),
                                                        miopen::deref(inputGradDesc),
                                                        DataCast(input_grad));
    });
}
